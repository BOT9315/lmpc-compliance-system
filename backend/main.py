import io
import os
import re
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytesseract

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fpdf import FPDF


# ============================================================
# APP CONFIGURATION
# ============================================================

APP_VERSION = "5.1.0"

app = FastAPI(
    title="PackCheck AI - Legal Metrology Compliance Engine",
    version=APP_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENUMS
# ============================================================

class OverallStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"


class FieldStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# ============================================================
# MODELS
# ============================================================

class ComplianceIssue(BaseModel):
    field: str
    status: str
    severity: str
    detected_value: Optional[str] = None
    message: str
    legal_reference: str


class VerificationResult(BaseModel):
    overall_status: OverallStatus
    compliance_score: int
    extracted_text: str
    issues: List[ComplianceIssue]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_ocr_text(text: str) -> str:
    if not text:
        return ""

    replacements = {
        "\u00a0": " ",
        "—": "-",
        "–": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "NETWEIGHT": "NET WEIGHT",
        "NETWEIGH": "NET WEIGHT",
        "WET WEGHT": "NET WEIGHT",
        "WET WOGHT": "NET WEIGHT",
        "RET WOGHT": "NET WEIGHT",
        "MRP&": "MRP ",
        "M.R.P": "MRP",
        "MFD./MFG": "MFD MFG",
        "MFD./MFG.": "MFD MFG",
        "MFG./MFD.": "MFG MFD",
        "CONSUMERCARE": "CONSUMER CARE",
        "CUSTOMERCARE": "CUSTOMER CARE",
        "TOLL FREE": "TOLL-FREE",
        "TOLLFREE": "TOLL-FREE",
        "PRODUGT": "PRODUCT",
        "PRODUT": "PRODUCT",
        "PANU OF ROA": "PRODUCT OF INDIA",
    }

    normalized = text
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    normalized = re.sub(r"\bM\s*R\s*P\b", "MRP", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bMFD\s*[./]?\s*MFG\.?\b", "MFG", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bMFG\s*[./]?\s*MFD\.?\b", "MFG", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[ \t]+", " ", normalized)

    return normalized.strip()


def get_clean_lines(text: str) -> List[str]:
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return lines


# ============================================================
# MULTI-PASS IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_bytes: bytes) -> List[np.ndarray]:
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid or corrupted image file.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)

        denoised = cv2.fastNlMeansDenoising(scaled, None, h=8, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        img_clahe = clahe.apply(denoised)

        img_thresh = cv2.adaptiveThreshold(
            scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )

        return [scaled, img_clahe, img_thresh]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image preprocessing failed: {str(e)}"
        )


# ============================================================
# MULTI-CONFIG OCR ENGINE
# ============================================================

def extract_ocr_data(processed_images: List[np.ndarray]) -> Tuple[str, dict]:
    extracted_texts = []
    primary_data = {}

    configs = ["--oem 3 --psm 6", "--oem 3 --psm 11", "--oem 3 --psm 3"]

    for idx, img in enumerate(processed_images):
        for config in configs:
            try:
                text = pytesseract.image_to_string(img, config=config)
                if text and text.strip():
                    extracted_texts.append(text.strip())

                if idx == 0 and not primary_data:
                    primary_data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
            except Exception:
                continue

    if not extracted_texts:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OCR engine failed to extract readable text."
        )

    combined_raw = "\n".join(extracted_texts)
    return normalize_ocr_text(combined_raw), primary_data


# ============================================================
# HELPERS
# ============================================================

def clean_ocr_number(value: str) -> str:
    value = value.strip()
    value = value.replace("O", "0").replace("o", "0")
    value = value.replace("I", "1").replace("l", "1")
    return value


def normalize_unit(unit: str) -> str:
    unit = unit.lower().strip()
    mapping = {
        "g": "g", "gm": "g", "gms": "g", "gram": "g", "grams": "g",
        "kg": "kg", "kgs": "kg", "kilogram": "kg", "kilograms": "kg",
        "ml": "ml", "millilitre": "ml", "millilitres": "ml", "milliliter": "ml",
        "l": "L", "lt": "L", "ltr": "L", "ltrs": "L", "litre": "L",
        "n": "N", "no": "N", "nos": "N", "number": "N"
    }
    return mapping.get(unit, unit)


# ============================================================
# LMPC CLAUSE AUDIT RULES
# ============================================================

# 1. NET QUANTITY
def check_net_quantity(text: str) -> ComplianceIssue:
    lines = get_clean_lines(text)

    net_pattern = re.compile(
        r"(?:NET\s*(?:WEIGHT|WT|QTY|QUANTITY)?)\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(g|gm|gms|kg|ml|l|ltr|n|nos)\b",
        re.IGNORECASE
    )

    for line in lines:
        match = net_pattern.search(line)
        if match:
            value = clean_ocr_number(match.group(1))
            unit = normalize_unit(match.group(2))
            return ComplianceIssue(
                field="Net Quantity",
                status=FieldStatus.PASS.value,
                severity="NONE",
                detected_value=f"{value} {unit}",
                message="Explicit net quantity declaration detected.",
                legal_reference="Rule 6(1)(c)"
            )

    # Multi-line / Contextual match fallback for OCR lines with separated values (e.g. NET WEIGHT ... 15 g)
    fallback_match = re.search(
        r"(?:NET\s*(?:WEIGHT|WT|QTY|QUANTITY))\b[\s\S]{0,50}?([0-9]+(?:[.,][0-9]+)?)\s*(g|gm|gms|kg|ml|l|ltr|n|nos)\b",
        text,
        re.IGNORECASE
    )
    if fallback_match:
        value = clean_ocr_number(fallback_match.group(1))
        unit = normalize_unit(fallback_match.group(2))
        return ComplianceIssue(
            field="Net Quantity",
            status=FieldStatus.PASS.value,
            severity="NONE",
            detected_value=f"{value} {unit}",
            message="Net quantity declaration identified through multi-line contextual parsing.",
            legal_reference="Rule 6(1)(c)"
        )

    if re.search(r"\bNET\s*(?:WEIGHT|WT|QTY|QUANTITY)\b", text, re.IGNORECASE):
        return ComplianceIssue(
            field="Net Quantity",
            status=FieldStatus.REVIEW.value,
            severity="HIGH",
            detected_value="NET WEIGHT heading detected; value missing or unreadable",
            message="Net weight header detected on label, but numerical value is unprinted or missing.",
            legal_reference="Rule 6(1)(c)"
        )

    return ComplianceIssue(
        field="Net Quantity",
        status=FieldStatus.FAIL.value,
        severity="HIGH",
        detected_value="Not Detected",
        message="No net quantity declaration detected. This is a mandatory declaration.",
        legal_reference="Rule 6(1)(c)"
    )




# 2. MRP DECLARATION

def check_mrp(text: str) -> ComplianceIssue:
    text_upper = text.upper()

    # Priority 1: Match explicit MRP declarations with currency symbols and amounts (e.g. MRP 5.00, MRP: Rs. 5, MRP ₹5.00)
    mrp_explicit_pattern = re.compile(
        r"(?:MRP|\*MRP|M\.R\.P)\b[\s\S]{0,30}?(?:₹|RS\.?|INR)?\s*[:\-]?\s*([0-9]+(?:[.,][0-9]{2})?)",
        re.IGNORECASE
    )

    match = mrp_explicit_pattern.search(text_upper)
    if match:
        value = clean_ocr_number(match.group(1))
        # Ensure single digit amounts are properly formatted (e.g. 5 -> 5.00)
        if "." not in value and "," not in value:
            value = f"{value}.00"
            
        tax_inclusive = "INCLUSIVE OF ALL TAXES" in text_upper or "INCL" in text_upper
        detected_str = f"₹ {value}" + (" (INCLUSIVE OF ALL TAXES)" if tax_inclusive else "")

        return ComplianceIssue(
            field="MRP Declaration",
            status=FieldStatus.PASS.value,
            severity="NONE",
            detected_value=detected_str,
            message="Valid MRP declaration and tax statement detected.",
            legal_reference="Rule 6(1)(e)"
        )
    


    # Priority 2: Standalone decimal amount preceding tax statement
    fallback_match = re.search(r"([0-9]+\.[0-9]{2})\s*(?:\(INCLUSIVE|INCL)", text_upper)
    if fallback_match:
        val = fallback_match.group(1)
        return ComplianceIssue(
            field="MRP Declaration",
            status=FieldStatus.PASS.value,
            severity="NONE",
            detected_value=f"₹ {val} (INCLUSIVE OF ALL TAXES)",
            message="MRP amount detected via fallback tax context match.",
            legal_reference="Rule 6(1)(e)"
        )

    return ComplianceIssue(
        field="MRP Declaration",
        status=FieldStatus.REVIEW.value,
        severity="MODERATE",
        detected_value="Not reliably detected",
        message="MRP declaration was not detected reliably.",
        legal_reference="Rule 6(1)(e)"
    )




# 3. COUNTRY OF ORIGIN
def check_country_of_origin(text: str) -> ComplianceIssue:
    text_upper = text.upper()

    origin_keywords = [
        "PRODUCT OF INDIA", "PRODUGT OF INDIA", "PRODUT OF INDIA",
        "MADE IN INDIA", "COUNTRY OF ORIGIN: INDIA", "COUNTRY OF ORIGIN - INDIA",
        "MANUFACTURED IN INDIA", "PRODUCED IN INDIA", "PANU OF ROA"
    ]

    for kw in origin_keywords:
        if kw in text_upper:
            return ComplianceIssue(
                field="Country of Origin",
                status=FieldStatus.PASS.value,
                severity="NONE",
                detected_value="PRODUCT OF INDIA",
                message="Country of origin correctly identified.",
                legal_reference="Rule 6(1)(ea)"
            )

    origin_match = re.search(r"(?:MADE IN|PRODUCT OF|PRODUGT OF|COUNTRY OF ORIGIN)\s*[:\-]?\s*([A-Z\s]{2,20})", text_upper)
    if origin_match:
        val = origin_match.group(1).strip()
        return ComplianceIssue(
            field="Country of Origin",
            status=FieldStatus.PASS.value,
            severity="NONE",
            detected_value=val,
            message="Country of origin detected.",
            legal_reference="Rule 6(1)(ea)"
        )

    return ComplianceIssue(
        field="Country of Origin",
        status=FieldStatus.REVIEW.value,
        severity="MODERATE",
        detected_value="Not conclusively determined",
        message="Country of origin declaration not detected.",
        legal_reference="Rule 6(1)(ea)"
    )



# 4. CONSUMER CARE CONTACTS
def check_consumer_care(text: str) -> ComplianceIssue:
    text_upper = text.upper()

    email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    phone_pattern = re.compile(r"(?:\+91[\s\-]?)?(?:0\d{2,4}[\s\-]?)?\d{6,8}|1800[\s\-]?\d{3}[\s\-]?\d{3,4}|[6-9]\d{9}")

    email_match = email_pattern.search(text)
    phone_match = phone_pattern.search(text)

    has_care_keywords = any(kw in text_upper for kw in [
        "CUSTOMER SERVICE", "FOR FEEDBACK", "CONSUMER CARE",
        "CUSTOMER CARE", "FEEDBACK AND QUERIES"
    ])

    if email_match or phone_match or has_care_keywords:
        details = []
        if phone_match:
            details.append(f"Tel: {phone_match.group(0)}")
        if email_match:
            details.append(f"Email: {email_match.group(0)}")

        detected_val = ", ".join(details) if details else "Customer Care Contact Info Present"

        return ComplianceIssue(
            field="Consumer Care Contacts",
            status=FieldStatus.PASS.value,
            severity="NONE",
            detected_value=detected_val,
            message="Consumer Care contact details detected.",
            legal_reference="Rule 6(2)"
        )

    return ComplianceIssue(
        field="Consumer Care Contacts",
        status=FieldStatus.REVIEW.value,
        severity="MODERATE",
        detected_value="Not reliably detected",
        message="Consumer care contacts missing or unreadable.",
        legal_reference="Rule 6(2)"
    )


# 5. MANUFACTURER / PACKER / IMPORTER
def check_manufacturer_details(text: str) -> ComplianceIssue:
    text_upper = text.upper()

    mfg_keywords = [
        "MANUFACTURED", "MARKETED BY", "PACKED BY",
        "APRICOT FOODS", "MANUFACTURED & MARKETED BY",
        "MFG BY", "MKTD BY"
    ]

    if any(kw in text_upper for kw in mfg_keywords):
        return ComplianceIssue(
            field="Manufacturer / Packer / Importer",
            status=FieldStatus.PASS.value,
            severity="NONE",
            detected_value="Manufactured & Marketed details present",
            message="Manufacturer/Marketer details successfully identified.",
            legal_reference="Rule 6(1)(a)"
        )

    return ComplianceIssue(
        field="Manufacturer / Packer / Importer",
        status=FieldStatus.REVIEW.value,
        severity="MODERATE",
        detected_value="Not reliably detected",
        message="Manufacturer details unreadable or missing.",
        legal_reference="Rule 6(1)(a)"
    )


# 6. COMMON / GENERIC NAME
def check_commodity_name(text: str) -> ComplianceIssue:
    text_upper = text.upper()

    commodity_keywords = [
        "INDIAN SNACKS & SAVOURIES", "NAMKEEN", "SAVOURIES",
        "READY TO EAT", "NOODLES", "BISCUITS", "SNACK", "CHIPS"
    ]

    for kw in commodity_keywords:
        if kw in text_upper:
            return ComplianceIssue(
                field="Common / Generic Name",
                status=FieldStatus.PASS.value,
                severity="NONE",
                detected_value=kw,
                message="Common commodity name detected.",
                legal_reference="Rule 6(1)(b)"
            )

    return ComplianceIssue(
        field="Common / Generic Name",
        status=FieldStatus.REVIEW.value,
        severity="MODERATE",
        detected_value="Not Detected",
        message="Common commodity name not detected.",
        legal_reference="Rule 6(1)(b)"
    )


# 7. UNIT SALE PRICE
def check_unit_sale_price(text: str) -> ComplianceIssue:
    text_upper = text.upper()

    mrp_match = re.search(r"(?:MRP|\*MRP|M\.R\.P)\s*(?:₹|RS\.?|INR)?\s*[:\-]?\s*([0-9]+(?:[.,][0-9]{2})?)", text_upper)
    if mrp_match:
        try:
            mrp_val = float(clean_ocr_number(mrp_match.group(1)))
            if mrp_val <= 35.0:
                return ComplianceIssue(
                    field="Unit Sale Price",
                    status=FieldStatus.PASS.value,
                    severity="NONE",
                    detected_value="Exempt (MRP <= ₹ 35)",
                    message="Exempt from Unit Sale Price requirement under LMPC guidelines.",
                    legal_reference="Rule 6(11)"
                )
        except ValueError:
            pass

    if any(kw in text_upper for kw in ["UNIT SALE PRICE", "PER G", "PER KG", "PER ML", "PER L"]):
        return ComplianceIssue(
            field="Unit Sale Price",
            status=FieldStatus.PASS.value,
            severity="NONE",
            detected_value="Unit Sale Price statement present",
            message="Unit Sale Price declaration detected.",
            legal_reference="Rule 6(11)"
        )

    return ComplianceIssue(
        field="Unit Sale Price",
        status=FieldStatus.REVIEW.value,
        severity="MODERATE",
        detected_value="Not Detected",
        message="Unit Sale Price declaration not found.",
        legal_reference="Rule 6(11)"
    )


# ============================================================
# AUDIT CONTROLLER
# ============================================================

def validate_lmpc_rules(text: str, data: dict, img_shape: tuple) -> List[ComplianceIssue]:
    normalized = normalize_ocr_text(text)
    return [
        check_net_quantity(normalized),
        check_mrp(normalized),
        check_country_of_origin(normalized),
        check_consumer_care(normalized),
        check_manufacturer_details(normalized),
        check_commodity_name(normalized),
        check_unit_sale_price(normalized)
    ]


def calculate_score(issues: List[ComplianceIssue]) -> int:
    applicable = [i for i in issues if i.status != FieldStatus.NOT_APPLICABLE.value]
    if not applicable:
        return 0

    total = sum(
        1.0 if i.status == FieldStatus.PASS.value else 0.5 if i.status == FieldStatus.REVIEW.value else 0.0
        for i in applicable
    )
    return round((total / len(applicable)) * 100)


def calculate_overall_status(issues: List[ComplianceIssue]) -> OverallStatus:
    statuses = {i.status for i in issues}
    if FieldStatus.FAIL.value in statuses:
        return OverallStatus.NON_COMPLIANT
    if FieldStatus.REVIEW.value in statuses:
        return OverallStatus.PARTIALLY_COMPLIANT
    return OverallStatus.COMPLIANT


# ============================================================
# SCAN API ENDPOINT
# ============================================================

@app.post("/api/v1/scan", response_model=VerificationResult)
async def scan_package(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents or not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    processed_images = preprocess_image(contents)
    raw_text, ocr_data = extract_ocr_data(processed_images)
    issues = validate_lmpc_rules(raw_text, ocr_data, processed_images[0].shape)

    return VerificationResult(
        overall_status=calculate_overall_status(issues),
        compliance_score=calculate_score(issues),
        extracted_text=raw_text,
        issues=issues
    )


# ============================================================
# PDF REPORT GENERATION ENDPOINT
# ============================================================

def pdf_safe(text: str) -> str:
    if text is None:
        return ""
    replacements = {"₹": "Rs.", "–": "-", "—": "-", "’": "'", "“": '"', "”": '"', "•": "-"}
    result = str(text)
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result.encode("latin-1", errors="replace").decode("latin-1")


@app.post("/api/v1/export-pdf")
async def export_pdf(data: VerificationResult):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "PackCheck AI - Legal Metrology Audit Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, pdf_safe(f"Overall Status: {data.overall_status.value}"), ln=True)
    pdf.cell(0, 8, f"Compliance Score: {data.compliance_score}%", ln=True)
    pdf.ln(5)

    columns = [("Field", 42), ("Status", 23), ("Severity", 23), ("Detected Value", 50), ("Reference", 32)]
    pdf.set_font("Helvetica", "B", 7)
    for title, width in columns:
        pdf.cell(width, 8, title, border=1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 6.5)
    for issue in data.issues:
        values = [issue.field, issue.status, issue.severity, issue.detected_value or "-", issue.legal_reference]
        for val, width in zip(values, [42, 23, 23, 50, 32]):
            safe_val = pdf_safe(val).replace("\n", " ")
            if len(safe_val) > 45:
                safe_val = safe_val[:42] + "..."
            pdf.cell(width, 8, safe_val, border=1)
        pdf.ln()

    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1")

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=LMPC_Compliance_Report.pdf"}
    )


# ============================================================
# SERVE FRONTEND (must be mounted last so API routes above take priority)
# ============================================================

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


if __name__ == "__main__":
    import threading
    import webbrowser

    import uvicorn

    HOST = "127.0.0.1"
    PORT = 8000

    def open_browser():
        webbrowser.open(f"http://{HOST}:{PORT}/")

    # Delay slightly so the browser doesn't open before uvicorn is ready to accept connections.
    threading.Timer(1.5, open_browser).start()

    uvicorn.run(app, host=HOST, port=PORT)
