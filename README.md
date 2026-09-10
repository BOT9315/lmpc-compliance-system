# PackCheck AI — LMPC Compliance Verifier

Automated packaging label auditor that checks scanned label images against
India's **Legal Metrology (Packaged Commodities) Rules, 2011** and produces a
compliance report with a downloadable PDF.

Upload a photo of a packaging label, and the app runs it through OCR, checks
it against the core mandatory declarations, and returns a pass/fail/review
verdict for each field along with an overall compliance score.

## Features

- **Drag-and-drop image upload** with live preview
- **Multi-pass OCR pipeline** — the image is preprocessed three different ways
  (grayscale, CLAHE-enhanced, adaptive threshold) and run through Tesseract
  with multiple page-segmentation modes to maximize text recovery from
  low-quality label photos
- **OCR text normalization** to correct common misreads (e.g. `NETWEIGHT` →
  `NET WEIGHT`, `M.R.P` → `MRP`)
- **Rule-by-rule compliance audit** against 7 mandatory label declarations:
  1. Net Quantity — Rule 6(1)(c)
  2. MRP Declaration — Rule 6(1)(e)
  3. Country of Origin — Rule 6(1)(ea)
  4. Consumer Care Contacts — Rule 6(2)
  5. Manufacturer / Packer / Importer — Rule 6(1)(a)
  6. Common / Generic Name — Rule 6(1)(b)
  7. Unit Sale Price — Rule 6(11)
- **Compliance scoring** — an overall `COMPLIANT` / `PARTIALLY_COMPLIANT` /
  `NON_COMPLIANT` verdict with a 0–100% score
- **PDF report export** — generates a downloadable audit report summarizing
  all findings

## Tech Stack

| Layer    | Technology |
|----------|------------|
| Backend  | FastAPI, OpenCV, Tesseract OCR (`pytesseract`), `fpdf2` |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Runtime  | Python 3.11, Docker |

## Project Structure

```
lmpc-compliance-system/
├── backend/
│   └── main.py           # FastAPI app: OCR pipeline, compliance rules, PDF export
├── frontend/
│   ├── index.html
│   ├── css/
│   └── js/
│       └── app.js        # Upload, scan, and results rendering logic
├── requirements.txt
└── Dockerfile
```

## Getting Started

### Prerequisites

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and
  available on your `PATH`
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - Windows: install from the [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki)

### Option 1 — Run locally

```bash
# Clone the repo
git clone https://github.com/BOT9315/lmpc-compliance-system.git
cd lmpc-compliance-system

# Install dependencies
pip install -r requirements.txt

# Run the app
python backend/main.py
```

This starts the server at `http://127.0.0.1:8000` and automatically opens it
in your default browser.

Alternatively, for auto-reload during development:

```bash
cd backend
uvicorn main:app --reload
```

### Running in VS Code

Open the repo folder in VS Code, open the integrated terminal (`` Ctrl+` ``),
and run the commands below for your OS.

**Windows (PowerShell / VS Code default terminal):**

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python backend/main.py
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python backend/main.py
```

This installs dependencies into an isolated virtual environment, starts the
server at `http://127.0.0.1:8000`, and automatically opens it in your default
browser.

> **Tesseract OCR must also be installed separately** (it's not a pip
> package) — see [Prerequisites](#prerequisites) above.

If VS Code prompts you to select `venv` as the workspace's Python
interpreter, click **Yes**.

To run with auto-reload during development instead:

```bash
cd backend
uvicorn main:app --reload
```

### Option 2 — Run with Docker

```bash
docker build -t packcheck-ai .
docker run -p 8000:8000 packcheck-ai
```

Then open `http://localhost:8000` in your browser.

## API Reference

### `POST /api/v1/scan`

Uploads a packaging image and returns a full compliance audit.

**Request:** `multipart/form-data` with a `file` field (image).

**Response:**

```json
{
  "overall_status": "PARTIALLY_COMPLIANT",
  "compliance_score": 71,
  "extracted_text": "...",
  "issues": [
    {
      "field": "Net Quantity",
      "status": "PASS",
      "severity": "NONE",
      "detected_value": "200 g",
      "message": "Explicit net quantity declaration detected.",
      "legal_reference": "Rule 6(1)(c)"
    }
  ]
}
```

### `POST /api/v1/export-pdf`

Accepts a `VerificationResult` JSON body (the response from `/scan`) and
returns a downloadable PDF audit report.

## Notes & Limitations

- OCR accuracy depends heavily on photo quality — low light, blur, or glare
  on the packaging can cause fields to be missed or misread.
- The rule checks currently rely on keyword and pattern matching rather than
  a full NLP/label-layout model, so unusual label phrasing may need manual
  review even when the information is technically present.
- This tool is intended as a first-pass screening aid, not a substitute for
  a full legal metrology compliance review.

## License

No license file is currently included in this repository. Add one (e.g. MIT)
if you intend to distribute or accept contributions.
