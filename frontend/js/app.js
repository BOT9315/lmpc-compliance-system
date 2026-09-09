const API_BASE_URL = "/api/v1";
let latestAuditResult = null;

// File Input & Drag-and-Drop Handler
const imageInput = document.getElementById('imageInput');
const dropZone = document.getElementById('dropZone');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const fileInfo = document.getElementById('fileInfo');
const scanBtn = document.getElementById('scanBtn');

imageInput.addEventListener('change', handleFileSelect);

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    }, false);
});

dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        handleFileSelect();
    }
});

function handleFileSelect() {
    const file = imageInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            previewContainer.style.display = 'block';
            fileInfo.innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            scanBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
}

// Upload & Scan Function
async function uploadImage() {
    const file = imageInput.files[0];
    if (!file) return alert("Please select a packaging image first.");

    const formData = new FormData();
    formData.append("file", file);

    setLoadingState(true);

    try {
        const response = await fetch(`${API_BASE_URL}/scan`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned HTTP ${response.status}`);
        }

        const data = await response.json();
        latestAuditResult = data;
        renderResults(data);
    } catch (error) {
        alert("Failed to process scan: " + error.message + ". Ensure 'python backend/main.py' is active.");
    } finally {
        setLoadingState(false);
    }
}

// UI State & Results Renderer
function setLoadingState(isLoading) {
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');

    if (isLoading) {
        scanBtn.disabled = true;
        btnText.innerText = "Scanning Label...";
        btnSpinner.style.display = 'inline-block';
    } else {
        scanBtn.disabled = false;
        btnText.innerText = "Scan Packaging Label";
        btnSpinner.style.display = 'none';
    }
}

function renderResults(data) {
    document.getElementById('resultsSection').style.display = 'block';

    // Status & Score Badge
    const badge = document.getElementById('statusBadge');
    badge.className = `metrics-banner ${data.overall_status}`;
    document.getElementById('scoreValue').innerText = `${data.compliance_score}%`;
    document.getElementById('statusTitle').innerText = data.overall_status;

    // Render Table Rows
    const tbody = document.querySelector("#findingsTable tbody");
    tbody.innerHTML = "";

    data.issues.forEach(issue => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${issue.field}</strong></td>
            <td><span class="status-pill ${issue.status}">${issue.status}</span></td>
            <td>${issue.severity}</td>
            <td><code>${issue.detected_value || 'N/A'}</code></td>
            <td><small>${issue.legal_reference}</small></td>
        `;
        tbody.appendChild(row);
    });

    // Display OCR Extracted Text
    document.getElementById('extractedTextDisplay').innerText = data.extracted_text || "No readable text extracted.";
}

// PDF Export Feature
async function exportPDF() {
    if (!latestAuditResult) return alert("No scan result available to export.");

    try {
        const response = await fetch(`${API_BASE_URL}/export-pdf`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(latestAuditResult)
        });

        if (!response.ok) throw new Error("PDF generation failed.");

        const blob = await response.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = `LMPC_Compliance_Report_${Date.now()}.pdf`;
        document.body.appendChild(link);
        link.click();
        link.remove();
    } catch (err) {

        alert("Error generating PDF: " + err.message);
    }
}