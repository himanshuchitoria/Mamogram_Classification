/**
 * AI4BCancer - API Client
 * Handles all communication with the FastAPI backend
 */

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080').replace(/\/$/, '');

/**
 * Check backend health status
 */
export async function checkHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Backend unavailable');
    return res.json();
}

/**
 * Get the list of 30 features expected by the model
 */
export async function getFeatureNames() {
    const res = await fetch(`${API_BASE}/features`);
    if (!res.ok) throw new Error('Failed to fetch feature names');
    return res.json();
}

/**
 * Submit prediction request — image upload or manual features
 */
export async function predict(file = null, features = null) {
    const formData = new FormData();

    if (file) {
        formData.append('file', file);
    } else if (features) {
        formData.append('features_json', JSON.stringify(features));
    } else {
        throw new Error('Provide either a file or feature values');
    }

    const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Prediction failed');
    }

    return res.json();
}

/**
 * Submit prediction from an Excel/CSV file with 30 WBCD features
 */
export async function predictExcel(file) {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/predict-excel`, {
        method: 'POST',
        body: formData,
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Excel prediction failed');
    }

    return res.json();
}

/**
 * Generate BI-RADS PDF report
 */
export async function generateReport(data) {
    const formData = new FormData();
    Object.entries(data).forEach(([key, value]) => {
        if (value !== null && value !== undefined) {
            formData.append(key, typeof value === 'object' ? JSON.stringify(value) : String(value));
        }
    });

    const res = await fetch(`${API_BASE}/generate-report`, {
        method: 'POST',
        body: formData,
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Report generation failed');
    }

    // Download the PDF
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AI4BCancer_Report_${data.patient_id || 'unknown'}.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Find nearby cancer hospitals
 */
export async function findHospitals(lat, lon, radius = 10000) {
    const params = new URLSearchParams({ lat, lon, radius });
    const res = await fetch(`${API_BASE}/find-hospitals?${params}`);

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Hospital search failed');
    }

    return res.json();
}
