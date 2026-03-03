import React, { useState, useCallback } from 'react';
import { predict, predictExcel } from '../utils/api';

/**
 * ImageUpload - Drag-and-drop mammogram upload with manual feature entry fallback
 */
export default function ImageUpload({ onResult, onLoading }) {
    const [dragOver, setDragOver] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [showManual, setShowManual] = useState(false);
    const [features, setFeatures] = useState({});
    const [error, setError] = useState(null);

    // Feature names for manual entry
    const featureNames = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se",
        "concave points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
    ];

    // Sample values from the dataset (first malignant case)
    const sampleMalignant = {
        "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8,
        "area_mean": 1001, "smoothness_mean": 0.1184, "compactness_mean": 0.2776,
        "concavity_mean": 0.3001, "concave points_mean": 0.1471, "symmetry_mean": 0.2419,
        "fractal_dimension_mean": 0.07871, "radius_se": 1.095, "texture_se": 0.9053,
        "perimeter_se": 8.589, "area_se": 153.4, "smoothness_se": 0.006399,
        "compactness_se": 0.04904, "concavity_se": 0.05373, "concave points_se": 0.01587,
        "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193, "radius_worst": 25.38,
        "texture_worst": 17.33, "perimeter_worst": 184.6, "area_worst": 2019,
        "smoothness_worst": 0.1622, "compactness_worst": 0.6656, "concavity_worst": 0.7119,
        "concave points_worst": 0.2654, "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.1189,
    };

    const sampleBenign = {
        "radius_mean": 13.54, "texture_mean": 14.36, "perimeter_mean": 87.46,
        "area_mean": 566.3, "smoothness_mean": 0.09779, "compactness_mean": 0.08129,
        "concavity_mean": 0.06664, "concave points_mean": 0.04781, "symmetry_mean": 0.1885,
        "fractal_dimension_mean": 0.05766, "radius_se": 0.2699, "texture_se": 0.7886,
        "perimeter_se": 2.058, "area_se": 23.56, "smoothness_se": 0.008462,
        "compactness_se": 0.0146, "concavity_se": 0.02387, "concave points_se": 0.01315,
        "symmetry_se": 0.0198, "fractal_dimension_se": 0.0023, "radius_worst": 15.11,
        "texture_worst": 19.26, "perimeter_worst": 99.7, "area_worst": 711.2,
        "smoothness_worst": 0.144, "compactness_worst": 0.1773, "concavity_worst": 0.239,
        "concave points_worst": 0.1288, "symmetry_worst": 0.2977, "fractal_dimension_worst": 0.07259,
    };

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    }, []);

    const handleFile = (file) => {
        if (!file.type.startsWith('image/')) {
            setError('Please upload a valid image file (PNG, JPEG)');
            return;
        }
        setSelectedFile(file);
        setPreview(URL.createObjectURL(file));
        setError(null);
    };

    const handleSubmitImage = async () => {
        if (!selectedFile) return;
        setError(null);
        onLoading(true);
        try {
            const result = await predict(selectedFile);
            onResult(result);
        } catch (err) {
            setError(err.message);
        } finally {
            onLoading(false);
        }
    };

    const handleSubmitFeatures = async () => {
        // Validate all features are filled
        const missing = featureNames.filter(f => !features[f] && features[f] !== 0);
        if (missing.length > 0) {
            setError(`Missing ${missing.length} features. Fill all fields or use a sample.`);
            return;
        }
        setError(null);
        onLoading(true);
        try {
            const numericFeatures = {};
            featureNames.forEach(f => { numericFeatures[f] = parseFloat(features[f]); });
            const result = await predict(null, numericFeatures);
            onResult(result);
        } catch (err) {
            setError(err.message);
        } finally {
            onLoading(false);
        }
    };

    const loadSample = (sample) => {
        setFeatures({ ...sample });
    };

    const handleExcelUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const validExts = ['.csv', '.xlsx', '.xls'];
        const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        if (!validExts.includes(ext)) {
            setError('Please upload a .csv or .xlsx file');
            return;
        }
        setError(null);
        onLoading(true);
        try {
            const result = await predictExcel(file);
            onResult(result);
        } catch (err) {
            setError(err.message);
        } finally {
            onLoading(false);
        }
    };

    return (
        <div className="fade-in">
            <h2 className="section-title">
                <span className="section-title-icon">📤</span>
                Upload Mammogram or Enter Features
            </h2>

            {error && (
                <div style={{
                    background: 'var(--danger-bg)', border: '1px solid rgba(239,68,68,0.2)',
                    borderRadius: 'var(--radius-md)', padding: '12px 16px', marginBottom: '20px',
                    color: 'var(--danger)', fontSize: '14px', fontWeight: 500,
                }}>
                    ⚠️ {error}
                </div>
            )}

            {/* Image Upload Zone */}
            <div
                className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input').click()}
            >
                <input
                    id="file-input" type="file" accept="image/*"
                    style={{ display: 'none' }}
                    onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
                />

                {preview ? (
                    <div>
                        <img src={preview} alt="Preview" style={{
                            maxWidth: '300px', maxHeight: '200px', borderRadius: 'var(--radius-md)',
                            border: '2px solid var(--border)', marginBottom: '16px',
                        }} />
                        <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
                            {selectedFile?.name} ({(selectedFile?.size / 1024).toFixed(1)} KB)
                        </p>
                    </div>
                ) : (
                    <>
                        <span className="upload-icon">🔬</span>
                        <p className="upload-title">Drop mammogram image here</p>
                        <p className="upload-subtitle">Supports PNG, JPEG, DICOM formats</p>
                        <button className="upload-btn" type="button">
                            📁 Browse Files
                        </button>
                    </>
                )}
            </div>

            {preview && (
                <div style={{ textAlign: 'center', marginTop: '16px' }}>
                    <button className="btn btn-primary btn-lg" onClick={handleSubmitImage}>
                        🧬 Analyze Image
                    </button>
                    <button className="btn btn-outline" style={{ marginLeft: '12px' }}
                        onClick={() => { setSelectedFile(null); setPreview(null); }}>
                        ✕ Clear
                    </button>
                </div>
            )}

            <div className="divider">OR UPLOAD EXCEL/CSV WITH FEATURES</div>

            {/* Excel Upload Section */}
            <div className="glass-card" style={{ padding: '24px', marginBottom: '24px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
                    <div style={{ flex: 1 }}>
                        <h3 style={{ margin: '0 0 8px', fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)' }}>
                            📊 Upload Features from Excel/CSV
                        </h3>
                        <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)' }}>
                            Upload a .csv or .xlsx file with the 30 WBCD feature columns (radius_mean, texture_mean, ...)
                        </p>
                    </div>
                    <label className="btn btn-outline" style={{ cursor: 'pointer' }}>
                        📂 Choose Excel/CSV File
                        <input
                            type="file" accept=".csv,.xlsx,.xls"
                            style={{ display: 'none' }}
                            onChange={handleExcelUpload}
                        />
                    </label>
                </div>
            </div>

            <div className="divider">OR ENTER FEATURES MANUALLY</div>

            {/* Manual Feature Entry */}
            <div className="glass-card" style={{ padding: '24px' }}>
                <div style={{ display: 'flex', gap: '12px', marginBottom: '20px', flexWrap: 'wrap' }}>
                    <button className="btn btn-outline" onClick={() => setShowManual(!showManual)}>
                        {showManual ? '▼ Hide' : '▶ Show'} Feature Input Form
                    </button>
                    <button className="btn btn-outline" onClick={() => loadSample(sampleMalignant)}>
                        🔴 Load Malignant Sample
                    </button>
                    <button className="btn btn-outline" onClick={() => loadSample(sampleBenign)}>
                        🟢 Load Benign Sample
                    </button>
                </div>

                {showManual && (
                    <div className="form-grid fade-in">
                        {featureNames.map((name) => (
                            <div className="form-group" key={name}>
                                <label className="form-label">{name.replace(/_/g, ' ')}</label>
                                <input
                                    className="form-input" type="number" step="any"
                                    placeholder="0.0"
                                    value={features[name] ?? ''}
                                    onChange={(e) => setFeatures(prev => ({ ...prev, [name]: e.target.value }))}
                                />
                            </div>
                        ))}
                    </div>
                )}

                {Object.keys(features).length > 0 && (
                    <div style={{ textAlign: 'center', marginTop: '20px' }}>
                        <button className="btn btn-primary btn-lg" onClick={handleSubmitFeatures}>
                            🧬 Classify with Features ({Object.keys(features).length}/30)
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
