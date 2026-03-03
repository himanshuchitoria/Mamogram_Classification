import React, { useState } from 'react';
import { generateReport } from '../utils/api';

/**
 * ReportForm - Patient demographics form + PDF report generation
 */
export default function ReportForm({ result }) {
    const [patientId, setPatientId] = useState('');
    const [patientName, setPatientName] = useState('');
    const [patientDob, setPatientDob] = useState('');
    const [clinicalNotes, setClinicalNotes] = useState('');
    const [generating, setGenerating] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(false);

    const handleGenerate = async () => {
        if (!patientId.trim() || !patientName.trim()) {
            setError('Patient ID and Name are required');
            return;
        }

        setError(null);
        setSuccess(false);
        setGenerating(true);

        try {
            await generateReport({
                patient_id: patientId,
                patient_name: patientName,
                patient_dob: patientDob,
                clinical_notes: clinicalNotes,
                prediction: result?.prediction || 'Benign',
                confidence: result?.confidence || 0.95,
                birads_category: result?.birads_category,
                feature_importance: result?.xai?.lime?.feature_importance
                    ? JSON.stringify(result.xai.lime.feature_importance)
                    : null,
                original_image: result?.original_image || null,
                xai_plot: result?.xai?.lime?.plot_base64 || null,
            });
            setSuccess(true);
        } catch (err) {
            setError(err.message);
        } finally {
            setGenerating(false);
        }
    };

    return (
        <div className="fade-in">
            <h2 className="section-title">
                <span className="section-title-icon">📋</span>
                Generate Official BI-RADS Report
            </h2>

            {!result && (
                <div className="glass-card" style={{ padding: '24px', textAlign: 'center' }}>
                    <p style={{ color: 'var(--text-muted)', fontSize: '15px' }}>
                        ⚠️ Run a classification first to generate a report with results.
                    </p>
                </div>
            )}

            <div className="glass-card" style={{ padding: '24px' }}>
                <div className="result-label" style={{ marginBottom: '16px' }}>Patient Demographics</div>

                <div className="form-grid">
                    <div className="form-group">
                        <label className="form-label">Patient ID *</label>
                        <input className="form-input" type="text" placeholder="e.g. PAT-2024-001"
                            value={patientId} onChange={(e) => setPatientId(e.target.value)} />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Patient Name *</label>
                        <input className="form-input" type="text" placeholder="Full Name"
                            value={patientName} onChange={(e) => setPatientName(e.target.value)} />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Date of Birth</label>
                        <input className="form-input" type="date"
                            value={patientDob} onChange={(e) => setPatientDob(e.target.value)} />
                    </div>
                </div>

                <div className="form-group" style={{ marginTop: '16px' }}>
                    <label className="form-label">Clinical Notes / Indications</label>
                    <textarea className="form-input" placeholder="Enter clinical history, symptoms, or indications..."
                        value={clinicalNotes} onChange={(e) => setClinicalNotes(e.target.value)}
                        rows={3} />
                </div>

                {/* Classification Summary */}
                {result && (
                    <div style={{
                        marginTop: '20px', padding: '16px', borderRadius: 'var(--radius-md)',
                        background: result.prediction === 'Malignant' ? 'var(--danger-bg)' : 'var(--success-bg)',
                        border: `1px solid ${result.prediction === 'Malignant' ? 'rgba(239,68,68,0.2)' : 'rgba(16,185,129,0.2)'}`,
                    }}>
                        <div className="result-label" style={{ marginBottom: '8px' }}>Classification Summary (will be included in report)</div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', fontSize: '14px' }}>
                            <div>
                                <strong>Prediction:</strong>{' '}
                                <span style={{ color: result.prediction === 'Malignant' ? 'var(--danger)' : 'var(--success)' }}>
                                    {result.prediction}
                                </span>
                            </div>
                            <div>
                                <strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%
                            </div>
                            <div>
                                <strong>BI-RADS:</strong> Category {result.birads_category}
                            </div>
                        </div>
                    </div>
                )}

                {error && (
                    <div style={{
                        marginTop: '16px', padding: '12px', borderRadius: 'var(--radius-md)',
                        background: 'var(--danger-bg)', color: 'var(--danger)', fontSize: '14px',
                    }}>
                        ⚠️ {error}
                    </div>
                )}

                {success && (
                    <div style={{
                        marginTop: '16px', padding: '12px', borderRadius: 'var(--radius-md)',
                        background: 'var(--success-bg)', color: 'var(--success)', fontSize: '14px',
                    }}>
                        ✅ PDF report generated and downloaded successfully!
                    </div>
                )}

                <div style={{ marginTop: '24px', textAlign: 'center' }}>
                    <button
                        className="btn btn-success btn-lg"
                        onClick={handleGenerate}
                        disabled={generating || !result}
                    >
                        {generating ? (
                            <><span className="spinner" style={{ width: '18px', height: '18px', margin: 0, borderWidth: '2px' }} /> Generating...</>
                        ) : (
                            <>📄 Generate Official PDF Report</>
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
}
