import React from 'react';

/**
 * ResultsView - Displays classification results, confidence, BI-RADS, and XAI plots
 */
export default function ResultsView({ result, onGoToHospitals }) {
    if (!result) return null;

    const {
        prediction, confidence, malignant_probability, benign_probability,
        birads_category, birads_label, xai, original_image, heatmap_image,
        processing_notes = []
    } = result;

    const isMalignant = prediction === 'Malignant';
    const confidencePercent = (confidence * 100).toFixed(1);

    // LIME feature importance data
    const limeData = xai?.lime?.feature_importance || [];
    const limePlot = xai?.lime?.plot_base64 || '';
    const topFeatures = xai?.lime?.top_features || [];

    // Find max absolute weight for scaling bars
    const maxWeight = limeData.length > 0
        ? Math.max(...limeData.map(f => Math.abs(f.weight)))
        : 1;

    return (
        <div className="fade-in">
            <h2 className="section-title">
                <span className="section-title-icon">📊</span>
                Classification Results
            </h2>

            {/* Pipeline Routing Info */}
            {processing_notes.length > 0 && (
                <div style={{
                    padding: '12px 16px', borderRadius: 'var(--radius-md)',
                    background: 'rgba(56, 189, 248, 0.1)', color: 'var(--primary)',
                    fontSize: '14px', marginBottom: '24px', borderLeft: '4px solid var(--primary)'
                }}>
                    <strong>Pipeline Info:</strong> {processing_notes.join(', ')}
                </div>
            )}

            {/* Main Results Grid */}
            <div className="results-grid">
                {/* Prediction Card */}
                <div className="glass-card result-card stagger-1 fade-in">
                    <div className="result-label">AI Classification</div>
                    <div className={`result-value ${isMalignant ? 'malignant' : 'benign'}`}>
                        {isMalignant ? '⚠️' : '✅'} {prediction}
                    </div>
                    <div style={{ marginTop: '12px', fontSize: '14px', color: 'var(--text-secondary)' }}>
                        Malignant: {(malignant_probability * 100).toFixed(1)}% &nbsp;|&nbsp;
                        Benign: {(benign_probability * 100).toFixed(1)}%
                    </div>
                </div>

                {/* Confidence Gauge */}
                <div className="glass-card result-card stagger-2 fade-in" style={{ textAlign: 'center' }}>
                    <div className="result-label">Confidence Score</div>
                    <div className="confidence-gauge" style={{ '--confidence': confidencePercent }}>
                        <span className="confidence-text">{confidencePercent}%</span>
                    </div>
                </div>

                {/* BI-RADS Category */}
                <div className="glass-card result-card stagger-3 fade-in">
                    <div className="result-label">BI-RADS Assessment</div>
                    <div className={`birads-badge birads-${birads_category}`} style={{ marginTop: '8px' }}>
                        {birads_category <= 2 ? '🟢' : birads_category === 3 ? '🟡' : '🔴'}
                        &nbsp; {birads_label}
                    </div>
                    <div style={{ marginTop: '12px', fontSize: '13px', color: 'var(--text-secondary)' }}>
                        {birads_category <= 2 && 'Routine screening recommended'}
                        {birads_category === 3 && 'Short-interval follow-up recommended'}
                        {birads_category === 4 && 'Biopsy recommended'}
                        {birads_category >= 5 && 'Immediate action recommended'}
                    </div>
                </div>

                {/* Top Contributing Features */}
                <div className="glass-card result-card stagger-4 fade-in">
                    <div className="result-label">Top Contributing Features</div>
                    {topFeatures.length > 0 ? (
                        <div style={{ marginTop: '8px' }}>
                            {topFeatures.map((feat, i) => (
                                <div key={i} style={{
                                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                    padding: '6px 0', borderBottom: i < topFeatures.length - 1 ? '1px solid var(--border)' : 'none',
                                }}>
                                    <span style={{ fontSize: '13px', color: 'var(--text-primary)' }}>
                                        {feat.feature}
                                    </span>
                                    <span style={{
                                        fontSize: '12px', fontWeight: 600,
                                        color: feat.direction === 'Malignant' ? 'var(--danger)' : 'var(--success)',
                                    }}>
                                        {feat.weight > 0 ? '+' : ''}{feat.weight.toFixed(4)}
                                    </span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                            Feature importance data not available
                        </p>
                    )}
                </div>
            </div>

            {/* ── Find Hospitals CTA ── */}
            {onGoToHospitals && (
                <div className="glass-card fade-in" style={{
                    padding: '20px 24px', marginTop: '20px', display: 'flex',
                    alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px',
                    border: `1px solid ${isMalignant ? 'rgba(239,68,68,0.25)' : 'rgba(16,185,129,0.25)'}`,
                    background: isMalignant ? 'rgba(239,68,68,0.05)' : 'rgba(16,185,129,0.05)',
                }}>
                    <div>
                        <div style={{ fontWeight: 700, fontSize: '15px', color: 'var(--text-primary)', marginBottom: '4px' }}>
                            {isMalignant ? '🎗️ Oncology Centers Recommended' : '🏥 Follow-up Centers Near You'}
                        </div>
                        <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                            {isMalignant
                                ? 'Please consult an oncology specialist immediately. Find cancer centres near your location.'
                                : 'Routine follow-up is recommended. Explore nearby breast care facilities.'}
                        </div>
                    </div>
                    <button onClick={onGoToHospitals} className="btn btn-primary" style={{
                        background: isMalignant ? 'var(--danger)' : 'var(--success)',
                        border: 'none', whiteSpace: 'nowrap', minWidth: '200px',
                    }}>
                        🏥 Find Nearby Hospitals →
                    </button>
                </div>
            )}

            {/* Image Comparison (if image was uploaded) */}
            {(original_image || heatmap_image) && (
                <div className="glass-card fade-in" style={{ padding: '24px', marginTop: '24px' }}>
                    <div className="result-label" style={{ marginBottom: '16px' }}>Visual Analysis</div>
                    <div className="image-comparison">
                        {original_image && (
                            <div>
                                <img src={`data:image/png;base64,${original_image}`} alt="Original mammogram" />
                                <p className="image-comparison-label">Original Image</p>
                            </div>
                        )}
                        {heatmap_image && (
                            <div>
                                <img src={`data:image/png;base64,${heatmap_image}`} alt="Heatmap overlay" />
                                <p className="image-comparison-label">Attention Heatmap</p>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* LIME Feature Importance Chart */}
            {limePlot && (
                <div className="glass-card fade-in" style={{ padding: '24px', marginTop: '24px' }}>
                    <div className="result-label" style={{ marginBottom: '16px' }}>
                        LIME Feature Importance Analysis
                    </div>
                    <img
                        className="xai-image"
                        src={`data:image/png;base64,${limePlot}`}
                        alt="LIME Feature Importance"
                    />
                </div>
            )}

            {/* Feature Importance Bars (custom rendering) */}
            {limeData.length > 0 && (
                <div className="glass-card fade-in" style={{ padding: '24px', marginTop: '24px' }}>
                    <div className="result-label" style={{ marginBottom: '16px' }}>
                        Feature Contribution Breakdown ({limeData.length} features)
                    </div>
                    {limeData.slice(0, 15).map((feat, i) => (
                        <div className="feature-bar" key={i}>
                            <span className="feature-name">{feat.feature}</span>
                            <div className="feature-bar-track">
                                <div
                                    className={`feature-bar-fill ${feat.weight > 0 ? 'positive' : 'negative'}`}
                                    style={{ width: `${(Math.abs(feat.weight) / maxWeight) * 100}%` }}
                                />
                            </div>
                            <span className="feature-weight">
                                {feat.weight > 0 ? '+' : ''}{feat.weight.toFixed(4)}
                            </span>
                        </div>
                    ))}
                </div>
            )}

            {/* Extracted Features Table (shown when image was analyzed) */}
            {result.features_used && typeof result.features_used === 'object' && Object.keys(result.features_used).length > 5 && (
                <div className="glass-card fade-in" style={{ padding: '24px', marginTop: '24px' }}>
                    <div className="result-label" style={{ marginBottom: '16px' }}>
                        📋 Extracted WBCD Features ({Object.keys(result.features_used).length} features)
                    </div>
                    <div style={{
                        display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
                        gap: '8px', maxHeight: '400px', overflowY: 'auto',
                        padding: '8px', borderRadius: 'var(--radius-md)', background: 'var(--bg-input)',
                    }}>
                        {Object.entries(result.features_used)
                            .filter(([key]) => key.includes('_'))
                            .map(([key, val]) => (
                                <div key={key} style={{
                                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                    padding: '6px 12px', borderRadius: '6px',
                                    background: 'rgba(255,255,255,0.03)', borderBottom: '1px solid var(--border)',
                                }}>
                                    <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 500 }}>
                                        {key.replace(/_/g, ' ')}
                                    </span>
                                    <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-accent)', fontFamily: 'monospace' }}>
                                        {typeof val === 'number' ? val.toFixed(4) : val}
                                    </span>
                                </div>
                            ))}
                    </div>
                </div>
            )}
        </div>
    );
}
