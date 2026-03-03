import React, { useState, useEffect } from 'react';
import ImageUpload from './components/ImageUpload';
import ResultsView from './components/ResultsView';
import ReportForm from './components/ReportForm';
import HospitalLocator from './components/HospitalLocator';
import { checkHealth } from './utils/api';

/**
 * AI4BCancer - Main Application Shell
 * Radiologist Dashboard with tabbed navigation
 */
export default function App() {
    const [activeTab, setActiveTab] = useState('upload');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [backendStatus, setBackendStatus] = useState('checking');

    // Check backend health on mount
    useEffect(() => {
        const check = async () => {
            try {
                const health = await checkHealth();
                const isReady = health.hybrid_model_loaded && health.vision_model_loaded;
                setBackendStatus(isReady ? 'online' : 'no-models');
            } catch {
                setBackendStatus('offline');
            }
        };
        check();
        const interval = setInterval(check, 30000); // Re-check every 30s
        return () => clearInterval(interval);
    }, []);

    // Auto-switch to results tab when prediction completes
    const handleResult = (data) => {
        setResult(data);
        setActiveTab('results');
    };

    const tabs = [
        { id: 'upload', label: '📤 Upload & Classify', icon: '📤' },
        { id: 'results', label: '📊 Results', icon: '📊' },
        { id: 'report', label: '📋 Report', icon: '📋' },
        { id: 'hospitals', label: '🏥 Hospitals', icon: '🏥' },
    ];

    return (
        <div>
            {/* ===== HEADER ===== */}
            <header className="header">
                <div className="header-logo">
                    <div className="header-logo-icon">🧬</div>
                    <div>
                        <div className="header-title">AI4BCancer</div>
                        <div className="header-subtitle">Radiologist Classification Dashboard</div>
                    </div>
                </div>
                <div className="header-status">
                    <span className={`status-dot ${backendStatus === 'online' ? 'online' : 'offline'}`} />
                    {backendStatus === 'online' && 'AI Models Ready'}
                    {backendStatus === 'offline' && 'Backend Offline'}
                    {backendStatus === 'no-models' && 'Models Not Trained'}
                    {backendStatus === 'checking' && 'Connecting...'}
                </div>
            </header>

            {/* ===== NAVIGATION TABS ===== */}
            <nav className="nav-tabs">
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        {tab.label}
                        {tab.id === 'results' && result && (
                            <span style={{
                                width: 8, height: 8, borderRadius: '50%',
                                background: result.prediction === 'Malignant' ? 'var(--danger)' : 'var(--success)',
                                display: 'inline-block',
                            }} />
                        )}
                    </button>
                ))}
            </nav>

            {/* ===== MAIN CONTENT ===== */}
            <main className="main-content">
                {/* Loading Overlay */}
                {loading && (
                    <div style={{
                        position: 'fixed', inset: 0, zIndex: 200,
                        background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(8px)',
                        display: 'flex', flexDirection: 'column',
                        alignItems: 'center', justifyContent: 'center',
                    }}>
                        <div className="spinner" style={{ width: '56px', height: '56px', borderWidth: '4px' }} />
                        <p className="loading-text" style={{ marginTop: '20px', fontSize: '16px', color: '#fff' }}>
                            🧬 Analyzing with AI Ensemble Model...
                        </p>
                        <p className="loading-text" style={{ fontSize: '13px' }}>
                            Running 5 models with weighted voting
                        </p>
                    </div>
                )}

                {/* Backend Offline Warning */}
                {backendStatus === 'offline' && (
                    <div className="glass-card fade-in" style={{
                        padding: '32px', textAlign: 'center', marginBottom: '24px',
                        border: '1px solid rgba(239,68,68,0.2)',
                    }}>
                        <p style={{ fontSize: '18px', fontWeight: 600, color: 'var(--danger)', marginBottom: '12px' }}>
                            ⚠️ Backend Server Not Available
                        </p>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.8 }}>
                            Please start the backend server:<br />
                            <code style={{
                                background: 'var(--bg-input)', padding: '8px 16px',
                                borderRadius: '6px', display: 'inline-block', marginTop: '8px',
                                fontSize: '13px', color: 'var(--text-accent)',
                            }}>
                                cd backend && pip install -r requirements.txt && python -m ml.train && uvicorn main:app --reload
                            </code>
                        </p>
                    </div>
                )}

                {backendStatus === 'no-models' && (
                    <div className="glass-card fade-in" style={{
                        padding: '24px', textAlign: 'center', marginBottom: '24px',
                        border: '1px solid rgba(245,158,11,0.2)',
                    }}>
                        <p style={{ fontSize: '16px', fontWeight: 600, color: 'var(--warning)', marginBottom: '8px' }}>
                            ⚠️ Models Not Trained Yet
                        </p>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
                            Run: <code style={{ color: 'var(--text-accent)' }}>cd backend && python -m ml.train</code>
                        </p>
                    </div>
                )}

                {/* Tab Content */}
                {activeTab === 'upload' && (
                    <ImageUpload onResult={handleResult} onLoading={setLoading} />
                )}
                {activeTab === 'results' && (
                    result ? (
                        <ResultsView result={result} onGoToHospitals={() => setActiveTab('hospitals')} />
                    ) : (
                        <div className="glass-card fade-in" style={{ padding: '48px', textAlign: 'center' }}>
                            <p style={{ fontSize: '48px', marginBottom: '16px' }}>📊</p>
                            <p style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px' }}>
                                No Results Yet
                            </p>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
                                Upload a mammogram image or enter features in the Upload tab to see classification results.
                            </p>
                            <button className="btn btn-outline" style={{ marginTop: '16px' }}
                                onClick={() => setActiveTab('upload')}>
                                Go to Upload →
                            </button>
                        </div>
                    )
                )}
                {activeTab === 'report' && <ReportForm result={result} />}
                {activeTab === 'hospitals' && <HospitalLocator prediction={result?.prediction || null} />}
            </main>

            {/* ===== FOOTER ===== */}
            <footer style={{
                textAlign: 'center', padding: '24px 32px',
                borderTop: '1px solid var(--border)', color: 'var(--text-muted)',
                fontSize: '12px',
            }}>
                AI4BCancer v1.0 — AI-Assisted Classification System &nbsp;|&nbsp;
                For screening assistance only. Not a substitute for professional medical diagnosis.
            </footer>
        </div>
    );
}
