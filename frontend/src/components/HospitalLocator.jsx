import React, { useState, useEffect, useRef, useCallback } from 'react';
import { findHospitals } from '../utils/api';

/**
 * HospitalLocator - Integrated hospital finder with interactive map
 * Props:
 *   prediction: 'Malignant' | 'Benign' | null – auto-triggers search when provided
 */
export default function HospitalLocator({ prediction }) {
    const [hospitals, setHospitals] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [location, setLocation] = useState(null);
    const [searchRadius, setSearchRadius] = useState(50);
    const [selectedHospital, setSelectedHospital] = useState(null);
    const [autoSearched, setAutoSearched] = useState(false);
    const mapRef = useRef(null);
    const leafletMapRef = useRef(null);
    const markersRef = useRef([]);

    // ── Detect location ──────────────────────────────────────────────────────
    useEffect(() => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (pos) => setLocation({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
                () => setLocation({ lat: 28.6139, lon: 77.2090 }) // Default: New Delhi
            );
        } else {
            setLocation({ lat: 28.6139, lon: 77.2090 });
        }
    }, []);

    // ── Load Leaflet CSS/JS from CDN ─────────────────────────────────────────
    useEffect(() => {
        if (!document.getElementById('leaflet-css')) {
            const link = document.createElement('link');
            link.id = 'leaflet-css';
            link.rel = 'stylesheet';
            link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
            document.head.appendChild(link);
        }
        if (!document.getElementById('leaflet-js')) {
            const script = document.createElement('script');
            script.id = 'leaflet-js';
            script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
            document.head.appendChild(script);
        }
    }, []);

    // ── Initialize map once we have location ────────────────────────────────
    useEffect(() => {
        if (!location || !mapRef.current) return;
        const tryInit = setInterval(() => {
            if (window.L && mapRef.current && !leafletMapRef.current) {
                clearInterval(tryInit);
                const map = window.L.map(mapRef.current, { zoomControl: true, scrollWheelZoom: true });
                window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors',
                    maxZoom: 19,
                }).addTo(map);
                map.setView([location.lat, location.lon], 12);
                // User pin
                const userIcon = window.L.divIcon({
                    html: '<div style="background:#6366f1;width:16px;height:16px;border-radius:50%;border:3px solid white;box-shadow:0 2px 8px rgba(0,0,0,0.4)"></div>',
                    className: '', iconAnchor: [8, 8],
                });
                window.L.marker([location.lat, location.lon], { icon: userIcon })
                    .bindPopup('<b>📍 Your Location</b>').addTo(map);
                leafletMapRef.current = map;
            }
        }, 200);
        return () => clearInterval(tryInit);
    }, [location, mapRef.current]);

    // ── Place hospital markers on map ────────────────────────────────────────
    useEffect(() => {
        if (!leafletMapRef.current || !window.L || hospitals.length === 0) return;
        const map = leafletMapRef.current;
        // Clear old markers
        markersRef.current.forEach(m => map.removeLayer(m));
        markersRef.current = [];

        const bounds = [[location.lat, location.lon]];
        hospitals.forEach((h, i) => {
            const color = h.is_specialist ? '#ef4444' : '#3b82f6';
            const icon = window.L.divIcon({
                html: `<div style="background:${color};color:white;width:28px;height:28px;border-radius:50%;border:2px solid white;box-shadow:0 2px 8px rgba(0,0,0,0.4);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700">${i + 1}</div>`,
                className: '', iconAnchor: [14, 14],
            });
            const popup = `
                <div style="min-width:200px;font-family:system-ui">
                    <b style="font-size:14px">${h.is_specialist ? '🎗️' : '🏥'} ${h.name}</b><br/>
                    <span style="color:#666;font-size:12px">${h.type}</span><br/>
                    ${h.address && h.address !== 'Address not available' ? `<span style="font-size:12px">📍 ${h.address}</span><br/>` : ''}
                    ${h.phone && h.phone !== 'N/A' ? `<a href="tel:${h.phone}" style="color:#6366f1;font-size:12px">📞 ${h.phone}</a><br/>` : ''}
                    <span style="font-size:12px;color:#3b82f6">📏 ${h.distance_km} km away</span>
                    ${h.website ? `<br/><a href="${h.website}" target="_blank" style="color:#6366f1;font-size:12px">🌐 Website</a>` : ''}
                </div>`;
            const marker = window.L.marker([h.lat, h.lon], { icon })
                .bindPopup(popup).addTo(map);
            marker.on('click', () => setSelectedHospital(h));
            markersRef.current.push(marker);
            bounds.push([h.lat, h.lon]);
        });
        if (bounds.length > 1) map.fitBounds(bounds, { padding: [40, 40] });
    }, [hospitals]);

    // ── Draw route line to selected hospital ────────────────────────────────
    const routeLineRef = useRef(null);
    useEffect(() => {
        if (!leafletMapRef.current || !window.L || !selectedHospital || !location) return;
        if (routeLineRef.current) leafletMapRef.current.removeLayer(routeLineRef.current);
        routeLineRef.current = window.L.polyline(
            [[location.lat, location.lon], [selectedHospital.lat, selectedHospital.lon]],
            { color: '#f59e0b', weight: 3, dashArray: '8,6', opacity: 0.9 }
        ).addTo(leafletMapRef.current);
        leafletMapRef.current.fitBounds(routeLineRef.current.getBounds(), { padding: [60, 60] });
    }, [selectedHospital]);

    // ── Search function ──────────────────────────────────────────────────────
    const handleSearch = useCallback(async (loc = location, rad = searchRadius) => {
        if (!loc) return;
        setLoading(true);
        setError(null);
        try {
            const data = await findHospitals(loc.lat, loc.lon, rad * 1000);
            const list = data.hospitals || [];
            setHospitals(list);
            if (list.length === 0) {
                // Expand radius automatically
                if (rad < 200) {
                    const bigger = Math.min(rad * 3, 200);
                    setSearchRadius(bigger);
                    const retry = await findHospitals(loc.lat, loc.lon, bigger * 1000);
                    setHospitals(retry.hospitals || []);
                    if ((retry.hospitals || []).length === 0) {
                        setError('No hospitals found within 200 km. Showing nearest major hospitals.');
                    }
                } else {
                    setError('No hospitals found nearby. Try a different location.');
                }
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [location, searchRadius]);

    // ── Auto-search when prediction arrives and location is ready ────────────
    useEffect(() => {
        if (prediction && location && !autoSearched) {
            setAutoSearched(true);
            handleSearch(location, searchRadius);
        }
    }, [prediction, location, autoSearched, handleSearch, searchRadius]);

    const specialists = hospitals.filter(h => h.is_specialist);
    const general = hospitals.filter(h => !h.is_specialist);
    const isMalignant = prediction === 'Malignant';

    return (
        <div className="fade-in">
            {/* ── Header ── */}
            <h2 className="section-title">
                <span className="section-title-icon">🏥</span>
                Nearby Cancer Hospitals &amp; Breast Care Centers
            </h2>

            {/* ── Prediction context banner ── */}
            {prediction && (
                <div style={{
                    padding: '14px 20px', borderRadius: '12px', marginBottom: '20px',
                    background: isMalignant ? 'rgba(239,68,68,0.10)' : 'rgba(16,185,129,0.10)',
                    border: `1px solid ${isMalignant ? 'rgba(239,68,68,0.3)' : 'rgba(16,185,129,0.3)'}`,
                    display: 'flex', alignItems: 'center', gap: '12px',
                }}>
                    <span style={{ fontSize: '28px' }}>{isMalignant ? '🎗️' : '💚'}</span>
                    <div>
                        <div style={{ fontWeight: 700, color: isMalignant ? '#ef4444' : '#10b981', fontSize: '15px' }}>
                            {isMalignant ? 'Malignant Detected — Oncology Specialists Recommended' : 'Benign Finding — Routine Follow-up Centers'}
                        </div>
                        <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                            {isMalignant
                                ? 'Showing cancer centres and oncology specialists near you.'
                                : 'Showing breast care and general hospital facilities near you.'}
                        </div>
                    </div>
                </div>
            )}

            {/* ── Search Controls ── */}
            <div className="glass-card" style={{ padding: '20px', marginBottom: '20px' }}>
                <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-end', flexWrap: 'wrap' }}>
                    <div className="form-group" style={{ flex: 1, minWidth: '180px' }}>
                        <label className="form-label">Your Location</label>
                        <input className="form-input" type="text" readOnly
                            value={location ? `${location.lat.toFixed(4)}°N, ${location.lon.toFixed(4)}°E` : 'Detecting…'}
                        />
                    </div>
                    <div className="form-group" style={{ width: '130px' }}>
                        <label className="form-label">Radius (km)</label>
                        <input className="form-input" type="number" min="5" max="500"
                            value={searchRadius}
                            onChange={(e) => setSearchRadius(parseInt(e.target.value) || 25)}
                        />
                    </div>
                    <button className="btn btn-primary"
                        onClick={() => { setAutoSearched(true); handleSearch(); }}
                        disabled={loading || !location}
                        style={{ height: '42px', minWidth: '160px' }}>
                        {loading ? '🔄 Searching…' : '🔍 Search Hospitals'}
                    </button>
                </div>
            </div>

            {/* ── Error ── */}
            {error && (
                <div style={{
                    padding: '12px 16px', borderRadius: '10px', fontSize: '14px',
                    background: 'rgba(245,158,11,0.1)', color: '#f59e0b',
                    border: '1px solid rgba(245,158,11,0.25)', marginBottom: '16px',
                }}>⚠️ {error}</div>
            )}

            {/* ── Loading ── */}
            {loading && (
                <div style={{ textAlign: 'center', padding: '48px' }}>
                    <div className="spinner" />
                    <p className="loading-text" style={{ marginTop: '16px' }}>Searching for nearby hospitals…</p>
                </div>
            )}

            {/* ── Map + Results Layout ── */}
            {!loading && hospitals.length > 0 && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: '20px', alignItems: 'start' }}>

                    {/* Left: Map */}
                    <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                            <div style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                                Found <strong style={{ color: 'var(--text-primary)' }}>{hospitals.length}</strong> hospitals
                                {specialists.length > 0 && <> · <span style={{ color: '#ef4444' }}>{specialists.length} specialists</span></>}
                            </div>
                            {selectedHospital && (
                                <div style={{ fontSize: '13px', color: '#f59e0b' }}>
                                    📏 {selectedHospital.distance_km} km to {selectedHospital.name.split(' ').slice(0, 3).join(' ')}
                                </div>
                            )}
                        </div>
                        <div ref={mapRef} style={{
                            width: '100%', height: '460px', borderRadius: '14px',
                            overflow: 'hidden', border: '1px solid var(--border)',
                            boxShadow: '0 4px 24px rgba(0,0,0,0.15)',
                        }} />
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px', textAlign: 'right' }}>
                            Map data © OpenStreetMap · Click a marker or card to show route
                        </div>
                    </div>

                    {/* Right: Hospital Cards */}
                    <div style={{ maxHeight: '520px', overflowY: 'auto', paddingRight: '4px' }}>
                        {specialists.length > 0 && (
                            <>
                                <div style={{ fontSize: '13px', fontWeight: 700, color: '#ef4444', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                    🎗️ Cancer Specialists
                                </div>
                                {specialists.map((h, i) => (
                                    <HospitalCard key={`s-${i}`} hospital={h} index={i}
                                        selected={selectedHospital?.name === h.name}
                                        onSelect={() => setSelectedHospital(h)}
                                        userLat={location?.lat} userLon={location?.lon}
                                    />
                                ))}
                                {general.length > 0 && <div style={{ marginTop: '16px', marginBottom: '10px', borderTop: '1px solid var(--border)', paddingTop: '14px', fontSize: '13px', fontWeight: 700, color: '#3b82f6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                    🏥 General Hospitals
                                </div>}
                            </>
                        )}
                        {general.map((h, i) => (
                            <HospitalCard key={`g-${i}`} hospital={h} index={specialists.length + i}
                                selected={selectedHospital?.name === h.name}
                                onSelect={() => setSelectedHospital(h)}
                                userLat={location?.lat} userLon={location?.lon}
                            />
                        ))}
                    </div>
                </div>
            )}

            {/* ── Empty state ── */}
            {!loading && hospitals.length === 0 && !error && (
                <div className="glass-card" style={{ padding: '48px', textAlign: 'center' }}>
                    <p style={{ fontSize: '48px', marginBottom: '12px' }}>🏥</p>
                    <p style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px' }}>
                        {prediction ? 'Auto-searching nearby hospitals…' : 'Search for Nearby Hospitals'}
                    </p>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
                        {prediction
                            ? 'Please wait while we locate hospitals near you.'
                            : 'Click "Search Hospitals" to find cancer facilities near your location.'}
                    </p>
                </div>
            )}
        </div>
    );
}

/* ── Hospital Card ────────────────────────────────────────────────────────── */
function HospitalCard({ hospital: h, index, selected, onSelect, userLat, userLon }) {
    const googleMapsUrl = `https://www.google.com/maps/dir/${userLat},${userLon}/${h.lat},${h.lon}`;
    const callUrl = h.phone && h.phone !== 'N/A' ? `tel:${h.phone}` : null;

    return (
        <div onClick={onSelect} style={{
            marginBottom: '12px', borderRadius: '12px', padding: '14px 16px',
            background: selected ? 'rgba(245,158,11,0.06)' : 'var(--bg-card)',
            border: `1px solid ${selected ? '#f59e0b' : 'var(--border)'}`,
            cursor: 'pointer', transition: 'all 0.18s ease',
            boxShadow: selected ? '0 0 0 2px rgba(245,158,11,0.25)' : 'none',
        }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                {/* Number badge */}
                <div style={{
                    minWidth: '26px', height: '26px', borderRadius: '50%',
                    background: h.is_specialist ? '#ef4444' : '#3b82f6',
                    color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '12px', fontWeight: 700, marginTop: '2px', flexShrink: 0,
                }}>
                    {index + 1}
                </div>

                <div style={{ flex: 1, minWidth: 0 }}>
                    {/* Hospital name */}
                    <div style={{ fontWeight: 700, fontSize: '14px', color: 'var(--text-primary)', lineHeight: 1.3, marginBottom: '4px' }}>
                        {h.is_specialist ? '🎗️' : '🏥'} {h.name}
                    </div>

                    {/* Type badge */}
                    <span style={{
                        display: 'inline-block', padding: '2px 8px', borderRadius: '20px', fontSize: '11px', fontWeight: 600,
                        background: h.is_specialist ? 'rgba(239,68,68,0.12)' : 'rgba(59,130,246,0.12)',
                        color: h.is_specialist ? '#ef4444' : '#3b82f6', marginBottom: '6px',
                    }}>
                        {h.type}
                    </span>

                    {/* Address */}
                    {h.address && h.address !== 'Address not available' && (
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                            📍 {h.address}
                        </div>
                    )}

                    {/* Distance */}
                    <div style={{ fontSize: '13px', color: '#f59e0b', fontWeight: 600, marginBottom: '8px' }}>
                        📏 {h.distance_km} km away
                    </div>

                    {/* Action buttons */}
                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                        {callUrl && (
                            <a href={callUrl} onClick={e => e.stopPropagation()} style={{
                                display: 'inline-flex', alignItems: 'center', gap: '4px',
                                padding: '5px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: 600,
                                background: 'rgba(16,185,129,0.12)', color: '#10b981',
                                textDecoration: 'none', border: '1px solid rgba(16,185,129,0.25)',
                            }}>📞 {h.phone}</a>
                        )}
                        <a href={googleMapsUrl} target="_blank" rel="noopener noreferrer"
                            onClick={e => e.stopPropagation()} style={{
                                display: 'inline-flex', alignItems: 'center', gap: '4px',
                                padding: '5px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: 600,
                                background: 'rgba(59,130,246,0.12)', color: '#3b82f6',
                                textDecoration: 'none', border: '1px solid rgba(59,130,246,0.25)',
                            }}>🗺️ Directions</a>
                        {h.website && (
                            <a href={h.website} target="_blank" rel="noopener noreferrer"
                                onClick={e => e.stopPropagation()} style={{
                                    display: 'inline-flex', alignItems: 'center', gap: '4px',
                                    padding: '5px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: 600,
                                    background: 'rgba(99,102,241,0.12)', color: '#6366f1',
                                    textDecoration: 'none', border: '1px solid rgba(99,102,241,0.25)',
                                }}>🌐 Website</a>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
