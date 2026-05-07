// ForecastPanel.jsx — Subdivision selector + forecast controls + results
import { useState, useEffect } from 'react';
import { fetchSubdivisions, fetchHistory, fetchPrediction } from '../api';
import TimeseriesChart from './TimeseriesChart';

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

function RainBar({ mm, maxMm = 800 }) {
  const pct = Math.min((mm / maxMm) * 100, 100);
  const color = mm > 300 ? '#00D4FF' : mm > 100 ? '#7B2FFF' : '#FF6B6B';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
      <div style={{
        flex: 1, height: 6, background: 'var(--color-bg-elevated)',
        borderRadius: 3, overflow: 'hidden',
      }}>
        <div style={{
          height: '100%', width: `${pct}%`, borderRadius: 3,
          background: color, transition: 'width 0.6s ease',
        }} />
      </div>
      <span style={{ fontSize: '0.8rem', fontFamily: 'monospace', color, minWidth: 60, textAlign: 'right' }}>
        {mm.toFixed(1)} mm
      </span>
    </div>
  );
}

export default function ForecastPanel({ initialSubdivision }) {
  const [subdivisions, setSubdivisions] = useState([]);
  const [selected, setSelected]         = useState(initialSubdivision || '');
  const [startYear, setStartYear]       = useState(2010);
  const [startMonth, setStartMonth]     = useState(1);
  const [horizon, setHorizon]           = useState(12);
  const [history, setHistory]           = useState(null);
  const [predictions, setPredictions]   = useState(null);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState(null);

  useEffect(() => {
    fetchSubdivisions()
      .then(subs => { setSubdivisions(subs); if (subs.length) setSelected(subs[0]); })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!selected) return;
    fetchHistory(selected).then(d => setHistory(d.data)).catch(() => {});
  }, [selected]);

  const handleForecast = async () => {
    if (!selected) return;
    setLoading(true); setError(null); setPredictions(null);
    try {
      const res = await fetchPrediction(selected, startYear, startMonth, horizon);
      setPredictions(res.predictions);
    } catch (e) {
      setError(e.response?.data?.detail || 'Forecast failed. Is the model trained?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Controls */}
      <div className="card">
        <div className="card-header">
          <span style={{ fontSize: '1.1rem' }}>🎛️</span>
          <h3>Forecast Controls</h3>
        </div>

        <div className="grid grid-2" style={{ gap: '1rem', marginBottom: '1.25rem' }}>
          {/* Subdivision */}
          <div className="input-group" style={{ gridColumn: '1 / -1' }}>
            <label className="input-label">Meteorological Subdivision</label>
            <select className="input" value={selected} onChange={e => setSelected(e.target.value)}>
              {subdivisions.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          {/* Start year */}
          <div className="input-group">
            <label className="input-label">Start Year</label>
            <input
              className="input" type="number"
              min={1925} max={2030} value={startYear}
              onChange={e => setStartYear(+e.target.value)}
            />
          </div>

          {/* Start month */}
          <div className="input-group">
            <label className="input-label">Start Month</label>
            <select className="input" value={startMonth} onChange={e => setStartMonth(+e.target.value)}>
              {MONTHS.map((m, i) => <option key={i} value={i + 1}>{m}</option>)}
            </select>
          </div>
        </div>

        {/* Horizon slider */}
        <div className="input-group" style={{ marginBottom: '1.25rem' }}>
          <div className="flex-between">
            <label className="input-label">Forecast Horizon</label>
            <span style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--color-accent)', fontFamily: 'monospace' }}>
              {horizon} months
            </span>
          </div>
          <input type="range" min={1} max={60} value={horizon}
            onChange={e => setHorizon(+e.target.value)} />
          <div className="flex-between" style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)' }}>
            <span>1 month</span><span>5 years</span>
          </div>
        </div>

        <button className="btn btn-primary btn-lg" style={{ width: '100%' }}
          onClick={handleForecast} disabled={loading || !selected}>
          {loading ? <><div className="spinner" style={{ width: 18, height: 18 }} /> Forecasting…</> : '🌧️ Generate Forecast'}
        </button>

        {error && (
          <div style={{
            marginTop: '1rem', padding: '0.75rem 1rem',
            background: 'rgba(255,107,107,0.08)', border: '1px solid rgba(255,107,107,0.2)',
            borderRadius: 8, color: 'var(--color-accent-3)', fontSize: '0.85rem',
          }}>
            ⚠️ {error}
          </div>
        )}
      </div>

      {/* Chart */}
      <div className="card">
        <div className="card-header">
          <span style={{ fontSize: '1.1rem' }}>📈</span>
          <h3>Rainfall Time-series</h3>
        </div>
        <TimeseriesChart
          historyData={history}
          predictions={predictions}
          subdivision={selected}
        />
      </div>

      {/* Monthly results table */}
      {predictions && (
        <div className="card anim-fade-up">
          <div className="card-header">
            <span style={{ fontSize: '1.1rem' }}>📋</span>
            <h3>Monthly Forecast — {selected}</h3>
            <span className="badge badge-accent" style={{ marginLeft: 'auto' }}>
              {predictions.length} months
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
            {predictions.map((p, i) => (
              <div key={i} style={{
                padding: '0.75rem 1rem',
                background: 'var(--color-bg-surface)',
                borderRadius: 8, border: '1px solid var(--color-border)',
              }}>
                <div className="flex-between" style={{ marginBottom: '0.4rem' }}>
                  <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>
                    {MONTHS[p.month - 1]} {p.year}
                  </span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                    CI: [{p.lower_mm.toFixed(1)} – {p.upper_mm.toFixed(1)}] mm
                  </span>
                </div>
                <RainBar mm={p.predicted_mm} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
