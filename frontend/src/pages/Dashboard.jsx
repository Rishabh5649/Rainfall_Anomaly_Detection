// Dashboard.jsx — Main dashboard page
import { useState, useEffect } from 'react';
import { useState as useS } from 'react';
import { fetchSubdivisions, fetchHistory, fetchMetrics, fetchTrainStatus } from '../api';
import MapView from '../components/MapView';
import TimeseriesChart from '../components/TimeseriesChart';
import MetricsCards from '../components/MetricsCards';
import TrainModal from '../components/TrainModal';
import { useNavigate } from 'react-router-dom';

export default function Dashboard() {
  const navigate = useNavigate();
  const [selectedSub, setSelectedSub] = useState(null);
  const [history, setHistory]         = useState(null);
  const [showTrainModal, setShowTrain]= useState(false);
  const [metrics, setMetrics]         = useState(null);
  const [trainStatus, setTrainStatus] = useState(null);

  useEffect(() => {
    fetchMetrics().then(setMetrics).catch(() => {});
    fetchTrainStatus().then(setTrainStatus).catch(() => {});
    fetchSubdivisions().then(subs => {
      if (subs.length) setSelectedSub(subs[0]);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!selectedSub) return;
    fetchHistory(selectedSub).then(d => setHistory(d.data)).catch(() => setHistory(null));
  }, [selectedSub]);

  const isModelReady = trainStatus?.status === 'complete';

  return (
    <div className="page">
      {/* Header */}
      <div className="page-header flex-between anim-fade-up">
        <div>
          <h1 className="page-title">🌧️ RainSight India</h1>
          <p className="page-subtitle">
            CNN+LSTM Hybrid Forecasting · 36 IMD Meteorological Subdivisions · 1901–2015
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button className="btn btn-secondary" onClick={() => setShowTrain(true)}>
            🧠 Train Model
          </button>
          <button className="btn btn-primary" onClick={() => navigate('/forecast')}>
            🌧️ Forecast →
          </button>
        </div>
      </div>

      {/* Model not trained notice */}
      {trainStatus && trainStatus.status === 'idle' && (
        <div className="anim-fade-up delay-1" style={{
          marginBottom: '1.5rem',
          padding: '1rem 1.5rem',
          background: 'rgba(255,179,71,0.08)', border: '1px solid rgba(255,179,71,0.2)',
          borderRadius: 12, display: 'flex', alignItems: 'center', gap: '1rem',
        }}>
          <span style={{ fontSize: '1.3rem' }}>⚠️</span>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 600, color: 'var(--color-warning)' }}>Model not trained yet</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
              Click <b>Train Model</b> to start, or run <code style={{ fontFamily: 'monospace', color: 'var(--color-accent)' }}>python src/train.py</code> manually.
            </div>
          </div>
          <button className="btn btn-secondary btn-sm" onClick={() => setShowTrain(true)}>Train Now</button>
        </div>
      )}

      {/* Metrics */}
      <div className="anim-fade-up delay-1" style={{ marginBottom: '1.5rem' }}>
        <div style={{ fontWeight: 700, fontSize: '1rem', marginBottom: '0.75rem', color: 'var(--color-text-secondary)' }}>
          MODEL PERFORMANCE
        </div>
        <MetricsCards />
      </div>

      <div className="glow-line" />

      {/* Two-column: map + chart */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: '1.5rem' }}>
        <div className="card anim-fade-up delay-2">
          <div className="card-header">
            <span style={{ fontSize: '1.1rem' }}>🗺️</span>
            <h3>Subdivision Rainfall Map</h3>
            <span className="badge badge-accent" style={{ marginLeft: 'auto' }}>Avg 2012–2015</span>
          </div>
          <MapView
            onSelectSubdivision={setSelectedSub}
            selectedSubdivision={selectedSub}
          />
        </div>

        <div className="card anim-fade-up delay-3">
          <div className="card-header">
            <span style={{ fontSize: '1.1rem' }}>📈</span>
            <h3>Historical Rainfall</h3>
            {selectedSub && <span className="badge badge-accent" style={{ marginLeft: 'auto' }}>{selectedSub}</span>}
          </div>
          <TimeseriesChart historyData={history} predictions={null} />
          {selectedSub && (
            <div style={{ marginTop: '1rem', textAlign: 'center' }}>
              <button className="btn btn-primary btn-sm"
                onClick={() => navigate(`/forecast?sub=${encodeURIComponent(selectedSub)}`)}>
                Forecast {selectedSub} →
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Training modal */}
      {showTrainModal && <TrainModal onClose={() => setShowTrain(false)} />}
    </div>
  );
}
