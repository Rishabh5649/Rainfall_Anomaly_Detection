// Analytics.jsx — Model performance analytics page
import { useMemo, useState, useEffect } from 'react';
import { fetchMetrics, fetchTrainHistory } from '../api';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, BarChart, Bar,
} from 'recharts';
import MetricsCards from '../components/MetricsCards';
import TrainModal from '../components/TrainModal';

function GaugeCard({ label, value, min = 0, max = 1, color, unit, icon }) {
  const pct = value !== null && value !== undefined
    ? Math.max(0, Math.min((value - min) / (max - min), 1))
    : null;

  return (
    <div className="stat-card" style={{ padding: '1.25rem' }}>
      <div className="flex-between">
        <div className="stat-label">{label}</div>
        <span style={{ fontSize: '1.1rem' }}>{icon}</span>
      </div>
      <div className="stat-value" style={{ color, fontSize: '1.6rem', margin: '0.5rem 0' }}>
        {value !== null && value !== undefined ? value.toFixed(3) : '—'}
        {unit && <span style={{ fontSize: '0.75rem', marginLeft: 4, color: 'var(--color-text-muted)' }}>{unit}</span>}
      </div>
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: pct !== null ? `${pct * 100}%` : '0%', background: color }} />
      </div>
    </div>
  );
}

export default function Analytics() {
  const [metrics, setMetrics]           = useState(null);
  const [trainHistory, setTrainHistory] = useState([]);
  const [showTrain, setShowTrain]       = useState(false);
  const [selectedRatio, setSelectedRatio] = useState('80:20');

  useEffect(() => {
    fetchMetrics().then(setMetrics).catch(() => {});
    fetchTrainHistory().then(setTrainHistory).catch(() => {});
  }, []);

  useEffect(() => {
    if (!metrics) return;
    const preferred = metrics.selected_ratio || '80:20';
    if (metrics.ratios?.[preferred]) {
      setSelectedRatio(preferred);
      return;
    }
    const first = metrics.available_ratios?.[0];
    if (first) setSelectedRatio(first);
  }, [metrics]);

  const m = metrics;
  const ratioRows = useMemo(() => {
    if (!m?.ratios) return [];
    return (m.available_ratios || Object.keys(m.ratios)).map((ratio) => ({
      ratio,
      test: m.ratios?.[ratio]?.splits?.test,
      threshold: m.ratios?.[ratio]?.threshold_mm,
    }));
  }, [m]);

  const selected = m?.ratios?.[selectedRatio] || null;
  const selectedTest = selected?.splits?.test || null;
  const selectedCm = selectedTest?.confusion_matrix || [[0, 0], [0, 0]];

  const checklist = [
    { label: 'Confusion Matrix', ok: !!selectedTest?.confusion_matrix },
    { label: 'F1-Score', ok: selectedTest?.f1_score !== null && selectedTest?.f1_score !== undefined },
    { label: 'Precision', ok: selectedTest?.precision !== null && selectedTest?.precision !== undefined },
    { label: 'Recall', ok: selectedTest?.recall !== null && selectedTest?.recall !== undefined },
    { label: 'Sensitivity', ok: selectedTest?.sensitivity !== null && selectedTest?.sensitivity !== undefined },
    { label: 'Accuracy', ok: selectedTest?.accuracy !== null && selectedTest?.accuracy !== undefined },
    { label: 'Area Under Curve (AUC)', ok: selectedTest?.auc !== null && selectedTest?.auc !== undefined },
    { label: 'Error Rate', ok: selectedTest?.error_rate !== null && selectedTest?.error_rate !== undefined },
  ];

  return (
    <div className="page">
      <div className="page-header flex-between anim-fade-up">
        <div>
          <h1 className="page-title">📊 Model Analytics</h1>
          <p className="page-subtitle">CNN+LSTM performance on IMD rainfall test set</p>
        </div>
        <button className="btn btn-primary" onClick={() => setShowTrain(true)}>
          🧠 Re-train Model
        </button>
      </div>

      {/* Headline metrics */}
      <div style={{ marginBottom: '1.5rem' }} className="anim-fade-up delay-1">
        <MetricsCards />
      </div>

      <div className="glow-line" />

      <div className="card anim-fade-up delay-2" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header">
          <span>🧪</span><h3>Evaluation Checklist & Split Selector</h3>
          <div style={{ marginLeft: 'auto', minWidth: 180 }}>
            <select
              className="input"
              value={selectedRatio}
              onChange={(e) => setSelectedRatio(e.target.value)}
            >
              {(m?.available_ratios || ['60:40', '70:30', '80:20', '90:10']).map((ratio) => (
                <option key={ratio} value={ratio}>{ratio}</option>
              ))}
            </select>
          </div>
        </div>
        <div className="grid grid-4">
          {checklist.map((item) => (
            <div key={item.label} className="stat-card" style={{ padding: '1rem' }}>
              <div className="flex-between">
                <div className="stat-label">{item.label}</div>
                <span className={item.ok ? 'badge badge-success' : 'badge badge-error'}>
                  {item.ok ? 'Ready' : 'Missing'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Split charts */}
      <div className="grid grid-2" style={{ marginBottom: '1.5rem' }}>
        {/* Gauge cards for train/val/test */}
        <div className="card anim-fade-up delay-2">
          <div className="card-header"><span>🎯</span><h3>Selected Ratio Classification Metrics ({selectedRatio})</h3></div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <GaugeCard label="Accuracy" value={selectedTest?.accuracy} min={0} max={1} color="#00E5A0" icon="✅" />
            <GaugeCard label="F1-Score" value={selectedTest?.f1_score} min={0} max={1} color="#00D4FF" icon="🎯" />
            <GaugeCard label="AUC"      value={selectedTest?.auc}      min={0} max={1} color="#7B2FFF" icon="📈" />
            <GaugeCard label="Error Rate" value={selectedTest?.error_rate} min={0} max={1} color="#FF6B6B" icon="❌" />
          </div>
        </div>

        {/* Confusion matrix */}
        <div className="card anim-fade-up delay-3">
          <div className="card-header"><span>🧩</span><h3>Confusion Matrix ({selectedRatio} Test)</h3></div>
          {selectedTest ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
              <div className="stat-card" style={{ padding: '1rem', borderColor: 'rgba(0,229,160,0.35)' }}>
                <div className="stat-label">TN</div>
                <div className="stat-value" style={{ color: 'var(--color-success)' }}>{selectedCm[0][0]}</div>
              </div>
              <div className="stat-card" style={{ padding: '1rem', borderColor: 'rgba(255,107,107,0.35)' }}>
                <div className="stat-label">FP</div>
                <div className="stat-value" style={{ color: 'var(--color-accent-3)' }}>{selectedCm[0][1]}</div>
              </div>
              <div className="stat-card" style={{ padding: '1rem', borderColor: 'rgba(255,107,107,0.35)' }}>
                <div className="stat-label">FN</div>
                <div className="stat-value" style={{ color: 'var(--color-accent-3)' }}>{selectedCm[1][0]}</div>
              </div>
              <div className="stat-card" style={{ padding: '1rem', borderColor: 'rgba(0,229,160,0.35)' }}>
                <div className="stat-label">TP</div>
                <div className="stat-value" style={{ color: 'var(--color-success)' }}>{selectedCm[1][1]}</div>
              </div>
            </div>
          ) : (
            <div className="loading-state" style={{ height: 220 }}>
              <div className="spinner" />
              <span>Run evaluation first</span>
            </div>
          )}
        </div>
      </div>

      <div className="card anim-fade-up delay-3" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header"><span>📋</span><h3>Test Metrics by Train:Test Ratio</h3></div>
        {ratioRows.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--color-border)' }}>
                  <th style={{ padding: '0.6rem' }}>Ratio</th>
                  <th style={{ padding: '0.6rem' }}>Accuracy</th>
                  <th style={{ padding: '0.6rem' }}>F1</th>
                  <th style={{ padding: '0.6rem' }}>Precision</th>
                  <th style={{ padding: '0.6rem' }}>Recall</th>
                  <th style={{ padding: '0.6rem' }}>Sensitivity</th>
                  <th style={{ padding: '0.6rem' }}>AUC</th>
                  <th style={{ padding: '0.6rem' }}>Error Rate</th>
                  <th style={{ padding: '0.6rem' }}>Threshold (mm)</th>
                </tr>
              </thead>
              <tbody>
                {ratioRows.map((row) => (
                  <tr key={row.ratio} style={{ borderBottom: '1px solid var(--color-border)' }}>
                    <td style={{ padding: '0.6rem', fontWeight: 700 }}>{row.ratio}</td>
                    <td style={{ padding: '0.6rem' }}>{row.test?.accuracy ?? '—'}</td>
                    <td style={{ padding: '0.6rem' }}>{row.test?.f1_score ?? '—'}</td>
                    <td style={{ padding: '0.6rem' }}>{row.test?.precision ?? '—'}</td>
                    <td style={{ padding: '0.6rem' }}>{row.test?.recall ?? '—'}</td>
                    <td style={{ padding: '0.6rem' }}>{row.test?.sensitivity ?? '—'}</td>
                    <td style={{ padding: '0.6rem' }}>{row.test?.auc ?? '—'}</td>
                    <td style={{ padding: '0.6rem' }}>{row.test?.error_rate ?? '—'}</td>
                    <td style={{ padding: '0.6rem' }}>{row.threshold ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="loading-state" style={{ height: 120 }}>
            <div className="spinner" />
            <span>No ratio metrics yet</span>
          </div>
        )}
      </div>

      <div className="card anim-fade-up delay-4" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header"><span>📏</span><h3>RMSE by Ratio (Test Split)</h3></div>
        {ratioRows.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart
              data={ratioRows.map((row) => ({ ratio: row.ratio, RMSE: row.test?.rmse, MAE: row.test?.mae }))}
              margin={{ top: 8, right: 12, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis dataKey="ratio" tick={{ fill: 'var(--color-text-muted)', fontSize: 12 }} />
              <YAxis tick={{ fill: 'var(--color-text-muted)', fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  background: 'var(--color-bg-elevated)',
                  border: '1px solid var(--color-border-hover)',
                  borderRadius: 8,
                }}
                formatter={v => [`${v?.toFixed?.(2) ?? v} mm`]}
              />
              <Legend />
              <Bar dataKey="RMSE" fill="#00D4FF" radius={[4,4,0,0]} />
              <Bar dataKey="MAE" fill="#7B2FFF" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="loading-state" style={{ height: 220 }}>
            <div className="spinner" />
            <span>Run evaluation first</span>
          </div>
        )}
      </div>

      {/* Training curves */}
      <div className="card anim-fade-up delay-5">
        <div className="card-header">
          <span>📉</span><h3>Training & Validation Loss</h3>
          {trainHistory.length > 0 && (
            <span className="badge badge-success" style={{ marginLeft: 'auto' }}>
              {trainHistory.length} epochs
            </span>
          )}
        </div>
        {trainHistory.length > 0 ? (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={trainHistory} margin={{ top: 8, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis dataKey="epoch" tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', fill: 'var(--color-text-muted)', dy: 10 }} />
              <YAxis tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: 'var(--color-bg-elevated)',
                  border: '1px solid var(--color-border-hover)',
                  borderRadius: 8,
                }}
                formatter={v => [v?.toFixed(6)]}
              />
              <Legend />
              <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#00D4FF" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="val_loss"   name="Val Loss"   stroke="#FF6B6B" strokeWidth={2} dot={false} strokeDasharray="5 3" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="loading-state" style={{ height: 300 }}>
            <div style={{ fontSize: '2.5rem' }}>📉</div>
            <div>No training history yet</div>
            <button className="btn btn-primary btn-sm" onClick={() => setShowTrain(true)}>
              Train Model
            </button>
          </div>
        )}
      </div>

      {/* Val RMSE over epochs */}
      {trainHistory.length > 0 && (
        <div className="card anim-fade-up delay-5" style={{ marginTop: '1.5rem' }}>
          <div className="card-header"><span>📐</span><h3>Val RMSE over Training</h3></div>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={trainHistory} margin={{ top: 8, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis dataKey="epoch" tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} />
              <YAxis tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: 'var(--color-bg-elevated)', border: '1px solid var(--color-border-hover)', borderRadius: 8 }}
                formatter={v => [v?.toFixed(4)]}
              />
              <Line type="monotone" dataKey="val_rmse" name="Val RMSE" stroke="#00E5A0" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {showTrain && <TrainModal onClose={() => setShowTrain(false)} />}
    </div>
  );
}
