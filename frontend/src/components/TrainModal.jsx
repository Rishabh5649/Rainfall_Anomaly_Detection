// TrainModal.jsx — Modal to trigger model training with params
import { useState } from 'react';
import { startTraining } from '../api';

export default function TrainModal({ onClose }) {
  const [params, setParams] = useState({
    epochs: 80,
    batch_size: 256,
    lr: 0.003,
    patience: 15,
    split_ratio: '80:20',
  });
  const [loading, setLoading] = useState(false);
  const [done, setDone]       = useState(false);
  const [error, setError]     = useState(null);

  const update = (k, v) => setParams(p => ({ ...p, [k]: v }));

  const handleTrain = async () => {
    setLoading(true); setError(null);
    try {
      await startTraining(params);
      setDone(true);
    } catch (e) {
      setError(e.response?.data?.detail || 'Training request failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal">
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <span style={{ fontSize: '1.3rem' }}>🧠</span>
            <h3>Train CNN+LSTM Model</h3>
          </div>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        {done ? (
          <div style={{ textAlign: 'center', padding: '2rem 0' }}>
            <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🚀</div>
            <div style={{ fontWeight: 700, color: 'var(--color-success)', marginBottom: '0.5rem' }}>
              Training started in background!
            </div>
            <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
              Monitor progress in the header status bar. Training will auto-save the best checkpoint.
            </div>
            <button className="btn btn-secondary" onClick={onClose}>Close</button>
          </div>
        ) : (
          <>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginBottom: '1.5rem' }}>
              <div className="input-group">
                <label className="input-label">Epochs (max)</label>
                <input className="input" type="number" min={5} max={500}
                  value={params.epochs} onChange={e => update('epochs', +e.target.value)} />
              </div>
              <div className="input-group">
                <label className="input-label">Batch Size</label>
                <input className="input" type="number" min={16} max={1024}
                  value={params.batch_size} onChange={e => update('batch_size', +e.target.value)} />
              </div>
              <div className="input-group">
                <label className="input-label">Learning Rate</label>
                <input className="input" type="number" step="0.0001" min={0.00001}
                  value={params.lr} onChange={e => update('lr', +e.target.value)} />
              </div>
              <div className="input-group">
                <label className="input-label">Early Stopping Patience (epochs)</label>
                <input className="input" type="number" min={3} max={100}
                  value={params.patience} onChange={e => update('patience', +e.target.value)} />
              </div>
              <div className="input-group">
                <label className="input-label">Train:Test Ratio</label>
                <select
                  className="input"
                  value={params.split_ratio}
                  onChange={e => update('split_ratio', e.target.value)}
                >
                  <option value="60:40">60:40</option>
                  <option value="70:30">70:30</option>
                  <option value="80:20">80:20</option>
                  <option value="90:10">90:10</option>
                </select>
              </div>
            </div>

            <div style={{
              padding: '0.75rem 1rem', marginBottom: '1rem',
              background: 'rgba(0,212,255,0.06)', border: '1px solid rgba(0,212,255,0.15)',
              borderRadius: 8, fontSize: '0.82rem', color: 'var(--color-text-secondary)',
            }}>
              ℹ️ Training runs in background on your CPU. It may take several minutes.
              The API will still respond during training. Best model auto-saved to <code style={{ fontFamily: 'monospace', color: 'var(--color-accent)' }}>checkpoints/best_model.pt</code>.
            </div>

            {error && (
              <div style={{
                padding: '0.75rem 1rem', marginBottom: '1rem',
                background: 'rgba(255,107,107,0.08)', border: '1px solid rgba(255,107,107,0.2)',
                borderRadius: 8, color: 'var(--color-accent-3)', fontSize: '0.85rem',
              }}>⚠️ {error}</div>
            )}

            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <button className="btn btn-ghost" style={{ flex: 1 }} onClick={onClose}>Cancel</button>
              <button className="btn btn-primary" style={{ flex: 2 }}
                onClick={handleTrain} disabled={loading}>
                {loading ? <><div className="spinner" style={{ width: 16, height: 16 }} /> Sending…</> : '🚀 Start Training'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
