// Header.jsx
import { useState, useEffect } from 'react';
import { fetchTrainStatus } from '../api';

export default function Header() {
  const [status, setStatus] = useState({ status: 'idle' });

  useEffect(() => {
    const poll = async () => {
      try { setStatus(await fetchTrainStatus()); } catch { /* ignore */ }
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, []);

  const isTraining = status.status === 'training';

  return (
    <header style={{
      position: 'fixed', top: 0, left: 'var(--sidebar-width)', right: 0,
      height: 'var(--header-height)', zIndex: 100,
      background: 'rgba(8, 15, 30, 0.85)',
      backdropFilter: 'blur(20px)',
      borderBottom: '1px solid var(--color-border)',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 2rem',
    }}>
      {/* Left: logo mark */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <div style={{
          width: 36, height: 36, borderRadius: 10,
          background: 'linear-gradient(135deg, var(--color-accent), var(--color-accent-2))',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '1.1rem', boxShadow: 'var(--shadow-glow)',
        }}>🌧️</div>
        <div>
          <div style={{ fontWeight: 800, fontSize: '1rem', letterSpacing: '-0.01em' }}>
            RainSight <span className="text-accent">India</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)' }}>
            CNN+LSTM Rainfall Forecasting
          </div>
        </div>
      </div>

      {/* Right: status + badges */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {isTraining && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: '0.5rem',
            background: 'rgba(0,229,160,0.08)', border: '1px solid rgba(0,229,160,0.2)',
            borderRadius: 8, padding: '0.4rem 0.9rem',
          }}>
            <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />
            <span style={{ fontSize: '0.8rem', color: 'var(--color-success)', fontWeight: 600 }}>
              Training Epoch {status.epoch}/{status.total_epochs}
            </span>
          </div>
        )}

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span className="badge badge-accent">
            <span>⚡</span> Real IMD Data
          </span>
          <span className="badge badge-success">
            <div className="pulse-dot" />
            {status.status === 'complete' ? 'Model Ready' : status.status === 'idle' ? 'Untrained' : 'Live'}
          </span>
        </div>
      </div>
    </header>
  );
}
