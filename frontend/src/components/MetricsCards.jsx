// MetricsCards.jsx — Classification metric stat cards
import { useState, useEffect } from 'react';
import { fetchMetrics } from '../api';

function StatCard({ label, value, unit, icon, color, delay }) {
  return (
    <div className={`stat-card anim-fade-up delay-${delay}`}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div className="stat-label">{label}</div>
        <div className="stat-icon" style={{ background: `${color}20`, color }}>
          {icon}
        </div>
      </div>
      <div className="stat-value" style={{ color }}>
        {value !== null && value !== undefined
          ? typeof value === 'number' ? value.toFixed(2) : value
          : <span style={{ color: 'var(--color-text-muted)', fontSize: '1rem' }}>—</span>
        }
      </div>
      {unit && <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>{unit}</div>}
    </div>
  );
}

export default function MetricsCards() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetchMetrics().then(setMetrics).catch(() => {});
  }, []);

  const m = metrics;

  return (
    <div className="grid grid-4">
      <StatCard
        label="Accuracy" delay={1} icon="✅" color="var(--color-success)"
        value={m?.test_accuracy} unit="test split"
      />
      <StatCard
        label="F1-Score" delay={2} icon="🎯" color="var(--color-accent)"
        value={m?.test_f1_score} unit="test split"
      />
      <StatCard
        label="Precision" delay={3} icon="📌" color="var(--color-accent-2)"
        value={m?.test_precision} unit="test split"
      />
      <StatCard
        label="Recall / Sensitivity" delay={4} icon="🔎" color="var(--color-warning)"
        value={m?.test_recall} unit="test split"
      />
      <StatCard
        label="AUC" delay={1} icon="📈" color="var(--color-accent)"
        value={m?.test_auc} unit="test split"
      />
      <StatCard
        label="Error Rate" delay={2} icon="❌" color="var(--color-accent-3)"
        value={m?.test_error_rate} unit="test split"
      />
    </div>
  );
}
