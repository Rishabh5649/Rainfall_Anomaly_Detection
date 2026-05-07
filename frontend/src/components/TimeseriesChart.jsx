// TimeseriesChart.jsx — Historical + predicted rainfall chart (Recharts)
import {
  ResponsiveContainer, ComposedChart, Area, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine,
} from 'recharts';

const MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

function formatLabel(year, month) {
  return `${MONTH_NAMES[month - 1]} ${year}`;
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'var(--color-bg-elevated)',
      border: '1px solid var(--color-border-hover)',
      borderRadius: 8, padding: '0.75rem 1rem',
      boxShadow: 'var(--shadow-md)',
    }}>
      <div style={{ fontWeight: 600, marginBottom: '0.4rem', color: 'var(--color-text-primary)' }}>
        {label}
      </div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, fontSize: '0.85rem', display: 'flex', gap: '0.5rem' }}>
          <span>{p.name}:</span>
          <span style={{ fontWeight: 600, fontFamily: 'monospace' }}>
            {typeof p.value === 'number' ? `${p.value.toFixed(1)} mm` : '—'}
          </span>
        </div>
      ))}
    </div>
  );
};

export default function TimeseriesChart({ historyData, predictions, title, subdivision }) {
  // Combine historical and predicted data
  const historicalPoints = (historyData || [])
    .filter(d => d.year >= 1990)  // show recent history
    .map(d => ({
      label:       formatLabel(d.year, d.month),
      year:        d.year,
      month:       d.month,
      historical:  d.rainfall_mm,
      predicted:   null,
      lower:       null,
      upper:       null,
    }));

  const predPoints = (predictions || []).map(d => ({
    label:       formatLabel(d.year, d.month),
    year:        d.year,
    month:       d.month,
    historical:  null,
    predicted:   d.predicted_mm,
    lower:       d.lower_mm,
    upper:       d.upper_mm,
  }));

  const data = [...historicalPoints, ...predPoints];

  if (data.length === 0) {
    return (
      <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-text-muted)' }}>
        Select a subdivision to view data
      </div>
    );
  }

  return (
    <div>
      {title && (
        <div style={{ fontWeight: 700, fontSize: '1.05rem', marginBottom: '1rem', color: 'var(--color-text-primary)' }}>
          {title}
          {subdivision && (
            <span className="badge badge-accent" style={{ marginLeft: '0.75rem', verticalAlign: 'middle' }}>
              {subdivision}
            </span>
          )}
        </div>
      )}
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 8, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
          <XAxis
            dataKey="label"
            tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }}
            tickFormatter={(v, i) => (i % 12 === 0 ? v.split(' ')[1] : '')}
            interval="preserveStartEnd"
          />
          <YAxis tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} tickFormatter={v => `${v}`} />
          <Tooltip content={<CustomTooltip />} />
          <Legend/>

          {/* Confidence band */}
          {predPoints.length > 0 && (
            <Area
              dataKey="upper" data={predPoints}
              fill="rgba(0,212,255,0.08)" stroke="none" name="Upper CI"
            />
          )}

          {/* Historical */}
          <Area
            type="monotone" dataKey="historical"
            stroke="#44D9E6" strokeWidth={1.5}
            fill="rgba(68,217,230,0.08)"
            dot={false} name="Historical"
            connectNulls={false}
          />

          {/* Predicted */}
          <Line
            type="monotone" dataKey="predicted"
            stroke="var(--color-accent)" strokeWidth={2.5}
            strokeDasharray="6 3"
            dot={{ r: 3, fill: 'var(--color-accent)', strokeWidth: 0 }}
            activeDot={{ r: 6 }}
            name="Forecast"
            connectNulls={false}
          />

          {/* Divider between historical and forecast */}
          {predPoints.length > 0 && (
            <ReferenceLine
              x={formatLabel(predPoints[0].year, predPoints[0].month)}
              stroke="rgba(0,212,255,0.4)" strokeDasharray="4 4"
              label={{ value: 'Forecast start', fill: 'var(--color-accent)', fontSize: 10 }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
