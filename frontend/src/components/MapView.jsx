// MapView.jsx — India subdivision heatmap (SVG simplified)
// Each subdivision is a labelled box, coloured by rainfall intensity
import { useState, useEffect } from 'react';
import { fetchSubdivisions, fetchHistory, fetchPrediction } from '../api';

// Colour scale: 0 → dull blue, 800+ → vivid cyan
function rainfallToColor(mm) {
  if (mm === null || mm === undefined) return '#1a2540';
  const t = Math.min(mm / 800, 1);
  // interpolate from #1a2540 to #00D4FF
  const r = Math.round(26  + t * (0   - 26));
  const g = Math.round(37  + t * (212 - 37));
  const b = Math.round(64  + t * (255 - 64));
  return `rgb(${r},${g},${b})`;
}

function RainfallLegend() {
  const stops = [0, 100, 200, 400, 600, 800];
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', justifyContent: 'flex-end', marginBottom: '0.75rem' }}>
      <span style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)' }}>0 mm</span>
      <div style={{
        width: 120, height: 8, borderRadius: 4,
        background: 'linear-gradient(90deg, #1a2540 0%, #00D4FF 100%)',
        border: '1px solid var(--color-border)',
      }} />
      <span style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)' }}>800+ mm</span>
    </div>
  );
}

// ─── Layout: group subdivisions by region ───────────────────────
const REGION_GROUPS = {
  'North-West': ['PUNJAB', 'HARYANA DELHI & CHANDIGARH', 'HIMACHAL PRADESH', 'JAMMU & KASHMIR', 'WEST RAJASTHAN', 'EAST RAJASTHAN'],
  'North': ['UTTARAKHAND', 'WEST UTTAR PRADESH', 'EAST UTTAR PRADESH', 'BIHAR'],
  'North-East': ['ARUNACHAL PRADESH', 'ASSAM & MEGHALAYA', 'NAGA MANI MIZO TRIPURA', 'SUB HIMALAYAN WEST BENGAL & SIKKIM'],
  'Central': ['EAST MADHYA PRADESH', 'WEST MADHYA PRADESH', 'CHATTISGARH', 'JHARKHAND', 'GANGETIC WEST BENGAL', 'ORISSA'],
  'West': ['GUJARAT REGION', 'SAURASHTRA KUTCH & DIU', 'KONKAN & GOA', 'MADHYA MAHARASHTRA', 'MARATHWADA'],
  'South': ['COASTAL ANDHRA PRADESH', 'TELANGANA', 'RAYALASEEMA', 'TAMIL NADU', 'COASTAL KARNATAKA', 'NORTH INTERIOR KARNATAKA', 'SOUTH INTERIOR KARNATAKA', 'KERALA', 'LAKSHADWEEP'],
  'Islands': ['ANDAMAN & NICOBAR ISLANDS'],
};

export default function MapView({ onSelectSubdivision, selectedSubdivision }) {
  const [subdivisions, setSubdivisions]   = useState([]);
  const [rainfallMap, setRainfallMap]     = useState({});
  const [hoveredSub, setHoveredSub]       = useState(null);
  const [loadingMap, setLoadingMap]       = useState(true);

  useEffect(() => {
    fetchSubdivisions().then(subs => {
      setSubdivisions(subs);
      // Load latest annual rainfall for each subdivision
      const loadAll = subs.map(sub =>
        fetchHistory(sub)
          .then(d => {
            // average the last 3 years
            const recent = d.data.filter(r => r.year >= 2012);
            const avg = recent.length > 0
              ? recent.reduce((s, r) => s + r.rainfall_mm, 0) / recent.length
              : null;
            return [sub, avg];
          })
          .catch(() => [sub, null])
      );
      Promise.all(loadAll).then(pairs => {
        setRainfallMap(Object.fromEntries(pairs));
        setLoadingMap(false);
      });
    }).catch(() => setLoadingMap(false));
  }, []);

  if (loadingMap) {
    return (
      <div className="loading-state">
        <div className="spinner" />
        <span>Loading subdivision data…</span>
      </div>
    );
  }

  return (
    <div>
      <RainfallLegend />
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {Object.entries(REGION_GROUPS).map(([region, subList]) => {
          const validSubs = subList.filter(s => subdivisions.includes(s));
          if (validSubs.length === 0) return null;
          return (
            <div key={region}>
              <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.4rem' }}>
                {region}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                {validSubs.map(sub => {
                  const rf  = rainfallMap[sub];
                  const bg  = rainfallToColor(rf);
                  const sel = sub === selectedSubdivision;
                  const hov = sub === hoveredSub;
                  return (
                    <button
                      key={sub}
                      onClick={() => onSelectSubdivision?.(sub)}
                      onMouseEnter={() => setHoveredSub(sub)}
                      onMouseLeave={() => setHoveredSub(null)}
                      title={`${sub}: ${rf !== null && rf !== undefined ? rf.toFixed(0) + ' mm avg' : 'No data'}`}
                      style={{
                        padding: '0.35rem 0.65rem',
                        borderRadius: 6,
                        border: `1px solid ${sel ? 'var(--color-accent)' : hov ? 'rgba(0,212,255,0.3)' : 'transparent'}`,
                        background: bg,
                        color: rf > 200 ? '#000' : 'var(--color-text-primary)',
                        fontSize: '0.72rem',
                        fontWeight: sel ? 700 : 500,
                        cursor: 'pointer',
                        transition: 'all 0.15s ease',
                        transform: sel || hov ? 'scale(1.05)' : 'scale(1)',
                        boxShadow: sel ? 'var(--shadow-glow)' : 'none',
                        whiteSpace: 'nowrap',
                        fontFamily: 'inherit',
                      }}
                    >
                      {sub.length > 22 ? sub.slice(0, 20) + '…' : sub}
                      {rf !== null ? (
                        <span style={{ marginLeft: '0.3rem', opacity: 0.75, fontSize: '0.65rem' }}>
                          {rf.toFixed(0)}
                        </span>
                      ) : null}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}

        {/* Show unmapped subdivisions */}
        {(() => {
          const mapped = Object.values(REGION_GROUPS).flat();
          const unmapped = subdivisions.filter(s => !mapped.includes(s));
          if (!unmapped.length) return null;
          return (
            <div>
              <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.4rem' }}>
                Other
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                {unmapped.map(sub => {
                  const rf  = rainfallMap[sub];
                  const bg  = rainfallToColor(rf);
                  const sel = sub === selectedSubdivision;
                  return (
                    <button key={sub} onClick={() => onSelectSubdivision?.(sub)}
                      style={{
                        padding: '0.35rem 0.65rem', borderRadius: 6,
                        border: `1px solid ${sel ? 'var(--color-accent)' : 'transparent'}`,
                        background: bg,
                        color: rf > 200 ? '#000' : 'var(--color-text-primary)',
                        fontSize: '0.72rem', fontWeight: sel ? 700 : 500,
                        cursor: 'pointer', transition: 'all 0.15s ease',
                        fontFamily: 'inherit',
                      }}>
                      {sub}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
}
