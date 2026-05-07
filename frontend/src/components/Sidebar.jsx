// Sidebar.jsx — Navigation sidebar
import { NavLink } from 'react-router-dom';

const NAV_ITEMS = [
  { path: '/',          icon: '🏠', label: 'Dashboard' },
  { path: '/forecast',  icon: '🌧️', label: 'Forecast' },
  { path: '/analytics', icon: '📊', label: 'Analytics' },
];

export default function Sidebar() {
  return (
    <aside style={{
      position: 'fixed', top: 0, left: 0, bottom: 0,
      width: 'var(--sidebar-width)', zIndex: 200,
      background: 'rgba(8, 15, 30, 0.95)',
      backdropFilter: 'blur(20px)',
      borderRight: '1px solid var(--color-border)',
      display: 'flex', flexDirection: 'column',
      padding: '1.5rem 0',
    }}>
      {/* Logo */}
      <div style={{
        padding: '0.5rem 1.5rem 1.5rem',
        borderBottom: '1px solid var(--color-border)',
        marginBottom: '1rem',
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: '0.75rem',
        }}>
          <div style={{
            width: 42, height: 42, borderRadius: 12,
            background: 'linear-gradient(135deg, #00D4FF 0%, #7B2FFF 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1.3rem', boxShadow: '0 0 20px rgba(0,212,255,0.4)',
            flexShrink: 0,
          }}>🌧️</div>
          <div>
            <div style={{ fontWeight: 800, fontSize: '1.05rem' }}>
              Rain<span style={{ color: 'var(--color-accent)' }}>Sight</span>
            </div>
            <div style={{ fontSize: '0.68rem', color: 'var(--color-text-muted)', lineHeight: 1 }}>
              INDIA • IMD DATA
            </div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: '0 0.75rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
        <div style={{ fontSize: '0.68rem', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase', letterSpacing: '0.12em', padding: '0 0.75rem', marginBottom: '0.5rem' }}>
          Navigation
        </div>
        {NAV_ITEMS.map(({ path, icon, label }) => (
          <NavLink
            key={path}
            to={path}
            end={path === '/'}
            style={({ isActive }) => ({
              display: 'flex', alignItems: 'center', gap: '0.75rem',
              padding: '0.65rem 0.9rem', borderRadius: 10,
              textDecoration: 'none', fontWeight: 500, fontSize: '0.9rem',
              transition: 'all 0.2s ease',
              color: isActive ? 'var(--color-accent)' : 'var(--color-text-secondary)',
              background: isActive ? 'var(--color-accent-dim)' : 'transparent',
              border: `1px solid ${isActive ? 'rgba(0,212,255,0.2)' : 'transparent'}`,
            })}
          >
            <span style={{ fontSize: '1.1rem' }}>{icon}</span>
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div style={{
        padding: '1rem 1.5rem',
        borderTop: '1px solid var(--color-border)',
        margin: '0 0',
      }}>
        <div style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)', lineHeight: 1.6 }}>
          <div>📡 36 Meteorological Subdivisions</div>
          <div>📅 1901 – 2015 (IMD Data)</div>
          <div style={{ marginTop: '0.5rem', color: 'var(--color-text-accent)', fontWeight: 600 }}>
            CNN+LSTM Model
          </div>
        </div>
      </div>
    </aside>
  );
}
