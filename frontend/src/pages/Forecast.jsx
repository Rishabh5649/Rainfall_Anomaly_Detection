// Forecast.jsx — Dedicated forecast page
import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import ForecastPanel from '../components/ForecastPanel';
import MapView from '../components/MapView';

export default function Forecast() {
  const [searchParams] = useSearchParams();
  const [selectedSub, setSelectedSub] = useState(searchParams.get('sub') || '');

  return (
    <div className="page">
      <div className="page-header anim-fade-up">
        <h1 className="page-title">🌧️ Rainfall Forecast</h1>
        <p className="page-subtitle">
          Select a subdivision, set the forecast horizon, and generate monthly predictions.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.6fr', gap: '1.5rem', alignItems: 'start' }}>
        {/* Subdivision chooser */}
        <div className="card anim-fade-up delay-1">
          <div className="card-header">
            <span>🗺️</span><h3>Select Subdivision</h3>
          </div>
          <MapView
            onSelectSubdivision={setSelectedSub}
            selectedSubdivision={selectedSub}
          />
        </div>

        {/* Forecast controls + output */}
        <div className="anim-fade-up delay-2">
          <ForecastPanel initialSubdivision={selectedSub} />
        </div>
      </div>
    </div>
  );
}
