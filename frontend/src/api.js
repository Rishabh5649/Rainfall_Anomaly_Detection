// api.js — Centralised Axios API client
import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${BASE_URL}/api`,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

/* ── Endpoints ─────────────────────────────────────────────────── */

export const fetchSubdivisions = () =>
  api.get('/subdivisions').then(r => r.data);

export const fetchHistory = (subdivision) =>
  api.get(`/history/${encodeURIComponent(subdivision)}`).then(r => r.data);

export const fetchMetrics = () =>
  api.get('/metrics').then(r => r.data);

export const fetchTrainHistory = () =>
  api.get('/train/history').then(r => r.data);

export const fetchTrainStatus = () =>
  api.get('/train/status').then(r => r.data);

export const startTraining = (params) =>
  api.post('/train', params).then(r => r.data);

export const fetchPrediction = (subdivision, startYear, startMonth, horizon) =>
  api.post('/predict', {
    subdivision,
    start_year:  startYear,
    start_month: startMonth,
    horizon,
  }).then(r => r.data);

export default api;
