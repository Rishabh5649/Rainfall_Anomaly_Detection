// App.jsx — Root app with routing
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import Forecast from './pages/Forecast';
import Analytics from './pages/Analytics';

export default function App() {
  return (
    <BrowserRouter>
      <div className="layout">
        <Sidebar />
        <div className="main-content">
          <Header />
          <Routes>
            <Route path="/"          element={<Dashboard />} />
            <Route path="/forecast"  element={<Forecast />} />
            <Route path="/analytics" element={<Analytics />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
}
