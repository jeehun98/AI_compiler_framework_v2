import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import report from '../public/report.json';

createRoot(document.getElementById('root')!).render(<StrictMode><App report={report} /></StrictMode>);
