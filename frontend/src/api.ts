export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export const API_BASE = `${API_URL}/api`;

// Expose to window for legacy index.html scripts
(window as any).API_BASE = API_BASE;
