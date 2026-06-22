import React, { useRef, useState } from 'react';
import { API_BASE } from './api';

export function Uploader() {
  const [dragover, setDragover] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragover(true); };
  const handleDragLeave = (e: React.DragEvent) => { e.preventDefault(); setDragover(false); };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragover(false);
    const f = e.dataTransfer.files[0];
    if (f) uploadFile(f);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) uploadFile(f);
  };

  const uploadFile = async (file: File) => {
    const w = window as any;
    if (w.showProgress) w.showProgress('Uploading dataset...', 10);
    
    const fd = new FormData();
    fd.append('file', file);
    if (w.S && w.S.sessionId) fd.append('session_id', w.S.sessionId);

    try {
      if (w.updateProgress) w.updateProgress(35);
      const res = await fetch(`${API_BASE}/upload/`, { method: 'POST', body: fd });
      if (w.updateProgress) w.updateProgress(80);
      
      if (!res.ok) {
        const e = await res.json();
        throw new Error(e.detail || 'Upload failed');
      }
      
      const data = await res.json();
      if (w.updateProgress) w.updateProgress(100);
      
      setTimeout(() => {
        if (w.hideProgress) w.hideProgress();
        if (w.activateChatMode) w.activateChatMode(data);
      }, 400);
      
    } catch (err: any) {
      if (w.hideProgress) w.hideProgress();
      if (w.toast) w.toast(err.message || 'Upload failed', 'error');
      console.error(err);
    }
  };

  return (
    <div
      className={`drop-zone ${dragover ? 'dragover' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <div className="drop-icon">📁</div>
      <div className="drop-title">Drop your dataset here</div>
      <div className="drop-sub">CSV • Excel • SQLite — up to 100 MB</div>
      <div className="file-types">
        <span className="file-badge">CSV</span>
        <span className="file-badge">XLSX</span>
        <span className="file-badge">SQLite</span>
      </div>
      <input
        type="file"
        id="file-input"
        accept=".csv,.xlsx,.xls,.sqlite,.db"
        onChange={handleFileSelect}
        ref={fileInputRef}
        style={{ display: 'none' }}
      />
    </div>
  );
}
