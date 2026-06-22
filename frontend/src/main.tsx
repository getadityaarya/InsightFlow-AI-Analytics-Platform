import React from 'react';
import { createRoot } from 'react-dom/client';
import { Uploader } from './Uploader';

const rootElement = document.getElementById('upload-react-root');
if (rootElement) {
  const root = createRoot(rootElement);
  root.render(<Uploader />);
}
