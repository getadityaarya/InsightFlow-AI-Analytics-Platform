import re

with open('frontend/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Replace CSS
html = html.replace('html,body{height:100%}', 'html,body{min-height:100vh;overflow-y:auto}')
html = html.replace('body{font-family:var(--font-body);background:var(--bg);color:var(--text);overflow:hidden;-webkit-font-smoothing:antialiased}', 'body{font-family:var(--font-body);background:var(--bg);color:var(--text);-webkit-font-smoothing:antialiased;padding:2rem;margin-bottom:3rem}')
html = html.replace('#app{display:flex;height:100vh}', '#app{display:flex;min-height:100vh;min-height:0}')

# Replace API definition with script injection
html = re.sub(r'<script>\n// ─── Config ───────────────────────────────────────────────────────────────────\nconst API_URL = import.meta.env.VITE_API_URL \|\| \'http://localhost:8000\';\nconst API_BASE = `\$\{API_URL\}/api`;', '<script type="module" src="/src/main.tsx"></script>\n<script>\n// ─── Config ───────────────────────────────────────────────────────────────────\n// API_BASE is now provided by src/api.ts', html)

# Replace drop zone
html = re.sub(r'<div class="drop-zone" id="drop-zone".*?</div>\s*<button class="upload-btn".*?</button>\s*</div>', '<div id="upload-react-root"></div>', html, flags=re.DOTALL)

with open('frontend/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
