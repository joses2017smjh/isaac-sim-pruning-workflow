import { createReadStream } from 'node:fs';
import { realpath, stat } from 'node:fs/promises';
import { dirname, extname, isAbsolute, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Plugin } from 'vite';
import type { Connect } from 'vite';

const root = dirname(fileURLToPath(import.meta.url));
// This directory is deliberately NOT publicDir: licensed meshes must not be
// silently copied into a distributable production build.
const bundle = resolve(root, process.env.STUDIO_ASSET_DIR ?? '../artifacts/studio-assets');
const mime: Record<string, string> = {
  '.json': 'application/json', '.urdf': 'application/xml', '.dae': 'model/vnd.collada+xml',
  '.stl': 'application/octet-stream', '.glb': 'model/gltf-binary', '.gltf': 'model/gltf+json',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.bin': 'application/octet-stream',
};
function localAssets(): Plugin {
  const serve: Connect.NextHandleFunction = async (request, response, next) => {
    if (!request.url?.startsWith('/local-assets/')) return next();
    if (!['GET', 'HEAD'].includes(request.method ?? '')) { response.writeHead(405).end(); return; }
    try {
      const file = decodeURIComponent(request.url.split('?')[0].slice('/local-assets/'.length));
      const actualRoot = await realpath(bundle);
      const actualFile = await realpath(resolve(actualRoot, file));
      const child = relative(actualRoot, actualFile);
      if (!child || isAbsolute(child) || child === '..' || child.startsWith(`..${sep}`)) {
        response.writeHead(403).end('Asset path must remain inside the local bundle'); return;
      }
      const info = await stat(actualFile);
      if (!info.isFile()) { response.writeHead(404).end(); return; }
      response.writeHead(200, { 'Content-Type': mime[extname(actualFile).toLowerCase()] ?? 'application/octet-stream',
        'Content-Length': info.size, 'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff' });
      if (request.method === 'HEAD') response.end();
      else createReadStream(actualFile).on('error', () => response.destroy()).pipe(response);
    } catch { response.writeHead(404).end('Local asset missing. See studio/NOTES.md.'); }
  };
  return { name: 'local-licensed-assets', configureServer(server) { server.middlewares.use(serve); },
    configurePreviewServer(server) { server.middlewares.use(serve); } };
}
export default defineConfig({
  plugins: [localAssets()], publicDir: false,
  server: { host: '127.0.0.1', strictPort: true }, preview: { host: '127.0.0.1', strictPort: true },
  test: { include: ['tests/**/*.test.ts'] },
});
