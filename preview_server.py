#!/usr/bin/env python3
"""Minimal Audio & Ground-Truth Timeline Inspector for the Synthetic Voice Pipeline."""

from __future__ import annotations

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import sys
from urllib.parse import unquote

ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "output_dataset"

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Synthetic Voice Pipeline — Audio & Variation Inspector</title>
<style>
  :root {
    --bg: #fafafa;
    --card: #ffffff;
    --text: #111827;
    --muted: #6b7280;
    --border: #e5e7eb;
    --accent: #111827;
    --alert: #dc2626;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px 32px;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 24px;
  }
  h1 { font-size: 20px; font-weight: 600; margin: 0; }
  .subtitle { color: var(--muted); font-size: 13px; }
  .layout { display: grid; grid-template-columns: 320px 1fr; gap: 24px; }
  .panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }
  .call-item {
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: background 0.12s;
  }
  .call-item:hover { background: #f3f4f6; }
  .call-item.active {
    border-color: var(--accent);
    background: #f3f4f6;
    font-weight: 600;
  }
  .call-meta { font-size: 12px; color: var(--muted); margin-top: 4px; font-weight: 400; }
  .audio-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 12px;
    margin: 16px 0;
  }
  .audio-box {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    background: #fcfcfc;
  }
  .audio-label { font-size: 12px; font-weight: 600; margin-bottom: 6px; display: block; }
  audio { width: 100%; height: 34px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 12px; }
  th, td {
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
  .badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }
  .badge-voice { background: #111827; color: #fff; }
  .badge-block { background: #fef2f2; color: var(--alert); border: 1px solid #fecaca; }
  .play-seg-btn {
    border: 1px solid var(--border);
    background: #fff;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    cursor: pointer;
  }
  .play-seg-btn:hover { background: #f3f4f6; }
</style>
</head>
<body>
<header>
  <div>
    <h1>Synthetic Voice Data Pipeline — Audio & Ground-Truth Preview</h1>
    <div class="subtitle">Gemini 3.8 Flash TTS (`gemini-3.8-flash-tts`) • 4-Stem Composition • 16 kHz & 8 kHz Telephony</div>
  </div>
  <div class="subtitle" id="summary-stats"></div>
</header>
<div class="layout">
  <div class="panel">
    <div style="font-size:12px;font-weight:600;text-transform:uppercase;color:var(--muted);margin-bottom:10px;">Generated Showcase Calls</div>
    <div id="call-list"></div>
  </div>
  <div class="panel" id="detail-pane">
    <div id="call-header"></div>
    <div class="audio-grid" id="audio-players"></div>
    <div style="font-size:12px;font-weight:600;text-transform:uppercase;color:var(--muted);margin-top:20px;">Ground-Truth Segments (Click timestamp to play slice)</div>
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Window (s)</th>
          <th>Dur</th>
          <th>Segment Role</th>
          <th>Expected VAD</th>
          <th>Variation / Acoustic Tag & Transcript</th>
        </tr>
      </thead>
      <tbody id="seg-tbody"></tbody>
    </table>
  </div>
</div>
<script>
let manifest = [];
let currentAudio = null;

async function init() {
  const res = await fetch('/api/manifest');
  manifest = await res.json();
  const totDur = manifest.reduce((a, b) => a + b.duration_s, 0).toFixed(1);
  document.getElementById('summary-stats').textContent = `${manifest.length} Calls • ${totDur}s Total Audio`;
  const listEl = document.getElementById('call-list');
  manifest.forEach((item, idx) => {
    const div = document.createElement('div');
    div.className = 'call-item' + (idx === 0 ? ' active' : '');
    div.innerHTML = `<div>${item.id}</div><div class="call-meta">${item.language} • ${item.primary_voice} • ${item.duration_s.toFixed(1)}s (${item.n_total_segments} segs)</div>`;
    div.onclick = () => {
      document.querySelectorAll('.call-item').forEach(el => el.classList.remove('active'));
      div.classList.add('active');
      loadCall(item.id);
    };
    listEl.appendChild(div);
  });
  if (manifest.length > 0) loadCall(manifest[0].id);
}

async function loadCall(cid) {
  const res = await fetch(`/api/gt/${cid}`);
  const gt = await res.json();
  document.getElementById('call-header').innerHTML = `
    <div style="font-size:17px;font-weight:600;">${gt.title}</div>
    <div style="font-size:13px;color:var(--muted);margin-top:4px;"><strong>Variations Tested:</strong> ${gt.failure_focus}</div>
  `;
  const players = [
    { label: 'Full Mix — 8 kHz Telephony (audio_8k)', src: `/files/audio_8k/${cid}.wav`, id: 'main-player' },
    { label: 'Full Mix — 16 kHz Wideband (audio_16k)', src: `/files/audio_16k/${cid}.wav`, id: 'player-16k' },
    { label: 'Stem 1: Primary Caller Voice Only', src: `/files/stems/${cid}_primary_voice.wav` },
    { label: 'Stem 2: Competing Background Voice Only', src: `/files/stems/${cid}_background_voice.wav` },
    { label: 'Stem 3: Compound Multi-Noise Bursts Only', src: `/files/stems/${cid}_noise_bursts.wav` },
    { label: 'Stem 4: Continuous Ambient Bed Only', src: `/files/stems/${cid}_ambient_bed.wav` },
  ];
  document.getElementById('audio-players').innerHTML = players.map(p => `
    <div class="audio-box">
      <span class="audio-label">${p.label}</span>
      <audio controls preload="metadata" ${p.id ? `id="${p.id}"` : ''} src="${p.src}"></audio>
    </div>
  `).join('');

  const tbody = document.getElementById('seg-tbody');
  tbody.innerHTML = gt.segments.map(s => {
    const isVoice = s.should_trigger;
    const roleLabel = s.label === 'voice' ? 'Caller Speech' : (s.label === 'background_voice' ? 'Competing Background Voice' : 'Background Noise Burst');
    const expBadge = isVoice
      ? '<span class="badge badge-voice">Trigger (Caller Speech)</span>'
      : '<span class="badge badge-block">Block (Do Not Trigger)</span>';
    const extra = s.overlap_tags && s.overlap_tags.length ? ` • Overlaps: ${s.overlap_tags.join(', ')}` : '';
    const txt = s.transcript ? `<div style="margin-top:4px;color:#374151;">“${s.transcript}”</div>` : '';
    return `
      <tr>
        <td><code>${s.segment_id}</code></td>
        <td><button class="play-seg-btn" onclick="playSlice(${s.start}, ${s.end})">▶ ${s.start.toFixed(2)}s – ${s.end.toFixed(2)}s</button></td>
        <td>${s.duration_s.toFixed(2)}s</td>
        <td>${roleLabel}</td>
        <td>${expBadge}</td>
        <td><div><strong>${s.tag}</strong>${extra}</div>${txt}</td>
      </tr>
    `;
  }).join('');
}

function playSlice(start, end) {
  const player = document.getElementById('player-16k');
  if (!player) return;
  player.currentTime = start;
  player.play();
  const check = setInterval(() => {
    if (player.currentTime >= end || player.paused) {
      player.pause();
      clearInterval(check);
    }
  }, 40);
}

init();
</script>
</body>
</html>
"""


class PreviewHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        path = unquote(self.path.split("?")[0])
        if path in ("/", "/index.html"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/manifest":
            lines = (OUT_DIR / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
            rows = [json.loads(line) for line in lines if line.strip()]
            body = json.dumps(rows).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/api/gt/"):
            cid = path.split("/api/gt/", 1)[1]
            gt_path = OUT_DIR / "ground_truth" / f"{cid}.json"
            if gt_path.exists():
                body = gt_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
        if path.startswith("/files/"):
            rel = path.split("/files/", 1)[1]
            target = (OUT_DIR / rel).resolve()
            if target.exists() and str(target).startswith(str(OUT_DIR.resolve())):
                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        self.send_error(404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8792
    srv = HTTPServer(("0.0.0.0", port), PreviewHandler)
    print(f"Preview server listening on http://roshah.c.googlers.com:{port}")
    srv.serve_forever()
