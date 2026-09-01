"""driftlab dashboard: watch agents work, live or replayed, in a browser.

    python -m driftlab.viz.server              # http://localhost:8765
    python -m driftlab.viz.server --port 9000 --events runs/events.jsonl

The page tails the events file that every run appends to (in-process runs,
harness runs through the MCP server, all of them) and renders: the current
observation, the agent's reply and the outcome; world changes, notices and
memory rewrites as they happen; rolling and cumulative accuracy with event
markers; a per-request-key grid for RuleWorld; running metrics and the final
analysis table. Finished runs can be replayed from their JSONL logs.

No dependencies beyond the standard library.
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EVENTS = Path(os.environ.get("DRIFTLAB_EVENTS", ROOT / "runs" / "events.jsonl"))


def _clean_json(path: Path) -> str:
    """Python's json writes bare NaN, which browsers' JSON.parse rejects; serve null instead."""
    return json.dumps(json.loads(path.read_text(), parse_constant=lambda c: None))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/events":
            return self._sse()
        if u.path == "/registry":
            try:
                sys.path.insert(0, str(ROOT))
                from experiments.registry import REGISTRY
                return self._send(200, json.dumps(REGISTRY))
            except Exception:  # noqa: BLE001  (the page renders fine without it)
                return self._send(200, "{}")
        if u.path == "/profile":
            rel = parse_qs(u.query).get("dir", [""])[0]
            path = (ROOT / rel / "profile.json").resolve()
            if not str(path).startswith(str(ROOT / "runs")) or not path.exists():
                return self._send(200, "null")
            return self._send(200, _clean_json(path))
        if u.path == "/benchmarks":
            out = {p.stem: json.loads(_clean_json(p)) for p in sorted((ROOT / "runs" / "benchmark").glob("*.json"))} \
                if (ROOT / "runs" / "benchmark").is_dir() else {}
            return self._send(200, json.dumps(out))
        if u.path == "/hypotheses":
            items = []
            for p in sorted((ROOT / "research" / "hypotheses").glob("H*.md")):
                lines = p.read_text().splitlines()
                title = lines[0].lstrip("# ").split("—", 1)[-1].strip() if lines else p.stem
                status = next((ln.split(":", 1)[1].strip() for ln in lines if ln.lower().startswith("status:")), "")
                items.append({"id": p.name.split("-")[0], "file": p.name, "title": title,
                              "status": status, "body": p.read_text()})
            return self._send(200, json.dumps(items))
        if u.path == "/runs":
            logs = sorted(ROOT.joinpath("runs").rglob("*.jsonl"))
            return self._send(200, json.dumps([str(p.relative_to(ROOT)) for p in logs if p.name != "events.jsonl"]))
        if u.path == "/log":
            rel = parse_qs(u.query).get("path", [""])[0]
            path = (ROOT / rel).resolve()
            if not str(path).startswith(str(ROOT / "runs")) or not path.exists():
                return self._send(404, json.dumps({"error": "not found"}))
            lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            return self._send(200, json.dumps({"header": lines[0], "steps": lines[1:]}, default=str))
        self._send(404, json.dumps({"error": "not found"}))

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        pos, last_beat = 0, time.time()
        try:
            while True:
                if EVENTS.exists():
                    with EVENTS.open("rb") as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos = f.tell()
                    if chunk:
                        for line in chunk.splitlines():
                            if line.strip():
                                self.wfile.write(b"data: " + line + b"\n\n")
                        self.wfile.flush()
                if time.time() - last_beat > 15:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    last_beat = time.time()
                time.sleep(0.25)
        except (BrokenPipeError, ConnectionResetError):
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--events", default=None, help="events file to tail (default runs/events.jsonl)")
    ap.add_argument("--open", action="store_true", help="open the browser")
    args = ap.parse_args()
    global EVENTS
    if args.events:
        EVENTS = Path(args.events).resolve()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"driftlab dashboard at {url}  (tailing {EVENTS})", flush=True)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
