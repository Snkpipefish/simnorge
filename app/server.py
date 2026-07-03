"""HTTP-server for spørretaben — Modul 8. Ren stdlib, ingen nye avhengigheter.

    .venv/bin/python -m app.server [port]

GET  /                 -> spørretaben (app/static/index.html)
GET  /api/kommuner     -> [{kode, navn}]
POST /api/ask          -> {sporsmal, geografi} -> AskResult.as_dict()

Første poll for en ny (spørsmål × geografi) tar ~1 min (24 LLM-kall via
claude-CLI); alt er diskcachet, så gjentak er momentane.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .qspec import QSpecError
from .service import SimNorgeService

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("simnorge.server")

STATIC = Path(__file__).parent / "static"
_service: SimNorgeService | None = None
_service_lock = threading.Lock()


def get_service() -> SimNorgeService:
    global _service
    with _service_lock:
        if _service is None:
            logger.warning("Initialiserer tjenesten (SSB-data, cachet) …")
            _service = SimNorgeService()
        return _service


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # roligere logg
        logger.info("%s " + fmt, self.address_string(), *args)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/kommuner":
            self._json(200, get_service().kommuner())
        elif self.path == "/api/kart":
            from .batch import list_maps
            self._json(200, list_maps())
        elif self.path.startswith("/kart/"):
            # Kun kjente filnavn under static/kart (ingen sti-triksing).
            # unquote: æøå i slug prosent-kodes av nettleseren.
            from urllib.parse import unquote
            name = Path(unquote(self.path)).name
            f = STATIC / "kart" / name
            if f.is_file() and f.suffix in (".html", ".json"):
                ctype = ("text/html; charset=utf-8" if f.suffix == ".html"
                         else "application/json; charset=utf-8")
                self._send(200, f.read_bytes(), ctype)
            else:
                self._json(404, {"feil": "ukjent kart"})
        else:
            self._json(404, {"feil": "ukjent sti"})

    def do_POST(self):
        if self.path != "/api/ask":
            self._json(404, {"feil": "ukjent sti"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            q = str(req.get("sporsmal", "")).strip()
            geo = str(req.get("geografi", "0")).strip() or "0"
            res = get_service().ask(q, geo)
            self._json(200, res.as_dict())
        except QSpecError as e:
            self._json(422, {"feil": f"Kunne ikke tolke spørsmålet: {e}"})
        except Exception as e:  # noqa: BLE001 — vis feilen ærlig i UI-et
            logger.exception("ask feilet")
            self._json(500, {"feil": str(e)})


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"SimNorge spørretab: http://127.0.0.1:{port}/  (Ctrl-C stopper)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
