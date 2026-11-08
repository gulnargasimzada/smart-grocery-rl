"""Build the Grocery Coach as a static site for Netlify (or any static host).

    python build_static.py        # writes dist/index.html

API keys are never written into the static page. AI explanations go through the Netlify Function in
netlify/functions/explain.mjs, which keeps ANTHROPIC_API_KEY on the server.
"""
import os
from pathlib import Path

from ui_payload import ROOT, load_payload, render_page

OUT = ROOT / "dist"


def main():
    payload = load_payload(include_keys=False)
    payload["explain_url"] = os.environ.get("STATIC_EXPLAIN_URL", "/.netlify/functions/explain")
    OUT.mkdir(exist_ok=True)
    page = render_page(payload).replace("<head>", "<head><title>Smart Grocery Coach</title>", 1)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {OUT / 'index.html'} ({(OUT / 'index.html').stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
