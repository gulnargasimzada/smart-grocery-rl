"""Smart Grocery Coach - Streamlit app (Person 4).

Run from the repo root:
    streamlit run app.py

How it works
- ui_payload.py loads data/*.csv and, when it exists, results/results.json (see INTERFACES.md sections 2 and 9),
  and packs everything into one JSON payload for the designed interface in ui/coach.html.
- Streamlit shows that interface full-width through components.html.
- Until src/evaluate.py writes results/results.json, the interface runs a clearly labelled demo
  simulation of the four policies in the browser, on the same weekly demand data.
- The same page can be published as a static site with build_static.py (see netlify.toml).

Optional settings in .env: FDC_API_KEY, BARCODELOOKUP_API_KEY (barcode sources), EXPLAIN_URL (AI explanations).
"""
import streamlit as st
import streamlit.components.v1 as components

from ui_payload import DATA, DATA_FILES, EXTRA_FILES, OFFICIAL, PLANS, RESULTS, load_payload, render_page

APP_HEIGHT = 1500  # px; the interface scrolls inside this frame on smaller screens

st.set_page_config(page_title="Smart Grocery Coach", layout="wide", initial_sidebar_state="collapsed")

# Let the designed interface fill the page: hide Streamlit's header, footer and padding.
st.markdown(
    """
    <style>
      header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"] {display: none !important;}
      .block-container {padding: 0 !important; max-width: 100% !important;}
      [data-testid="stAppViewContainer"] > .main {padding: 0 !important;}
      iframe {border: 0; display: block;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_payload(mtimes: tuple) -> dict:
    """`mtimes` changes when a data file changes, which refreshes the cache."""
    return load_payload()


def main():
    files = [DATA / f for f in DATA_FILES]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        st.error(f"Missing data files: {', '.join(missing)}. Run `python -m src.data_gen` from the repo root.")
        return
    watched = files + [DATA / f for f in EXTRA_FILES] + [RESULTS, PLANS, OFFICIAL]
    mtimes = tuple(f.stat().st_mtime for f in watched if f.exists())
    components.html(render_page(cached_payload(mtimes)), height=APP_HEIGHT, scrolling=True)


if __name__ == "__main__":
    main()
