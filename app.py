import io
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from PIL import Image
from ultralytics import YOLO

BASE = Path(__file__).parent
MODEL_PATH = BASE / "best.pt"
SAMPLES = sorted((BASE / "samples").glob("*.jpg"))
APP_NAME = "TraceScan"

st.set_page_config(page_title=f"{APP_NAME} | PCB Defect Inspection", layout="wide")

# ---------------------------------------------------------------------------
# Background: faint copper traces on solder-mask green
# ---------------------------------------------------------------------------
TRACES = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='280' height='280' viewBox='0 0 280 280'>"
    "<g fill='none' stroke='rgba(217,164,65,0.12)' stroke-width='1.4' stroke-linejoin='round'>"
    "<path d='M0 60 H80 L105 85 H200 L225 60 H280'/>"
    "<path d='M0 170 H50 L75 145 H150 L175 170 H280'/>"
    "<path d='M0 235 H110 L135 210 H210 L235 235 H280'/>"
    "<path d='M140 0 V45 L160 65 V120 L140 140 V280'/>"
    "</g>"
    "<g fill='rgba(217,164,65,0.24)'>"
    "<circle cx='80' cy='60' r='3'/><circle cx='200' cy='85' r='3'/>"
    "<circle cx='75' cy='145' r='3'/><circle cx='175' cy='170' r='3'/>"
    "<circle cx='135' cy='210' r='3'/><circle cx='160' cy='65' r='3'/>"
    "</g></svg>"
)
PATTERN = "data:image/svg+xml;utf8," + quote(TRACES)

LOGO = (
    "<svg width='46' height='46' viewBox='0 0 48 48' xmlns='http://www.w3.org/2000/svg'>"
    "<rect x='1' y='1' width='46' height='46' rx='11' fill='#0f3328' stroke='#d9a441' stroke-width='1.5'/>"
    "<path d='M8 32 H18 L24 26 H32' stroke='#d9a441' stroke-width='2.2' fill='none' "
    "stroke-linecap='round' stroke-linejoin='round'/>"
    "<circle cx='35' cy='26' r='3.6' fill='#d9a441'/>"
    "<path d='M12 17 V12 H17 M31 12 H36 V17 M36 35 V40 H31 M17 40 H12 V35' stroke='#e8efe9' "
    "stroke-width='1.8' fill='none' stroke-linecap='round' stroke-linejoin='round'/>"
    "</svg>"
)

CSS = """
:root{--copper:#d9a441;--ok:#4ade80;--bad:#f87171;--text:#e8efe9;--muted:#8aa398;--line:rgba(255,255,255,.09);}
.stApp{background:url("__PATTERN__"),linear-gradient(180deg,#0c2b21 0%,#08201a 100%);background-attachment:fixed;color:var(--text);}
#MainMenu, footer{visibility:hidden;}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1160px;margin:1.6rem auto 2rem auto;padding:2.2rem 2.6rem 2.6rem 2.6rem !important;background:rgba(4,18,13,.84);border:1px solid var(--line);border-radius:18px;box-shadow:0 24px 70px rgba(0,0,0,.45);}
@media(max-width:800px){.block-container{padding:1.4rem 1.1rem 1.8rem 1.1rem !important;margin:.6rem;}}
.brand{display:flex;align-items:center;gap:14px;}
.brand .name{font-size:28px;font-weight:700;letter-spacing:-.01em;line-height:1.1;}
.brand .name span{color:var(--copper);}
.brand .tag{font-size:13px;color:var(--muted);margin-top:2px;}
.lede{margin:20px 0 6px;max-width:760px;font-size:16.5px;line-height:1.65;color:#cfdcd4;}
.lede b{color:var(--text);}
.rule{height:1px;background:linear-gradient(90deg,rgba(217,164,65,.5),transparent);margin:22px 0 4px;}
.label{font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:20px 0 8px;}
.verdict{padding:14px 16px;border-radius:10px;background:rgba(0,0,0,.30);border:1px solid var(--line);border-left:4px solid var(--ok);font-size:17px;font-weight:600;}
.verdict.fail{border-left-color:var(--bad);}
.verdict small{display:block;font-weight:400;font-size:12.5px;color:var(--muted);margin-top:2px;}
.nums{display:grid;grid-template-columns:repeat(3,1fr);margin:12px 0 4px;border:1px solid var(--line);border-radius:10px;background:rgba(0,0,0,.22);}
.num{padding:12px 14px;border-right:1px solid var(--line);}
.num:last-child{border-right:none;}
.num .v{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:21px;font-weight:600;}
.num .k{font-size:11.5px;color:var(--muted);margin-top:2px;}
.row{margin:10px 0;font-size:14px;}
.row .top{display:flex;justify-content:space-between;margin-bottom:5px;}
.row .cnt{font-family:ui-monospace,Menlo,Consolas,monospace;color:var(--copper);}
.row .bar{height:6px;border-radius:3px;background:rgba(255,255,255,.07);overflow:hidden;}
.row .bar i{display:block;height:100%;background:var(--copper);border-radius:3px;}
.note{font-size:13px;color:var(--muted);line-height:1.6;}
.foot{margin-top:30px;padding-top:14px;border-top:1px solid var(--line);font-size:12px;color:var(--muted);display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;}
[data-testid="stFileUploaderDropzone"]{background:rgba(0,0,0,.25);border:1.5px dashed rgba(217,164,65,.5);border-radius:12px;}
[data-testid="stImage"] img{border-radius:8px;border:1px solid rgba(217,164,65,.45);width:100%;}
[data-baseweb="tab-list"]{gap:6px;}
.stButton button{background:rgba(217,164,65,.12);border:1px solid rgba(217,164,65,.55);color:var(--text);border-radius:8px;font-weight:600;}
.stButton button:hover{background:rgba(217,164,65,.22);border-color:var(--copper);color:var(--text);}
.stDownloadButton button{background:var(--copper);border:none;border-radius:8px;font-weight:600;}
.stDownloadButton button p{color:#1a1204 !important;font-weight:600;}
.stDownloadButton button:hover{filter:brightness(1.08);}
"""
CSS = " ".join(l.strip() for l in CSS.splitlines()).replace("__PATTERN__", PATTERN)
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model + state
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


if not MODEL_PATH.exists():
    st.error("best.pt not found. Put the trained weights next to app.py.")
    st.stop()

model = load_model()
class_names = list(model.names.values())

if "sample" not in st.session_state:
    st.session_state["sample"] = None


def pick_sample(path):
    st.session_state["sample"] = str(path)


def clear_sample():
    st.session_state["sample"] = None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"<div class='brand'>{LOGO}<div><div class='name'>Trace<span>Scan</span></div>"
    "<div class='tag'>AI visual inspection for printed circuit boards</div></div></div>"
    "<div class='lede'>A single missed fault can scrap an entire production batch. "
    f"<b>{APP_NAME}</b> checks each board in seconds and marks exactly where the problem is, "
    "so defects are caught before assembly, not after.</div>"
    "<div class='rule'></div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input row: upload or sample
# ---------------------------------------------------------------------------
st.markdown("<div class='label'>Board image</div>", unsafe_allow_html=True)
up_col, sample_col = st.columns([3, 2], gap="large")

with up_col:
    file = st.file_uploader(
        "Upload a PCB image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

with sample_col:
    if SAMPLES:
        thumb, info = st.columns([1, 1.3])
        with thumb:
            st.image(str(SAMPLES[0]))
        with info:
            st.markdown(
                "<div class='note'>No board at hand?<br>Try a ready-made sample.</div>",
                unsafe_allow_html=True,
            )
            st.button("Try sample board", key="sample_btn", on_click=pick_sample, args=(SAMPLES[0],))
            if st.session_state["sample"] and not file:
                st.button("Clear", key="clear_btn", on_click=clear_sample)

# Which image to analyse
img = None
if file:
    img = Image.open(file).convert("RGB")
elif st.session_state["sample"]:
    img = Image.open(st.session_state["sample"]).convert("RGB")

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
if img is None:
    kinds = " &nbsp;&middot;&nbsp; ".join(n.replace("_", " ") for n in class_names)
    st.markdown(
        f"<div class='label'>What it detects</div><div class='note'>{kinds}</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Result: big image (left) + summary (right)
# ---------------------------------------------------------------------------
else:
    st.markdown("<div class='label'>Result</div>", unsafe_allow_html=True)
    left, right = st.columns([2, 1], gap="large")

    with right:
        conf = st.slider(
            "Sensitivity (confidence threshold)",
            0.10,
            0.90,
            0.40,
            0.05,
            help="Lower finds more defects but adds false alarms. Higher is cleaner but may miss some.",
        )
        with st.expander("Display options"):
            show_labels = st.toggle("Show labels", value=True)
            show_scores = st.toggle("Show scores", value=True)

    t0 = time.perf_counter()
    with st.spinner("Scanning..."):
        result = model.predict(img, imgsz=640, conf=conf, device="cpu", verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000

    n = len(result.boxes)
    confs = result.boxes.conf.tolist() if n else []
    labels = [model.names[int(c)] for c in result.boxes.cls.tolist()] if n else []
    counts = Counter(labels)
    avg = (sum(confs) / n * 100) if n else 0
    annotated = result.plot(labels=show_labels, conf=show_scores)[:, :, ::-1]

    with left:
        tab_det, tab_org = st.tabs(["Detected", "Original"])
        with tab_det:
            st.image(annotated)
        with tab_org:
            st.image(img)

    with right:
        if n == 0:
            verdict = "<div class='verdict'>No defects detected<small>Nothing found at this sensitivity.</small></div>"
        else:
            verdict = (
                f"<div class='verdict fail'>{n} defect{'s' if n != 1 else ''} detected"
                "<small>Marked on the image.</small></div>"
            )
        nums = (
            "<div class='nums'>"
            f"<div class='num'><div class='v'>{n}</div><div class='k'>Defects</div></div>"
            f"<div class='num'><div class='v'>{f'{avg:.0f}%' if n else '-'}</div><div class='k'>Avg conf.</div></div>"
            f"<div class='num'><div class='v'>{ms:.0f}</div><div class='k'>ms</div></div>"
            "</div>"
        )
        rows = ""
        if n:
            top = max(counts.values())
            rows = "<div class='label'>Defects by type</div>" + "".join(
                f"<div class='row'><div class='top'><span>{name.replace('_', ' ')}</span>"
                f"<span class='cnt'>{c}</span></div>"
                f"<div class='bar'><i style='width:{c / top * 100:.0f}%'></i></div></div>"
                for name, c in counts.most_common()
            )
        st.markdown(verdict + nums + rows, unsafe_allow_html=True)

        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="PNG")
        st.write("")
        st.download_button(
            "Download annotated image",
            data=buf.getvalue(),
            file_name="tracescan_result.png",
            mime="image/png",
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    f"<div class='foot'><span>{APP_NAME} &middot; automated PCB visual inspection</span>"
    "<span>Powered by YOLO26</span></div>",
    unsafe_allow_html=True,
)
