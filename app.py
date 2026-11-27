import io
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "best.pt"

st.set_page_config(page_title="PCB Defect Inspector", layout="wide")

# Faint copper traces on a solder-mask green background
TRACES = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='280' height='280' viewBox='0 0 280 280'>"
    "<g fill='none' stroke='rgba(217,164,65,0.10)' stroke-width='1.4' stroke-linejoin='round'>"
    "<path d='M0 60 H80 L105 85 H200 L225 60 H280'/>"
    "<path d='M0 170 H50 L75 145 H150 L175 170 H280'/>"
    "<path d='M0 235 H110 L135 210 H210 L235 235 H280'/>"
    "<path d='M140 0 V45 L160 65 V120 L140 140 V280'/>"
    "</g>"
    "<g fill='rgba(217,164,65,0.20)'>"
    "<circle cx='80' cy='60' r='3'/><circle cx='200' cy='85' r='3'/>"
    "<circle cx='75' cy='145' r='3'/><circle cx='175' cy='170' r='3'/>"
    "<circle cx='135' cy='210' r='3'/><circle cx='160' cy='65' r='3'/>"
    "</g></svg>"
)
PATTERN = "data:image/svg+xml;utf8," + quote(TRACES)

CSS = """
:root{--copper:#d9a441;--ok:#4ade80;--bad:#f87171;--text:#e8efe9;--muted:#8aa398;--line:rgba(255,255,255,.09);}
.stApp{background:url("__PATTERN__"),linear-gradient(180deg,#0c2b21 0%,#08201a 100%);background-attachment:fixed;color:var(--text);}
#MainMenu, footer{visibility:hidden;}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2.4rem;padding-bottom:3rem;max-width:1100px;}
[data-testid="stSidebar"]{background:#071a14;border-right:1px solid var(--line);}
.title{font-size:34px;font-weight:700;letter-spacing:-.01em;margin:0;}
.title span{color:var(--copper);}
.subtitle{color:var(--muted);font-size:15px;margin:6px 0 26px;}
.label{font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:26px 0 10px;}
.verdict{padding:14px 18px;border-radius:10px;background:rgba(0,0,0,.28);border:1px solid var(--line);border-left:4px solid var(--ok);font-size:17px;font-weight:600;}
.verdict.fail{border-left-color:var(--bad);}
.verdict small{display:block;font-weight:400;font-size:13px;color:var(--muted);margin-top:2px;}
.nums{display:flex;gap:0;margin:16px 0 4px;border:1px solid var(--line);border-radius:10px;background:rgba(0,0,0,.2);}
.num{flex:1;padding:14px 18px;border-right:1px solid var(--line);}
.num:last-child{border-right:none;}
.num .v{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:26px;font-weight:600;}
.num .k{font-size:12px;color:var(--muted);margin-top:2px;}
.row{display:flex;align-items:center;gap:12px;margin:8px 0;font-size:14px;}
.row .name{width:170px;color:var(--text);}
.row .bar{flex:1;height:6px;border-radius:3px;background:rgba(255,255,255,.07);overflow:hidden;}
.row .bar i{display:block;height:100%;background:var(--copper);border-radius:3px;}
.row .cnt{width:34px;text-align:right;font-family:ui-monospace,Menlo,Consolas,monospace;color:var(--copper);}
.cap{font-size:12px;color:var(--muted);margin:0 0 6px 2px;}
[data-testid="stFileUploaderDropzone"]{background:rgba(0,0,0,.22);border:1.5px dashed rgba(217,164,65,.5);border-radius:12px;}
[data-testid="stImage"] img{border-radius:10px;border:1px solid var(--line);width:100%;}
.stDownloadButton button{background:var(--copper);border:none;border-radius:8px;font-weight:600;}
.stDownloadButton button p{color:#1a1204 !important;font-weight:600;}
.stDownloadButton button:hover{filter:brightness(1.08);}
"""
CSS = " ".join(l.strip() for l in CSS.splitlines()).replace("__PATTERN__", PATTERN)
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


if not MODEL_PATH.exists():
    st.error("best.pt not found. Put the trained weights next to app.py.")
    st.stop()

model = load_model()

with st.sidebar:
    st.markdown("<div class='label' style='margin-top:6px'>Settings</div>", unsafe_allow_html=True)
    conf = st.slider(
        "Confidence threshold",
        0.10,
        0.90,
        0.40,
        0.05,
        help="Lower finds more defects but adds false alarms. Higher is cleaner but may miss some.",
    )
    show_labels = st.toggle("Show labels", value=True)
    show_scores = st.toggle("Show scores", value=True)

st.markdown(
    "<div class='title'>PCB Defect <span>Inspector</span></div>"
    "<div class='subtitle'>Upload a board photo and the model marks every defect it finds.</div>",
    unsafe_allow_html=True,
)

file = st.file_uploader(
    "Upload a PCB image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if not file:
    st.stop()

img = Image.open(file).convert("RGB")

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

if n == 0:
    st.markdown(
        "<div class='verdict'>No defects detected"
        "<small>Nothing found at the current threshold.</small></div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<div class='verdict fail'>{n} defect{'s' if n != 1 else ''} detected"
        "<small>Marked on the right image.</small></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='nums'>"
    f"<div class='num'><div class='v'>{n}</div><div class='k'>Defects</div></div>"
    f"<div class='num'><div class='v'>{f'{avg:.0f}%' if n else '-'}</div><div class='k'>Avg confidence</div></div>"
    f"<div class='num'><div class='v'>{ms:.0f} ms</div><div class='k'>Scan time</div></div>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("<div class='label'>Result</div>", unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.markdown("<div class='cap'>Original</div>", unsafe_allow_html=True)
    st.image(img)
with right:
    st.markdown("<div class='cap'>Detected</div>", unsafe_allow_html=True)
    st.image(annotated)

if n:
    st.markdown("<div class='label'>Defects by type</div>", unsafe_allow_html=True)
    top = max(counts.values())
    rows = "".join(
        f"<div class='row'><div class='name'>{name}</div>"
        f"<div class='bar'><i style='width:{c / top * 100:.0f}%'></i></div>"
        f"<div class='cnt'>{c}</div></div>"
        for name, c in counts.most_common()
    )
    st.markdown(rows, unsafe_allow_html=True)

buf = io.BytesIO()
Image.fromarray(annotated).save(buf, format="PNG")
st.write("")
st.download_button(
    "Download annotated image",
    data=buf.getvalue(),
    file_name="pcb_result.png",
    mime="image/png",
)
