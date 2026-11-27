import io
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "best.pt"
MODEL_MAP50 = "95.2%"  # test-set mAP@50 from the training run

st.set_page_config(
    page_title="PCB Defect Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme: circuit-board background + glass cards
# ---------------------------------------------------------------------------
PCB_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='260' height='260' viewBox='0 0 260 260'>"
    "<g fill='none' stroke='rgba(57,255,160,0.10)' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'>"
    "<path d='M0 50 H70 L95 75 H190 L215 50 H260'/>"
    "<path d='M0 150 H40 L65 125 H140 L165 150 H260'/>"
    "<path d='M0 215 H100 L125 190 H200 L225 215 H260'/>"
    "<path d='M130 0 V40 L150 60 V110 L130 130 V260'/>"
    "<path d='M40 0 V30 L20 50'/>"
    "</g>"
    "<g fill='rgba(255,200,87,0.22)'>"
    "<circle cx='70' cy='50' r='3.5'/><circle cx='190' cy='75' r='3.5'/>"
    "<circle cx='65' cy='125' r='3.5'/><circle cx='165' cy='150' r='3.5'/>"
    "<circle cx='125' cy='190' r='3.5'/><circle cx='150' cy='60' r='3.5'/>"
    "</g>"
    "<g fill='none' stroke='rgba(34,211,238,0.12)' stroke-width='1.2'>"
    "<rect x='205' y='95' width='34' height='22' rx='3'/>"
    "<rect x='18' y='170' width='26' height='26' rx='3'/>"
    "</g>"
    "</svg>"
)
PATTERN = "data:image/svg+xml;utf8," + quote(PCB_SVG)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root{--mint:#39ffa0;--gold:#ffc857;--cyan:#22d3ee;--red:#ff5d6c;--text:#e6fff4;--muted:#8fb8a6;--line:rgba(57,255,160,.22);--panel:rgba(13,42,34,.62);}
.stApp{
background:
radial-gradient(900px 500px at 12% -5%, rgba(57,255,160,.16), transparent 60%),
radial-gradient(800px 500px at 95% 10%, rgba(34,211,238,.12), transparent 60%),
radial-gradient(900px 600px at 50% 110%, rgba(255,200,87,.08), transparent 60%),
url("__PATTERN__"),
linear-gradient(180deg,#071a14 0%,#04100c 100%);
background-attachment:fixed;
font-family:'Space Grotesk',sans-serif;
color:var(--text);
}
#MainMenu, footer{visibility:hidden;}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1200px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(6,25,19,.97),rgba(4,16,12,.99));border-right:1px solid var(--line);}
.hero{position:relative;padding:36px 38px;border-radius:22px;border:1px solid var(--line);background:linear-gradient(135deg,rgba(13,42,34,.85),rgba(6,25,19,.7));backdrop-filter:blur(8px);overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.45);}
.hero::after{content:"";position:absolute;right:-90px;top:-90px;width:340px;height:340px;border-radius:50%;background:radial-gradient(circle,rgba(57,255,160,.25),transparent 65%);}
.eyebrow{display:inline-flex;align-items:center;gap:9px;font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--mint);padding:6px 13px;border:1px solid var(--line);border-radius:999px;background:rgba(57,255,160,.07);}
.dot{width:8px;height:8px;border-radius:50%;background:var(--mint);box-shadow:0 0 12px var(--mint);animation:pulse 1.8s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.hero-title{font-size:clamp(34px,5vw,58px);font-weight:700;line-height:1.05;margin:18px 0 12px;background:linear-gradient(90deg,#e6fff4 0%,#39ffa0 55%,#22d3ee 100%);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero-sub{color:var(--muted);font-size:17px;max-width:700px;line-height:1.6;}
.badges{display:flex;gap:10px;flex-wrap:wrap;margin-top:22px;}
.badge{font-family:'JetBrains Mono',monospace;font-size:12.5px;padding:7px 13px;border-radius:10px;border:1px solid var(--line);background:rgba(0,0,0,.28);color:var(--text);}
.badge b{color:var(--gold);}
.section-title{font-size:13px;letter-spacing:.16em;text-transform:uppercase;color:var(--mint);margin:30px 0 12px;font-family:'JetBrains Mono',monospace;display:flex;align-items:center;gap:12px;}
.section-title::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent);}
.metric{padding:18px 20px;border-radius:16px;border:1px solid var(--line);background:var(--panel);backdrop-filter:blur(6px);}
.metric .k{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);}
.metric .v{font-size:34px;font-weight:700;margin-top:4px;font-family:'JetBrains Mono',monospace;}
.metric .s{font-size:12px;color:var(--muted);margin-top:2px;}
.v.good{color:var(--mint);} .v.bad{color:var(--red);} .v.gold{color:var(--gold);} .v.cyan{color:var(--cyan);}
.status{display:flex;align-items:center;gap:14px;padding:16px 22px;border-radius:16px;margin:6px 0 18px;font-weight:600;font-size:19px;}
.status.pass{background:rgba(57,255,160,.10);border:1px solid rgba(57,255,160,.5);color:var(--mint);box-shadow:0 0 40px rgba(57,255,160,.12);}
.status.fail{background:rgba(255,93,108,.10);border:1px solid rgba(255,93,108,.55);color:var(--red);box-shadow:0 0 40px rgba(255,93,108,.14);}
.status small{display:block;font-weight:400;font-size:13px;opacity:.8;margin-top:2px;}
.img-label{font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 8px 2px;}
.chip{display:inline-flex;gap:8px;align-items:center;margin:0 8px 8px 0;padding:8px 14px;border-radius:999px;border:1px solid rgba(255,200,87,.4);background:rgba(255,200,87,.08);color:var(--gold);font-family:'JetBrains Mono',monospace;font-size:13px;}
.chip b{color:var(--text);}
.chip.mini{padding:5px 10px;font-size:11.5px;border-color:var(--line);background:rgba(57,255,160,.06);color:var(--mint);}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:6px;}
@media(max-width:800px){.steps{grid-template-columns:1fr}}
.step{padding:20px;border-radius:16px;border:1px solid var(--line);background:var(--panel);}
.step .n{font-family:'JetBrains Mono',monospace;color:var(--gold);font-size:13px;}
.step .t{font-weight:600;font-size:18px;margin:6px 0 4px;}
.step .d{color:var(--muted);font-size:14px;line-height:1.5;}
.side-title{font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--mint);margin:4px 0 10px;}
.side-card{padding:14px;border-radius:14px;border:1px solid var(--line);background:rgba(13,42,34,.5);margin-top:14px;font-size:13px;color:var(--muted);line-height:1.6;}
.side-card b{color:var(--text);}
.foot{margin-top:40px;text-align:center;color:var(--muted);font-size:12.5px;font-family:'JetBrains Mono',monospace;letter-spacing:.06em;}
[data-testid="stFileUploaderDropzone"]{background:rgba(13,42,34,.55);border:2px dashed rgba(57,255,160,.45);border-radius:18px;padding:30px;transition:.2s;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--mint);box-shadow:0 0 30px rgba(57,255,160,.15);}
[data-testid="stImage"] img{border-radius:14px;border:1px solid var(--line);box-shadow:0 12px 40px rgba(0,0,0,.5);width:100%;}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden;}
.stDownloadButton button{background:linear-gradient(90deg,#1fd58a,#22d3ee);border:none;border-radius:12px;padding:.65rem 1.3rem;font-weight:700;}
.stDownloadButton button p{color:#04100c !important;font-weight:700;}
.stDownloadButton button:hover{filter:brightness(1.1);transform:translateY(-1px);}
"""
CSS = " ".join(line.strip() for line in CSS.splitlines()).replace("__PATTERN__", PATTERN)
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def section(title):
    return f"<div class='section-title'>{title}</div>"


def metric(label, value, sub="", tone=""):
    return (
        f"<div class='metric'><div class='k'>{label}</div>"
        f"<div class='v {tone}'>{value}</div><div class='s'>{sub}</div></div>"
    )


def footer():
    st.markdown(
        "<div class='foot'>YOLO26 &middot; Ultralytics &middot; Streamlit &nbsp;|&nbsp; "
        "Automated PCB visual inspection</div>",
        unsafe_allow_html=True,
    )


if not MODEL_PATH.exists():
    st.error("best.pt not found. Put the trained weights next to app.py.")
    st.stop()

model = load_model()
class_names = list(model.names.values())

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='side-title'>Inspection settings</div>", unsafe_allow_html=True)
    conf = st.slider(
        "Confidence threshold",
        0.10,
        0.90,
        0.40,
        0.05,
        help="Lower = finds more defects but more false alarms. Higher = cleaner but may miss defects.",
    )
    show_labels = st.toggle("Show class labels", value=True)
    show_scores = st.toggle("Show confidence scores", value=True)

    chips = "".join(f"<span class='chip mini'>{n}</span>" for n in class_names)
    st.markdown(
        "<div class='side-card'><b>Model</b><br>YOLO26s &middot; trained on a custom PCB dataset"
        f"<br><b>mAP@50</b> {MODEL_MAP50} (test set)<br><b>Detects</b><br>{chips}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='side-card'><b>Tip</b><br>Use a sharp, well-lit, top-down photo of the board "
        "for the most reliable results.</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='hero'>"
    "<span class='eyebrow'><span class='dot'></span>Live inspection &middot; YOLO26</span>"
    "<div class='hero-title'>PCB Defect Inspector</div>"
    "<div class='hero-sub'>Upload a photo of a printed circuit board and the model scans it "
    "for manufacturing defects in seconds, marking every fault it finds.</div>"
    "<div class='badges'>"
    f"<span class='badge'>mAP@50 <b>{MODEL_MAP50}</b></span>"
    f"<span class='badge'>Classes <b>{len(class_names)}</b></span>"
    "<span class='badge'>Input <b>640 px</b></span>"
    "<span class='badge'>Runtime <b>CPU</b></span>"
    "</div></div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
st.markdown(section("01 / Upload board image"), unsafe_allow_html=True)
file = st.file_uploader(
    "Upload a PCB image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if not file:
    st.markdown(section("How it works"), unsafe_allow_html=True)
    st.markdown(
        "<div class='steps'>"
        "<div class='step'><div class='n'>STEP 01</div><div class='t'>Upload</div>"
        "<div class='d'>Drop a JPG or PNG photo of the board into the box above.</div></div>"
        "<div class='step'><div class='n'>STEP 02</div><div class='t'>Scan</div>"
        "<div class='d'>The YOLO26 model analyses the whole board and locates every defect.</div></div>"
        "<div class='step'><div class='n'>STEP 03</div><div class='t'>Review</div>"
        "<div class='d'>Get a pass / fail verdict, the defect list, and an annotated image to download.</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    footer()
    st.stop()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
img = Image.open(file).convert("RGB")

t0 = time.perf_counter()
with st.spinner("Scanning board..."):
    result = model.predict(img, imgsz=640, conf=conf, device="cpu", verbose=False)[0]
total_ms = (time.perf_counter() - t0) * 1000

n = len(result.boxes)
confs = result.boxes.conf.tolist() if n else []
labels = [model.names[int(c)] for c in result.boxes.cls.tolist()] if n else []
counts = Counter(labels)
avg_conf = (sum(confs) / n) if n else 0.0

annotated = result.plot(labels=show_labels, conf=show_scores)[:, :, ::-1]

# ---------------------------------------------------------------------------
# Verdict + metrics
# ---------------------------------------------------------------------------
st.markdown(section("02 / Inspection result"), unsafe_allow_html=True)

if n == 0:
    st.markdown(
        "<div class='status pass'><span style='font-size:28px'>&#10004;</span>"
        "<div>PASS - no defects detected<small>Nothing found at the current confidence threshold.</small></div></div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<div class='status fail'><span style='font-size:28px'>&#9888;</span>"
        f"<div>FAIL - {n} defect{'s' if n != 1 else ''} detected"
        f"<small>Review the marked areas below.</small></div></div>",
        unsafe_allow_html=True,
    )

c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric("Defects found", n, "total detections", "bad" if n else "good"), unsafe_allow_html=True)
c2.markdown(metric("Defect types", len(counts), f"of {len(class_names)} classes", "gold"), unsafe_allow_html=True)
c3.markdown(
    metric("Avg confidence", f"{avg_conf * 100:.0f}%" if n else "-", "mean score", "cyan"),
    unsafe_allow_html=True,
)
c4.markdown(metric("Scan time", f"{total_ms:.0f} ms", "on this server", "cyan"), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
st.markdown(section("03 / Original vs detected"), unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.markdown("<div class='img-label'>Original</div>", unsafe_allow_html=True)
    st.image(img)
with right:
    st.markdown("<div class='img-label'>Detected</div>", unsafe_allow_html=True)
    st.image(annotated)

# ---------------------------------------------------------------------------
# Breakdown
# ---------------------------------------------------------------------------
if n:
    st.markdown(section("04 / Defect breakdown"), unsafe_allow_html=True)
    st.markdown(
        "".join(f"<span class='chip'>{name} <b>x{c}</b></span>" for name, c in counts.most_common()),
        unsafe_allow_html=True,
    )

    xyxy = result.boxes.xyxy.tolist()
    df = pd.DataFrame(
        {
            "Defect": labels,
            "Confidence": [round(c, 3) for c in confs],
            "x1": [int(b[0]) for b in xyxy],
            "y1": [int(b[1]) for b in xyxy],
            "x2": [int(b[2]) for b in xyxy],
            "y2": [int(b[3]) for b in xyxy],
        }
    ).sort_values("Confidence", ascending=False)
    st.dataframe(df, hide_index=True)

buf = io.BytesIO()
Image.fromarray(annotated).save(buf, format="PNG")
st.download_button(
    "Download annotated image",
    data=buf.getvalue(),
    file_name="pcb_result.png",
    mime="image/png",
)

footer()
