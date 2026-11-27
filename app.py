import io
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path(__file__).parent
MODEL_PATH = BASE / "best.pt"
SAMPLES = sorted((BASE / "samples").glob("*.jpg"))
APP_NAME = "TraceScan"

# High-contrast box colors that stand out on a green board
PALETTE = [
    (255, 59, 48),
    (255, 159, 10),
    (255, 214, 10),
    (255, 45, 146),
    (0, 229, 255),
    (191, 90, 242),
    (255, 255, 255),
    (120, 255, 140),
]

st.set_page_config(page_title=f"{APP_NAME} | PCB Defect Inspection", layout="wide")

# ---------------------------------------------------------------------------
# Background and logo
# ---------------------------------------------------------------------------
TRACES = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='280' height='280' viewBox='0 0 280 280'>"
    "<g fill='none' stroke='rgba(240,180,60,0.13)' stroke-width='1.4' stroke-linejoin='round'>"
    "<path d='M0 60 H80 L105 85 H200 L225 60 H280'/>"
    "<path d='M0 170 H50 L75 145 H150 L175 170 H280'/>"
    "<path d='M0 235 H110 L135 210 H210 L235 235 H280'/>"
    "<path d='M140 0 V45 L160 65 V120 L140 140 V280'/>"
    "</g>"
    "<g fill='rgba(240,180,60,0.26)'>"
    "<circle cx='80' cy='60' r='3'/><circle cx='200' cy='85' r='3'/>"
    "<circle cx='75' cy='145' r='3'/><circle cx='175' cy='170' r='3'/>"
    "<circle cx='135' cy='210' r='3'/><circle cx='160' cy='65' r='3'/>"
    "</g></svg>"
)
PATTERN = "data:image/svg+xml;utf8," + quote(TRACES)

LOGO = (
    "<svg width='48' height='48' viewBox='0 0 48 48' xmlns='http://www.w3.org/2000/svg'>"
    "<rect x='1' y='1' width='46' height='46' rx='11' fill='#0f3a2c' stroke='#f0b43c' stroke-width='1.6'/>"
    "<path d='M8 32 H18 L24 26 H32' stroke='#f0b43c' stroke-width='2.2' fill='none' "
    "stroke-linecap='round' stroke-linejoin='round'/>"
    "<circle cx='35' cy='26' r='3.6' fill='#f0b43c'/>"
    "<path d='M12 17 V12 H17 M31 12 H36 V17 M36 35 V40 H31 M17 40 H12 V35' stroke='#f1f7f3' "
    "stroke-width='1.8' fill='none' stroke-linecap='round' stroke-linejoin='round'/>"
    "</svg>"
)

CSS = """
:root{--copper:#f0b43c;--ok:#3ddc84;--bad:#ff5c5c;--text:#f1f7f3;--muted:#9bb8aa;--line:rgba(255,255,255,.11);}
.stApp{background:url("__PATTERN__"),linear-gradient(180deg,#0b3427 0%,#06231b 100%);background-attachment:fixed;color:var(--text);}
#MainMenu, footer{visibility:hidden;}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1240px;margin:1.6rem auto 2rem auto;padding:2.2rem 2.6rem 2.6rem 2.6rem !important;background:rgba(3,20,15,.86);border:1px solid var(--line);border-radius:18px;box-shadow:0 24px 70px rgba(0,0,0,.5);}
@media(max-width:800px){.block-container{padding:1.4rem 1.1rem 1.8rem 1.1rem !important;margin:.6rem;}}
.brand{display:flex;align-items:center;gap:14px;}
.brand .name{font-size:30px;font-weight:700;letter-spacing:-.01em;line-height:1.1;}
.brand .name span{color:var(--copper);}
.brand .tag{font-size:13.5px;color:var(--muted);margin-top:2px;}
.lede{margin:20px 0 6px;max-width:780px;font-size:17px;line-height:1.65;color:#d6e4dc;}
.lede b{color:var(--text);}
.rule{height:1px;background:linear-gradient(90deg,rgba(240,180,60,.6),transparent);margin:22px 0 4px;}
.label{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:24px 0 10px;}
.pill{display:inline-block;font-size:12px;letter-spacing:.12em;text-transform:uppercase;padding:6px 14px;border-radius:999px;border:1px solid var(--line);color:var(--text);background:rgba(255,255,255,.06);margin-bottom:10px;}
.pill.hot{color:#1a1204;background:var(--copper);border-color:var(--copper);font-weight:700;}
.verdict{padding:16px 18px;border-radius:12px;background:rgba(61,220,132,.10);border:1px solid rgba(61,220,132,.45);border-left:5px solid var(--ok);font-size:20px;font-weight:700;}
.verdict.fail{background:rgba(255,92,92,.10);border-color:rgba(255,92,92,.5);border-left-color:var(--bad);}
.verdict small{display:block;font-weight:400;font-size:13px;color:var(--muted);margin-top:3px;}
.nums{display:grid;grid-template-columns:repeat(3,1fr);margin:14px 0 4px;border:1px solid var(--line);border-radius:12px;background:rgba(0,0,0,.25);}
.num{padding:14px 18px;border-right:1px solid var(--line);}
.num:last-child{border-right:none;}
.num .v{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:28px;font-weight:700;color:var(--text);}
.num .k{font-size:12px;color:var(--muted);margin-top:2px;}
.row{margin:11px 0;font-size:15px;}
.row .top{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}
.row .nm{display:flex;align-items:center;gap:9px;}
.row .dot{width:11px;height:11px;border-radius:3px;display:inline-block;}
.row .cnt{font-family:ui-monospace,Menlo,Consolas,monospace;font-weight:700;color:var(--text);}
.row .bar{height:7px;border-radius:4px;background:rgba(255,255,255,.08);overflow:hidden;}
.row .bar i{display:block;height:100%;border-radius:4px;}
.note{font-size:13.5px;color:var(--muted);line-height:1.6;}
.foot{margin-top:30px;padding-top:14px;border-top:1px solid var(--line);font-size:12px;color:var(--muted);display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;}
[data-testid="stFileUploaderDropzone"]{background:rgba(0,0,0,.28);border:1.5px dashed rgba(240,180,60,.6);border-radius:12px;}
[data-testid="stImage"] img{border-radius:10px;border:1px solid var(--line);width:100%;box-shadow:0 10px 30px rgba(0,0,0,.4);}
.stButton button{background:rgba(240,180,60,.14);border:1px solid rgba(240,180,60,.6);color:var(--text);border-radius:8px;font-weight:600;}
.stButton button:hover{background:rgba(240,180,60,.26);border-color:var(--copper);color:var(--text);}
.stDownloadButton button{background:var(--copper);border:none;border-radius:8px;font-weight:700;}
.stDownloadButton button p{color:#1a1204 !important;font-weight:700;}
.stDownloadButton button:hover{filter:brightness(1.08);}
"""
CSS = " ".join(l.strip() for l in CSS.splitlines()).replace("__PATTERN__", PATTERN)
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def rgb_css(c):
    return f"rgb({c[0]},{c[1]},{c[2]})"


def draw_boxes(img, boxes, names, show_labels, show_scores):
    """Draw high-contrast boxes with scaled line width and readable labels."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    lw = max(3, round(max(w, h) / 320))
    fs = max(16, round(max(w, h) / 55))
    try:
        font = ImageFont.load_default(size=fs)
    except TypeError:
        font = ImageFont.load_default()

    for xyxy, c, p in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
        col = PALETTE[int(c) % len(PALETTE)]
        x1, y1, x2, y2 = xyxy
        draw.rectangle([x1, y1, x2, y2], outline=col, width=lw)

        parts = []
        if show_labels:
            parts.append(names[int(c)].replace("_", " "))
        if show_scores:
            parts.append(f"{p:.2f}")
        if not parts:
            continue
        text = " ".join(parts)
        bb = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad = lw + 2
        ty = y1 - th - 2 * pad
        if ty < 0:
            ty = y1
        draw.rectangle([x1, ty, x1 + tw + 2 * pad, ty + th + 2 * pad], fill=col)
        draw.text((x1 + pad - bb[0], ty + pad - bb[1]), text, fill=(15, 15, 15), font=font)
    return out


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
# Input row
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
# Result
# ---------------------------------------------------------------------------
else:
    st.markdown("<div class='label'>Settings</div>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns([3, 1, 1], gap="large")
    with s1:
        conf = st.slider(
            "Sensitivity (confidence threshold)",
            0.10,
            0.90,
            0.40,
            0.05,
            help="Lower finds more defects but adds false alarms. Higher is cleaner but may miss some.",
        )
    with s2:
        show_labels = st.toggle("Show labels", value=True)
    with s3:
        show_scores = st.toggle("Show scores", value=True)

    t0 = time.perf_counter()
    with st.spinner("Scanning..."):
        result = model.predict(img, imgsz=640, conf=conf, device="cpu", verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000

    n = len(result.boxes)
    cls_ids = [int(c) for c in result.boxes.cls.tolist()] if n else []
    confs = result.boxes.conf.tolist() if n else []
    counts = Counter(cls_ids)
    avg = (sum(confs) / n * 100) if n else 0
    predicted = draw_boxes(img, result.boxes, model.names, show_labels, show_scores) if n else img

    # Original and predicted, side by side
    st.markdown("<div class='label'>Result</div>", unsafe_allow_html=True)
    col_o, col_p = st.columns(2, gap="medium")
    with col_o:
        st.markdown("<div class='pill'>Original</div>", unsafe_allow_html=True)
        st.image(img)
    with col_p:
        st.markdown(
            f"<div class='pill hot'>Predicted &middot; {n} defect{'s' if n != 1 else ''}</div>",
            unsafe_allow_html=True,
        )
        st.image(predicted)

    # Summary
    st.markdown("<div class='label'>Summary</div>", unsafe_allow_html=True)
    sum_l, sum_r = st.columns(2, gap="large")

    with sum_l:
        if n == 0:
            verdict = "<div class='verdict'>No defects detected<small>Nothing found at this sensitivity.</small></div>"
        else:
            verdict = (
                f"<div class='verdict fail'>{n} defect{'s' if n != 1 else ''} detected"
                "<small>Each one is boxed on the predicted image.</small></div>"
            )
        nums = (
            "<div class='nums'>"
            f"<div class='num'><div class='v'>{n}</div><div class='k'>Defects</div></div>"
            f"<div class='num'><div class='v'>{f'{avg:.0f}%' if n else '-'}</div><div class='k'>Avg confidence</div></div>"
            f"<div class='num'><div class='v'>{ms:.0f}</div><div class='k'>Scan time (ms)</div></div>"
            "</div>"
        )
        st.markdown(verdict + nums, unsafe_allow_html=True)

    with sum_r:
        if n:
            top = max(counts.values())
            rows = "".join(
                "<div class='row'><div class='top'>"
                f"<span class='nm'><span class='dot' style='background:{rgb_css(PALETTE[cid % len(PALETTE)])}'></span>"
                f"{model.names[cid].replace('_', ' ')}</span><span class='cnt'>{c}</span></div>"
                f"<div class='bar'><i style='width:{c / top * 100:.0f}%;"
                f"background:{rgb_css(PALETTE[cid % len(PALETTE)])}'></i></div></div>"
                for cid, c in counts.most_common()
            )
            st.markdown(rows, unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='note'>This board looks clean. Lower the sensitivity if you "
                "suspect a faint defect.</div>",
                unsafe_allow_html=True,
            )

        buf = io.BytesIO()
        predicted.save(buf, format="PNG")
        st.write("")
        st.download_button(
            "Download predicted image",
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
