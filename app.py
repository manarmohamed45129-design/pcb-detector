import base64
import io
import json
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path(__file__).parent
MODEL_PATH = BASE / "best.pt"
SAMPLES = sorted((BASE / "samples").glob("*.jpg"))
APP_NAME = "TraceScan"
MODEL_MAP50 = "95.2%"
RESULTS_DIR = BASE / "results"

# Real numbers from the test split (69 images, 292 instances) - see the training notebook
CLASS_STATS = [
    # class, images, instances, precision, recall, mAP50, mAP50-95
    ("missing_hole", 16, 67, 0.988, 0.985, 0.997, 0.554),
    ("mouse_bite", 13, 54, 0.998, 0.926, 0.959, 0.522),
    ("open_circuit", 9, 32, 0.927, 0.906, 0.914, 0.527),
    ("short", 11, 52, 0.981, 0.984, 0.996, 0.547),
    ("spur", 6, 27, 0.897, 0.815, 0.945, 0.449),
    ("spurious_copper", 14, 60, 0.977, 0.833, 0.902, 0.469),
]
# Repair knowledge base: general shop-floor guidance for each defect type.
# Always confirm against your own quality standard (e.g. IPC-A-600) before reworking a board.
SEV_RANK = {"Critical": 0, "High": 1, "Medium": 2}
SEV_COLOR = {"Critical": "#ff5c5c", "High": "#ff9f43", "Medium": "#f0b43c"}
KNOWLEDGE = {
    "missing_hole": {
        "sev": "High",
        "hint": "Re-drill at the marked position with the correct bit size, then check the pad is intact.",
        "what": "A hole that should have been drilled is absent. Component leads or vias have nowhere to go.",
        "impact": "A through-hole part cannot be mounted, or a via fails to connect the layers, so the circuit is incomplete.",
        "causes": ["Drill bit broke or was worn out", "Drill program skipped a position", "Wrong tool or panel shifted during drilling"],
        "fix": ["Compare the position with the drawing or Gerber to confirm the hole is really required",
                "Re-drill with the correct bit size, aligned to the pad center",
                "Deburr the hole and clean away dust",
                "If the pad or surrounding traces are damaged, scrap the board instead"],
        "verify": "Insert a lead or gauge pin. For vias, test continuity between the layers.",
        "tools": "Precision drill or drill press, correct bit, deburring tool, magnifier",
    },
    "mouse_bite": {
        "sev": "Medium",
        "hint": "Check the remaining trace width. If it is too thin, bridge it with solder wire or conductive epoxy.",
        "what": "Small bite-shaped notches along a copper edge or trace, leaving it narrower than designed.",
        "impact": "Higher resistance and local heating, and the trace can crack later under load or vibration. A deep bite can become an open circuit.",
        "causes": ["Over-etching", "Rough routing or breaking off panel tabs", "Mechanical damage during handling"],
        "fix": ["Measure the remaining trace width against the design minimum",
                "If it is still within tolerance, accept it and note it",
                "If it is too thin, reinforce it by flowing solder along the trace or by bridging with fine copper foil or conductive epoxy",
                "Clean with isopropyl alcohol and re-coat the area with solder mask or lacquer"],
        "verify": "Measure trace width under a magnifier and check continuity with a multimeter.",
        "tools": "Magnifier or microscope, soldering iron, copper foil or conductive epoxy, IPA",
    },
    "open_circuit": {
        "sev": "Critical",
        "hint": "Bridge the break with fine jumper wire or conductive epoxy, then confirm continuity with a multimeter.",
        "what": "A copper trace is broken, so current cannot flow from one end to the other.",
        "impact": "The connection is dead. The circuit will not work, or will work only intermittently.",
        "causes": ["Over-etching or a scratch through the copper", "Dust or a flaw in the resist during etching", "Thermal or mechanical stress cracks"],
        "fix": ["Locate both ends of the break and scrape the solder mask off about 1 mm on each side",
                "Bridge the gap with thin enameled wire (about 30 AWG), solder both ends",
                "For very fine traces use copper foil or silver conductive epoxy instead",
                "Secure the wire with a drop of epoxy, then re-coat the exposed copper"],
        "verify": "Continuity test with a multimeter between the two ends of the net.",
        "tools": "Fine soldering iron, 30 AWG wire, scalpel, multimeter, conductive epoxy",
    },
    "short": {
        "sev": "Critical",
        "hint": "Cut the unwanted copper bridge with a sharp knife, then confirm the two nets are isolated.",
        "what": "Copper (or solder) connects two conductors that must stay separate.",
        "impact": "A short circuit. It can stop the board working, overheat, or damage components when powered.",
        "causes": ["Under-etching that leaves copper between traces", "Solder bridge", "Conductive debris or a resist defect"],
        "fix": ["Power the board off and do not test it with power until fixed",
                "Cut through the bridge with a sharp scalpel or a fine rotary tool",
                "Remove all copper residue so no thin strand remains",
                "Clean with IPA and re-coat the area with solder mask"],
        "verify": "Resistance test between the two nets: they should read open, not near zero ohms.",
        "tools": "Scalpel or rotary tool, magnifier, multimeter, IPA",
    },
    "spur": {
        "sev": "Medium",
        "hint": "Trim the protrusion flush with the trace. It is urgent only if it nearly touches a neighbor.",
        "what": "A small spike of copper sticking out from a trace.",
        "impact": "It reduces the clearance to the next conductor and can turn into a short if it gets close. It can also add noise on sensitive signals.",
        "causes": ["Incomplete etching", "Flaw in the resist pattern"],
        "fix": ["Measure the gap between the spur and the nearest conductor",
                "Trim it flush with a sharp scalpel or a fine file",
                "Clean the area and check clearance against the design rule",
                "Re-coat with solder mask if the copper is left exposed"],
        "verify": "Check the clearance under magnification and test isolation between neighboring nets.",
        "tools": "Scalpel or fine file, magnifier, IPA",
    },
    "spurious_copper": {
        "sev": "High",
        "hint": "Remove the stray copper completely, then test isolation of nearby traces.",
        "what": "Leftover copper where it should have been etched away: an island or a loose patch.",
        "impact": "It can short neighboring traces, break isolation in high-voltage areas, or act as an antenna that picks up noise.",
        "causes": ["Under-etching or exhausted etchant", "Contaminated resist", "Insufficient etch time"],
        "fix": ["Check the Gerber or drawing to confirm that copper should not be there",
                "Scrape or cut away the copper with a scalpel or fine rotary tool",
                "Make sure no copper dust remains, then clean with IPA",
                "Re-coat with solder mask or lacquer"],
        "verify": "Test isolation between the nearby nets with a multimeter.",
        "tools": "Scalpel or rotary tool, magnifier, multimeter, IPA",
    },
}

PLOTS = {
    "Confusion matrix": "confusion_matrix.png",
    "Confusion (normalized)": "confusion_matrix_normalized.png",
    "PR curve": "BoxPR_curve.png",
    "F1 curve": "BoxF1_curve.png",
    "Precision curve": "BoxP_curve.png",
    "Recall curve": "BoxR_curve.png",
}

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
    "<g fill='none' stroke='rgba(240,180,60,0.14)' stroke-width='1.4' stroke-linejoin='round'>"
    "<path d='M0 60 H80 L105 85 H200 L225 60 H280'/>"
    "<path d='M0 170 H50 L75 145 H150 L175 170 H280'/>"
    "<path d='M0 235 H110 L135 210 H210 L235 235 H280'/>"
    "<path d='M140 0 V45 L160 65 V120 L140 140 V280'/>"
    "</g>"
    "<g fill='rgba(240,180,60,0.28)'>"
    "<circle cx='80' cy='60' r='3'/><circle cx='200' cy='85' r='3'/>"
    "<circle cx='75' cy='145' r='3'/><circle cx='175' cy='170' r='3'/>"
    "<circle cx='135' cy='210' r='3'/><circle cx='160' cy='65' r='3'/>"
    "</g></svg>"
)
PATTERN = "data:image/svg+xml;utf8," + quote(TRACES)

LOGO = (
    "<svg width='50' height='50' viewBox='0 0 48 48' xmlns='http://www.w3.org/2000/svg'>"
    "<rect x='1' y='1' width='46' height='46' rx='11' fill='#0f3a2c' stroke='#f0b43c' stroke-width='1.6'/>"
    "<path d='M8 32 H18 L24 26 H32' stroke='#f0b43c' stroke-width='2.2' fill='none' "
    "stroke-linecap='round' stroke-linejoin='round'/>"
    "<circle cx='35' cy='26' r='3.6' fill='#f0b43c'/>"
    "<path d='M12 17 V12 H17 M31 12 H36 V17 M36 35 V40 H31 M17 40 H12 V35' stroke='#f1f7f3' "
    "stroke-width='1.8' fill='none' stroke-linecap='round' stroke-linejoin='round'/>"
    "<rect class='lscan' x='10' y='10' width='28' height='2' rx='1' fill='#5ad7ff'/>"
    "</svg>"
)

CSS = """
:root{--copper:#f0b43c;--cyan:#5ad7ff;--ok:#3ddc84;--bad:#ff5c5c;--text:#f1f7f3;--muted:#9bb8aa;--line:rgba(255,255,255,.11);}
@keyframes drift{from{background-position:0 0,0 0,0 0,0 0;}to{background-position:280px 280px,0 0,0 0,0 0;}}
@keyframes fadeUp{from{opacity:0;transform:translateY(18px);}to{opacity:1;transform:none;}}
@keyframes popIn{0%{opacity:0;transform:scale(.92);}60%{transform:scale(1.03);}100%{opacity:1;transform:none;}}
@keyframes flow{from{background-position:0 0;}to{background-position:300% 0;}}
@keyframes shine{from{background-position:0 0;}to{background-position:200% 0;}}
@keyframes bob{0%,100%{transform:translateY(0);}50%{transform:translateY(-4px);}}
@keyframes lscan{0%,100%{transform:translateY(0);opacity:.2;}50%{transform:translateY(27px);opacity:1;}}
@keyframes slide{from{left:-30%;}to{left:110%;}}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.25;}}
@keyframes sweep{0%{top:-6px;opacity:0;}10%{opacity:1;}90%{opacity:1;}100%{top:100%;opacity:0;}}
@keyframes grow{from{transform:scaleX(0);}to{transform:scaleX(1);}}
@keyframes alert{0%,100%{box-shadow:0 0 0 0 rgba(255,92,92,0);}50%{box-shadow:0 0 30px 2px rgba(255,92,92,.38);}}
@keyframes okglow{0%,100%{box-shadow:0 0 0 0 rgba(61,220,132,0);}50%{box-shadow:0 0 30px 2px rgba(61,220,132,.30);}}
@keyframes ring{0%{box-shadow:0 0 0 0 rgba(240,180,60,.55);}70%{box-shadow:0 0 0 12px rgba(240,180,60,0);}100%{box-shadow:0 0 0 0 rgba(240,180,60,0);}}
@keyframes dz{0%,100%{box-shadow:0 0 0 0 rgba(240,180,60,0);border-color:rgba(240,180,60,.55);}50%{box-shadow:0 0 28px 0 rgba(240,180,60,.22);border-color:rgba(90,215,255,.8);}}
.stApp{
background:
radial-gradient(900px 520px at 8% -8%, rgba(90,215,255,.13), transparent 60%),
radial-gradient(900px 560px at 100% 105%, rgba(240,180,60,.12), transparent 60%),
url("__PATTERN__"),
linear-gradient(180deg,#0b3427 0%,#06231b 100%);
background-attachment:fixed;
background-position:0 0,0 0,0 0,0 0;
animation:drift 45s linear infinite;
color:var(--text);}
#MainMenu, footer{visibility:hidden;}
[data-testid="stHeader"]{background:transparent;}
.block-container{position:relative;max-width:1240px;margin:1.6rem auto 2rem auto;padding:2.4rem 2.6rem 2.6rem 2.6rem !important;background:rgba(3,20,15,.86);border:1px solid var(--line);border-radius:18px;box-shadow:0 24px 70px rgba(0,0,0,.5);animation:fadeUp .7s cubic-bezier(.2,.7,.2,1) backwards;}
.block-container::before{content:"";position:absolute;top:0;left:22px;right:22px;height:3px;border-radius:0 0 4px 4px;background:linear-gradient(90deg,#f0b43c,#5ad7ff,#3ddc84,#f0b43c);background-size:300% 100%;animation:flow 7s linear infinite;}
@media(max-width:800px){.block-container{padding:1.5rem 1.1rem 1.8rem 1.1rem !important;margin:.6rem;}}
.top{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;}
.brand{display:flex;align-items:center;gap:14px;animation:fadeUp .8s .05s backwards;}
.logo{filter:drop-shadow(0 0 12px rgba(240,180,60,.5));animation:bob 4s ease-in-out infinite;display:flex;}
.lscan{animation:lscan 2.4s ease-in-out infinite;}
.brand .name{font-size:32px;font-weight:700;letter-spacing:-.01em;line-height:1.1;}
.brand .name span{background:linear-gradient(90deg,#f0b43c,#fff0b8,#f0b43c);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:shine 3.5s linear infinite;}
.brand .tag{font-size:13.5px;color:var(--muted);margin-top:2px;}
.live{display:inline-flex;align-items:center;gap:9px;padding:7px 14px;border-radius:999px;border:1px solid rgba(61,220,132,.45);background:rgba(61,220,132,.09);font-size:12.5px;color:var(--text);animation:fadeUp .8s .2s backwards;}
.live i{width:8px;height:8px;border-radius:50%;background:var(--ok);box-shadow:0 0 10px var(--ok);animation:blink 1.6s infinite;}
.live b{color:var(--copper);}
.lede{margin:20px 0 6px;max-width:780px;font-size:17px;line-height:1.65;color:#d6e4dc;animation:fadeUp .8s .15s backwards;}
.lede b{color:var(--text);}
.rule{position:relative;overflow:hidden;height:2px;background:linear-gradient(90deg,rgba(240,180,60,.65),rgba(90,215,255,.25),transparent);margin:22px 0 4px;}
.rule::after{content:"";position:absolute;top:0;left:-30%;width:30%;height:100%;background:linear-gradient(90deg,transparent,#fff,transparent);animation:slide 3.2s ease-in-out infinite;}
.label{display:flex;align-items:center;font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:26px 0 10px;animation:fadeUp .6s backwards;}
.label::before{content:"";display:inline-block;width:20px;height:3px;border-radius:2px;margin-right:10px;background:linear-gradient(90deg,#f0b43c,#5ad7ff);}
.pill{display:inline-block;font-size:12px;letter-spacing:.12em;text-transform:uppercase;padding:6px 14px;border-radius:999px;margin-bottom:10px;color:var(--cyan);border:1px solid rgba(90,215,255,.5);background:rgba(90,215,255,.09);animation:popIn .6s backwards;}
.pill.hot{color:#1a1204;background:var(--copper);border-color:var(--copper);font-weight:700;box-shadow:0 0 18px rgba(240,180,60,.4);}
.verdict{padding:16px 18px;border-radius:12px;background:rgba(61,220,132,.10);border:1px solid rgba(61,220,132,.45);border-left:5px solid var(--ok);font-size:20px;font-weight:700;animation:popIn .6s backwards,okglow 2.8s 1s ease-in-out infinite;}
.verdict.fail{background:rgba(255,92,92,.10);border-color:rgba(255,92,92,.5);border-left-color:var(--bad);animation:popIn .6s backwards,alert 2.2s 1s ease-in-out infinite;}
.verdict small{display:block;font-weight:400;font-size:13px;color:var(--muted);margin-top:3px;}
.nums{display:grid;grid-template-columns:repeat(3,1fr);margin:14px 0 4px;border:1px solid var(--line);border-radius:12px;background:rgba(0,0,0,.25);animation:fadeUp .7s .1s backwards;}
.num{padding:14px 18px;border-right:1px solid var(--line);}
.num:last-child{border-right:none;}
.num .v{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:30px;font-weight:700;animation:popIn .7s .25s backwards;}
.num .k{font-size:12px;color:var(--muted);margin-top:2px;}
.v.bad{color:var(--bad);} .v.ok{color:var(--ok);} .v.cy{color:var(--cyan);} .v.go{color:var(--copper);}
.row{margin:11px 0;font-size:15px;animation:fadeUp .6s backwards;}
.row:nth-child(2){animation-delay:.08s;} .row:nth-child(3){animation-delay:.16s;} .row:nth-child(4){animation-delay:.24s;} .row:nth-child(5){animation-delay:.32s;}
.row .top2{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}
.row .nm{display:flex;align-items:center;gap:9px;}
.row .dot{width:11px;height:11px;border-radius:3px;display:inline-block;}
.row .cnt{font-family:ui-monospace,Menlo,Consolas,monospace;font-weight:700;color:var(--text);}
.row .bar{height:7px;border-radius:4px;background:rgba(255,255,255,.08);overflow:hidden;}
.row .bar i{display:block;height:100%;border-radius:4px;transform-origin:left;animation:grow 1s .3s cubic-bezier(.2,.7,.2,1) backwards;}
.note{font-size:13.5px;color:var(--muted);line-height:1.6;}
.tags{display:flex;flex-wrap:wrap;gap:10px;}
.tag{display:inline-flex;align-items:center;gap:9px;padding:8px 15px;border-radius:999px;font-size:14px;background:rgba(255,255,255,.05);animation:fadeUp .6s backwards,bob 4.5s ease-in-out infinite;}
.tag:nth-child(2){animation-delay:.08s,.4s;} .tag:nth-child(3){animation-delay:.16s,.8s;} .tag:nth-child(4){animation-delay:.24s,1.2s;} .tag:nth-child(5){animation-delay:.32s,1.6s;} .tag:nth-child(6){animation-delay:.4s,2s;}
.tag i{width:10px;height:10px;border-radius:3px;display:inline-block;}
.foot{margin-top:30px;padding-top:14px;border-top:1px solid var(--line);font-size:12px;color:var(--muted);display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;}
[data-testid="stFileUploader"],[data-testid="stSlider"],[data-testid="stToggle"],.stButton,.stDownloadButton{animation:fadeUp .7s .1s backwards;}
[data-testid="stFileUploaderDropzone"]{background:rgba(0,0,0,.28);border:1.5px dashed rgba(240,180,60,.6);border-radius:12px;animation:dz 3.4s ease-in-out infinite;}
[data-testid="stImage"]{position:relative;overflow:hidden;border-radius:10px;animation:popIn .7s .1s backwards;}
[data-testid="stImage"] img{border-radius:10px;border:1px solid var(--line);width:100%;box-shadow:0 10px 30px rgba(0,0,0,.4);transition:transform .35s ease,box-shadow .35s ease;}
[data-testid="stImage"]:hover img{transform:translateY(-3px) scale(1.012);box-shadow:0 18px 44px rgba(0,0,0,.55);}
[data-testid="stColumn"]:has(.mark-pred) [data-testid="stImage"]::after,[data-testid="column"]:has(.mark-pred) [data-testid="stImage"]::after{content:"";position:absolute;left:0;right:0;top:-6px;height:4px;pointer-events:none;background:linear-gradient(90deg,transparent,#f0b43c,#fff6d6,#f0b43c,transparent);box-shadow:0 0 22px 8px rgba(240,180,60,.55);animation:sweep 2.4s .3s ease-in-out 1 both;}
.stButton button{background:rgba(240,180,60,.14);border:1px solid rgba(240,180,60,.6);color:var(--text);border-radius:8px;font-weight:600;transition:all .25s ease;}
.stButton button:hover{background:rgba(240,180,60,.28);border-color:var(--copper);color:var(--text);transform:translateY(-2px);}
.st-key-sample_btn button{animation:ring 2.4s ease-out infinite;}
.stDownloadButton button{background:linear-gradient(90deg,#f0b43c,#ffd479);border:none;border-radius:8px;font-weight:700;transition:all .25s ease;}
.stDownloadButton button p{color:#1a1204 !important;font-weight:700;}
.stDownloadButton button:hover{transform:translateY(-2px);box-shadow:0 10px 26px rgba(240,180,60,.4);}
.nums{grid-template-columns:repeat(var(--n,3),1fr);}
@media(max-width:800px){.nums{grid-template-columns:repeat(2,1fr);}.num{border-bottom:1px solid var(--line);}}
.num .v.sm{font-size:24px;}
.tbl{width:100%;border-collapse:collapse;font-size:14px;margin:6px 0 4px;animation:fadeUp .7s backwards;}
.tbl th{font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:600;text-align:right;padding:10px 12px;border-bottom:1px solid var(--line);}
.tbl td{padding:11px 12px;text-align:right;border-bottom:1px solid rgba(255,255,255,.06);font-family:ui-monospace,Menlo,Consolas,monospace;}
.tbl th:first-child,.tbl td:first-child{text-align:left;font-family:inherit;}
.tbl tr:hover td{background:rgba(240,180,60,.06);}
.tbl .cl{display:flex;align-items:center;gap:9px;}
.tbl .cl i{width:10px;height:10px;border-radius:3px;display:inline-block;}
.tbl .mb{display:inline-block;width:70px;height:6px;border-radius:3px;background:rgba(255,255,255,.08);vertical-align:middle;margin-left:8px;overflow:hidden;}
.tbl .mb i{display:block;height:100%;border-radius:3px;transform-origin:left;animation:grow 1s .3s backwards;}
.tbl .low{color:var(--copper);font-weight:700;}
.tbl .wrap{overflow-x:auto;}
.insight{margin:14px 0;padding:14px 16px;border-radius:12px;border:1px solid rgba(240,180,60,.4);border-left:5px solid var(--copper);background:rgba(240,180,60,.07);font-size:14.5px;line-height:1.65;color:#e4ece7;}
.insight b{color:var(--copper);}
.spec{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:8px 0;}
.spec div{padding:11px 14px;border:1px solid var(--line);border-radius:10px;background:rgba(0,0,0,.22);font-size:12px;color:var(--muted);}
.spec b{display:block;font-size:15.5px;color:var(--text);margin-top:3px;font-weight:600;}
[data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line);}
[data-baseweb="tab"]{padding:10px 18px;border-radius:8px 8px 0 0;font-weight:600;}
[data-baseweb="tab"][aria-selected="true"]{color:var(--copper) !important;}
[data-baseweb="tab-highlight"]{background:var(--copper) !important;}
iframe{border:0 !important;}
.gh{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px;}
.gh i{width:12px;height:12px;border-radius:3px;display:inline-block;}
.gh b{font-size:16px;text-transform:capitalize;}
.sv{padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:#1a1204;}
.gc{font-size:12.5px;color:var(--muted);margin-left:auto;}
.gg{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;}
.gg h5{margin:0 0 6px;font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--copper);font-weight:700;}
.gg p,.gg li{font-size:13.5px;line-height:1.6;color:#dbe6df;margin:0;}
.gg ul,.gg ol{margin:0;padding-left:18px;}
.gv{margin-top:12px;padding:10px 14px;border-radius:10px;border:1px solid var(--line);background:rgba(0,0,0,.22);font-size:13px;line-height:1.6;color:#dbe6df;}
.gv b{color:var(--cyan);}
.disc{margin:8px 0 4px;font-size:12px;color:var(--muted);}
[data-baseweb="tab"]{color:#c9d9d0 !important;}
[data-baseweb="tab"] p{color:#c9d9d0 !important;font-size:15px;font-weight:600;}
[data-baseweb="tab"][aria-selected="true"],[data-baseweb="tab"][aria-selected="true"] p{color:var(--copper) !important;}
[data-baseweb="tab-border"]{background:var(--line) !important;}
[data-testid="stExpander"]{border:1px solid var(--line) !important;border-radius:12px !important;background:rgba(0,0,0,.22) !important;margin-bottom:10px;}
[data-testid="stExpander"] details,[data-testid="stExpander"] summary{background:transparent !important;color:var(--text) !important;}
[data-testid="stExpander"] summary p,[data-testid="stExpander"] summary span{color:var(--text) !important;font-weight:600;}
[data-testid="stExpander"] summary svg{color:var(--copper) !important;fill:var(--copper) !important;}
[data-baseweb="tab"],[data-baseweb="tab"] *{opacity:1 !important;}
[data-baseweb="tab"]{color:#e3ede7 !important;background:transparent !important;}
[data-baseweb="tab"] p,[data-baseweb="tab"] span,[data-baseweb="tab"] div{color:#e3ede7 !important;}
[data-baseweb="tab"]:hover,[data-baseweb="tab"]:hover p{color:#ffd479 !important;}
[data-baseweb="tab"][aria-selected="true"],[data-baseweb="tab"][aria-selected="true"] p,[data-baseweb="tab"][aria-selected="true"] span{color:var(--copper) !important;}
[data-baseweb="tab-highlight"]{background:var(--copper) !important;height:3px !important;}
[data-testid="stFileUploaderDropzone"] button{background:rgba(240,180,60,.16) !important;color:var(--text) !important;border:1px solid rgba(240,180,60,.7) !important;border-radius:8px !important;}
[data-testid="stFileUploaderDropzone"] button:hover{background:rgba(240,180,60,.32) !important;}
[data-testid="stFileUploaderDropzone"] button *{color:var(--text) !important;}
[data-testid="stFileUploaderDropzone"] small,[data-testid="stFileUploaderDropzone"] span,[data-testid="stFileUploaderDropzone"] div{color:#c9d9d0 !important;}
[data-testid="stFileUploaderDropzone"] svg{color:var(--copper) !important;fill:var(--copper) !important;}
[data-testid="stFileUploaderFile"] *{color:var(--text) !important;}
/* ---- tabs fix v2: new Streamlit markup (role=tab / data-testid=stTab) ---- */
[role="tablist"]{gap:8px !important;}
[role="tab"],button[data-testid="stTab"]{background:rgba(255,255,255,.07) !important;border:1px solid rgba(255,255,255,.2) !important;border-bottom:none !important;border-radius:10px 10px 0 0 !important;padding:10px 20px !important;opacity:1 !important;}
[role="tab"],[role="tab"] *{color:#e9f2ec !important;opacity:1 !important;}
[role="tab"] p{font-size:15px !important;font-weight:600 !important;}
[role="tab"]:hover{background:rgba(240,180,60,.18) !important;border-color:rgba(240,180,60,.7) !important;}
[role="tab"]:hover,[role="tab"]:hover *{color:#ffd479 !important;}
[role="tab"][aria-selected="true"]{background:rgba(240,180,60,.20) !important;border-color:#f0b43c !important;box-shadow:inset 0 -3px 0 #f0b43c;}
[role="tab"][aria-selected="true"],[role="tab"][aria-selected="true"] *{color:#f0b43c !important;font-weight:700 !important;}
@media (prefers-reduced-motion: reduce){*,*::before,*::after{animation:none !important;transition:none !important;}}
"""
CSS = " ".join(l.strip() for l in CSS.splitlines()).replace("__PATTERN__", PATTERN)
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Lightbox + magnifier lens (pure HTML/JS, rendered in components)
# ---------------------------------------------------------------------------
# The lightbox is injected once into the parent page so it can cover the whole screen.
LB_SRC = r"""
(function(){
if(window.tsLB)return;
var css='#ts-lb{position:fixed;inset:0;z-index:2147483000;display:none;background:radial-gradient(1200px 700px at 50% 0%,rgba(240,180,60,.10),transparent 60%),rgba(2,12,9,.94);backdrop-filter:blur(8px);font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#f1f7f3;user-select:none;-webkit-user-select:none}'
+'#ts-lb.on{display:block;animation:tsin .25s ease}@keyframes tsin{from{opacity:0}to{opacity:1}}'
+'#ts-lb .bar{position:absolute;left:0;right:0;top:0;height:58px;display:flex;align-items:center;justify-content:space-between;gap:12px;padding:0 18px;border-bottom:1px solid rgba(255,255,255,.1);background:rgba(3,20,15,.8)}'
+'#ts-lb .cap{font-size:15px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}'
+'#ts-lb .cap:before{content:"";display:inline-block;width:18px;height:3px;border-radius:2px;margin-right:10px;vertical-align:middle;background:linear-gradient(90deg,#f0b43c,#5ad7ff)}'
+'#ts-lb .tools{display:flex;align-items:center;gap:8px}'
+'#ts-lb button{cursor:pointer;min-width:38px;height:36px;padding:0 12px;border-radius:9px;border:1px solid rgba(240,180,60,.55);background:rgba(240,180,60,.12);color:#f1f7f3;font-size:16px;font-weight:700;transition:all .2s}'
+'#ts-lb button:hover{background:rgba(240,180,60,.3);transform:translateY(-1px)}'
+'#ts-lb button.x{border-color:rgba(255,92,92,.7);background:rgba(255,92,92,.14)}'
+'#ts-lb .z{min-width:54px;text-align:center;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;color:#5ad7ff}'
+'#ts-lb .stage{position:absolute;left:0;right:0;top:58px;bottom:0;overflow:hidden;cursor:grab;touch-action:none}'
+'#ts-lb .stage.drag{cursor:grabbing}'
+'#ts-lb img{position:absolute;left:0;top:0;max-width:none;transform-origin:0 0;border-radius:4px;box-shadow:0 20px 80px rgba(0,0,0,.7);will-change:transform}'
+'#ts-lb .hint{position:absolute;left:50%;bottom:16px;transform:translateX(-50%);padding:7px 16px;border-radius:999px;font-size:12px;color:#9bb8aa;background:rgba(0,0,0,.55);border:1px solid rgba(255,255,255,.1);pointer-events:none;animation:tsfade 6s forwards}'
+'@keyframes tsfade{0%,70%{opacity:1}100%{opacity:0}}';
var st=document.createElement('style');st.textContent=css;document.head.appendChild(st);
var lb=document.createElement('div');lb.id='ts-lb';
lb.innerHTML='<div class="bar"><div class="cap"></div><div class="tools"><button data-a="out">&minus;</button><span class="z">100%</span><button data-a="in">+</button><button data-a="fit">Fit</button><button data-a="x" class="x">&times;</button></div></div><div class="stage"><img draggable="false" alt=""></div><div class="hint">Scroll to zoom &middot; drag to pan &middot; double-click to zoom in &middot; Esc to close</div>';
document.body.appendChild(lb);
var stage=lb.querySelector('.stage'),img=lb.querySelector('img'),cap=lb.querySelector('.cap'),zl=lb.querySelector('.z');
var s=1,tx=0,ty=0,fit=1,nw=1,nh=1,last={};
function apply(){img.style.transform='translate('+tx+'px,'+ty+'px) scale('+s+')';zl.textContent=Math.round(s/fit*100)+'%';}
function calcFit(){fit=Math.min(stage.clientWidth/nw,stage.clientHeight/nh);}
function zoomAt(mx,my,ns){ns=Math.max(fit*.4,Math.min(fit*16,ns));tx=mx-(mx-tx)*(ns/s);ty=my-(my-ty)*(ns/s);s=ns;apply();}
function place(o){
  calcFit();var W=stage.clientWidth,H=stage.clientHeight;
  var fx=o.fx==null?.5:o.fx,fy=o.fy==null?.5:o.fy,ns;
  if(o.px){ns=.42*Math.min(W,H)/o.px;ns=Math.max(fit,Math.min(ns,fit*14));}else{ns=fit*(o.z||1);}
  s=ns;tx=W/2-fx*nw*s;ty=H/2-fy*nh*s;apply();
}
function open(src,caption,fx,fy,z,px){
  last={fx:fx,fy:fy,z:z,px:px};cap.textContent=caption||'';lb.classList.add('on');
  var h=lb.querySelector('.hint');h.style.animation='none';void h.offsetWidth;h.style.animation='';
  img.onload=function(){nw=img.naturalWidth;nh=img.naturalHeight;img.style.width=nw+'px';img.style.height=nh+'px';place(last);};
  img.src=src;if(img.complete&&img.naturalWidth)img.onload();
}
function close(){lb.classList.remove('on');}
lb.addEventListener('click',function(e){var a=e.target.getAttribute&&e.target.getAttribute('data-a');if(!a)return;
  var cx=stage.clientWidth/2,cy=stage.clientHeight/2;
  if(a==='x')close();else if(a==='in')zoomAt(cx,cy,s*1.35);else if(a==='out')zoomAt(cx,cy,s/1.35);else if(a==='fit')place({z:1});});
stage.addEventListener('wheel',function(e){e.preventDefault();var r=stage.getBoundingClientRect();zoomAt(e.clientX-r.left,e.clientY-r.top,s*Math.exp(-e.deltaY*.0018));},{passive:false});
stage.addEventListener('dblclick',function(e){var r=stage.getBoundingClientRect();zoomAt(e.clientX-r.left,e.clientY-r.top,s*2);});
var ptrs={},pd=0;
stage.addEventListener('pointerdown',function(e){stage.setPointerCapture(e.pointerId);ptrs[e.pointerId]=[e.clientX,e.clientY];stage.classList.add('drag');
  var k=Object.keys(ptrs);if(k.length===2){pd=Math.hypot(ptrs[k[0]][0]-ptrs[k[1]][0],ptrs[k[0]][1]-ptrs[k[1]][1]);}});
stage.addEventListener('pointermove',function(e){if(!ptrs[e.pointerId])return;var k=Object.keys(ptrs);
  if(k.length===2){ptrs[e.pointerId]=[e.clientX,e.clientY];var d=Math.hypot(ptrs[k[0]][0]-ptrs[k[1]][0],ptrs[k[0]][1]-ptrs[k[1]][1]);
    if(pd){var r=stage.getBoundingClientRect();zoomAt((ptrs[k[0]][0]+ptrs[k[1]][0])/2-r.left,(ptrs[k[0]][1]+ptrs[k[1]][1])/2-r.top,s*d/pd);}pd=d;return;}
  tx+=e.clientX-ptrs[e.pointerId][0];ty+=e.clientY-ptrs[e.pointerId][1];ptrs[e.pointerId]=[e.clientX,e.clientY];apply();});
function up(e){delete ptrs[e.pointerId];pd=0;if(!Object.keys(ptrs).length)stage.classList.remove('drag');}
stage.addEventListener('pointerup',up);stage.addEventListener('pointercancel',up);
window.addEventListener('keydown',function(e){if(!lb.classList.contains('on'))return;
  if(e.key==='Escape')close();else if(e.key==='+'||e.key==='=')zoomAt(stage.clientWidth/2,stage.clientHeight/2,s*1.35);
  else if(e.key==='-')zoomAt(stage.clientWidth/2,stage.clientHeight/2,s/1.35);else if(e.key==='0')place({z:1});});
window.addEventListener('resize',function(){if(lb.classList.contains('on'))place(last);});
window.tsLB={open:open,close:close};
})();
"""

# Runs inside each component iframe: makes sure the lightbox exists, exposes tsOpen(), auto-sizes the iframe.
PRELUDE = (
    "const LB_SRC=" + json.dumps(LB_SRC).replace("</", "<\\/") + ";"
    r"""
let HOST=window;try{void window.parent.document.body;HOST=window.parent;}catch(e){HOST=window;}
(function(){const D=HOST.document;
 if(!HOST.tsLB&&!D.getElementById('ts-lb-script')){const s=D.createElement('script');s.id='ts-lb-script';s.textContent=LB_SRC;D.head.appendChild(s);}
})();
function tsOpen(src,cap,fx,fy,z,px){if(HOST.tsLB)HOST.tsLB.open(src,cap,fx,fy,z,px);}
function fitFrame(){try{const fe=window.frameElement;if(fe){fe.style.height=(Math.ceil(document.body.getBoundingClientRect().height)+4)+'px';}}catch(e){}}
window.addEventListener('load',fitFrame);window.addEventListener('resize',fitFrame);setTimeout(fitFrame,300);setTimeout(fitFrame,1200);
if(window.ResizeObserver){new ResizeObserver(fitFrame).observe(document.body);}
"""
)

BASE_CSS = """
*{box-sizing:border-box}html,body{margin:0;padding:0;background:transparent;font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#f1f7f3}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes sweep{0%{top:8%}50%{top:88%}100%{top:8%}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 3px #f0b43c,0 0 0 7px rgba(240,180,60,.22),0 14px 44px rgba(0,0,0,.75),0 0 30px rgba(240,180,60,.5)}50%{box-shadow:0 0 0 3px #ffd479,0 0 0 9px rgba(90,215,255,.22),0 14px 44px rgba(0,0,0,.75),0 0 46px rgba(90,215,255,.55)}}
@keyframes up{from{opacity:0;transform:translateY(16px) scale(.97)}to{opacity:1;transform:none}}
.lens{position:absolute;display:none;pointer-events:none;border-radius:50%;transform:translate(-50%,-50%);z-index:5;animation:pulse 2.2s ease-in-out infinite}
.lens .glass{position:absolute;inset:0;border-radius:50%;overflow:hidden;background-repeat:no-repeat;background-color:#041510}
.lens .glass:after{content:"";position:absolute;inset:0;border-radius:50%;background:radial-gradient(circle at 30% 25%,rgba(255,255,255,.28),transparent 38%),radial-gradient(circle,transparent 62%,rgba(0,0,0,.38) 100%)}
.lens .scan{position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#5ad7ff,transparent);box-shadow:0 0 10px #5ad7ff;animation:sweep 2.4s ease-in-out infinite;z-index:2}
.lens .ring{position:absolute;inset:-12px;border-radius:50%;border:2px dashed rgba(90,215,255,.85);animation:spin 9s linear infinite}
.lens .ring2{position:absolute;inset:-20px;border-radius:50%;border:1px solid transparent;border-top-color:#f0b43c;border-bottom-color:#f0b43c;animation:spin 3.5s linear infinite reverse}
.lens .cross{position:absolute;inset:0;z-index:3;background:linear-gradient(#5ad7ff,#5ad7ff) center/1.5px 26% no-repeat,linear-gradient(#5ad7ff,#5ad7ff) center/26% 1.5px no-repeat;opacity:.8}
.lens .cross:after{content:"";position:absolute;left:50%;top:50%;width:7px;height:7px;margin:-3.5px;border-radius:50%;border:1.5px solid #f0b43c}
.lens .badge{position:absolute;left:50%;bottom:-34px;transform:translateX(-50%);padding:3px 10px;border-radius:999px;background:#f0b43c;color:#1a1204;font:700 11px ui-monospace,Menlo,Consolas,monospace;letter-spacing:.04em;white-space:nowrap;z-index:6;box-shadow:0 0 14px rgba(240,180,60,.6)}
.hint{margin-top:8px;font-size:12px;color:#9bb8aa;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.hint kbd{font:600 11px ui-monospace,Menlo,Consolas,monospace;padding:2px 7px;border-radius:5px;border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.06);color:#f1f7f3}
"""

# JS shared by anything that wants a lens: attachLens(wrapEl, imgEl, lensEl, {R, Z, src, onClick})
LENS_JS = r"""
function attachLens(wrap,im,lens,o){
  const glass=lens.querySelector('.glass'),badge=lens.querySelector('.badge');
  let Z=o.Z,R=o.R;glass.style.backgroundImage='url("'+o.src+'")';
  function upd(e){
    const r=im.getBoundingClientRect(),x=Math.max(0,Math.min(r.width,e.clientX-r.left)),y=Math.max(0,Math.min(r.height,e.clientY-r.top));
    lens.style.left=x+'px';lens.style.top=y+'px';
    glass.style.backgroundSize=(r.width*Z)+'px '+(r.height*Z)+'px';
    glass.style.backgroundPosition=(-(x*Z-R))+'px '+(-(y*Z-R))+'px';
    return [x/r.width,y/r.height];
  }
  function show(e){if(e.pointerType&&e.pointerType!=='mouse')return;lens.style.display='block';wrap.style.cursor='none';upd(e);}
  wrap.addEventListener('pointerenter',show);wrap.addEventListener('pointermove',show);
  wrap.addEventListener('pointerleave',function(){lens.style.display='none';wrap.style.cursor='';});
  wrap.addEventListener('wheel',function(e){if(lens.style.display!=='block')return;e.preventDefault();Z=Math.max(1.5,Math.min(9,Z*Math.exp(-e.deltaY*.0018)));badge.textContent='\u00d7'+Z.toFixed(1);upd(e);},{passive:false});
  wrap.addEventListener('click',function(e){const f=upd(e);o.onClick(f[0],f[1],Z);});
  badge.textContent='\u00d7'+Z.toFixed(1);
}
const LENS_HTML='<div class="glass"></div><div class="ring"></div><div class="ring2"></div><div class="cross"></div><div class="scan"></div><div class="badge"></div>';
"""


def viewer_html(src, caption, lens_zoom=3.0, base_src=None, badge=None, badge_ok=False):
    """Image with a magnifier lens. Click opens the full-screen viewer.
    With base_src, a scan beam sweeps once over the original and reveals the predicted image (src) under it."""
    css = (
        BASE_CSS
        + "#wrap{position:relative;line-height:0;border-radius:10px;border:1px solid rgba(255,255,255,.12);box-shadow:0 10px 30px rgba(0,0,0,.4);overflow:hidden;transition:box-shadow .4s}"
        + "#wrap.flash{animation:flash .95s ease-out}"
        + "@keyframes flash{0%{box-shadow:0 0 0 2px #5ad7ff,0 0 46px rgba(90,215,255,.8)}100%{box-shadow:0 10px 30px rgba(0,0,0,.4)}}"
        + "#im{width:100%;display:block;transition:filter .5s}"
        + "#pr{position:absolute;left:0;top:0;width:100%;height:100%;display:none;clip-path:inset(0 0 100% 0)}"
        + ".beam{position:absolute;left:0;right:0;top:0;height:0;z-index:4;pointer-events:none;display:none}"
        + ".beam:before{content:'';position:absolute;left:0;right:0;bottom:0;height:90px;background:linear-gradient(to top,rgba(90,215,255,.38),transparent)}"
        + ".beam:after{content:'';position:absolute;left:0;right:0;bottom:-1.5px;height:3px;background:#bff1ff;box-shadow:0 0 12px 3px #5ad7ff,0 0 44px 10px rgba(90,215,255,.55)}"
        + ".found{position:absolute;left:12px;top:12px;z-index:6;padding:6px 14px;border-radius:999px;font:700 12px ui-monospace,Menlo,Consolas,monospace;letter-spacing:.05em;line-height:1.4;"
        + "background:#ff5c5c;color:#1c0707;box-shadow:0 0 22px rgba(255,92,92,.6);transform:scale(0);animation:pop .55s cubic-bezier(.2,1.7,.4,1) forwards}"
        + ".found.ok{background:#4ee08a;color:#05210f;box-shadow:0 0 22px rgba(78,224,138,.6)}"
        + "@keyframes pop{from{transform:scale(0) rotate(-6deg)}to{transform:scale(1)}}"
        + "#lens{width:190px;height:190px}"
        + "#rp{margin-left:auto;cursor:pointer;padding:3px 11px;border-radius:999px;border:1px solid rgba(240,180,60,.6);background:rgba(240,180,60,.12);color:#f1f7f3;font:600 11.5px system-ui}"
        + "#rp:hover{background:rgba(240,180,60,.3)}"
    )
    scan = base_src is not None
    html = (
        "<style>" + css + "</style>"
        "<div id='wrap'><img id='im' alt=''>"
        + ("<img id='pr' alt=''><div class='beam' id='beam'></div>" if scan else "")
        + "<div class='lens' id='lens'></div></div>"
        "<div class='hint'><span>Hover to magnify</span><kbd>scroll</kbd><span>changes zoom</span><kbd>click</kbd><span>opens full view</span>"
        + ("<button id='rp'>Replay scan</button>" if scan else "")
        + "</div><script>" + PRELUDE + LENS_JS
        + "const SRC=" + json.dumps(src) + ",CAP=" + json.dumps(caption) + ",BASE=" + json.dumps(base_src) + ",BADGE=" + json.dumps(badge) + ",OK=" + ("true" if badge_ok else "false") + ";"
        + r"""
const im=document.getElementById('im'),lens=document.getElementById('lens'),wrap=document.getElementById('wrap');
lens.innerHTML=LENS_HTML;
const PR=document.getElementById('pr'),beam=document.getElementById('beam');
let raf=0;
function showBadge(){
  const old=wrap.querySelector('.found');if(old)old.remove();
  if(!BADGE)return;const b=document.createElement('div');b.className='found'+(OK?' ok':'');b.textContent=BADGE;wrap.appendChild(b);
}
function runScan(){
  cancelAnimationFrame(raf);const old=wrap.querySelector('.found');if(old)old.remove();
  const reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  PR.style.display='block';
  if(reduce){PR.style.clipPath='none';im.style.filter='none';showBadge();return;}
  im.style.filter='brightness(.7) saturate(.55)';PR.style.clipPath='inset(0 0 100% 0)';beam.style.display='block';beam.style.top='0%';
  const D=2000,t0=performance.now();
  (function step(t){
    const p=Math.min(1,Math.max(0,(t-t0)/D)),e=p<.5?2*p*p:1-Math.pow(-2*p+2,2)/2;
    PR.style.clipPath='inset(0 0 '+((1-e)*100)+'% 0)';beam.style.top=(e*100)+'%';
    if(p<1){raf=requestAnimationFrame(step);}
    else{beam.style.display='none';PR.style.clipPath='none';im.style.filter='none';wrap.classList.remove('flash');void wrap.offsetWidth;wrap.classList.add('flash');showBadge();}
  })(t0);
}
if(BASE){
  im.src=BASE;
  PR.addEventListener('load',function(){runScan();fitFrame();});
  PR.src=SRC;
  document.getElementById('rp').addEventListener('click',runScan);
}else{
  im.src=SRC;
}
attachLens(wrap,im,lens,{R:95,Z:""" + str(lens_zoom) + r""",src:SRC,onClick:function(fx,fy,z){tsOpen(SRC,CAP,fx,fy,Math.max(2,z*.9));}});
im.addEventListener('load',fitFrame);
</script>"""
    )
    return html


def gallery_html(big_src, items):
    """Grid of zoomed defect cards. Hover shows a lens, click opens the defect in the full-screen viewer."""
    css = (
        BASE_CSS
        + "#g{display:grid;grid-template-columns:repeat(4,200px);justify-content:space-between;gap:16px;padding:6px 2px 8px;overflow-x:auto;overflow-y:hidden}"
        + ".card{--c:#f0b43c;position:relative;border-radius:14px;border:1px solid rgba(255,255,255,.12);background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(0,0,0,.22));overflow:hidden;cursor:zoom-in;"
        + "animation:up .55s cubic-bezier(.2,.7,.2,1) backwards;transition:transform .3s,box-shadow .3s,border-color .3s}"
        + ".card:hover{transform:translateY(-5px);border-color:var(--c);box-shadow:0 16px 40px rgba(0,0,0,.55),0 0 28px -4px var(--c)}"
        + ".pic{position:relative;aspect-ratio:1/1;background:#031510;overflow:hidden;line-height:0}"
        + ".pic img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .5s}"
        + ".card:hover .pic img{transform:scale(1.04)}"
        + ".idx{position:absolute;left:10px;top:10px;z-index:4;padding:3px 9px;border-radius:999px;font:700 11px ui-monospace,Menlo,monospace;background:var(--c);color:#161006;line-height:1.4}"
        + ".sev{position:absolute;right:10px;top:10px;z-index:4;padding:3px 9px;border-radius:999px;font:600 10.5px system-ui;background:rgba(0,0,0,.6);border:1px solid rgba(255,255,255,.2);line-height:1.4;letter-spacing:.04em;text-transform:uppercase}"
        + ".meta{padding:12px 14px 14px}"
        + ".nm{display:flex;align-items:center;gap:8px;font-weight:700;font-size:15px;text-transform:capitalize}"
        + ".nm i{width:11px;height:11px;border-radius:3px;background:var(--c);display:inline-block}"
        + ".cf{display:flex;justify-content:space-between;align-items:baseline;margin:9px 0 6px;font-size:12px;color:#9bb8aa}"
        + ".cf b{font:700 20px ui-monospace,Menlo,monospace;color:#f1f7f3}"
        + ".bar{height:6px;border-radius:3px;background:rgba(255,255,255,.09);overflow:hidden}"
        + ".bar i{display:block;height:100%;border-radius:3px;background:var(--c);transform-origin:left;animation:grow 1s .3s backwards}"
        + "@keyframes grow{from{transform:scaleX(0)}to{transform:none}}"
        + ".sub{margin-top:9px;font:11.5px ui-monospace,Menlo,monospace;color:#9bb8aa;display:flex;justify-content:space-between;gap:6px}"
        + ".fx{margin-top:auto;padding-top:9px;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden;border-top:1px dashed rgba(255,255,255,.16);font-size:12px;line-height:1.5;color:#c9d9d0}.fx b{color:#f0b43c}"
        + ".card{width:200px}.meta{height:205px;display:flex;flex-direction:column}"
        + ".lens{width:96px;height:96px}.lens .badge{bottom:-30px}"
        + ".lens .ring{inset:-8px}.lens .ring2{inset:-14px}"
    )
    body = (
        "<div id='g'></div>"
        "<script>" + PRELUDE + LENS_JS
        + "const BIG=" + json.dumps(big_src) + ";"
        + "const ITEMS=" + json.dumps(items).replace("</", "<\\/") + ";"
        + r"""
const g=document.getElementById('g');g.style.gridTemplateColumns='repeat('+Math.min(ITEMS.length,4)+',200px)';if(ITEMS.length<4)g.style.justifyContent='start';
ITEMS.forEach(function(it,i){
  const c=document.createElement('div');c.className='card';c.style.setProperty('--c',it.color);c.style.animationDelay=(i*.05)+'s';
  const sc={Critical:'#ff5c5c',High:'#ff9f43',Medium:'#f0b43c'}[it.sev]||'#f0b43c';
  c.innerHTML='<div class="pic"><img alt=""><span class="idx">#'+it.idx+'</span><span class="sev" style="border-color:'+sc+';color:'+sc+'">'+it.sev+'</span><div class="lens"></div></div>'
   +'<div class="meta"><div class="nm"><i></i>'+it.name+'</div><div class="cf"><span>confidence</span><b>'+Math.round(it.conf*100)+'%</b></div>'
   +'<div class="bar"><i style="width:'+Math.round(it.conf*100)+'%"></i></div>'
   +'<div class="sub"><span>'+it.w+' &times; '+it.h+' px</span><span>x '+it.cx+' &middot; y '+it.cy+'</span></div><div class="fx"><b>Fix:</b> '+it.hint+'</div></div>';
  g.appendChild(c);
  const im=c.querySelector('img'),lens=c.querySelector('.lens'),pic=c.querySelector('.pic');im.src=it.crop;lens.innerHTML=LENS_HTML;
  attachLens(pic,im,lens,{R:48,Z:2.2,src:it.crop,onClick:function(){tsOpen(BIG,'#'+it.idx+' \u00b7 '+it.name+' \u00b7 '+Math.round(it.conf*100)+'% confidence',it.fx,it.fy,1,it.px);}});
  c.addEventListener('click',function(e){if(!pic.contains(e.target))tsOpen(BIG,'#'+it.idx+' \u00b7 '+it.name+' \u00b7 '+Math.round(it.conf*100)+'% confidence',it.fx,it.fy,1,it.px);});
});
"""
        + "</script>"
    )
    return "<style>" + css + "</style>" + body

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def rgb_css(c):
    return f"rgb({c[0]},{c[1]},{c[2]})"


def pretty(name):
    return name.replace("_", " ")


def data_uri(img, max_side=1800, quality=86):
    """PIL image -> JPEG data URI (downscaled so the page stays light)."""
    im = img.copy()
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), im.size


def file_uri(path):
    raw = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode()


@st.cache_data(show_spinner=False, max_entries=128)
def detect(_model, data: bytes, conf: float):
    """Run the model once per (image, threshold). Returns plain lists, so it is cacheable."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    t0 = time.perf_counter()
    r = _model.predict(img, imgsz=640, conf=conf, device="cpu", verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    return {
        "xyxy": r.boxes.xyxy.tolist(),
        "cls": [int(c) for c in r.boxes.cls.tolist()],
        "conf": r.boxes.conf.tolist(),
        "size": img.size,
        "ms": ms,
    }


def draw_boxes(img, det, names, show_labels, show_scores):
    """Draw high-contrast boxes with thick lines and large, readable labels."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    big = max(w, h)
    lw = max(4, round(big / 200))
    fs = max(30, round(big / 28))
    try:
        font = ImageFont.load_default(size=fs)
    except TypeError:
        font = ImageFont.load_default()

    for xyxy, c, p in zip(det["xyxy"], det["cls"], det["conf"]):
        col = PALETTE[int(c) % len(PALETTE)]
        x1, y1, x2, y2 = xyxy
        draw.rectangle([x1, y1, x2, y2], outline=col, width=lw)

        parts = []
        if show_labels:
            parts.append(pretty(names[int(c)]))
        if show_scores:
            parts.append(f"{p:.2f}")
        if not parts:
            continue
        text = " ".join(parts)
        bb = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad = lw + 4
        box_w, box_h = tw + 2 * pad, th + 2 * pad
        bx = max(0, min(x1, w - box_w))
        by = y1 - box_h
        if by < 0:
            by = y1
        draw.rectangle([bx, by, bx + box_w, by + box_h], fill=col)
        draw.text((bx + pad - bb[0], by + pad - bb[1]), text, fill=(15, 15, 15), font=font)
    return out


def make_crop(img, xyxy, color, out=560):
    """Square close-up around a defect, upscaled, with a thin box drawn on it."""
    x1, y1, x2, y2 = xyxy
    W, H = img.size
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(max(x2 - x1, y2 - y1) * 2.6, 96)
    side = int(min(side, W, H))
    left = int(min(max(cx - side / 2, 0), W - side))
    top = int(min(max(cy - side / 2, 0), H - side))
    crop = img.crop((left, top, left + side, top + side)).resize((out, out), Image.LANCZOS)
    k = out / side
    d = ImageDraw.Draw(crop)
    d.rectangle([(x1 - left) * k, (y1 - top) * k, (x2 - left) * k, (y2 - top) * k], outline=color, width=3)
    return crop


def count_classes(det):
    return Counter(det["cls"])


def stats_of(det):
    n = len(det["cls"])
    W, H = det["size"]
    confs = det["conf"]
    area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in det["xyxy"])
    return {
        "n": n,
        "avg": (sum(confs) / n * 100) if n else 0,
        "top": (max(confs) * 100) if n else 0,
        "coverage": area / (W * H) * 100 if n else 0,
        "types": len(set(det["cls"])),
    }


def class_rows(counts):
    top = max(counts.values())
    return "<div class='rows'>" + "".join(
        "<div class='row'><div class='top2'>"
        f"<span class='nm'><span class='dot' style='background:{rgb_css(PALETTE[cid % len(PALETTE)])}'></span>"
        f"{pretty(model.names[cid])}</span><span class='cnt'>{c}</span></div>"
        f"<div class='bar'><i style='width:{c / top * 100:.0f}%;"
        f"background:{rgb_css(PALETTE[cid % len(PALETTE)])}'></i></div></div>"
        for cid, c in counts.most_common()
    ) + "</div>"


def verdict_html(n):
    if n == 0:
        return "<div class='verdict'>PASS &middot; no defects detected<small>Nothing found at this sensitivity.</small></div>"
    return (
        f"<div class='verdict fail'>FAIL &middot; {n} defect{'s' if n != 1 else ''} detected"
        "<small>Each one is boxed on the predicted image.</small></div>"
    )


def show_html(html, height):
    """Render an interactive HTML/JS block. st.iframe is the current API; fall back for older Streamlit."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=int(height))
    else:
        components.html(html, height=int(height))


def render_viewer(predicted, caption, width_guess=540, original=None, n=None):
    src, (w, h) = data_uri(predicted)
    base = badge = None
    if original is not None:
        base, _ = data_uri(original)
        badge = "CLEAN \u00b7 no defects" if not n else f"{n} DEFECT{'S' if n != 1 else ''} FOUND"
    show_html(viewer_html(src, caption, base_src=base, badge=badge, badge_ok=(n == 0)), int(width_guess * h / w) + 40)


def render_gallery(img, predicted, det):
    """Close-up cards for every defect (most confident first)."""
    if not det["cls"]:
        return
    big_src, (bw, bh) = data_uri(predicted)
    sc = bw / img.size[0]
    order = sorted(range(len(det["cls"])), key=lambda i: -det["conf"][i])
    shown = order[:24]
    items = []
    for rank, i in enumerate(shown, 1):
        x1, y1, x2, y2 = det["xyxy"][i]
        col = PALETTE[det["cls"][i] % len(PALETTE)]
        crop_src, _ = data_uri(make_crop(img, (x1, y1, x2, y2), col), max_side=560, quality=88)
        items.append(
            {
                "idx": rank,
                "name": pretty(model.names[det["cls"][i]]),
                "color": rgb_css(col),
                "conf": round(det["conf"][i], 4),
                "crop": crop_src,
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "cx": int((x1 + x2) / 2),
                "cy": int((y1 + y2) / 2),
                "fx": ((x1 + x2) / 2) / img.size[0],
                "fy": ((y1 + y2) / 2) / img.size[1],
                "px": max(x2 - x1, y2 - y1) * sc,
                "sev": KNOWLEDGE.get(model.names[det["cls"][i]], {}).get("sev", "Medium"),
                "hint": KNOWLEDGE.get(model.names[det["cls"][i]], {}).get("hint", "Inspect and rework per your standard."),
            }
        )
    st.markdown("<div class='label'>Defect close-ups</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='note' style='margin-bottom:8px'>Hover a card to use the lens. "
        "Click to open that defect on the full board, zoom and pan freely.</div>",
        unsafe_allow_html=True,
    )
    rows = -(-len(items) // 4)
    show_html(gallery_html(big_src, items), rows * 421 + 24)
    render_repair_guide(det)
    if len(order) > len(shown):
        st.caption(f"Showing the {len(shown)} most confident of {len(order)} defects.")


def render_repair_guide(det):
    """Per-defect-type repair notes for the technician, most severe first."""
    counts = count_classes(det)
    found = [(model.names[cid], c) for cid, c in counts.items() if model.names[cid] in KNOWLEDGE]
    if not found:
        return
    found.sort(key=lambda t: (SEV_RANK[KNOWLEDGE[t[0]]["sev"]], -t[1]))
    st.markdown("<div class='label'>Repair guide</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='note'>What each defect is, why it matters and how to fix it. Defect types are sorted by severity, "
        "so repair the Critical ones first.</div>",
        unsafe_allow_html=True,
    )
    for k, (cname, c) in enumerate(found):
        info = KNOWLEDGE[cname]
        cid = [i for i, v in model.names.items() if v == cname][0]
        col = rgb_css(PALETTE[cid % len(PALETTE)])
        label = f"{pretty(cname).title()}  -  {c} found  -  {info['sev']}"
        with st.expander(label, expanded=(k == 0)):
            st.markdown(
                f"<div class='gh'><i style='background:{col}'></i><b>{pretty(cname)}</b>"
                f"<span class='sv' style='background:{SEV_COLOR[info['sev']]}'>{info['sev']}</span>"
                f"<span class='gc'>{c} found on this board</span></div>"
                "<div class='gg'>"
                f"<div><h5>What it is</h5><p>{info['what']}</p></div>"
                f"<div><h5>Why it matters</h5><p>{info['impact']}</p></div>"
                "<div><h5>Likely causes</h5><ul>" + "".join(f"<li>{x}</li>" for x in info["causes"]) + "</ul></div>"
                "<div><h5>How to repair</h5><ol>" + "".join(f"<li>{x}</li>" for x in info["fix"]) + "</ol></div>"
                "</div>"
                f"<div class='gv'><b>Verify:</b> {info['verify']}<br><b>Tools:</b> {info['tools']}</div>",
                unsafe_allow_html=True,
            )
    st.markdown(
        "<div class='disc'>General guidance only. Confirm each repair against your quality standard "
        "(for example IPC-A-600) and your board's design rules. Boards for critical applications may need to be scrapped, not reworked.</div>",
        unsafe_allow_html=True,
    )


def defect_table(det):
    rows = []
    for k, i in enumerate(sorted(range(len(det["cls"])), key=lambda i: -det["conf"][i]), 1):
        x1, y1, x2, y2 = det["xyxy"][i]
        rows.append(
            {
                "#": k,
                "Defect": pretty(model.names[det["cls"][i]]),
                "Confidence %": round(det["conf"][i] * 100, 1),
                "X": int((x1 + x2) / 2),
                "Y": int((y1 + y2) / 2),
                "Width px": int(x2 - x1),
                "Height px": int(y2 - y1),
                "Severity": KNOWLEDGE.get(model.names[det["cls"][i]], {}).get("sev", ""),
                "Suggested fix": KNOWLEDGE.get(model.names[det["cls"][i]], {}).get("hint", ""),
            }
        )
    return pd.DataFrame(rows)


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
    "<div class='top'>"
    f"<div class='brand'><div class='logo'>{LOGO}</div><div>"
    "<div class='name'>Trace<span>Scan</span></div>"
    "<div class='tag'>AI visual inspection for printed circuit boards</div></div></div>"
    f"<div class='live'><i></i>Model online &middot; mAP@50 <b>{MODEL_MAP50}</b></div>"
    "</div>"
    "<div class='lede'>A single missed fault can scrap an entire production batch. "
    f"<b>{APP_NAME}</b> checks each board in seconds and marks exactly where the problem is, "
    "so defects are caught before assembly, not after.</div>"
    "<div class='rule'></div>",
    unsafe_allow_html=True,
)

tab_one, tab_batch, tab_about = st.tabs(["Inspect a board", "Batch inspection", "About the model"])

# ===========================================================================
# TAB 1 - single board
# ===========================================================================
with tab_one:
    st.markdown("<div class='label'>Board image</div>", unsafe_allow_html=True)
    up_col, sample_col = st.columns([3, 2], gap="large")

    with up_col:
        file = st.file_uploader(
            "Upload a PCB image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="single_up",
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

    raw, fname = None, "board"
    if file:
        raw, fname = file.getvalue(), Path(file.name).stem
    elif st.session_state["sample"]:
        raw, fname = Path(st.session_state["sample"]).read_bytes(), Path(st.session_state["sample"]).stem
    img = Image.open(io.BytesIO(raw)).convert("RGB") if raw else None

    if img is None:
        tags = "".join(
            f"<span class='tag' style='border:1px solid {rgb_css(PALETTE[i % len(PALETTE)])}'>"
            f"<i style='background:{rgb_css(PALETTE[i % len(PALETTE)])}'></i>{pretty(n)}</span>"
            for i, n in enumerate(class_names)
        )
        st.markdown(
            "<div class='label'>What it detects</div>"
            f"<div class='tags'>{tags}</div>"
            "<div class='note' style='margin-top:12px'>Each defect type gets its own color on the result image.</div>",
            unsafe_allow_html=True,
        )
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

        with st.spinner("Scanning..."):
            det = detect(model, raw, conf)
        S = stats_of(det)
        n = S["n"]
        counts = count_classes(det)
        predicted = draw_boxes(img, det, model.names, show_labels, show_scores) if n else img

        st.markdown("<div class='label'>Result</div>", unsafe_allow_html=True)
        col_o, col_p = st.columns(2, gap="medium")
        with col_o:
            st.markdown("<div class='pill'>Original</div>", unsafe_allow_html=True)
            st.image(img)
        with col_p:
            st.markdown(
                f"<div class='pill hot'>Predicted &middot; {n} defect{'s' if n != 1 else ''}</div>"
                "<span class='mark-pred'></span>",
                unsafe_allow_html=True,
            )
            render_viewer(predicted, f"{fname} \u00b7 {n} defect{'s' if n != 1 else ''}", original=img, n=n)

        st.markdown("<div class='label'>Summary</div>", unsafe_allow_html=True)
        sum_l, sum_r = st.columns(2, gap="large")
        with sum_l:
            avg_t = f"{S['avg']:.0f}%" if n else "-"
            top_t = f"{S['top']:.0f}%" if n else "-"
            cov_t = f"{S['coverage']:.2f}%" if n else "-"
            tone = "bad" if n else "ok"
            nums = (
                "<div class='nums' style='--n:3'>"
                f"<div class='num'><div class='v {tone}'>{n}</div><div class='k'>Defects</div></div>"
                f"<div class='num'><div class='v cy'>{avg_t}</div><div class='k'>Avg confidence</div></div>"
                f"<div class='num'><div class='v go'>{det['ms']:.0f}</div><div class='k'>Scan time (ms)</div></div>"
                f"<div class='num'><div class='v {tone}'>{S['types']}</div><div class='k'>Defect types</div></div>"
                f"<div class='num'><div class='v cy'>{top_t}</div><div class='k'>Top confidence</div></div>"
                f"<div class='num'><div class='v go'>{cov_t}</div><div class='k'>Board area affected</div></div>"
                "</div>"
            )
            st.markdown(verdict_html(n) + nums, unsafe_allow_html=True)

        with sum_r:
            if n:
                st.markdown(class_rows(counts), unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='note'>This board looks clean. Lower the sensitivity if you "
                    "suspect a faint defect.</div>",
                    unsafe_allow_html=True,
                )
            buf = io.BytesIO()
            predicted.save(buf, format="PNG")
            st.write("")
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download image",
                    data=buf.getvalue(),
                    file_name=f"tracescan_{fname}.png",
                    mime="image/png",
                )
            with d2:
                if n:
                    st.download_button(
                        "Download report (CSV)",
                        data=defect_table(det).to_csv(index=False).encode(),
                        file_name=f"tracescan_{fname}_defects.csv",
                        mime="text/csv",
                    )

        if n:
            render_gallery(img, predicted, det)
            with st.expander("All detections (table)"):
                st.dataframe(defect_table(det), hide_index=True)

# ===========================================================================
# TAB 2 - batch
# ===========================================================================
with tab_batch:
    st.markdown("<div class='label'>Upload several boards</div>", unsafe_allow_html=True)
    files = st.file_uploader(
        "Upload PCB images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="batch_up",
    )
    bconf = st.slider("Sensitivity (confidence threshold)", 0.10, 0.90, 0.40, 0.05, key="batch_conf")

    if not files:
        st.markdown(
            "<div class='note' style='margin-top:10px'>Drop a whole batch of boards here. "
            "You get a PASS / FAIL verdict for each one, a defect breakdown, and a CSV you can download. "
            "A board passes when no defect is found at the chosen sensitivity.</div>",
            unsafe_allow_html=True,
        )
    else:
        prog = st.progress(0.0, text="Scanning boards...")
        results = []
        for i, f in enumerate(files):
            data = f.getvalue()
            d = detect(model, data, bconf)
            results.append((f.name, data, d))
            prog.progress((i + 1) / len(files), text=f"Scanning boards... {i + 1}/{len(files)}")
        prog.empty()

        rows = []
        total_counts = Counter()
        for i, (nm, _, d) in enumerate(results, 1):
            c = count_classes(d)
            total_counts.update(c)
            row = {
                "#": i,
                "Board": nm,
                "Status": "FAIL" if d["cls"] else "PASS",
                "Defects": len(d["cls"]),
                "Avg confidence %": round(sum(d["conf"]) / len(d["conf"]) * 100, 1) if d["conf"] else 0.0,
                "Main defect": pretty(model.names[c.most_common(1)[0][0]]) if c else "-",
            }
            for cid, cname in model.names.items():
                row[pretty(cname)] = c.get(cid, 0)
            rows.append(row)
        df = pd.DataFrame(rows)

        total = len(df)
        n_fail = int((df["Status"] == "FAIL").sum())
        n_pass = total - n_fail
        yield_pct = n_pass / total * 100
        st.markdown("<div class='label'>Batch summary</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='nums' style='--n:5'>"
            f"<div class='num'><div class='v cy'>{total}</div><div class='k'>Boards scanned</div></div>"
            f"<div class='num'><div class='v ok'>{n_pass}</div><div class='k'>Passed</div></div>"
            f"<div class='num'><div class='v {'bad' if n_fail else 'ok'}'>{n_fail}</div><div class='k'>Failed</div></div>"
            f"<div class='num'><div class='v go'>{yield_pct:.0f}%</div><div class='k'>Yield</div></div>"
            f"<div class='num'><div class='v {'bad' if total_counts else 'ok'}'>{sum(total_counts.values())}</div><div class='k'>Total defects</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='label'>Results</div>", unsafe_allow_html=True)
        show_df = df.copy()
        show_df["Status"] = show_df["Status"].map({"PASS": "\u2705 PASS", "FAIL": "\u274c FAIL"})
        st.dataframe(show_df, hide_index=True)
        st.download_button(
            "Download results (CSV)",
            data=df.to_csv(index=False).encode(),
            file_name="tracescan_batch_results.csv",
            mime="text/csv",
            key="batch_csv",
        )

        if total_counts:
            st.markdown("<div class='label'>Defects by type</div>", unsafe_allow_html=True)
            chart = pd.DataFrame(
                {"Defects": [total_counts.get(cid, 0) for cid in model.names]},
                index=[pretty(v) for v in model.names.values()],
            )
            st.bar_chart(chart, color="#f0b43c", height=260)

        st.markdown("<div class='label'>Inspect one board</div>", unsafe_allow_html=True)
        pick = st.selectbox(
            "Board",
            range(total),
            format_func=lambda i: f"{i + 1}. {results[i][0]}  -  {rows[i]['Status']}  ({rows[i]['Defects']} defects)",
            label_visibility="collapsed",
            key="batch_pick",
        )
        nm, data, d = results[pick]
        bimg = Image.open(io.BytesIO(data)).convert("RGB")
        bpred = draw_boxes(bimg, d, model.names, True, True) if d["cls"] else bimg
        bl, br = st.columns([3, 2], gap="large")
        with bl:
            render_viewer(bpred, f"{nm} \u00b7 {len(d['cls'])} defects", width_guess=700, original=bimg, n=len(d['cls']))
        with br:
            st.markdown(verdict_html(len(d["cls"])), unsafe_allow_html=True)
            if d["cls"]:
                st.write("")
                st.markdown(class_rows(count_classes(d)), unsafe_allow_html=True)
        if d["cls"]:
            render_gallery(bimg, bpred, d)

# ===========================================================================
# TAB 3 - about the model
# ===========================================================================
with tab_about:
    best_recall = min(CLASS_STATS, key=lambda r: r[4])
    best_map = min(CLASS_STATS, key=lambda r: r[5])
    st.markdown(
        "<div class='label'>Test-set performance</div>"
        "<div class='nums' style='--n:4'>"
        "<div class='num'><div class='v go'>95.2%</div><div class='k'>mAP@50</div></div>"
        "<div class='num'><div class='v cy'>51.1%</div><div class='k'>mAP@50-95</div></div>"
        "<div class='num'><div class='v ok'>96.1%</div><div class='k'>Precision</div></div>"
        "<div class='num'><div class='v ok'>90.8%</div><div class='k'>Recall</div></div>"
        "</div>"
        "<div class='note' style='margin-top:8px'>Measured on a held-out test split the model never saw during training: "
        "69 images with 292 labeled defects.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='label'>Model card</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='spec'>"
        "<div>Architecture<b>YOLO26s</b></div>"
        "<div>Parameters<b>9.47 M</b></div>"
        "<div>Compute<b>20.8 GFLOPs</b></div>"
        "<div>Inference (T4 GPU)<b>12.7 ms / image</b></div>"
        "<div>Input size<b>640 &times; 640</b></div>"
        "<div>Training<b>up to 80 epochs, early stop 20</b></div>"
        "<div>Defect classes<b>6</b></div>"
        "<div>Dataset<b>Roboflow PCB defects</b></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='label'>Per-class results</div>", unsafe_allow_html=True)
    trs = ""
    for i, (cn, ims, inst, p, r, m50, m95) in enumerate(CLASS_STATS):
        col = rgb_css(PALETTE[i % len(PALETTE)])
        low_r = " class='low'" if cn == best_recall[0] else ""
        low_m = " class='low'" if cn == best_map[0] else ""
        trs += (
            f"<tr><td><span class='cl'><i style='background:{col}'></i>{pretty(cn)}</span></td>"
            f"<td>{ims}</td><td>{inst}</td><td>{p:.3f}</td><td{low_r}>{r:.3f}</td>"
            f"<td{low_m}>{m50:.3f}<span class='mb'><i style='width:{m50 * 100:.0f}%;background:{col}'></i></span></td>"
            f"<td>{m95:.3f}</td></tr>"
        )
    st.markdown(
        "<div class='wrap'><table class='tbl'><thead><tr><th>Class</th><th>Images</th><th>Instances</th>"
        "<th>Precision</th><th>Recall</th><th>mAP@50</th><th>mAP@50-95</th></tr></thead>"
        f"<tbody>{trs}</tbody></table></div>"
        "<div class='insight'><b>Honest read of the numbers.</b> "
        f"The weakest recall is <b>{pretty(best_recall[0])}</b> ({best_recall[4]:.3f}) and the lowest mAP@50 is "
        f"<b>{pretty(best_map[0])}</b> ({best_map[5]:.3f}). These are small, low-contrast copper defects that are easy to "
        "confuse with normal traces. Lowering the sensitivity slider recovers some of them, at the cost of more false alarms.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='label'>Training plots</div>", unsafe_allow_html=True)
    avail = {k: v for k, v in PLOTS.items() if (RESULTS_DIR / v).exists()}
    if avail:
        choice = st.radio("Plot", list(avail), horizontal=True, label_visibility="collapsed", key="plot_pick")
        pth = RESULTS_DIR / avail[choice]
        pw, ph = Image.open(pth).size
        show_html(viewer_html(file_uri(pth), choice, lens_zoom=2.5), int(1100 * ph / pw) + 40)
    else:
        st.info("Put the evaluation plots (confusion_matrix.png, BoxPR_curve.png, ...) in a `results/` folder next to app.py.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    f"<div class='foot'><span>{APP_NAME} &middot; automated PCB visual inspection</span>"
    "<span>Powered by YOLO26</span></div>",
    unsafe_allow_html=True,
)
