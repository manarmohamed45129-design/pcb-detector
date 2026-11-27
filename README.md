# TraceScan: PCB Defect Detection (YOLO26 + Streamlit)

AI visual inspection for printed circuit boards. Upload a board photo and TraceScan marks every defect, magnifies it, tells you how to fix it, and gives a PASS / FAIL verdict in seconds.

**Live demo:** https://pcb-detector-r9jv2gjjykycvz2aas7yfj.streamlit.app/

No board at hand? Click **Try a random sample** in the app and it runs on one of the bundled test images.

## Why this project matters

A printed circuit board (PCB) is the foundation of almost every electronic device, and its copper traces are only a fraction of a millimeter wide. A single hairline break or a stray bridge between two tracks can kill a device, or worse, let it pass testing and fail later in the field.

Finding these faults is harder than it sounds:

- **They are tiny.** Many defects are only a few pixels wide and look almost identical to normal traces, even under good lighting.
- **People get tired.** After hours of repetitive inspection, attention drops and results vary from one inspector to the next.
- **Manual inspection does not scale.** Production lines make thousands of boards, and a human cannot check each one with the same care.
- **Late detection is expensive.** A fault found after components are soldered means scrapping or reworking a finished assembly. On the bare board, the same fault costs almost nothing to catch.

TraceScan shows how a small, fast object detector can act as a **first-line quality gate**: it applies the same standard to every board at any hour, points to the exact location of each fault, suggests how to repair it, and keeps a record of every result.

## Features

| Feature | What it does |
|---|---|
| **Defect detection** | Finds and classifies 6 PCB defect types, each drawn in its own color. |
| **Scan animation** | A scan beam sweeps over the board and reveals the detections, with a badge showing how many defects were found. |
| **Magnifier lens** | Hover over the result to magnify it, scroll to change the zoom, click to open a full-screen zoom and pan viewer. |
| **Defect close-ups** | Every defect is cropped and enlarged in its own card with name, confidence, size and severity. Click a card to see it on the full board. |
| **Repair guide** | For each defect type found: what it is, why it matters, likely causes, step-by-step repair, how to verify, and tools needed. Sorted by severity, so Critical defects come first. |
| **Batch inspection** | Upload many boards at once. Get PASS / FAIL per board, yield, defects by type, and a CSV export. |
| **Random sample boards** | One click loads a random sample board so you can try the app without any files. |
| **Adjustable sensitivity** | A confidence slider trades missed defects against false alarms. Labels and scores can be toggled on or off. |
| **Reports** | Download the annotated image, or a CSV with position, size, confidence, severity and suggested fix for every detection. |
| **About the model** | Real test-set metrics, per-class table, confusion matrix and PR / F1 curves, so the numbers can be verified. |

## Defect classes

`missing_hole` · `mouse_bite` · `open_circuit` · `short` · `spur` · `spurious_copper`

## How it works

1. **Upload** a board image (JPG or PNG), or load a random sample.
2. **Detect:** YOLO26s runs at 640 px and returns boxes, classes and confidence scores.
3. **Annotate:** each defect type gets its own high-contrast color.
4. **Explain:** close-ups and a repair guide are generated for every defect found.
5. **Decide:** the board passes if nothing is found at the chosen sensitivity, otherwise it fails.

## Model results

Evaluated on a held-out test split the model never saw during training (69 images, 292 labeled defects).

| Metric | Value |
|---|---|
| mAP@50 | **95.2%** |
| mAP@50-95 | 51.1% |
| Precision | 96.1% |
| Recall | 90.8% |
| Inference speed | about 12.7 ms per image (Tesla T4) |

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| missing_hole | 0.988 | 0.985 | 0.997 | 0.554 |
| mouse_bite | 0.998 | 0.926 | 0.959 | 0.522 |
| open_circuit | 0.927 | 0.906 | 0.914 | 0.527 |
| short | 0.981 | 0.984 | 0.996 | 0.547 |
| spur | 0.897 | 0.815 | 0.945 | 0.449 |
| spurious_copper | 0.977 | 0.833 | 0.902 | 0.469 |

**Model:** YOLO26s, 9.47 M parameters, 20.8 GFLOPs.

## Resources

- **Dataset (Roboflow):** [pcb-defect-detection](https://app.roboflow.com/manar-nqk6x/pcb-defect-detection-9ewqw-rzuwm/1)
- **Training notebook in this repo:** [`yolo26_PCB Defection.ipynb`](<./yolo26_PCB Defection.ipynb>)
- **Based on the Roboflow template:** [Train YOLO26 on a custom dataset (Colab)](https://colab.research.google.com/drive/16Q61DbOEVEIuMkxFmNUhOG7NF-SpYTB9?usp=sharing)
- **Live demo:** https://pcb-detector-r9jv2gjjykycvz2aas7yfj.streamlit.app/

## Project structure

```
pcb-detector/
├── app.py                  # Streamlit app
├── best.pt                 # trained weights (YOLO26s)
├── requirements.txt        # dependencies
├── packages.txt            # system packages for Streamlit Cloud
├── .streamlit/
│   └── config.toml         # theme and server settings
├── samples/                # sample boards for the random-sample button
├── results/                # evaluation plots shown in "About the model"
└── yolo26_PCB Defection.ipynb
```

> `config.toml` must live inside a folder named `.streamlit`, or Streamlit will ignore it.

## How it was built

1. Labeled PCB defect images were exported from Roboflow in YOLO26 format.
2. A pretrained `yolo26s` model was fine-tuned for up to 80 epochs (batch 16, image size 640, early stopping with patience 20).
3. The best weights were evaluated on the test split, and the evaluation plots were exported.
4. `best.pt` is loaded by the Streamlit app and runs on CPU.

## Run locally

```bash
git clone https://github.com/manarmohamed45129-design/pcb-detector
cd pcb-detector
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Download `best.pt` from Colab and place it next to `app.py`.
2. In `requirements.txt`, pin the exact ultralytics version you trained with, for example `ultralytics==8.4.220`.
3. Push everything to a GitHub repository, including `samples/` and `results/`.
4. Go to [share.streamlit.io](https://share.streamlit.io), pick the repository and `app.py`, then deploy.



## Tech stack

Python · Ultralytics YOLO26 · Streamlit · Pillow · pandas · Roboflow
