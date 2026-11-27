# TraceScan: PCB Defect Detection (YOLO26 + Streamlit)

AI visual inspection for printed circuit boards. Upload a board photo and TraceScan marks every defect, magnifies it, and gives a PASS / FAIL verdict in seconds.

## Why this project matters

A printed circuit board (PCB) is the backbone of almost every electronic device. A tiny fault, such as a hairline break in a copper trace or a stray bridge between two tracks, can make a whole device fail or behave unpredictably.

- **Defects are tiny.** Many are only a few pixels wide, and they are easy to miss by eye, especially after hours of repetitive inspection.
- **Manual inspection does not scale.** It is slow, tiring, and varies from one inspector to another.
- **Late detection is expensive.** A fault found after components are soldered means scrapping or reworking the whole board. Caught at the bare-board stage, it costs almost nothing.
- **Automation gives consistency.** The model applies the same standard to every board, at any hour, and records every result.

TraceScan shows how a small, fast object detector can act as a first-line quality gate on a production line.

## Features

| Feature | What it does |
|---|---|
| **Defect detection** | Finds and classifies 6 PCB defect types, each drawn in its own color. |
| **Magnifier lens** | Hover over the result to magnify it, scroll to change zoom, click to open a full-screen zoom and pan viewer. |
| **Defect close-ups** | Every defect is cropped and enlarged in its own card with name, confidence and size. Click a card to see it on the full board. |
| **Batch inspection** | Upload many boards at once. Get PASS / FAIL per board, yield, defects by type, and a CSV export. |
| **About the model** | Real test-set metrics, per-class table, confusion matrix and PR / F1 curves, so the numbers can be verified. |
| **Adjustable sensitivity** | A confidence slider trades missed defects against false alarms. |
| **Reports** | Download the annotated image or a CSV of all detections. |

## Defect classes

`missing_hole` · `mouse_bite` · `open_circuit` · `short` · `spur` · `spurious_copper`

## Model results

Evaluated on a held-out test split (69 images, 292 labeled defects).

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

## Resources

- **Dataset (Roboflow):** [pcb-defect-detection](https://app.roboflow.com/manar-nqk6x/pcb-defect-detection-9ewqw-rzuwm/1)
- **Training notebook in this repo:** [`yolo26_PCB Defection.ipynb`](<./yolo26_PCB Defection.ipynb>)
- **Based on the Roboflow template:** [Train YOLO26 on a custom dataset (Colab)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolo26-object-detection-on-custom-dataset.ipynb)
- **Live demo:** Try the app here: [https://pcb-detector-r9jv2gjjykycvz2aas7yfj.streamlit.app/]

## Project structure

```
pcb-detector/
├── app.py                  # Streamlit app
├── best.pt                 # trained weights (YOLO26s)
├── requirements.txt        # dependencies
├── packages.txt            # system packages for Streamlit Cloud
├── .streamlit/
│   └── config.toml         # theme and server settings
├── samples/                # sample board images
├── results/                # evaluation plots shown in "About the model"
└── yolo26_PCB Defection.ipynb
```

> `config.toml` must live inside a folder named `.streamlit` or Streamlit will ignore it.

## How it was built

1. Labeled PCB defect images were exported from Roboflow in YOLO26 format.
2. A pretrained `yolo26s` model was fine-tuned for up to 80 epochs (batch 16, image size 640, early stopping with patience 20).
3. The best weights were evaluated on the test split, and the plots were exported.
4. `best.pt` is loaded by the Streamlit app and runs on CPU.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Download `best.pt` from Colab and place it next to `app.py`.
2. In `requirements.txt`, pin the exact ultralytics version you trained with, for example `ultralytics==8.4.220`.
3. Push everything to a GitHub repository, including `samples/` and `results/`.
4. Go to [share.streamlit.io](https://share.streamlit.io), pick the repository and `app.py`, then deploy.


- Training on more boards and lighting conditions to improve `spur` and `spurious_copper`

## Tech stack

Python · Ultralytics YOLO26 · Streamlit · Pillow · pandas · Roboflow
