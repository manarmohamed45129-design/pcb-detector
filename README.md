# PCB Defect Detection (YOLO26 + Streamlit)

## Files
- app.py: Streamlit app
- requirements.txt: dependencies (pin the ultralytics version, see below)
- best.pt: trained weights (add this file yourself)

## Steps
1. Download best.pt from Colab and place it next to app.py.
2. Replace the ultralytics line in requirements.txt with the exact version
   you trained with, for example: ultralytics==8.4.220
3. Upload everything to a GitHub repository.
4. Go to share.streamlit.io, pick the repository and app.py, then Deploy.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py
