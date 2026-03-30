# Demo App (UI Prototype)

Prototype Streamlit untuk visual layout perbandingan 4 skenario:
- S1 Raw
- S2 HVI-CIDNet
- S3 RetinexFormer
- S4 LYT-Net

Status saat ini:
- Fokus pada UI layout dan flow.
- Data masih mock (dummy), belum terhubung backend artifact parser.

## Run

1. Install deps:
   pip install -r Demo/requirements.txt

2. Run app:
   streamlit run Demo/app.py

## Planned Next

- Hubungkan ke artifact hasil eksperimen pada outputs/scenarios/*
- Replace mock images with real original/enhanced/prediction/ground-truth
- Add export CSV/PNG from actual experiment tables
