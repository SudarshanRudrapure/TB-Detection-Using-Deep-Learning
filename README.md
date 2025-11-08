# 🫁 TBX-Hybrid: Tuberculosis Detection

## Setup
```bash
conda create -n tbx-hybrid python=3.9 -y
conda activate tbx-hybrid
pip install -r requirements.txt
```

## Data
- Place TBX11K dataset inside `data/raw/`
- Preprocessed CSVs go into `data/processed/`

## Training
```bash
python src/train.py
```

## Inference
```bash
python src/inference.py --image data/raw/sample_tb.jpg
```

## Run Streamlit App
```bash
streamlit run deployment/app_streamlit.py
```
