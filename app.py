"""
BF-Dx · CytoDiff & CytoMal — Streamlit app.
Loads both models from a Hugging Face model repo and runs:
  CytoDiff  -> 25-class cell differential (% of each cell type)
  CytoMal   -> binary malignancy screening (CA / Non-CA)
"""
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torchvision.transforms as T
from huggingface_hub import hf_hub_download

from model_cytodiff import build_cytodiff

st.set_page_config(page_title="CytoDiff & CytoMal", layout="wide")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
_RM = np.array([148.60, 169.30, 105.97], np.float32)
_RS = np.array([41.56, 9.01, 6.67], np.float32)


# --- OpenCV-equivalent 8-bit LAB in pure numpy (no opencv dependency) ---
_WHITE = np.array([0.950456, 1.0, 1.088754], np.float32)
_M_RGB2XYZ = np.array([[0.412453, 0.357580, 0.180423],
                       [0.212671, 0.715160, 0.072169],
                       [0.019334, 0.119193, 0.950227]], np.float32)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)

def _f(t):     return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)
def _finv(t):
    t3 = t ** 3
    return np.where(t3 > 0.008856, t3, (t - 16.0 / 116.0) / 7.787)

def _rgb2lab(rgb_u8):
    rgb = rgb_u8.astype(np.float32) / 255.0
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = (lin.reshape(-1, 3) @ _M_RGB2XYZ.T) / _WHITE
    fx, fy, fz = _f(xyz[:, 0]), _f(xyz[:, 1]), _f(xyz[:, 2])
    L = 116 * fy - 16; a = 500 * (fx - fy); b = 200 * (fy - fz)
    return np.stack([L * 255.0 / 100.0, a + 128.0, b + 128.0], 1)

def _lab2rgb(lab):
    L = lab[:, 0] * 100.0 / 255.0; a = lab[:, 1] - 128.0; b = lab[:, 2] - 128.0
    fy = (L + 16.0) / 116.0; fx = fy + a / 500.0; fz = fy - b / 200.0
    xyz = np.stack([_finv(fx), _finv(fy), _finv(fz)], 1) * _WHITE
    lin = xyz @ _M_XYZ2RGB.T
    rgb = np.where(lin > 0.0031308, 1.055 * np.clip(lin, 0, None) ** (1/2.4) - 0.055, 12.92 * lin)
    return np.clip(rgb, 0, 1) * 255.0

def reinhard(img):
    arr = np.asarray(img.convert("RGB"))
    lab = _rgb2lab(arr)
    m = lab.mean(0); s = lab.std(0) + 1e-6
    lab = (lab - m) / s * _RS + _RM
    lab = np.clip(lab, 0, 255)
    rgb = _lab2rgb(lab).reshape(arr.shape).astype(np.uint8)
    return Image.fromarray(rgb)


def make_tfm(stain):
    pre = [T.Lambda(reinhard)] if stain else []
    return T.Compose(pre + [T.Resize(255), T.CenterCrop(224),
                            T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


@st.cache_resource(show_spinner="Loading models from Hugging Face…")
def load_task(repo, task):
    wpath = hf_hub_download(repo, f"{task}_best.pt")
    cpath = hf_hub_download(repo, f"{task}_classes.json")
    classes = json.load(open(cpath))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_cytodiff(len(classes), pretrained=False).to(dev).eval()
    ck = torch.load(wpath, map_location=dev)
    state = ck.get("model", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(state)
    return model, classes, dev


@torch.no_grad()
def predict(model, tfm, dev, classes, img):
    x = tfm(img.convert("RGB")).unsqueeze(0).to(dev)
    p = torch.softmax(model(x), 1).cpu().numpy()[0]
    return {classes[i]: float(p[i]) for i in range(len(classes))}


# ---------------- UI ----------------
st.title("🩸 CytoDiff & CytoMal")
st.caption("An AI System for Automated Body-Fluid Cell Differential and Malignancy Screening "
           "(Wright–Giemsa, Olympus CX33). Research prototype — not a diagnostic device.")

default_repo = st.secrets.get("HF_REPO", os.environ.get("HF_REPO", ""))
with st.sidebar:
    st.header("Settings")
    repo = st.text_input("Hugging Face model repo", default_repo,
                         placeholder="username/bf-dx-models")
    st.caption("Preprocessing must match how each model was trained:")
    stain_diff = st.checkbox("CytoDiff stain normalization", False)
    stain_mal  = st.checkbox("CytoMal stain normalization", True)
    conf = st.slider("Min confidence for differential", 0.0, 1.0, 0.5, 0.05)

if not repo:
    st.info("Enter your Hugging Face model repo in the sidebar (e.g. `username/bf-dx-models`) "
            "or set it as the `HF_REPO` secret in Streamlit. See README_DEPLOY.md.")
    st.stop()

try:
    diff_model, diff_cls, dev = load_task(repo, "cytodiff")
    mal_model, mal_cls, _ = load_task(repo, "cytomal")
except Exception as e:
    st.error(f"Could not load models from `{repo}`. Details: {e}")
    st.stop()

tfm_diff = make_tfm(stain_diff)
tfm_mal = make_tfm(stain_mal)
# which malignancy class name means "cancer"
ca_name = next((c for c in mal_cls if c.strip().upper() in ("CA", "MALIGNANT", "CANCER")), mal_cls[0])
st.success(f"Loaded · device **{dev}** · CytoDiff {len(diff_cls)} classes · "
           f"CytoMal classes {mal_cls} (malignant = {ca_name})")

files = st.file_uploader("Upload single-cell images (10–20 per patient)",
                         type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
                         accept_multiple_files=True)

if not files:
    st.info("Upload cropped single-cell images to get a differential and a malignancy screen.")
    st.stop()

rows = []
cols = st.columns(5)
for i, f in enumerate(files):
    img = Image.open(f)
    dp = predict(diff_model, tfm_diff, dev, diff_cls, img)
    mp = predict(mal_model, tfm_mal, dev, mal_cls, img)
    top = max(dp, key=dp.get)
    rows.append({"file": f.name, "cell_type": top, "cell_conf": dp[top],
                 "P(malignant)": mp[ca_name]})
    with cols[i % 5]:
        st.image(img, caption=f"{top} ({dp[top]:.2f}) · CA {mp[ca_name]:.2f}", width=120)

df = pd.DataFrame(rows)

c1, c2 = st.columns(2)
with c1:
    st.subheader("CytoDiff — cell differential")
    conf_df = df[df.cell_conf >= conf]
    if len(conf_df):
        diff = conf_df["cell_type"].value_counts()
        pct = (diff / diff.sum() * 100).round(1)
        st.dataframe(pd.DataFrame({"count": diff, "percent": pct}), use_container_width=True)
        st.bar_chart(pct)
        st.caption(f"{len(conf_df)}/{len(df)} cells ≥ {conf:.2f} confidence")
    else:
        st.info("No cells passed the confidence threshold.")
with c2:
    st.subheader("CytoMal — malignancy screen")
    pca = df["P(malignant)"]
    n_ca = int((pca >= 0.5).sum())
    st.metric("Cells flagged malignant (P≥0.5)", f"{n_ca} / {len(df)}")
    st.metric("Max P(malignant)", f"{pca.max():.2f}")
    st.metric("Mean P(malignant)", f"{pca.mean():.2f}")
    flag = "⚠️ Suspicious for malignancy" if (pca.max() >= 0.5) else "No malignant cells flagged"
    st.write(f"**Patient-level: {flag}**")

st.subheader("Per-cell results")
st.dataframe(df, use_container_width=True)
st.download_button("Download results CSV", df.to_csv(index=False).encode(),
                   "bf-dx_results.csv", "text/csv")
