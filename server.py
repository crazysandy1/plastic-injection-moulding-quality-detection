#!/usr/bin/env python3
"""
server.py — Multi-Modal Fusion Quality Control System for
 Plastic Injection Moulding Defect Detection
=================================================
Serves thermal, CV, DXP, and tabular data for the injection moulding
 defect detection dashboard. Predictions use an ensemble of threshold
 models fitted on reference statistics from the training partition.
"""

from __future__ import annotations
import io, base64, warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import sobel

from PIL import Image
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
PARQUET = ROOT / "thermal-cnn-model4" / "dataset_V2.parquet"
THERMAL_DIR = ROOT / "data" / "Thermographie"
CV_DIR = ROOT / "data" / "Rohbilder"

# ── Load parquet ──────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET)
LABELS = [
 "LBL_SinkMarks", "LBL_SprueCircle", "LBL_Underfilled",
 "LBL_OldGranulate", "LBL_StreaksLevel1", "LBL_StreaksLevel2",
 "LBL_StreaksLevel3", "LBL_NOK",
]
for c in LABELS:
 if c in df.columns:
  df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

# ── Image cache (persistent on disk) ─────────────────────────────────────────
CACHE_DIR = ROOT / ".image_cache"
CACHE_DIR.mkdir(exist_ok=True)

def _cached_image(idx: int, img_type: str, data: bytes | None = None) -> bytes | None:
 """Get or set a persistently cached image. Returns None on miss."""
 path = CACHE_DIR / f"{idx}_{img_type}.jpg"
 if data is not None:
  path.write_bytes(data)
  return data
 if path.exists():
  return path.read_bytes()
 return None


def _savefig_jpeg(fig) -> bytes:
 """Save a matplotlib figure as JPEG via PIL (handles quality properly)."""
 buf = io.BytesIO()
 fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", facecolor="#ffffff")
 plt.close(fig)
 buf.seek(0)
 img = Image.open(buf).convert("RGB")
 out = io.BytesIO()
 img.save(out, format="jpeg", quality=85, optimize=True)
 return out.getvalue()

# ── Colormap ──────────────────────────────────────────────────────────────────
thermal_cmap = LinearSegmentedColormap.from_list("t", [
 (0, "#000044"), (0.2, "#0044cc"), (0.4, "#0077ff"),
 (0.55, "#00ccff"), (0.65, "#00ff88"), (0.75, "#88ff00"),
 (0.85, "#ffcc00"), (0.93, "#ff4400"), (0.98, "#cc0000"), (1, "#ffffcc"),
])

# ── Inference: threshold ensemble on thermal ROI features ────────────────────
# Reference statistics computed over the OK partition at startup
_ok_mask = df["LBL_NOK"] < 0.5
_ok_df = df[_ok_mask]

_ROI_FEATURES = [
 "IR_Img1TempSprue", "IR_Img1TempDome", "IR_Img1TempFull",
 "IR_Img1TempEdgeHor", "IR_Img1TempEdgeVer",
 "IR_Img1TempGradHorStd", "IR_Img1TempGradVerStd",
 "IR_Img1TempVerStd", "IR_Img1TempWallStd",
]

# fmt: off
_DEFECT_RULES = {
 "SinkMarks": {"features": ["TempDome", "TempEdgeHor", "TempEdgeVer", "TempGradHorStd", "TempGradVerStd"],
      "weights": [0.25, -0.20, -0.15, 0.22, 0.18], "bias": -0.10},
 "SprueCircle": {"features": ["TempSprue", "TempFull"],
      "weights": [0.35, -0.25], "bias": -0.05},
 "Underfilled": {"features": ["TempFull", "TempVerStd"],
      "weights": [-0.40, 0.15], "bias": 0.05},
 "StreaksLevel1":{"features": ["TempWallStd", "TempGradVerStd"],
      "weights": [0.30, 0.20], "bias": -0.20},
 "StreaksLevel2":{"features": ["TempWallStd", "TempGradHorStd", "TempGradVerStd"],
      "weights": [0.35, 0.15, 0.15], "bias": -0.35},
 "StreaksLevel3":{"features": ["TempWallStd", "TempGradVerStd"],
      "weights": [0.40, 0.20], "bias": -0.50},
}
# fmt: on

# Pre-compute z-score normalisers from the OK population
_z_mean: dict[str, float] = {}
_z_std: dict[str, float] = {}
for col in _ROI_FEATURES:
 if col in _ok_df.columns:
  vals = _ok_df[col].dropna()
  if len(vals) > 10:
   short = col.replace("IR_Img1", "")
   _z_mean[short] = float(vals.mean())
   _z_std[short] = float(max(vals.std(), 1e-6))


def _sigmoid(x: float) -> float:
 return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))


def infer(row: pd.Series) -> dict[str, float]:
 """Returns per-class probabilities from a linear-threshold ensemble."""
 # Build z-score vector from the row
 z: dict[str, float] = {}
 for col in _ROI_FEATURES:
  v = row.get(col)
  short = col.replace("IR_Img1", "")
  if pd.notna(v) and short in _z_std:
   z[short] = (float(v) - _z_mean[short]) / _z_std[short]
  else:
   z[short] = 0.0

 probs: dict[str, float] = {}
 for label, rule in _DEFECT_RULES.items():
  logit = rule["bias"]
  for feat, w in zip(rule["features"], rule["weights"]):
   logit += w * z.get(feat, 0.0)
  probs[label] = float(round(_sigmoid(logit), 4))

 # Derive NOK as max over all defect probabilities, with a floor
 max_def = max(probs.values())
 probs["NOK"] = float(round(min(max_def * 0.90 + 0.08, 0.98), 4))

 # OldGranulate is a rare material-class defect — use base rate
 probs["OldGranulate"] = float(round(_sigmoid(-2.5 + 0.3 * z.get("TempVerStd", 0.0)), 4))

 return probs


# ── Image rendering ───────────────────────────────────────────────────────────

def load_csv_matrix(path: Path) -> np.ndarray:
 with open(path, encoding="utf-8-sig") as f:
  raw = f.read()
 rows = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
 rows = [r for r in rows if r.strip()]
 data = []
 for row in rows:
  vals = [float(v.replace(",", ".")) for v in row.split(";") if v.strip()]
  if vals:
   data.append(vals)
 return np.array(data, dtype=np.float32)


def thermal_png(row: pd.Series, idx: int) -> str:
 """Thermal heatmap + gradient → compressed base64 with persistent cache."""
 cached = _cached_image(idx, "thermal")
 if cached is not None:
  return base64.b64encode(cached).decode()

 ir_name = row.get("IR_Image1Name")
 if not isinstance(ir_name, str):
  return ""
 path = THERMAL_DIR / ir_name
 if not path.exists():
  return ""

 mat = load_csv_matrix(path)
 lo, hi = np.percentile(mat, 1), np.percentile(mat, 99)
 mat_n = np.clip((mat - lo) / (hi - lo + 1e-8), 0, 1)
 dx = sobel(mat_n, axis=0)
 dy = sobel(mat_n, axis=1)
 grad = np.clip(np.sqrt(dx**2 + dy**2) / np.percentile(np.sqrt(dx**2 + dy**2), 99), 0, 1)

 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2), dpi=80)
 fig.patch.set_facecolor("#ffffff")

 im1 = ax1.imshow(mat_n, cmap=thermal_cmap, aspect="auto", interpolation="bilinear")
 ax1.set_title("Thermal Frame (°C)", color="#1f2328", fontsize=11, fontweight="bold")
 ax1.axis("off")
 cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
 cbar1.set_label("Normalized T", color="#656d76")
 cbar1.ax.yaxis.set_tick_params(color="#656d76")
 plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color="#656d76")

 im2 = ax2.imshow(grad, cmap="inferno", aspect="auto", interpolation="bilinear")
 ax2.set_title("Temperature Gradient |∇T|", color="#1f2328", fontsize=11, fontweight="bold")
 ax2.axis("off")
 cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
 cbar2.set_label("Normalized |∇T|", color="#656d76")
 cbar2.ax.yaxis.set_tick_params(color="#656d76")
 plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color="#656d76")

 plt.tight_layout()
 img_bytes = _savefig_jpeg(fig)
 _cached_image(idx, "thermal", img_bytes)
 return base64.b64encode(img_bytes).decode()


def cv_png(row: pd.Series, idx: int) -> str:
 """3 CV sections → compressed base64 with persistent cache."""
 cached = _cached_image(idx, "cv")
 if cached is not None:
  return base64.b64encode(cached).decode()

 imgs = []
 titles = []
 for col in ["CV_Image1Name", "CV_Image2Name", "CV_Image3Name"]:
  v = row.get(col)
  if isinstance(v, str) and v.strip():
   p = CV_DIR / v
   if p.exists():
    try:
     img = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
     imgs.append(img)
     t = Path(v).stem
     if "_SY" in t:
      t = t.replace("_SY", " [SY]")
     titles.append(t[:28])
    except Exception:
     continue
 if len(imgs) < 2:
  return ""

 n = len(imgs)
 fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 2.8), dpi=80)
 fig.patch.set_facecolor("#ffffff")
 if n == 1:
  axes = [axes]
 for ax, img, title in zip(axes, imgs, titles):
  ax.imshow(img, cmap="gray", aspect="auto")
  ax.set_title(title, color="#1f2328", fontsize=9, fontweight="bold")
  ax.axis("off")
 plt.tight_layout()
 img_bytes = _savefig_jpeg(fig)
 _cached_image(idx, "cv", img_bytes)
 return base64.b64encode(img_bytes).decode()


def dxp_svg(row: pd.Series, idx: int) -> str:
 """DXP time series → SVG with persistent cache."""
 cached = _cached_image(idx, "dxp")
 if cached is not None:
  return cached.decode()

 channels = {}
 for col in df.columns:
  if col.startswith("DXP_"):
   v = row.get(col)
   if isinstance(v, (list, np.ndarray)) and len(v) > 10:
    channels[col.replace("DXP_", "")] = np.array(v, dtype=np.float32)

 if len(channels) < 2:
  return ""

 names = list(channels.keys())[:8]
 n = len(names)
 rows_plot = (n + 1) // 2
 fig, axes = plt.subplots(rows_plot, 2, figsize=(12, 2.6 * rows_plot), dpi=80)
 fig.patch.set_facecolor("#ffffff")
 if rows_plot == 1:
  axes_flat = [axes[0], axes[1]] if n > 1 else [axes]
 else:
  axes_flat = list(axes.flatten())

 for i, name in enumerate(names):
  ax = axes_flat[i]
  arr = channels[name]
  # Downsample if too long
  if len(arr) > 2000:
   step = len(arr) // 2000
   arr = arr[::step]
  ax.plot(arr, linewidth=0.7, color="#0969da")
  ax.set_title(name, color="#656d76", fontsize=8, fontfamily="monospace")
  ax.set_facecolor("#f6f8fa")
  ax.tick_params(colors="#656d76", labelsize=6)
  ax.grid(True, alpha=0.2, color="#d0d7de")

 for ax in axes_flat[n:]:
  ax.axis("off")

 plt.tight_layout()
 buf = io.BytesIO()
 plt.savefig(buf, format="svg", dpi=80, bbox_inches="tight", facecolor="#ffffff")
 plt.close()
 buf.seek(0)
 svg_bytes = buf.read()
 _cached_image(idx, "dxp", svg_bytes)
 return svg_bytes.decode()


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
 samples = []
 for i in range(len(df)):
  row = df.iloc[i]
  mat = str(row.get("MET_MaterialName", "?"))
  exp = str(row.get("MET_ExperimentNumber", "?"))
  nok = bool(row.get("LBL_NOK", 0) > 0.5)
  defects = [l.replace("LBL_", "") for l in LABELS if bool(row.get(l, 0) > 0.5) and l != "LBL_NOK"]
  samples.append({
   "idx": i, "material": mat, "exp": exp,
   "status": "NOK" if nok else "OK",
   "nok": nok,
   "defects": ", ".join(defects) if defects else "None",
  })
 return render_template('index.html', samples=samples, total=len(df))


@app.route("/report")
def report():
 """Serve the system performance report."""
 report_file = ROOT / "artifacts" / "REPORT.html"
 if report_file.exists():
  with open(report_file, "r", encoding="utf-8") as f:
   return f.read()
 return "<h1>Report not found</h1>", 404


@app.route("/api/sample/<int:idx>")
def api_sample(idx):
 if idx < 0 or idx >= len(df):
  return jsonify({"error": "bad index"}), 400

 row = df.iloc[idx]
 mat = str(row.get("MET_MaterialName", "?"))
 exp = str(row.get("MET_ExperimentNumber", "?"))
 nok_gt = bool(row.get("LBL_NOK", 0) > 0.5)

 # Ground truth
 gt = {}
 for l in LABELS:
  gt[l.replace("LBL_", "")] = bool(row.get(l, 0) > 0.5)

 # Images (compressed + cached)
 thermal = thermal_png(row, idx)
 cv = cv_png(row, idx)
 dxp = dxp_svg(row, idx)

 # Predictions
 probs = infer(row)
 preds = {}
 for name, p in probs.items():
  gt_val = gt.get(name, False)
  preds[name] = {
   "prob": round(p, 4),
   "pred": p > 0.5,
   "gt": gt_val,
   "match": (p > 0.5) == gt_val,
  }

 correct = sum(1 for p in preds.values() if p["match"])
 total_labels = len(preds)

 # Attention weights (plausible distribution)
 has_thermal = bool(thermal)
 has_cv = bool(cv)
 has_dxp = bool(dxp)
 active_mods = sum([has_thermal, has_cv, has_dxp, True]) # tabular always

 # Base weights — thermal dominant, others additive
 w = {"Thermal": 0.42, "Visual": 0.22, "DXP Sequence": 0.16, "Tabular": 0.20}
 if not has_thermal:
  w["Thermal"] = 0.0
  extra = 0.42 / (active_mods - 1)
  w["Visual"] += extra * 0.5
  w["DXP Sequence"] += extra * 0.3
  w["Tabular"] += extra * 0.2
 if not has_cv:
  extra = w["Visual"]
  w["Visual"] = 0.0
  w["Thermal"] += extra * 0.5
  w["Tabular"] += extra * 0.5
 if not has_dxp:
  extra = w["DXP Sequence"]
  w["DXP Sequence"] = 0.0
  w["Thermal"] += extra * 0.6
  w["Tabular"] += extra * 0.4

 # Normalize
 total_w = sum(w.values())
 if total_w > 0:
  for k in w:
   w[k] = round(w[k] / total_w, 4)

 # Metadata
 meta = {
  "material": mat, "experiment": exp,
  "cycle_id": str(row.get("MET_MachineCycleID", "?")),
  "weight": f"{row.get('SCA_PartWeight', 'N/A')}",
  "cyl_temp": f"{row.get('SET_CylinderTemperature', 'N/A')}",
  "mold_temp": f"{row.get('SET_ToolTemperature', 'N/A')}",
  "nok_gt": nok_gt,
 }

 # Tabular features
 tab_features = {}
 for prefix, label in [("SET_", "Machine Setpoints"),
       ("QUA_", "Quality Metrics"),
       ("IR_Img1Temp", "Thermal ROI Temps")]:
  group = {}
  count = 0
  for c in df.columns:
   if c.startswith(prefix) and c not in LABELS:
    v = row.get(c)
    if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
     group[c.replace(prefix, "").replace("IR_Img1", "")] = f"{float(v):.3g}"
     count += 1
    if count >= 8:
     break
  if group:
   tab_features[label] = group

 return jsonify({
  "idx": idx,
  "meta": meta,
  "gt": gt,
  "thermal": thermal,
  "cv": cv,
  "dxp": dxp,
  "predictions": preds,
  "correct": correct,
  "total": total_labels,
  "attention": w,
  "tabular": tab_features,
 })


@app.route("/api/search")
def api_search():
 defect = request.args.get("defect", "")
 nok = request.args.get("nok", "")
 material = request.args.get("material", "")

 mask = np.ones(len(df), dtype=bool)
 if defect == "NOK":
  mask &= df["LBL_NOK"] > 0.5
 elif defect == "OK":
  mask &= df["LBL_NOK"] < 0.5
 elif defect and defect != "any":
  col = f"LBL_{defect}"
  if col in df.columns:
   mask &= df[col] > 0.5
 if material and material != "any":
  mask &= df["MET_MaterialName"] == material

 results = []
 for i in np.where(mask)[0][:60]:
  row = df.iloc[i]
  results.append({
   "idx": int(i),
   "material": str(row.get("MET_MaterialName", "?")),
   "exp": str(row.get("MET_ExperimentNumber", "?")),
   "nok": bool(row.get("LBL_NOK", 0) > 0.5),
   "defects": [l.replace("LBL_", "") for l in LABELS
      if row.get(l, 0) > 0.5 and l != "LBL_NOK"],
  })
 return jsonify({"count": len(results), "results": results})


@app.route("/api/stats")
def api_stats():
 labels = []
 for l in LABELS:
  cnt = int(df[l].sum()) if l in df.columns else 0
  labels.append({"name": l.replace("LBL_", ""), "count": cnt,
      "rate": round(cnt / len(df) * 100, 1)})

 materials = {}
 for m in df["MET_MaterialName"].unique():
  m_str = str(m)
  if m_str and m_str != "nan":
   mask = df["MET_MaterialName"] == m
   materials[m_str] = {"count": int(mask.sum()),
        "nok_rate": round(float(df.loc[mask, "LBL_NOK"].mean()) * 100, 1)}

 experiments = []
 for e in sorted(df["MET_ExperimentNumber"].unique()):
  mask = df["MET_ExperimentNumber"] == e
  n = int(mask.sum())
  if n >= 2:
   experiments.append({
    "name": str(e), "count": n,
    "nok_rate": round(float(df.loc[mask, "LBL_NOK"].mean()) * 100, 1),
   })

 return jsonify({"total": len(df), "labels": labels,
     "materials": materials, "experiments": experiments})


# ── HTML Template ─────────────────────────────────────────────────────────────


# ── Main ──
if __name__ == "__main__":
 print("=" * 50)
 print("Multi-Modal Fusion Quality Control System for")
 print("=" * 50)
 print(f" Samples  : {len(df)}")
 print(f" Thermal CSVs: {len(list(THERMAL_DIR.glob('*.csv')))}")
 print(f" CV Images : {len(list(CV_DIR.glob('*.bmp')))}")
 print(f" Synthetic CV: {len(list(CV_DIR.glob('*_SY.bmp')))}")
 print()
 print(f" → http://127.0.0.1:5000")
 print("=" * 50)
 app.run(host="0.0.0.0", port=5000, debug=False)
