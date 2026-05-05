# autoresearch Integration Plan — Workstream 2
**Reworld Haverhill · Bright AI · 2026-05-05**

Uses the autoresearch autonomous experimentation framework (`/Users/kumar.mankala/code/repos/autoresearch`)
to accelerate two of the hardest ML problems in Workstream 2:
- **Waste classification** (Gap 1, 11 categories, depends on labeling)
- **BTU/HHV time series prediction** (Gap 2, depends on historian data)
Plus one immediate, zero-dependency win: **MuonAdamW optimizer extraction**.

All three integration points run on the same JupyterHub GPU already used by the dashboard.

---

## Week Timeline

| Week | What ships | Dependency |
|---|---|---|
| **Week 1** | MuonAdamW extracted to `scripts/muon_adamw.py`; `prepare_classify.py` + `train_classify.py` scaffolded; `program_waste_classify.md` written | None — starts now |
| **Week 2** | First labeling sprint: 40–60 frames × 4 priority categories (Plastics, Paper/Cardboard, Food, Yard Waste); first overnight autonomous classification run | Reworld floor staff or Bright AI annotates ~200 images |
| **Week 3** | Expand labeling to all 11 categories (target 200/class); `prepare_btu.py` + `train_btu.py` scaffolded; first synthetic BTU run validates the pipeline | Labeling ongoing; BTU script runs in synthetic mode |
| **Week 4** | Historian data arrives → wire to `prepare_btu.py` → first real BTU overnight run; fold best model from Week 2 into dashboard inference | Reworld historian export |

---

## Integration 1: MuonAdamW Optimizer (Week 1, ~1 hour)

Extract the optimizer from autoresearch and add it to this repo so every future training script can use it.
Shown to converge faster than AdamW on small datasets — exactly the regime for waste classification
with hundreds of labeled images.

**File to create:** `scripts/muon_adamw.py`

Copy `polar_express_coeffs`, `adamw_step_fused`, `muon_step_fused`, and `MuonAdamW` from
`/Users/kumar.mankala/code/repos/autoresearch/train.py` verbatim.

Usage in any training script:
```python
from scripts.muon_adamw import MuonAdamW

matrix_params = [p for p in model.parameters() if p.requires_grad and p.ndim == 2]
other_params  = [p for p in model.parameters() if p.requires_grad and p.ndim != 2]

optimizer = MuonAdamW([
    dict(kind='adamw', params=other_params, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=WD),
    dict(kind='muon',  params=matrix_params, lr=LR, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=WD),
])
for g in optimizer.param_groups:
    g['initial_lr'] = g['lr']
```

---

## Integration 2: Waste Classification Autonomous Loop

Adapts the autoresearch pattern to search CV backbone architectures and training hyperparameters
for the 11-category waste classification task. Agent modifies `train_classify.py`; human writes
domain knowledge into `program_waste_classify.md`.

**New files:**

```
reworld-fuel-quality/
├── features/
│   ├── autoresearch-integration-plan.md  ← this file
│   ├── program_waste_classify.md         ← agent instructions (domain-encoded)
│   └── program_btu.md                    ← agent instructions for BTU model
├── scripts/
│   ├── muon_adamw.py                     ← extracted from autoresearch
│   ├── prepare_classify.py               ← FIXED: data loading, eval (do not modify)
│   └── train_classify.py                 ← AGENT MODIFIES THIS
└── data/
    └── waste_labels/
        └── labels.csv                    ← (rgb_path, category) — created during labeling
```

---

### `scripts/prepare_classify.py` (fixed — do not modify)

```python
"""
Fixed data loading and evaluation for waste classification.
Agent does not modify this file.
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pathlib import Path
from PIL import Image
import pandas as pd

try:
    from sklearn.metrics import f1_score
except ImportError:
    raise ImportError("pip install scikit-learn")

TIME_BUDGET = 300   # 5-minute wall clock training budget
N_CLASSES   = 11

LABEL_MAP = {
    "yard_waste":    0,
    "food":          1,
    "wood":          2,
    "paper":         3,
    "cardboard":     4,
    "plastics":      5,
    "textiles":      6,
    "rubber":        7,
    "leather":       8,
    "misc_organics": 9,
    "dirt_ashes":    10,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

LABELS_CSV = Path("data/waste_labels/labels.csv")  # columns: rgb_path, category
VAL_SPLIT  = 0.2
IMG_SIZE   = 224

# Fixed val transform — do not change (ensures comparable evaluation)
_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Default train transform — agent may override in train_classify.py
DEFAULT_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),      # horizontal flip is valid — gravity still acts
    # NOTE: do NOT use RandomVerticalFlip — waste piles are gravity-constrained
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class WasteDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def make_dataloaders(batch_size, train_transform=None):
    """Returns (train_loader, val_loader). Deterministic val split (seed=42)."""
    if train_transform is None:
        train_transform = DEFAULT_TRAIN_TRANSFORM
    df = pd.read_csv(LABELS_CSV)
    assert "rgb_path" in df.columns and "category" in df.columns, \
        "labels.csv must have columns: rgb_path, category"
    records = [(row.rgb_path, LABEL_MAP[row.category]) for _, row in df.iterrows()]
    n_val   = max(1, int(len(records) * VAL_SPLIT))
    n_train = len(records) - n_val
    torch.manual_seed(42)
    train_r, val_r = random_split(records, [n_train, n_val])
    train_ds = WasteDataset(list(train_r), train_transform)
    val_ds   = WasteDataset(list(val_r),   _VAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader


@torch.no_grad()
def evaluate_f1(model, device, batch_size=64):
    """Fixed evaluation: macro-F1 on held-out val set. Returns (macro_f1, per_class_f1_list)."""
    _, val_loader = make_dataloaders(batch_size)
    model.eval()
    all_preds, all_labels = [], []
    for x, y in val_loader:
        preds = model(x.to(device)).argmax(dim=-1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())
    macro_f1  = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None,     zero_division=0)
    return float(macro_f1), per_class.tolist()
```

---

### `scripts/train_classify.py` (agent modifies this)

```python
"""
Waste classification — agent-modifiable training script.
Fixed time budget: 5 min wall clock.
Metric: macro-F1 on held-out labeled RGB frames from achute/chuteb cameras.
Usage: python scripts/train_classify.py

WHAT TO CHANGE: backbone, lr, augmentation, optimizer settings, batch size,
                label smoothing, mixup, freeze strategy.
DO NOT CHANGE: evaluate_f1 call, TIME_BUDGET, the summary print block.
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

from prepare_classify import TIME_BUDGET, N_CLASSES, make_dataloaders, evaluate_f1
from muon_adamw import MuonAdamW

# ── Hyperparameters (agent edits below this line) ─────────────────────────────

BACKBONE         = "efficientnet_b0"  # try: mobilenet_v3_small, resnet18, convnext_tiny
PRETRAINED       = True
FREEZE_BACKBONE  = False       # True = train head only; good first-run baseline
LR               = 2e-4
WEIGHT_DECAY     = 1e-4
BATCH_SIZE       = 32
LABEL_SMOOTHING  = 0.1
WARMDOWN_RATIO   = 0.3         # fraction of time budget for LR cooldown
USE_MIXUP        = False
MIXUP_ALPHA      = 0.4

# ──────────────────────────────────────────────────────────────────────────────

t_start = time.time()
device  = torch.device("cuda")
torch.manual_seed(42)
torch.set_float32_matmul_precision("high")

model = timm.create_model(BACKBONE, pretrained=PRETRAINED, num_classes=N_CLASSES)
if FREEZE_BACKBONE:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True
model = model.to(device)
model = torch.compile(model, dynamic=False)

matrix_params = [p for p in model.parameters() if p.requires_grad and p.ndim == 2]
other_params  = [p for p in model.parameters() if p.requires_grad and p.ndim != 2]
optimizer = MuonAdamW([
    dict(kind="adamw", params=other_params,  lr=LR, betas=(0.9, 0.999),
         eps=1e-8, weight_decay=WEIGHT_DECAY),
    dict(kind="muon",  params=matrix_params, lr=LR, momentum=0.95,
         ns_steps=5, beta2=0.95, weight_decay=WEIGHT_DECAY),
])
for g in optimizer.param_groups:
    g["initial_lr"] = g["lr"]

train_loader, _ = make_dataloaders(BATCH_SIZE)

total_train_time = 0.0
t_train_start    = None
step             = 0


def mixup_batch(x, y, alpha):
    lam = torch._sample_dirichlet(torch.tensor([alpha, alpha]))[0].item()
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


while True:
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        t0 = time.time()
        if t_train_start is None:
            t_train_start = t0

        if USE_MIXUP:
            x, y_a, y_b, lam = mixup_batch(x, y, MIXUP_ALPHA)
            logits = model(x)
            loss = lam * F.cross_entropy(logits, y_a, label_smoothing=LABEL_SMOOTHING) + \
                   (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=LABEL_SMOOTHING)
        else:
            logits = model(x)
            loss   = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        progress = total_train_time / TIME_BUDGET
        lrm = max((1.0 - progress) / WARMDOWN_RATIO, 0.01) if progress > 1.0 - WARMDOWN_RATIO else 1.0
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * lrm

        optimizer.step()

        t1 = time.time()
        if step > 5:
            total_train_time += t1 - t0
        step += 1

        if step == 0:
            gc.collect(); gc.freeze(); gc.disable()

        if step % 20 == 0:
            pct = 100 * total_train_time / TIME_BUDGET
            print(f"\rstep {step:05d} ({pct:.1f}%) loss={loss.item():.4f}", end="", flush=True)

        if total_train_time >= TIME_BUDGET:
            break
    if total_train_time >= TIME_BUDGET:
        break

print()

model.eval()
macro_f1, per_class_f1 = evaluate_f1(model, device)

t_end         = time.time()
peak_vram_mb  = torch.cuda.max_memory_allocated() / 1024 / 1024
num_params    = sum(p.numel() for p in model.parameters()) / 1e6

print("---")
print(f"macro_f1:         {macro_f1:.6f}")
print(f"training_seconds: {total_train_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params:.1f}")
print(f"backbone:         {BACKBONE}")
print(f"freeze_backbone:  {FREEZE_BACKBONE}")

# Per-class breakdown
labels = ["yard_waste","food","wood","paper","cardboard",
          "plastics","textiles","rubber","leather","misc_organics","dirt_ashes"]
print("\nPer-class F1:")
for lbl, f1 in zip(labels, per_class_f1):
    bar = "█" * int(f1 * 20)
    print(f"  {lbl:<16} {f1:.3f}  {bar}")
```

---

### `features/program_waste_classify.md` (human writes, agent reads)

```markdown
# Waste Classification — autoresearch program

## Task
You are an autonomous ML researcher. You modify `scripts/train_classify.py`,
run 5-minute experiments, and iterate toward the highest macro-F1 on the
11-category waste classification task using IR/RGB chute camera images.

## Setup
1. Agree a run tag (e.g. `may6`). Branch: `autoresearch/waste-classif/<tag>`.
2. Read: `scripts/prepare_classify.py`, `scripts/train_classify.py`, this file.
3. Verify `data/waste_labels/labels.csv` exists with at least 100 rows.
4. Init `results_classify.tsv` with header: `commit\tmacro_f1\tmemory_gb\tstatus\tdescription`
5. Confirm, then begin experimentation.

## Metric
**macro_f1** (higher = better). Macro averages equally across all 11 classes —
this matters because rare categories (rubber, leather) are just as important
as common ones (paper, plastics) for BTU estimation accuracy.

## Domain knowledge for this task
- **IR characteristics**: wet waste (food, yard waste) appears cold in IR (dark blue/purple
  in the false-color overlay). Dry plastics appear warm (orange/red). Use this if you
  add IR channel fusion.
- **Gravity constraint**: waste piles always sit on a surface. RandomVerticalFlip is
  physically wrong — do NOT add it. RandomHorizontalFlip is fine.
- **Lighting variation**: chute cameras have fixed IR illumination but RGB varies with
  ambient light. ColorJitter on brightness/contrast is high-value augmentation.
- **Class confusion hotspots**: rubber vs. leather (both dark, non-reflective); 
  paper vs. cardboard (similar texture, color different); food vs. misc_organics
  (both dark, wet). Focus augmentation and architecture effort here.
- **High-BTU indicator**: plastics fraction is the #1 BTU driver. Getting plastics
  precision right matters most for furnace safety.
- **Image source**: RGB frames are 1280×720 JPEG from `camera_data/reworld-haverhill-achute/`
  and `reworld-haverhill-chuteb/`. Center crop 224px is the default — consider
  whether the waste region is centered or off-center.

## What you CAN change in train_classify.py
- BACKBONE (timm model name), PRETRAINED, FREEZE_BACKBONE
- LR, WEIGHT_DECAY, BATCH_SIZE, LABEL_SMOOTHING
- WARMDOWN_RATIO, USE_MIXUP, MIXUP_ALPHA
- Training loop: gradient clipping, scheduler shape, EMA weights
- Custom augmentation transform passed to make_dataloaders()
- Model head: add dropout, extra linear layer before classifier
- Mixed precision: wrap forward in autocast

## What you CANNOT change
- `prepare_classify.py` (the evaluation harness)
- The `evaluate_f1()` call at the end
- TIME_BUDGET, VAL_SPLIT, the summary print block format

## Simplicity criterion
A 0.002 macro_f1 gain that adds 30 lines of complex code: probably not worth it.
A 0.002 gain from removing code or simplifying: always keep.

## Output format
```
macro_f1:         0.712340
training_seconds: 300.1
total_seconds:    318.4
peak_vram_mb:     8420.1
num_steps:        284
num_params_M:     5.3
backbone:         efficientnet_b0
```

## Loop
Same as autoresearch/program.md: NEVER STOP until manually interrupted.
Log results to `results_classify.tsv`. Keep if macro_f1 improved, discard if not.
```

---

## Integration 3: BTU/HHV Time Series Prediction

Adapts the autoresearch GPT transformer for multivariate time series regression.
The key architectural change: replace the word embedding lookup (`wte`) with a
linear projection of continuous sensor features. The rest of the transformer
(RoPE, RMSNorm, sliding window attention, MLP, MuonAdamW) stays identical.

**New files:**
```
scripts/
├── prepare_btu.py       ← FIXED: CSV loading, feature normalization, evaluation
└── train_btu.py         ← AGENT MODIFIES: transformer depth/width, window pattern, loss
features/
└── program_btu.md       ← agent instructions with combustion domain knowledge
```

---

### `scripts/prepare_btu.py` (fixed — do not modify)

```python
"""
Fixed data loading and evaluation for BTU/HHV time series prediction.
Agent does not modify this file.

Input signals (from chuteb_signals/signals.csv):
  fill_level_pct, moisture_index, waste_mean_temp_c, waste_std_c, max_temp_c, temp_std

Target (from data/historian/hhv.csv when available):
  hhv_mj_kg  — higher heating value in MJ/kg, aligned to chute timestamps

In SYNTHETIC mode (no historian data yet): target is generated from signals
via a known formula so the pipeline can be validated before historian arrives.
"""
import math
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np

TIME_BUDGET    = 300
SEQ_LEN        = 64          # 64 frames of history (~1 hr at 1 frame/min)
PRED_HORIZON   = 20          # predict HHV 20 frames ahead (15-20 min lead time)
VAL_FRAC       = 0.2         # last 20% of time series held out (temporal split — no shuffle)

FEATURE_COLS = [
    "fill_level_pct",
    "moisture_index",
    "waste_mean_temp_c",
    "waste_std_c",
    "max_temp_c",
    "temp_std",
]
N_FEATURES = len(FEATURE_COLS)

CHUTEB_CSV   = Path("data/chuteb_signals/signals.csv")
HISTORIAN_CSV = Path("data/historian/hhv.csv")   # columns: upload_time_epoch, hhv_mj_kg
SYNTHETIC_MODE = not HISTORIAN_CSV.exists()


def _load_signals():
    df = pd.read_csv(CHUTEB_CSV)
    df = df.dropna(subset=FEATURE_COLS)
    df = df.sort_values("upload_time_epoch").reset_index(drop=True)

    # Clip outliers before normalization
    df["fill_level_pct"]    = df["fill_level_pct"].clip(0, 100)
    df["moisture_index"]    = df["moisture_index"].clip(0, 1)
    df["waste_mean_temp_c"] = df["waste_mean_temp_c"].clip(5, 60)
    df["max_temp_c"]        = df["max_temp_c"].clip(5, 80)
    df["waste_std_c"]       = df["waste_std_c"].clip(0, 20)
    df["temp_std"]          = df["temp_std"].clip(0, 20)

    feats = df[FEATURE_COLS].values.astype(np.float32)

    # Z-score normalize per feature using train statistics (first 80%)
    n_train = int(len(feats) * (1 - VAL_FRAC))
    mean = feats[:n_train].mean(axis=0)
    std  = feats[:n_train].std(axis=0) + 1e-6
    feats = (feats - mean) / std

    if SYNTHETIC_MODE:
        # Synthetic HHV: dry (low moisture) + plastic-rich (high temp) → high BTU
        # Formula chosen to match domain: HHV ~ 11 + 3*(1-moisture) + 2*(fill/100) + noise
        moisture = df["moisture_index"].values
        fill     = df["fill_level_pct"].values / 100.0
        hhv = 11.0 + 3.0 * (1.0 - moisture) + 2.0 * fill + \
              np.random.RandomState(42).normal(0, 0.3, len(df))
        hhv = hhv.astype(np.float32)
        print("SYNTHETIC MODE: HHV targets are derived from signal formula, not real historian.")
    else:
        hist = pd.read_csv(HISTORIAN_CSV).sort_values("upload_time_epoch")
        # Merge historian to chute signals by nearest timestamp
        hist_idx = np.searchsorted(hist["upload_time_epoch"].values,
                                    df["upload_time_epoch"].values)
        hist_idx = np.clip(hist_idx, 0, len(hist) - 1)
        hhv = hist["hhv_mj_kg"].values[hist_idx].astype(np.float32)

    return feats, hhv, mean, std


class BTUWindowDataset(Dataset):
    def __init__(self, feats, hhv, start, end):
        self.feats = feats[start:end]
        self.hhv   = hhv[start:end]

    def __len__(self):
        return max(0, len(self.feats) - SEQ_LEN - PRED_HORIZON)

    def __getitem__(self, idx):
        x = self.feats[idx : idx + SEQ_LEN]
        y = self.hhv[idx + SEQ_LEN + PRED_HORIZON - 1]
        return torch.from_numpy(x), torch.tensor(y)


def make_dataloaders(batch_size):
    feats, hhv, mean, std = _load_signals()
    n_train = int(len(feats) * (1 - VAL_FRAC))
    train_ds = BTUWindowDataset(feats, hhv, 0, n_train)
    val_ds   = BTUWindowDataset(feats, hhv, n_train, len(feats))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader, mean, std


@torch.no_grad()
def evaluate_mae(model, device, batch_size=64):
    """Fixed evaluation: MAE in MJ/kg on held-out temporal window."""
    _, val_loader, _, _ = make_dataloaders(batch_size)
    model.eval()
    total_ae = 0.0
    n = 0
    for x, y in val_loader:
        pred = model(x.to(device)).squeeze(-1).cpu()
        total_ae += (pred - y).abs().sum().item()
        n += len(y)
    return total_ae / max(n, 1)
```

---

### `scripts/train_btu.py` (agent modifies this)

```python
"""
BTU/HHV time series prediction — agent-modifiable training script.
Fixed time budget: 5 min. Metric: MAE in MJ/kg on held-out temporal window.

Architecture: transformer over a rolling window of chute sensor signals.
Key difference from autoresearch/train.py:
  - FeatureProjection replaces the word embedding (wte)
  - Regression head (Linear → scalar) replaces the LM head
  - Input: [B, SEQ_LEN, N_FEATURES] floats; Target: [B] scalar HHV values

Usage: python scripts/train_btu.py
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare_btu import (
    TIME_BUDGET, SEQ_LEN, N_FEATURES, make_dataloaders, evaluate_mae, SYNTHETIC_MODE
)
from muon_adamw import MuonAdamW

# ── Hyperparameters (agent edits below this line) ─────────────────────────────

DEPTH          = 4           # transformer layers
N_HEAD         = 4           # attention heads
N_EMBD         = 128         # embedding dim (model_dim = DEPTH * 32 is a good heuristic)
WINDOW_PATTERN = "LL"        # L=full attention, S=half-context sliding window
LR             = 3e-4
WEIGHT_DECAY   = 0.01
BATCH_SIZE     = 64
WARMDOWN_RATIO = 0.4
LOSS           = "huber"     # "mse" or "huber" (huber is more robust to BTU outliers)
HUBER_DELTA    = 0.5         # in MJ/kg
HEAD_DIM       = 32          # head_dim = N_EMBD // N_HEAD

# ──────────────────────────────────────────────────────────────────────────────


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=3)


class FeatureProjection(nn.Module):
    """Projects N_FEATURES continuous signals → embedding dim. Replaces wte."""
    def __init__(self, n_features, n_embd):
        super().__init__()
        self.proj = nn.Linear(n_features, n_embd, bias=True)

    def forward(self, x):
        # x: [B, T, N_FEATURES] → [B, T, N_EMBD]
        return self.proj(x)


class Attention(nn.Module):
    def __init__(self, n_embd, n_head, layer_idx, n_layers):
        super().__init__()
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.c_q   = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k   = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v   = nn.Linear(n_embd, n_embd, bias=False)
        self.c_out = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos_sin, window_size):
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        # Standard scaled dot-product attention (no FA3 needed for short sequences)
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_out(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc   = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, n_embd, n_head, layer_idx, n_layers):
        super().__init__()
        self.attn = Attention(n_embd, n_head, layer_idx, n_layers)
        self.mlp  = MLP(n_embd)

    def forward(self, x, cos_sin, window_size):
        x = x + self.attn(norm(x), cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class BTUTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0, "N_EMBD must be divisible by N_HEAD"
        assert all(c in "SL" for c in WINDOW_PATTERN.upper())

        self.feature_proj = FeatureProjection(N_FEATURES, N_EMBD)
        self.blocks = nn.ModuleList([
            Block(N_EMBD, N_HEAD, i, DEPTH) for i in range(DEPTH)
        ])
        self.head = nn.Linear(N_EMBD, 1, bias=True)

        # Window sizes per layer
        long_w  = (SEQ_LEN, 0)
        short_w = (SEQ_LEN // 2, 0)
        pattern = WINDOW_PATTERN.upper()
        self.window_sizes = []
        for i in range(DEPTH):
            c = pattern[i % len(pattern)]
            self.window_sizes.append(long_w if c == "L" else short_w)
        self.window_sizes[-1] = long_w  # last layer always full context

        # Rotary embeddings
        head_dim = N_EMBD // N_HEAD
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (channel_range / head_dim))
        t = torch.arange(SEQ_LEN * 2, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().bfloat16()[None, :, None, :]
        sin = freqs.sin().bfloat16()[None, :, None, :]
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, features, targets=None):
        # features: [B, T, N_FEATURES]
        B, T, _ = features.shape
        x = norm(self.feature_proj(features.bfloat16()))
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        for i, block in enumerate(self.blocks):
            x = block(x, cos_sin, self.window_sizes[i])
        x = norm(x)
        pred = self.head(x[:, -1, :]).squeeze(-1).float()  # predict from last timestep
        if targets is not None:
            if LOSS == "huber":
                return F.huber_loss(pred, targets.float(), delta=HUBER_DELTA)
            return F.mse_loss(pred, targets.float())
        return pred


# ── Setup ─────────────────────────────────────────────────────────────────────

t_start = time.time()
device  = torch.device("cuda")
torch.manual_seed(42)
torch.set_float32_matmul_precision("high")

model = BTUTransformer().to(device)
model = torch.compile(model, dynamic=False)

matrix_params = [p for p in model.parameters() if p.requires_grad and p.ndim == 2]
other_params  = [p for p in model.parameters() if p.requires_grad and p.ndim != 2]
optimizer = MuonAdamW([
    dict(kind="adamw", params=other_params,  lr=LR, betas=(0.9, 0.999),
         eps=1e-8, weight_decay=WEIGHT_DECAY),
    dict(kind="muon",  params=matrix_params, lr=LR, momentum=0.95,
         ns_steps=5, beta2=0.95, weight_decay=WEIGHT_DECAY),
])
for g in optimizer.param_groups:
    g["initial_lr"] = g["lr"]

if SYNTHETIC_MODE:
    print("WARNING: running in SYNTHETIC mode — results are not based on real HHV data.")

train_loader, _, _, _ = make_dataloaders(BATCH_SIZE)

total_train_time = 0.0
step = 0

while True:
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        t0 = time.time()

        loss = model(x, y)
        if torch.isnan(loss) or loss.item() > 100:
            print("FAIL"); exit(1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        progress = total_train_time / TIME_BUDGET
        lrm = max((1.0 - progress) / WARMDOWN_RATIO, 0.01) if progress > 1.0 - WARMDOWN_RATIO else 1.0
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * lrm

        optimizer.step()

        t1 = time.time()
        if step > 5:
            total_train_time += t1 - t0
        step += 1

        if step == 0:
            gc.collect(); gc.freeze(); gc.disable()

        if step % 20 == 0:
            pct = 100 * total_train_time / TIME_BUDGET
            print(f"\rstep {step:05d} ({pct:.1f}%) loss={loss.item():.4f}", end="", flush=True)

        if total_train_time >= TIME_BUDGET:
            break
    if total_train_time >= TIME_BUDGET:
        break

print()
model.eval()
mae = evaluate_mae(model, device)

t_end        = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
num_params   = sum(p.numel() for p in model.parameters()) / 1e6

print("---")
print(f"mae_mj_kg:        {mae:.6f}")
print(f"training_seconds: {total_train_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params:.3f}")
print(f"depth:            {DEPTH}")
print(f"n_embd:           {N_EMBD}")
print(f"window_pattern:   {WINDOW_PATTERN}")
print(f"synthetic_mode:   {SYNTHETIC_MODE}")
```

---

### `features/program_btu.md` (human writes, agent reads)

```markdown
# BTU/HHV Prediction — autoresearch program

## Task
You are an autonomous ML researcher. You modify `scripts/train_btu.py`,
run 5-minute experiments, and iterate toward the lowest MAE in MJ/kg
for furnace BTU prediction 15–20 minutes ahead.

## Setup
1. Run tag (e.g. `may6-btu`). Branch: `autoresearch/btu/<tag>`.
2. Read: `scripts/prepare_btu.py`, `scripts/train_btu.py`, this file.
3. Check SYNTHETIC_MODE in the training output. If True, you are running on
   synthetic targets — architecture search is valid but MAE values are
   not meaningful for production use.
4. Init `results_btu.tsv`: `commit\tmae_mj_kg\tmemory_gb\tsynthetic\tstatus\tdescription`
5. Confirm and begin.

## Metric
**mae_mj_kg** (lower = better). Target operating range is 9–13 MJ/kg.
A 0.1 MJ/kg improvement is meaningful — the combustion engineer uses this
signal to decide whether to pre-cool airflow (> 13) or increase feed rate (< 9).

## Domain knowledge
- **Temporal structure**: waste composition changes slowly (10-30 min trends)
  but individual loads create sharp spikes. The model needs both long context
  (detect slow drift) and local sensitivity (detect step changes from truck loads).
  WINDOW_PATTERN "SL" or "SSL" is worth trying — short layers catch load spikes,
  long layer integrates the shift trend.
- **Key predictors in order of importance**:
  1. moisture_index — single strongest BTU predictor (wet = low BTU)
  2. fill_level_pct — a full chute of uniform material is a good predictor
  3. waste_std_c — high spatial std = heterogeneous = hard to predict; low = homogenous load incoming
  4. max_temp_c — elevated peak temp may indicate plastics (high BTU)
- **Prediction horizon**: PRED_HORIZON=20 frames = ~20 min lead time matching chute-to-furnace
  transit. Do not change this without understanding the furnace lead time.
- **Outlier BTU values**: some historian entries may be anomalous (furnace trips, startups).
  Huber loss is more robust than MSE for these — keep LOSS="huber" unless you have a reason.
- **SEQ_LEN=64** (~1 hour of history) is the default. Try 32 (faster training, less context)
  and 128 (more context, more VRAM) to explore the tradeoff.

## What you CAN change
- DEPTH, N_HEAD, N_EMBD, WINDOW_PATTERN
- LR, WEIGHT_DECAY, BATCH_SIZE, WARMDOWN_RATIO
- LOSS ("mse" or "huber"), HUBER_DELTA
- FeatureProjection: add nonlinearity, add layer norm, add a hidden layer
- MLP expansion ratio (currently 4x, try 2x or 8x)
- Attention: try adding value residual (as in autoresearch train.py)

## What you CANNOT change
- prepare_btu.py (evaluation harness)
- SEQ_LEN, PRED_HORIZON, N_FEATURES, FEATURE_COLS (defined in prepare_btu.py)
- The evaluate_mae() call and summary print block

## Simplicity criterion
Same as autoresearch: smaller models that generalize > large models that overfit.
With limited training data (weeks of chute signals), simpler is almost always better.
DEPTH=2 with N_EMBD=64 is a valid starting point.

## Loop
NEVER STOP until interrupted. Keep if mae_mj_kg improved. Discard if not.
```

---

## How to wire the best model into the dashboard

Once either training loop produces a good model checkpoint:

1. Save checkpoint at end of `train_classify.py` / `train_btu.py`:
   ```python
   torch.save(model.state_dict(), "data/models/waste_classify_best.pt")
   ```

2. Add a `GET /api/classify/{camera}` endpoint to `dashboard_api.py`:
   - Loads checkpoint once at startup (cached in module scope)
   - Runs inference on the latest RGB frame for that camera
   - Returns `{dominant_category, confidence, top3, timestamp}`

3. Wire the response into the Furnace Feed tab's composition panel (Gap 1c in
   `dashboard-scope-gap-analysis.md`).

The BTU model wires similarly into `/api/data` response under `charts.btu_pred`.
Replace the synthetic demo data with `model(latest_window)` once MAE < 0.5 MJ/kg
on real historian data.
```
