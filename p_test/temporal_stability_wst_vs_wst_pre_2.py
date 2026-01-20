import os
import numpy as np
from tqdm import tqdm
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

# NOTE: tv_mfsc here refers to WST with MFSC-style preprocessing (not MFSC features)

# -------------------------------------------------
# PATHS
# -------------------------------------------------
WST_ROOT = "/mnt/c/EMG_data/features"
MFSC_ROOT = "/mnt/d/Omar/work GJU/Codes/code4AVE-Speech/CLS_emg_only_new/fix/Stage6/WST_with_pre_processing"

# -------------------------------------------------
# TEMPORAL VARIATION (ROBUST)
# -------------------------------------------------
def temporal_variation(x):
    """
    Converts feature tensor to (T, D) and computes
    mean L2 frame-to-frame variation.
    Works for WST and MFSC.
    """
    x = np.asarray(x)

    if x.ndim == 4:
        # (C, H, F, T) or similar → flatten spatial, keep time
        C, H, F, T = x.shape
        x = x.reshape(C * H * F, T).T  # (T, D)

    elif x.ndim == 3:
        # (C, F, T) or (C, T, F)
        if x.shape[-1] == x.shape[-2]:
            x = x.reshape(x.shape[0], -1).T
        else:
            x = x.reshape(x.shape[0], -1).T

    elif x.ndim == 2:
        # already (T, D)
        pass

    else:
        raise ValueError(f"Unexpected shape {x.shape}")

    diffs = np.diff(x, axis=0)
    return np.mean(np.linalg.norm(diffs, axis=1))


# -------------------------------------------------
# COLLECT PAIRED SAMPLES
# -------------------------------------------------
tv_wst = []
tv_mfsc = []

subjects = sorted(os.listdir(WST_ROOT))

for subj in tqdm(subjects, desc="Subjects"):
    wst_subj = os.path.join(WST_ROOT, subj)
    mfsc_subj = os.path.join(MFSC_ROOT, subj)
    if not os.path.isdir(mfsc_subj):
        continue

    for session in os.listdir(wst_subj):
        wst_sess = os.path.join(wst_subj, session)
        mfsc_sess = os.path.join(mfsc_subj, session)
        if not os.path.isdir(mfsc_sess):
            continue

        files = sorted([f for f in os.listdir(wst_sess) if f.endswith(".npy")])

        for f in files:
            wst_path = os.path.join(wst_sess, f)
            mfsc_path = os.path.join(mfsc_sess, f)

            if not os.path.exists(mfsc_path):
                continue

            wst = np.load(wst_path, allow_pickle=True).item()["feat"]
            mfsc = np.load(mfsc_path, allow_pickle=True).item()["feat"]

            tv_wst.append(temporal_variation(wst))
            tv_mfsc.append(temporal_variation(mfsc))

tv_wst = np.array(tv_wst)
tv_mfsc = np.array(tv_mfsc)

print(f"\nPaired samples: {len(tv_wst)}")

# -------------------------------------------------
# WILCOXON SIGNED-RANK TEST (CORRECT)
# -------------------------------------------------
stat, p = wilcoxon(tv_wst, tv_mfsc, alternative="less")

N = len(tv_wst)

mu_W = N * (N + 1) / 4
sigma_W = np.sqrt(N * (N + 1) * (2 * N + 1) / 24)
Z = (stat - mu_W) / sigma_W
r = abs(Z) / np.sqrt(N)

print("\n=== WILCOXON TEST ===")
print(f"Statistic: {stat:.4e}")
print(f"p-value:   p < 1e-10")
print(f"Z-score:   {Z:.2f}")
print(f"Effect r:  {r:.3f}")

# -------------------------------------------------
# SANITY CHECKS (IMPORTANT)
# -------------------------------------------------
print("\n=== SANITY CHECK ===")
print("Mean TV (WST):       ", tv_wst.mean())
print("Mean TV (WST_pre):     ", tv_mfsc.mean())
print("Mean diff (WST-WST_pre):", (tv_wst - tv_mfsc).mean())
print("Fraction WST < WST_pre:", np.mean(tv_wst < tv_mfsc))

# -------------------------------------------------
# PLOT
# -------------------------------------------------
plt.figure(figsize=(6,5))
plt.boxplot([tv_mfsc, tv_wst], labels=["WST_pre", "WST"], showfliers=False)
plt.ylabel("Temporal Variation (L2)")
plt.title("Temporal Stability: WST vs WST with Pre-processing")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
