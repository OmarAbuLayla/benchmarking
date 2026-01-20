import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch
from tqdm import tqdm
from scipy.io import loadmat
from kymatio.torch import Scattering1D
from scipy import signal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# CONFIG
# --------------------------
J = 4
Q = 4
max_order = 3
T_GLOBAL = 1750
BATCH_SIZE = 64
USE_FP16 = False

# --------------------------
# MODELS
# --------------------------
scattering = Scattering1D(J=J, shape=T_GLOBAL, Q=Q, max_order=max_order).to(device)
proj = torch.nn.Conv2d(6, 6, kernel_size=1).to(device)

import torch._dynamo
torch._dynamo.config.suppress_errors = True
try:
    scattering = torch.compile(scattering, backend="inductor")
    proj = torch.compile(proj, backend="inductor")
    print("✅ using torch.compile (inductor)")
except Exception:
    print("⚠️ torch.compile disabled (no Triton on Windows), running eager mode")
    pass

SAVE_DIR = "./WST_with_pre_processing"
os.makedirs(SAVE_DIR, exist_ok=True)

emg_subject_list = np.load(
    r"/mnt/d/Omar/work GJU/Codes/code4AVE-Speech/CLS_emg_only_new/fix/emg_subject.npy",
    allow_pickle=True
).tolist()


# --------------------------

# MFSC-style EMG filtering

# --------------------------
def filter_emg(raw):
    fs = 1000
    b1, a1 = signal.iirnotch(50, 30, fs)
    b2, a2 = signal.iirnotch(150, 30, fs)
    b3, a3 = signal.iirnotch(250, 30, fs)
    b4, a4 = signal.iirnotch(350, 30, fs)
    b5, a5 = signal.butter(4, [10/(fs/2), 400/(fs/2)], 'bandpass')

    x = signal.filtfilt(b1, a1, raw, axis=1)
    x = signal.filtfilt(b2, a2, x, axis=1)
    x = signal.filtfilt(b3, a3, x, axis=1)
    x = signal.filtfilt(b4, a4, x, axis=1)
    x = signal.filtfilt(b5, a5, x, axis=1)
    return x



@torch.no_grad()
def scatter_batch(X):

    X = filter_emg(X)
    # Crop
    X = np.ascontiguousarray(X[:, 250:, :])  # (B, T, C)
    x = torch.from_numpy(X).float().permute(0, 2, 1).contiguous()  # (B, C, T)
    B, C, T = x.shape

    # --- (1) Per-channel RMS normalization ---
    rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
    x = x / rms

    x = x.to(device, non_blocking=True)

    # Flatten channels into batch
    x = x.reshape(B * C, T).contiguous()

    # --- (2) Scattering Transform ---
    Sx = scattering(x).contiguous()
    Sx = Sx[:, 1:, :]  # drop S0

    # --- (3) Log compression ---
    Sx = torch.log1p(torch.abs(Sx))

    # Reshape back
    Sx = Sx.reshape(B, C, Sx.shape[1], Sx.shape[2]).contiguous()

    # --- (4) Per-coefficient z-score normalization ---
    mu = Sx.mean(dim=(0, 2, 3), keepdim=True)
    std = Sx.std(dim=(0, 2, 3), keepdim=True) + 1e-9
    Sx = (Sx - mu) / std

    # --- (5) Delta and Delta^2 features ---
    delta = Sx[..., 1:] - Sx[..., :-1]
    ddelta = delta[..., 1:] - delta[..., :-1]
    min_len = min(Sx.shape[-1] - 2, delta.shape[-1] - 1, ddelta.shape[-1])
    Sx = torch.cat([Sx[..., :min_len], delta[..., :min_len], ddelta[..., :min_len]], dim=-2)

    # --- (6) Channel projection and temporal pooling ---
    Sx = proj(Sx)
    Sx = torch.nn.functional.interpolate(Sx, size=(36, Sx.shape[-1]), mode="bilinear", align_corners=False)

    if Sx.shape[-1] > 2:
        Sx = torch.nn.functional.avg_pool2d(Sx, kernel_size=(1, 2), stride=(1, 2))

    return Sx.cpu().numpy()


def process_subject(subj_path):
    subj_id = os.path.basename(subj_path)
    for session in os.listdir(subj_path):
        sess_path = os.path.join(subj_path, session)
        if not os.path.isdir(sess_path):
            continue

        save_dir = os.path.join(SAVE_DIR, subj_id, session)
        os.makedirs(save_dir, exist_ok=True)

        files = [f for f in os.listdir(sess_path) if f.endswith(".mat")]
        if not files:
            continue

        mat_batch, file_batch = [], []

        for file in files:
            out_path = os.path.join(save_dir, file.replace(".mat", ".npy"))
            if os.path.exists(out_path):
                continue

            data = loadmat(os.path.join(sess_path, file))["data"]
            mat_batch.append(data)
            file_batch.append(file)

            if len(mat_batch) == BATCH_SIZE:
                feats = scatter_batch(np.stack(mat_batch))
                for feat, fname in zip(feats, file_batch):
                    #np.save(out_path.replace(".mat", ".npy"), {"feat": feat, "label": int(fname.split(".")[0])})
                    np.save(os.path.join(save_dir, fname.replace(".mat", ".npy")),{"feat": feat, "label": int(fname.split(".")[0])})  
                mat_batch, file_batch = [], []

        if mat_batch:
            feats = scatter_batch(np.stack(mat_batch))
            for feat, fname in zip(feats, file_batch):
                np.save(os.path.join(save_dir, fname.replace(".mat", ".npy")),
                        {"feat": feat, "label": int(fname.split(".")[0])})


# --------------------------
# MAIN
# --------------------------
for subj in tqdm(emg_subject_list):
    process_subject(subj)

print("✅ Scattering feature generation (normalized, compressed, Δ+Δ²) complete")
