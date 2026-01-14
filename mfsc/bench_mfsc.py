import argparse
import torch
import time
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis

# -------------------------------------------------
# ARGUMENTS
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["remamba", "bimamba", "finetuneGRU", "transformer"],
    help="Model architecture to benchmark (MFSC)"
)
parser.add_argument(
    "--ckpt",
    type=str,
    required=True,
    help="Path to checkpoint (.pt)"
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="cuda or cpu"
)
args = parser.parse_args()

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DEVICE = args.device if torch.cuda.is_available() else "cpu"
WARMUP = 20
RUNS = 100

# MFSC input: (B, C, H, W)
INPUT_SHAPE = (1, 6, 36, 36)
T = INPUT_SHAPE[-1]   # temporal frames = 36

# -------------------------------------------------
# MODEL FACTORY (MFSC)
# -------------------------------------------------
if args.model == "remamba":
    from emg_model_with_mamba5 import EMGNet
    model = EMGNet(
        mode="mamba",
        nClasses=101,
        every_frame=False
    )

elif args.model == "bimamba":
    from emg_model_mamba_mfsc import EMGMamba
    model = EMGMamba(
        n_classes=101,
        every_frame=False
    )

elif args.model == "finetuneGRU":
    from emg_model_with_trans import emg_model
    model = emg_model(
        mode="finetuneGRU",
        nClasses=101,
        every_frame=False
    )

elif args.model == "transformer":
    from emg_model_with_trans import emg_model
    model = emg_model(
        mode="transformer",
        nClasses=101,
        every_frame=False
    )

else:
    raise ValueError("Invalid model type")

# -------------------------------------------------
# LOAD CHECKPOINT (FILTERED, SAFE)
# -------------------------------------------------
ckpt = torch.load(args.ckpt, map_location="cpu")

model_dict = model.state_dict()
filtered_ckpt = {
    k: v for k, v in ckpt.items()
    if k in model_dict and model_dict[k].shape == v.shape
}

model_dict.update(filtered_ckpt)
model.load_state_dict(model_dict, strict=False)

model.to(DEVICE)
model.eval()

dummy = torch.randn(*INPUT_SHAPE).to(DEVICE)

# -------------------------------------------------
# PARAM COUNT
# -------------------------------------------------
params = sum(p.numel() for p in model.parameters())
params_m = params / 1e6

# -------------------------------------------------
# FLOPs (STATIC)
# -------------------------------------------------
with torch.no_grad():
    flops = FlopCountAnalysis(model, dummy)
    flops_g = flops.total() / 1e9

# -------------------------------------------------
# MACs (STATIC)
# -------------------------------------------------
macs, _ = get_model_complexity_info(
    model,
    INPUT_SHAPE[1:],   # exclude batch dim
    as_strings=False,
    print_per_layer_stat=False,
    verbose=False
)
macs_g = macs / 1e9

# -------------------------------------------------
# LATENCY
# -------------------------------------------------
with torch.no_grad():
    for _ in range(WARMUP):
        _ = model(dummy)

if DEVICE == "cuda":
    torch.cuda.synchronize()

start = time.time()
with torch.no_grad():
    for _ in range(RUNS):
        _ = model(dummy)

if DEVICE == "cuda":
    torch.cuda.synchronize()

latency_ms = (time.time() - start) / RUNS * 1000
latency_per_frame = latency_ms / T

# -------------------------------------------------
# RESULTS
# -------------------------------------------------
print("=" * 60)
print(f"MODEL:              {args.model}")
print(f"DEVICE:             {DEVICE}")
print(f"CHECKPOINT:         {args.ckpt}")
print("-" * 60)
print(f"Params:             {params_m:.2f} M")
print(f"FLOPs:              {flops_g:.2f} G  (static)")
print(f"MACs:               {macs_g:.2f} G  (static)")
print(f"Latency:            {latency_ms:.2f} ms")
print(f"Latency / frame:    {latency_per_frame:.4f} ms")
print("=" * 60)
