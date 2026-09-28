import time
import torch
from mamba_ssm import Mamba3

# Define hyper-parameters
batch_size = 2
seq_len = 128
d_model = 256  # Model dimension

# Ensure device is CUDA and using fp16/bf16
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

# Reset CUDA state and stats before starting
if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# 1. Instantiate the Mamba3 layer
model = Mamba3(
    d_model=d_model,
    d_state=128,       # Hidden state dimension per head
    expand=2,          # Inner dimension expand ratio (d_inner = expand * d_model = 512)
    headdim=64,        # Head dimension (512 / 64 = 8 heads)
    rope_fraction=0.5, # Fractional RoPE angle allocation
    is_mimo=False,     # Set to True if testing MIMO configuration (requires TileLang)
).to(device=device, dtype=dtype)

# 2. Create dummy input tensor: (batch_size, sequence_length, d_model)
x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

# Measure baseline memory (model weights + input tensor)
if device == "cuda":
    model_mem = torch.cuda.memory_allocated() / (1024**2)
    print(f"Model + Input static memory: {model_mem:.2f} MB")

# 1. Warm-up run (triggers Triton JIT compilation & CUDA setup)
print("\nWarming up / compiling kernels...")
start_warmup = time.time()
_ = model(x)
torch.cuda.synchronize()
print(f"Warm-up finished in {time.time() - start_warmup:.2f} seconds.")

# Reset peak memory stats after warm-up to isolate benchmark VRAM usage
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

# 2. Benchmark run (subsequent executions use cached compiled kernels)
start_bench = time.time()
for _ in range(100):
    output = model(x)
torch.cuda.synchronize()
elapsed = (time.time() - start_bench) / 100

print(f"Average execution time per pass: {elapsed * 1000:.3f} ms")

# Measure peak VRAM allocation during benchmark
if device == "cuda":
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    reserved_mem = torch.cuda.max_memory_reserved() / (1024**2)
    print(f"Peak VRAM allocated: {peak_mem:.2f} MB")
    print(f"Peak VRAM reserved by PyTorch cache: {reserved_mem:.2f} MB")
