# Cloud GPU setup notes — RL robotics project (updated 2026-07-27)

## Standard rebuild recipe (Lambda VM, Ubuntu 22.04 Lambda Stack image)

```bash
# 1. apt deps
sudo apt-get update && sudo apt-get install -y libxt6 libglu1-mesa libxrandr2 \
    libxinerama1 libxcursor1 libxi6 git wget libvulkan1 vulkan-tools ffmpeg

# 2. miniconda + env (ToS acceptance REQUIRED before env creation)
cd ~ && wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n env_isaaclab python=3.11 -y && conda activate env_isaaclab
echo 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab' >> ~/.bashrc

# 3. Isaac Sim (pip route) + moviepy for video recording
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install moviepy

# 4. IsaacLab — PIN THE COMMIT, do not use default branch!
#    (default branch moved to release/3.0.0-beta2 requiring Python>=3.12 as of ~Jul 2026;
#     our stack needs a July-2026-era main commit)
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
git fetch origin main
git checkout $(git rev-list -n 1 --before="2026-07-23" origin/main)
./isaaclab.sh --install     # accepts EULA interactively; torch auto-corrected to 2.7.0
```

Smoke test: `python ~/min_test.py` -> expect "50 STEPS OK" (~0.03 s/step on A10 = GPU physics).

## ISAAC RENDERING ON LAMBDA — the hard-won three-part Vulkan fix (2026-07-27)

Symptom: every Isaac launch spams `VkResult: ERROR_INCOMPATIBLE_DRIVER /
vkCreateInstance failed`; vulkaninfo shows only llvmpipe (CPU renderer);
Lambda Stack image ships kernel driver + CUDA but an incoherent/absent
GL-Vulkan userspace. nvidia-smi may not even exist. ALL THREE fixes needed:

```bash
# (0) establish the running kernel driver version FIRST — do NOT guess:
cat /proc/driver/nvidia/version          # e.g. 580.105.08
sudo apt-get remove -y libnvidia-gl-580  # remove any mismatched apt GL package

# (1) EXACT-VERSION userspace via NVIDIA .run, kernel modules untouched:
wget https://download.nvidia.com/XFree86/Linux-x86_64/580.105.08/NVIDIA-Linux-x86_64-580.105.08.run
chmod +x NVIDIA-Linux-x86_64-*.run
sudo ./NVIDIA-Linux-x86_64-*.run -s --no-kernel-modules
sudo ldconfig
# writes the ICD to /etc/vulkan/icd.d/nvidia_icd.json (NOT /usr/share/vulkan/icd.d/)
# if missing, create it by hand: {"file_format_version":"1.0.0","ICD":
#   {"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}

# (2) MODERN VULKAN LOADER — Ubuntu 22.04's stock 1.3.204 loader CANNOT
#     negotiate with 1.4-era NVIDIA ICDs (this was the invisible third layer):
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc >/dev/null
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
sudo apt-get update && (sudo apt-get install -y vulkan-sdk || sudo apt-get install -y libvulkan1 vulkan-tools)

# verify — must show NVIDIA A10 as PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
vulkaninfo --summary | grep -A8 "Devices:"
```

Recording then works via: AppLauncher(headless=True, enable_cameras=True) +
gym.make(..., render_mode="rgb_array") + gym.wrappers.RecordVideo (see
week3_code/record_cells.py). One cell per process.

## Lessons / gotchas ledger

- **Volumes + recipes > pet machines.** Treat every machine as disposable;
  durable state on volume/Mac; machine-specific setup as a written recipe.
- **Pin everything**: IsaacLab commit (default branch WILL move), gymnasium,
  isaacsim version. Dates are not pins; hashes are.
- **Verify `nvidia-smi` exists before ANY GL package operation.** Its absence
  means bare-compute image; blind apt installs of `libnvidia-gl-*` with empty
  version substitution can install MISMATCHED builds (this bit us).
- **Never `apt install pkg-$(cmd)` where cmd can fail/return empty.**
- **RunPod (containers) 2026-07-23: fleet-wide futex abort** — "The futex
  facility returned an unexpected error code" + core dump at first physics
  step, glibc vs container-runtime/seccomp, unfixable in-container, reproduced
  across EU-RO-1 + US-IL-1, 4090/PRO4500, ubuntu2404+2204 images, CPU-only,
  stock train.py. Worked 2026-07-16. => full VMs (Lambda) for Isaac; ticket filed.
- **Isaac quirks**: pip-installed Isaac Sim needs AppLauncher before ANY
  isaaclab import (pxr unresolvable otherwise); sequential env creation in one
  process wedges (one cell per process); OnPolicyRunner requires
  RslRlVecEnvWrapper-wrapped env; torch.inference_mode() clashes with wrapper
  reset (use torch.no_grad()); kit traps SIGINT (kill -9 from second terminal);
  kit shutdown is slow/whiny — Vulkan errors at teardown are noise.
- **Conda ToS**: new conda requires `conda tos accept` before env creation.
- **First-run buffering**: launch python with `-u` when piping/tee-ing, or
  cell-summary prints die in the buffer on timeout-kill.
