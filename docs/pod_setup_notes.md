<!-- # Cloud GPU setup notes — RL robotics project (updated 2026-07-27)

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
  cell-summary prints die in the buffer on timeout-kill. -->


# Cloud GPU setup notes

Setup used to train and render the Isaac Lab experiments on a Lambda GPU VM
(Ubuntu 22.04 Lambda Stack image). These notes document the final working setup
and the main issues encountered along the way.

## Standard setup

```bash
# 1. System dependencies
sudo apt-get update && sudo apt-get install -y \
    libxt6 libglu1-mesa libxrandr2 libxinerama1 libxcursor1 libxi6 \
    git wget libvulkan1 vulkan-tools ffmpeg


# 2. Miniconda + Isaac Lab environment
cd ~
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh

# New conda versions require ToS acceptance before creating an environment
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab

echo 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab' >> ~/.bashrc


# 3. Isaac Sim + video dependencies
pip install "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com
pip install moviepy


# 4. Isaac Lab
# IMPORTANT: pin the commit rather than using the current default branch.
# Around July 2026, the default branch moved to release/3.0.0-beta2 and
# required Python >=3.12, while this project uses the earlier Python 3.11 stack.

git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
git fetch origin main
git checkout $(git rev-list -n 1 --before="2026-07-23" origin/main)

# Accepts the EULA interactively; torch was automatically corrected to 2.7.0
./isaaclab.sh --install
```

# Rendering on Lambda

Isaac Sim initially ran physics but could not render. vkCreateInstance failed
with ERROR_INCOMPATIBLE_DRIVER, and vulkaninfo detected only llvmpipe
(CPU rendering).

The Lambda Stack image had a working NVIDIA kernel driver/CUDA installation,
but the Vulkan userspace stack was incomplete. Three pieces needed to be
consistent: the NVIDIA kernel driver, NVIDIA userspace libraries/Vulkan ICD,
and the Vulkan loader.

```bash
# 1. Check the RUNNING kernel driver version first — do not assume it
cat /proc/driver/nvidia/version
# Example: 580.105.08

# Remove a potentially mismatched NVIDIA GL package
sudo apt-get remove -y libnvidia-gl-580


# 2. Install userspace libraries matching the kernel driver exactly
# --no-kernel-modules is important: the working kernel driver is left untouched

wget https://download.nvidia.com/XFree86/Linux-x86_64/580.105.08/NVIDIA-Linux-x86_64-580.105.08.run
chmod +x NVIDIA-Linux-x86_64-*.run
sudo ./NVIDIA-Linux-x86_64-*.run -s --no-kernel-modules
sudo ldconfig

# The installer should create:
# /etc/vulkan/icd.d/nvidia_icd.json
#
# If it does not, the working configuration was:
# {
#   "file_format_version": "1.0.0",
#   "ICD": {
#     "library_path": "libGLX_nvidia.so.0",
#     "api_version": "1.3.277"
#   }
# }


# 3. Install a recent Vulkan loader
# Ubuntu 22.04's stock loader (1.3.204) was not compatible with the
# newer NVIDIA Vulkan stack used on this machine.

wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc \
    | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc >/dev/null

sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list \
    https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list

sudo apt-get update
sudo apt-get install -y vulkan-sdk || \
    sudo apt-get install -y libvulkan1 vulkan-tools


# Verify that Vulkan sees the physical NVIDIA GPU rather than llvmpipe
vulkaninfo --summary | grep -A8 "Devices:"
```

The output should identify the NVIDIA A10 as PHYSICAL_DEVICE_TYPE_DISCRETE_GPU.

Once Vulkan is configured correctly, recording works with:

AppLauncher(headless=True, enable_cameras=True)
gym.make(..., render_mode="rgb_array")
gym.wrappers.RecordVideo(...)

See code/record_cells.py. I run one recording cell per process.

## Practical notes

- **Pin the software stack.** Pin the Isaac Sim version, Isaac Lab commit, and other important dependencies. The Isaac Lab default branch can move to a release with different Python/dependency requirements; dates are useful for reconstruction, but commit hashes are the actual pins.

- **Check the NVIDIA installation before changing graphics packages.** `cat /proc/driver/nvidia/version` gives the running kernel driver version. Installing a mismatched `libnvidia-gl-*` package can leave CUDA/compute working while breaking Vulkan rendering.

- **Do not construct package names from commands that may return nothing.** Avoid patterns such as `apt install pkg-$(cmd)` unless the command output has been checked explicitly.

- **Containerized Isaac Sim was not stable in this setup.** On RunPod (2026-07-23), Isaac Sim consistently aborted at the first physics step with `The futex facility returned an unexpected error code`. The same failure was reproduced across multiple regions, GPUs, Ubuntu 22.04/24.04 images, CPU-only execution, and the stock training script. The setup had worked on 2026-07-16. Moving to a full Lambda VM resolved the issue, suggesting a container-runtime/glibc interaction rather than an Isaac Lab configuration problem.

- **Initialize Isaac Sim before importing Isaac Lab.** With the pip installation, `AppLauncher` needs to run before imports that depend on `pxr`.

- **Create one recording environment per process.** Sequential environment creation in the same process was unreliable.

- **rsl-rl expects the Isaac Lab wrapper.** `OnPolicyRunner` requires an `RslRlVecEnvWrapper`-wrapped environment.

- **Use `torch.no_grad()` rather than `torch.inference_mode()`** around the evaluation loop; `inference_mode()` caused problems during wrapper reset.

- **Isaac/Kit process behavior can be awkward.** Kit traps `SIGINT`, so a stuck process may need to be killed from another terminal. Shutdown can also produce Vulkan warnings that did not affect completed runs.

- **Conda requires ToS acceptance before environment creation** on recent versions.

- **Use unbuffered Python for long cloud jobs.** Run with `python -u` when piping output through `tee`; otherwise useful final logs can remain buffered if the job is terminated.

