"""
record_cells.py — record video of selected policy x perturbation cells

Companion to eval_sweep.py: builds the same frozen-perturbation environments
(with only a few robots so the frame is readable) and records a short clip of
a chosen policy in each.

Usage (one cell per process):
  python record_cells.py <cell_index>      # 0 .. len(CELLS)-1
Videos are written to ~/IsaacLab/videos/ as mp4.
"""

from isaaclab.app import AppLauncher
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import os
import sys

import gymnasium as gym
import torch
import yaml

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.ablation_env_cfg import (
    _delayed_pd_actuators,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)
from rsl_rl.runners import OnPolicyRunner

HOME = os.path.expanduser("~")
RUNS = f"{HOME}/IsaacLab/logs/rsl_rl/unitree_go2_flat"

# Checkpoints from run_ablation.sh. (no_sensor_noise is omitted: its behaviour
# is not visually distinguishable at this clip length.)
POLICIES = {
    "baseline_full_dr":  f"{RUNS}/2026-07-23_19-36-02/model_999.pt",
    "no_friction":       f"{RUNS}/2026-07-23_19-53-27/model_999.pt",
    "no_mass":           f"{RUNS}/2026-07-23_20-11-31/model_999.pt",
    "no_motor_strength": f"{RUNS}/2026-07-23_20-29-13/model_999.pt",
    "no_action_latency": f"{RUNS}/2026-07-23_21-11-02/model_999.pt",
}

# (policy, perturb_type, value) — values match the eval_sweep.py grid.
CELLS = [
    ("baseline_full_dr",  "nominal",        None),   # 0  reference: unperturbed
    ("no_friction",       "motor_strength", 0.45),   # 1  worst cell in the sweep
    ("baseline_full_dr",  "action_latency", 20),     # 2  latency pair: trained with latency DR
    ("no_action_latency", "action_latency", 20),     # 3  latency pair: trained without
    ("no_mass",           "mass",           4.5),    # 4  heavy payload without mass DR
    ("no_motor_strength", "friction",       0.20),   # 5  low friction without motor DR
    ("no_mass",           "motor_strength", 0.45),   # 6  cross-term: mass DR vs weak motors
    ("no_action_latency", "mass",           4.5),    # 7  cross-term: latency DR vs heavy payload
    ("no_action_latency", "sensor_noise",   1.1),    # 8  noise pair: trained without latency DR
    ("baseline_full_dr",  "sensor_noise",   1.1),    # 9  noise pair: full-DR baseline
]

N_ENVS = 4            # few robots -> readable frame
VIDEO_STEPS = 500     # ~10 s at 50 Hz control
EVAL_SEED = 42


def make_perturbed_env(perturb_type, value):
    """Same frozen-world construction as eval_sweep.make_perturbed_env, with
    fewer envs and rgb_array rendering enabled."""
    env_cfg = UnitreeGo2FlatEnvCfg()
    env_cfg.scene.num_envs = N_ENVS

    # disable every randomization sampler
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_external_force_torque = None
    for attr in ("push_robot", "base_com", "actuator_gains"):
        if hasattr(env_cfg.events, attr):
            setattr(env_cfg.events, attr, None)
    env_cfg.observations.policy.enable_corruption = False

    # actuator model identical to training; delay frozen at the cell value
    # for the latency cell and at 0 otherwise
    delay = int(value) if perturb_type == "action_latency" else 0
    _delayed_pd_actuators(env_cfg, min_delay=delay, max_delay=delay)

    # re-enable exactly one term, frozen
    if perturb_type == "friction":
        env_cfg.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (value, value),
                    "dynamic_friction_range": (value * 0.75, value * 0.75),
                    "restitution_range": (0.0, 0.0), "num_buckets": 64})
    elif perturb_type == "mass":
        env_cfg.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                    "mass_distribution_params": (value, value), "operation": "add"})
    elif perturb_type == "motor_strength":
        env_cfg.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "stiffness_distribution_params": (value, value),
                    "damping_distribution_params": (value, value),
                    "operation": "scale", "distribution": "uniform"})
    elif perturb_type == "sensor_noise":
        env_cfg.observations.policy.enable_corruption = True
        for term_name in ["base_lin_vel", "base_ang_vel", "projected_gravity",
                          "joint_pos", "joint_vel"]:
            term = getattr(env_cfg.observations.policy, term_name, None)
            if term is not None and getattr(term, "noise", None) is not None:
                term.noise.n_min = term.noise.n_min * value
                term.noise.n_max = term.noise.n_max * value
    # nominal / action_latency: nothing further (latency is set on the actuator)

    # optional camera framing, if robots leave the frame:
    # env_cfg.viewer.eye = (4.0, 4.0, 2.5)
    # env_cfg.viewer.lookat = (0.0, 0.0, 0.4)

    return gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=env_cfg,
                    render_mode="rgb_array")


def load_policy(checkpoint_path, env):
    """Checkpoint on disk -> deterministic (mean-action) policy callable."""
    log_dir = os.path.dirname(checkpoint_path)
    with open(os.path.join(log_dir, "params", "agent.yaml")) as f:
        agent_cfg = yaml.safe_load(f)
    device = env.unwrapped.device
    runner = OnPolicyRunner(RslRlVecEnvWrapper(env), agent_cfg, log_dir=None, device=device)
    runner.load(checkpoint_path)
    return runner.get_inference_policy(device=device)


def main():
    policy_name, perturb_type, value = CELLS[int(sys.argv[1])]
    tag = f"{policy_name}__{perturb_type}" + (f"_{value}" if value is not None else "")
    print(f"=== RECORDING {tag} ===", flush=True)

    env = make_perturbed_env(perturb_type, value)
    env = gym.wrappers.RecordVideo(
        env, video_folder=f"{HOME}/IsaacLab/videos",
        step_trigger=lambda s: s == 0, video_length=VIDEO_STEPS,
        name_prefix=tag, disable_logger=True)

    policy = load_policy(POLICIES[policy_name], env)

    obs, _ = env.reset(seed=EVAL_SEED)
    with torch.no_grad():
        for step in range(VIDEO_STEPS + 10):   # a few extra steps so the recorder flushes
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
    env.close()
    print(f"=== DONE {tag} -> ~/IsaacLab/videos/ ===", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
