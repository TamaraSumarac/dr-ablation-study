"""
record_cells.py — film selected policy x perturbation cells (Week 3, video session)
v2: full 10-clip list + sensor_noise branch.

One cell per process:  python record_cells.py <index 0-9>
Videos land in ~/IsaacLab/videos/ as mp4.
Indices 0-5 unchanged from v1 (already filmed — no need to re-run unless reshooting).
"""

from isaaclab.app import AppLauncher
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import os
import sys

import torch
import yaml
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.ablation_env_cfg import (
    _delayed_pd_actuators,
)

HOME = os.path.expanduser("~")
RUNS = f"{HOME}/IsaacLab/logs/rsl_rl/unitree_go2_flat"

POLICIES = {
    "baseline_full_dr":  f"{RUNS}/2026-07-23_19-36-02/model_999.pt",
    "no_friction":       f"{RUNS}/2026-07-23_19-53-27/model_999.pt",
    "no_mass":           f"{RUNS}/2026-07-23_20-11-31/model_999.pt",
    "no_motor_strength": f"{RUNS}/2026-07-23_20-29-13/model_999.pt",
    "no_action_latency": f"{RUNS}/2026-07-23_21-11-02/model_999.pt",
}

# (policy, perturb_type, value) — values = sweep grid
CELLS = [
    ("baseline_full_dr",  "nominal",        None),   # 0 anchor
    ("no_friction",       "motor_strength", 0.45),   # 1 catastrophe cell
    ("baseline_full_dr",  "action_latency", 20),     # 2 latency pair: copes
    ("no_action_latency", "action_latency", 20),     # 3 latency pair: collapses
    ("no_mass",           "mass",           4.5),    # 4 graceful bias (subtle)
    ("no_motor_strength", "friction",       0.20),   # 5 slipping
    ("no_mass",           "motor_strength", 0.45),   # 6 coupling check
    ("no_action_latency", "mass",           4.5),    # 7 the 0.72 curiosity
    ("no_action_latency", "sensor_noise",   1.1),    # 8 subtle pair a
    ("baseline_full_dr",  "sensor_noise",   1.1),    # 9 subtle pair b
]

N_ENVS = 4            # few robots -> readable frame
VIDEO_STEPS = 500     # ~10 s at 50 Hz control
EVAL_SEED = 42


def make_perturbed_env(perturb_type, value):
    env_cfg = UnitreeGo2FlatEnvCfg()
    env_cfg.scene.num_envs = N_ENVS
    # kill all DR (same as eval_sweep)
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_external_force_torque = None
    for attr in ("push_robot", "base_com", "actuator_gains"):
        if hasattr(env_cfg.events, attr):
            setattr(env_cfg.events, attr, None)
    env_cfg.observations.policy.enable_corruption = False

    # all six policies trained with DelayedPD -> eval matches
    delay = int(value) if perturb_type == "action_latency" else 0
    _delayed_pd_actuators(env_cfg, min_delay=delay, max_delay=delay)

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
    # nominal / action_latency: nothing further (latency handled via actuator)

    # camera framing — tune if robots leave frame:
    # env_cfg.viewer.eye = (4.0, 4.0, 2.5)
    # env_cfg.viewer.lookat = (0.0, 0.0, 0.4)

    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=env_cfg,
                   render_mode="rgb_array")
    return env


def load_policy(checkpoint_path, env):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
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
        for step in range(VIDEO_STEPS + 10):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
    env.close()
    print(f"=== DONE {tag} -> ~/IsaacLab/videos/ ===", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
