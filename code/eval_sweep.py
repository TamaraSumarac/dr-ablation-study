"""
eval_sweep.py — post-training robustness sweep for the Go2 DR-ablation study

Evaluates each trained policy under a grid of frozen physics perturbations that
lie outside the domain-randomization ranges used during training. Produces one
CSV row per (policy, perturbation, episode); the table/heatmap is built from
that file downstream.

Design:
  - Every eval env has ALL domain randomization disabled, then exactly ONE
    physical property re-enabled and frozen at an out-of-distribution value.
  - Every eval env uses the same DelayedPDActuator model the six policies were
    trained with (delay fixed at 0 unless the cell is the latency cell), so
    eval and training share an identical actuator implementation.
  - Each cell rolls out N_EPISODES_PER_CELL parallel envs and records each
    env's FIRST episode only, from a fixed seed.

Usage (one grid cell per process, so a bad cell can't take down the rest):
  python eval_sweep.py <cell_index>      # 0 .. len(PERTURBATION_GRID)-1
Rows are appended to results_raw.csv; run cell 0 (nominal) first — it acts as
a sanity gate for the whole pipeline.
"""

# ── Isaac Sim bootstrap — must precede every isaaclab import ────────────────
from isaaclab.app import AppLauncher
simulation_app = AppLauncher(headless=True).app

# ── normal imports ──────────────────────────────────────────────────────────
import csv
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


# ─────────────────────────────────────────────────────────────────────────────
# PART 0 — experiment manifest
# ─────────────────────────────────────────────────────────────────────────────

# The six policies from run_ablation.sh (same seed, same iterations; each
# differs from the full-DR baseline by exactly one randomization term).
POLICIES = {
    "baseline_full_dr":  "/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-07-23_19-36-02/model_999.pt",
    "no_friction":       "/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-07-23_19-53-27/model_999.pt",
    "no_mass":           "/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-07-23_20-11-31/model_999.pt",
    "no_motor_strength": "/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-07-23_20-29-13/model_999.pt",
    "no_sensor_noise":   "/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-07-23_20-47-32/model_999.pt",
    "no_action_latency": "/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-07-23_21-11-02/model_999.pt",
}

# ── PERTURBATION GRID — anchored to the training DR ranges in ablation_env_cfg.py
# rule: moderate = 10% beyond the training-range edge, hard = 50% beyond
#   friction:  train U(0.4, 1.0)        -> eval 0.36 / 0.20   (slipperier)
#   mass add:  train U(-1, +3) kg       -> eval +3.3 / +4.5   (heavier)
#   motor:     train scale U(0.9, 1.1)  -> eval 0.81 / 0.45   (weaker)
#   noise:     train 1.0x stock bounds  -> eval 1.1x / 1.5x   (noisier)
#   latency:   train U{0..4} physics steps (0-20 ms) -> eval 8 / 20 steps
#              (40 ms / 100 ms; 1 physics step = 5 ms; 1 control step = 4 physics steps)
# Ordering matters: nominal must be index 0 (the sanity gate in main()).
PERTURBATION_GRID = [
    {"perturb_type": "nominal",        "severity": "none",     "value": None},
    {"perturb_type": "friction",       "severity": "moderate", "value": 0.36},
    {"perturb_type": "friction",       "severity": "hard",     "value": 0.20},
    {"perturb_type": "mass",           "severity": "moderate", "value": 3.3},
    {"perturb_type": "mass",           "severity": "hard",     "value": 4.5},
    {"perturb_type": "motor_strength", "severity": "moderate", "value": 0.81},
    {"perturb_type": "motor_strength", "severity": "hard",     "value": 0.45},
    {"perturb_type": "sensor_noise",   "severity": "moderate", "value": 1.1},
    {"perturb_type": "sensor_noise",   "severity": "hard",     "value": 1.5},
    {"perturb_type": "action_latency", "severity": "moderate", "value": 8},
    {"perturb_type": "action_latency", "severity": "hard",     "value": 20},
]

N_EPISODES_PER_CELL = 30
EVAL_SEED = 42
MAX_EPISODE_STEPS = 1100    # episode length is 20 s / (0.005 s * 4) = 1000 control
                            # steps; +100 headroom. hit_ceiling=True in the CSV
                            # means an env never terminated or truncated — a bug flag.

# Optional: mean nominal return of a known-good run (e.g. the TensorBoard
# plateau). If set, the nominal gate also fails on a large return mismatch.
REFERENCE_NOMINAL_RETURN = None


# ─────────────────────────────────────────────────────────────────────────────
# PART A — load a frozen policy
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(checkpoint_path, env):
    """Checkpoint on disk -> deterministic policy callable.

    Uses the mean action: the learned action std only served exploration during
    training. Observation-normalizer statistics (if any) are restored together
    with the network weights by runner.load()."""
    log_dir = os.path.dirname(checkpoint_path)
    with open(os.path.join(log_dir, "params", "agent.yaml"), "r") as f:
        agent_cfg = yaml.safe_load(f)

    device = env.unwrapped.device if hasattr(env, "unwrapped") else env.device

    runner = OnPolicyRunner(RslRlVecEnvWrapper(env), agent_cfg, log_dir=None, device=device)
    runner.load(checkpoint_path)
    return runner.get_inference_policy(device=device)


# ─────────────────────────────────────────────────────────────────────────────
# PART B — frozen-world factory
# ─────────────────────────────────────────────────────────────────────────────

def make_perturbed_env(perturb_type, value):
    """Go2 flat env with ALL domain randomization off and exactly ONE property
    frozen at `value` (chosen outside the training DR range)."""
    env_cfg = UnitreeGo2FlatEnvCfg()
    env_cfg.scene.num_envs = N_EPISODES_PER_CELL

    # ── disable every randomization sampler ─────────────────────────────
    # The stock physics_material term already has a degenerate (0.8, 0.8)
    # range, i.e. a frozen nominal, so it can stay. reset_base and
    # reset_robot_joints also stay: they are the reset mechanism, not DR.
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_external_force_torque = None
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "base_com"):
        env_cfg.events.base_com = None
    if hasattr(env_cfg.events, "actuator_gains"):
        env_cfg.events.actuator_gains = None
    env_cfg.observations.policy.enable_corruption = False

    # ── actuator model: identical to training (DelayedPDActuator) ───────
    # Delay is frozen at the cell's value for the latency cell and at 0 for
    # every other cell, so only the latency cell perturbs the actuator.
    delay = int(value) if perturb_type == "action_latency" else 0
    _delayed_pd_actuators(env_cfg, min_delay=delay, max_delay=delay)

    # ── re-enable exactly ONE term, frozen ──────────────────────────────
    if perturb_type in ("nominal", "action_latency"):
        pass  # nominal: nothing; latency: handled by the actuator above

    elif perturb_type == "friction":
        env_cfg.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (value, value),
                "dynamic_friction_range": (value * 0.75, value * 0.75),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )

    elif perturb_type == "mass":
        env_cfg.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (value, value),  # kg added to base
                "operation": "add",
            },
        )

    elif perturb_type == "motor_strength":
        # Same code path as the training DR term, with a degenerate range.
        env_cfg.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (value, value),
                "damping_distribution_params": (value, value),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

    elif perturb_type == "sensor_noise":
        # Scale every policy-observation noise bound by `value`.
        env_cfg.observations.policy.enable_corruption = True
        policy_obs = env_cfg.observations.policy
        for term_name in ["base_lin_vel", "base_ang_vel", "projected_gravity",
                          "joint_pos", "joint_vel"]:
            term = getattr(policy_obs, term_name, None)
            if term is not None and getattr(term, "noise", None) is not None:
                term.noise.n_min = term.noise.n_min * value
                term.noise.n_max = term.noise.n_max * value

    else:
        raise ValueError(f"unknown perturb_type: {perturb_type}")

    return gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=env_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# PART C — roll out N episodes and measure
# ─────────────────────────────────────────────────────────────────────────────

def rollout_cell(policy, env, n_episodes):
    """Returns per-env metric vectors. n_envs == n_episodes: each env
    contributes its FIRST episode only. All envs share one clock; `live`
    gates every accumulation so that once an env's first episode ends (and
    Isaac Lab auto-resets it into a second), nothing it does afterwards can
    change its row."""
    n_envs = n_episodes
    dev = env.unwrapped.device if hasattr(env, "unwrapped") else env.device

    obs, _ = env.reset(seed=EVAL_SEED)
    ep_return = torch.zeros(n_envs, device=dev)
    ep_len    = torch.zeros(n_envs, device=dev)
    track_err = torch.zeros(n_envs, device=dev)
    fell      = torch.zeros(n_envs, dtype=torch.bool, device=dev)
    live      = torch.ones(n_envs, dtype=torch.bool, device=dev)

    uenv = env.unwrapped if hasattr(env, "unwrapped") else env
    asset = uenv.scene["robot"]

    with torch.no_grad():
        for step in range(MAX_EPISODE_STEPS):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            reward     = reward.reshape(-1)
            terminated = terminated.reshape(-1).bool()
            truncated  = truncated.reshape(-1).bool()

            # Raw velocity-tracking error in the base frame (not the
            # exponential reward kernel), so it stays interpretable.
            cmd = uenv.command_manager.get_command("base_velocity")[:, :2]
            vel = asset.data.root_lin_vel_b[:, :2]
            step_err = torch.norm(cmd - vel, dim=1)

            ep_return += reward * live
            ep_len    += live
            track_err += step_err * live

            # terminated = base contact (fell); truncated = timeout (survived)
            just_ended = (terminated | truncated) & live
            fell |= just_ended & terminated
            live = live & ~just_ended
            if not live.any():
                break

    return {
        "ep_return": ep_return,
        "ep_len": ep_len,
        "track_err": track_err / ep_len,
        "fell": fell,
        "live": live,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PART D — sweep driver + CSV
# ─────────────────────────────────────────────────────────────────────────────

def main():
    results_file = "results_raw.csv"
    fieldnames = [
        "policy", "perturb_type", "severity", "perturb_value",
        "episode_idx", "ep_return", "ep_len", "track_err", "fell",
        "hit_ceiling", "eval_seed",
    ]

    if not os.path.exists(results_file):
        with open(results_file, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    # One grid cell per process (see module docstring).
    perturbation = PERTURBATION_GRID[int(sys.argv[1])]
    perturb_type = perturbation["perturb_type"]
    severity     = perturbation["severity"]
    value        = perturbation["value"]

    env = make_perturbed_env(perturb_type, value)

    for policy_name, ckpt_path in POLICIES.items():
        policy = load_policy(ckpt_path, env)
        out = rollout_cell(policy, env, n_episodes=N_EPISODES_PER_CELL)

        rows = []
        for i in range(N_EPISODES_PER_CELL):
            rows.append({
                "policy":        policy_name,
                "perturb_type":  perturb_type,
                "severity":      severity,
                "perturb_value": value,
                "episode_idx":   i,
                "ep_return":     out["ep_return"][i].item(),
                "ep_len":        int(out["ep_len"][i].item()),
                "track_err":     out["track_err"][i].item(),
                "fell":          bool(out["fell"][i].item()),
                "hit_ceiling":   bool(out["live"][i].item()),
                "eval_seed":     EVAL_SEED,
            })

        with open(results_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(rows)
            f.flush()

        mean_ret  = out["ep_return"].mean().item()
        fall_rate = out["fell"].float().mean().item()
        n_stuck   = int(out["live"].sum().item())
        print(f"[{policy_name:>20s} | {perturb_type:>14s}/{severity:<8s}] "
              f"return={mean_ret:8.2f}  fall_rate={fall_rate:.2f}  "
              f"stuck={n_stuck}")
        if n_stuck > 0:
            print(f"  !! {n_stuck} episodes hit the step ceiling — investigate")

        # Sanity gate: in frozen-nominal conditions no policy should fall.
        # A failure here means the pipeline (loading, wrapping, normalizer,
        # env config) is broken, not that the physics is hard.
        if perturb_type == "nominal":
            if fall_rate > 0.05:
                raise RuntimeError(
                    f"{policy_name} falls in NOMINAL conditions "
                    f"(fall_rate={fall_rate:.2f}) — pipeline broken, "
                    "not physics. Stop and debug.")
            if REFERENCE_NOMINAL_RETURN and mean_ret < 0.5 * REFERENCE_NOMINAL_RETURN:
                raise RuntimeError(
                    f"nominal return {mean_ret:.1f} far below reference "
                    f"(~{REFERENCE_NOMINAL_RETURN}) — normalizer/env mismatch?")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()