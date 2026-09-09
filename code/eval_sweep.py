"""
eval_sweep.py (v2, pod edition 2026-07-23) — post-training robustness sweep

CHANGES FROM v1 (all from pod reconnaissance):
  - AppLauncher bootstrap (pip-installed Isaac Sim needs kit before pxr)
  - grid values REAL, anchored to the ablation training ranges we chose
  - motor_strength + sensor_noise branches implemented (yaml gave attrs)
  - latency dual-mode: LatencyWrapper for TODAY's stock-actuator validation,
    DelayedPD actuator freeze for TOMORROW's six-policy sweep
  - POLICIES set up for today's validation (stock baseline only)

RUN MODES:
  TODAY (validation):   USE_DELAYED_PD = False, POLICIES = stock baseline model_999
  TOMORROW (the sweep): USE_DELAYED_PD = True, fill six checkpoint paths,
                        fill STOCK_BALLPARK_RETURN if known
"""

# ── Isaac Sim bootstrap — MUST precede every isaaclab import ────────────────
from isaaclab.app import AppLauncher
simulation_app = AppLauncher(headless=True).app

# ── normal imports (safe now) ───────────────────────────────────────────────
import csv
import sys
import os

import torch
import yaml
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.ablation_env_cfg import (
    _delayed_pd_actuators,
)


# ─────────────────────────────────────────────────────────────────────────────
# PART 0 — CONFIG: the experiment manifest
# ─────────────────────────────────────────────────────────────────────────────

# False = TODAY: stock actuators (matches stock baseline checkpoint); latency via wrapper.
## True  = TOMORROW: all cells use DelayedPD actuators (matches the six new
#         policies); latency via frozen actuator delay. FLIP BEFORE THE SWEEP.
USE_DELAYED_PD = True

POLICIES = {
    # TODAY — validation against the stock baseline:
    #"stock_baseline":
    #    "/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-07-16_21-33-44/model_999.pt",
    # TOMORROW — replace the entry above with the six fresh runs
    # (paths from: ls -d /home/ubuntu/IsaacLab/logs/rsl_rl/*/20*):
    "baseline_full_dr":  "/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-07-23_19-36-02/model_999.pt",
    "no_friction":       "/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-07-23_19-53-27/model_999.pt",
    "no_mass":           "/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-07-23_20-11-31/model_999.pt",
    "no_motor_strength": "/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-07-23_20-29-13/model_999.pt",
    "no_sensor_noise":   "/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-07-23_20-47-32/model_999.pt",
    "no_action_latency": "/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-07-23_21-11-02/model_999.pt", 
}

# ── PERTURBATION GRID — values REAL, anchored to ablation TRAINING ranges ────
# rule: moderate = 10% beyond training-range edge, hard = 50% beyond
#   friction:  train U(0.4, 1.0)  -> eval 0.36 / 0.20   (direction: slippery)
#   mass add:  train U(-1, +3) kg -> eval +3.3 / +4.5   (direction: heavier)
#   motor:     train scale U(0.9, 1.1) -> eval 0.81 / 0.45 (direction: weaker)
#   noise:     train 1.0x stock bounds -> eval 1.1x / 1.5x (direction: noisier)
#   latency:   train U{0..4} PHYSICS steps (0-20 ms) -> eval 8 / 20
#              (40 ms / 100 ms; 1 physics step = 5 ms; 1 control step = 4)
# ORDERING LOAD-BEARING: nominal first (sanity gate fires on cell one).
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
MAX_EPISODE_STEPS = 1100    # true length 20/(0.005*4)=1000 + headroom;
                            # live=True in output = hit ceiling = bug flag

STOCK_BALLPARK_RETURN = None   # optional: fill from TensorBoard plateau


# ─────────────────────────────────────────────────────────────────────────────
# PART A — load a frozen policy
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(checkpoint_path, env):
    """File on disk -> deterministic callable (mean action; the learned
    actor_logstd served exploration, i.e. data collection for the optimizer —
    at eval there is no optimizer, so sigma is just measurement jitter and
    we grade the mean). Normalizer stats restore with the weights."""
    log_dir = os.path.dirname(checkpoint_path)
    with open(os.path.join(log_dir, "params", "agent.yaml"), "r") as f:
        agent_cfg = yaml.safe_load(f)   # VERIFY: safe_load vs python-object tags

    device = env.unwrapped.device if hasattr(env, "unwrapped") else env.device

    runner = OnPolicyRunner(RslRlVecEnvWrapper(env), agent_cfg, log_dir=None, device=device)
    # VERIFY: may need the RslRlVecEnvWrapper-wrapped env for construction
    runner.load(checkpoint_path)
    return runner.get_inference_policy(device=device)


# ─────────────────────────────────────────────────────────────────────────────
# PART B — the frozen-world factory
# ─────────────────────────────────────────────────────────────────────────────

def make_perturbed_env(perturb_type, value):
    """Training env with ALL DR off and exactly ONE property frozen at
    `value` (a value outside the ablation training ranges)."""
    env_cfg = UnitreeGo2FlatEnvCfg()
    env_cfg.scene.num_envs = N_EPISODES_PER_CELL

    # ── kill ALL randomization ──────────────────────────────────────────
    # (stock cfg's physics_material is ALREADY degenerate (0.8,0.8) — a
    # frozen nominal — so leaving it equals nulling; we null the samplers
    # and keep resets, same reasoning as v1)
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_external_force_torque = None
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "base_com"):
        env_cfg.events.base_com = None
    if hasattr(env_cfg.events, "actuator_gains"):
        env_cfg.events.actuator_gains = None
    env_cfg.observations.policy.enable_corruption = False
    # reset_base / reset_robot_joints stay ON — reset mechanism, not DR

    # tomorrow's mode: every cell shares the DelayedPD actuator (delay 0
    # unless this IS the latency cell) so eval matches how the six trained
    if USE_DELAYED_PD:
        delay = int(value) if perturb_type == "action_latency" else 0
        _delayed_pd_actuators(env_cfg, min_delay=delay, max_delay=delay)

    # ── re-add exactly ONE frozen term ──────────────────────────────────
    if perturb_type == "nominal":
        pass

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
                "mass_distribution_params": (value, value),  # kg added
                "operation": "add",
            },
        )

    elif perturb_type == "motor_strength":
        # frozen gain scale — same code path as the training DR term
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
        # scale every obs term's stock noise bounds by `value`
        env_cfg.observations.policy.enable_corruption = True
        policy_obs = env_cfg.observations.policy
        for term_name in ["base_lin_vel", "base_ang_vel", "projected_gravity",
                          "joint_pos", "joint_vel"]:   # VERIFY: full list in env.yaml
            term = getattr(policy_obs, term_name, None)
            if term is not None and getattr(term, "noise", None) is not None:
                term.noise.n_min = term.noise.n_min * value
                term.noise.n_max = term.noise.n_max * value

    elif perturb_type == "action_latency":
        pass  # delayed-PD mode handled above; stock mode wrapped below

    else:
        raise ValueError(f"unknown perturb_type: {perturb_type}")

    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=env_cfg)

    if perturb_type == "action_latency" and not USE_DELAYED_PD:
        # stock-actuator fallback (TODAY): wrapper works in CONTROL steps;
        # grid value is PHYSICS steps -> convert (decimation = 4)
        env = LatencyWrapper(env, delay_steps=max(1, int(value) // 4))

    return env


class LatencyWrapper:
    """Delay actions by k CONTROL steps via FIFO (stock-actuator mode only).
    Wrapper because latency transforms the action stream over time; queue
    because delay-by-k IS a length-k FIFO; __getattr__ keeps env.device /
    env.scene / env.command_manager reachable. Pre-fill = zeros = default
    joint targets (actions are offsets)."""

    def __init__(self, env, delay_steps):
        self.env = env
        self.k = delay_steps
        self._queue = None

    def reset(self, **kwargs):
        self._queue = None
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._queue is None:
            self._queue = [torch.zeros_like(action) for _ in range(self.k)]
        self._queue.append(action)
        return self.env.step(self._queue.pop(0))

    def __getattr__(self, name):
        return getattr(self.env, name)


# ─────────────────────────────────────────────────────────────────────────────
# PART C — roll out N episodes and measure
# ─────────────────────────────────────────────────────────────────────────────

def rollout_cell(policy, env, n_episodes):
    """Dict of per-env metric vectors. n_envs = n_episodes: each env's FIRST
    episode only. One shared clock; `live` gates every write — once an env's
    first episode ends (auto-reset starts its second), nothing it does can
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
            # VERIFY: signature (raw env 5-tuple vs wrapped merged dones)

            reward     = reward.reshape(-1)          # (n,1) would broadcast
            terminated = terminated.reshape(-1).bool()
            truncated  = truncated.reshape(-1).bool()

            cmd = uenv.command_manager.get_command("base_velocity")[:, :2]
            vel = asset.data.root_lin_vel_b[:, :2]
            step_err = torch.norm(cmd - vel, dim=1)  # raw error, not exp kernel

            ep_return += reward * live
            ep_len    += live
            track_err += step_err * live

            # terminated = base contact = fell; truncated = timeout = survived
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

    for perturbation in [PERTURBATION_GRID[int(sys.argv[1])]]:
        perturb_type = perturbation["perturb_type"]
        severity     = perturbation["severity"]
        value        = perturbation["value"]

        env = make_perturbed_env(perturb_type, value)
        # VERIFY: seeded-reset determinism; sequential env creation per process

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

            # known-answer gate: policies in frozen-nominal must not fall
            if perturb_type == "nominal":
                if fall_rate > 0.05:
                    raise RuntimeError(
                        f"{policy_name} falls in NOMINAL conditions "
                        f"(fall_rate={fall_rate:.2f}) — pipeline broken, "
                        "not physics. Stop and debug.")
                if STOCK_BALLPARK_RETURN and mean_ret < 0.5 * STOCK_BALLPARK_RETURN:
                    raise RuntimeError(
                        f"nominal return {mean_ret:.1f} far below reference "
                        f"(~{STOCK_BALLPARK_RETURN}) — normalizer/env mismatch?")

        env.close()

    simulation_app.close()


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# TODAY'S DEFINITION OF DONE: this file x stock_baseline x full grid ->
# results_raw.csv, nominal gate passes, hard cells degraded, stuck=0.
# TOMORROW: USE_DELAYED_PD=True, six POLICIES paths, rerun -> the dataset.
# ─────────────────────────────────────────────────────────────────────────────
