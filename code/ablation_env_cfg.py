"""
ablation_env_cfg.py — DR-ablation configs (rebuilt against the REAL
stock-baseline env.yaml, 2026-07-23)

INSTALL: copy this file into
  /workspace/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/
      locomotion/velocity/config/go2/ablation_env_cfg.py
then append to that folder's __init__.py:
  from . import ablation_env_cfg
Stock train.py can then launch every task by name (see run_ablation.sh).

GROUND TRUTH (from logs/.../2026-07-16_21-33-44/params/env.yaml):
  stock training DR was ONLY: mass add U(-1,+3) kg + sensor noise
  (base_lin_vel +/-0.1, base_ang_vel +/-0.2, projected_gravity +/-0.05).
  Friction was FROZEN at 0.8/0.6 (degenerate range). No motor DR, no
  latency, push_robot null, force/torque term a (0,0) no-op.
  => the FULL-DR baseline below ADDS friction, motor-gain, and latency DR.

DESIGN INVARIANTS (the experiment dies if violated):
  - every run: same seed, same iterations, same reward/curriculum
  - ALL SIX configs use DelayedPDActuator (identical gains/limits) so the
    actuator model is constant across runs; only the delay range differs.
  - each ablation differs from FullDr by exactly ONE term.

VERIFY-ON-POD before launching (5 min):
  V1. Import path + field names of DelayedPDActuatorCfg
      (isaaclab.actuators; fields min_delay/max_delay in PHYSICS steps).
  V2. mdp.randomize_actuator_gains exists with these param names.
  V3. Actuator attr names copied in _delayed_pd_actuators (effort_limit,
      velocity_limit, stiffness, damping) exist on Go2's stock actuator cfg
      (print env_cfg.scene.robot.actuators to see keys + fields).
  V4. gym.register kwargs match the pattern in this folder's __init__.py
      (env_cfg_entry_point / rsl_rl_cfg_entry_point strings).
"""

import gymnasium as gym

import isaaclab.envs.mdp as mdp
from isaaclab.actuators import DelayedPDActuatorCfg          # V1
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass

from .flat_env_cfg import UnitreeGo2FlatEnvCfg
from . import agents


# ─────────────────────────────────────────────────────────────────────────────
# helper: swap every actuator group for DelayedPD with IDENTICAL parameters
# (delay in PHYSICS steps; sim.dt = 0.005 s -> 1 step = 5 ms; decimation 4
#  -> 1 control step = 4 physics steps = 20 ms)
# ─────────────────────────────────────────────────────────────────────────────

def _delayed_pd_actuators(cfg, min_delay, max_delay):
    new_actuators = {}
    for name, act in cfg.scene.robot.actuators.items():      # V3
        new_actuators[name] = DelayedPDActuatorCfg(
            joint_names_expr=act.joint_names_expr,
            effort_limit=act.effort_limit,
            velocity_limit=act.velocity_limit,
            stiffness=act.stiffness,
            damping=act.damping,
            min_delay=min_delay,
            max_delay=max_delay,
        )
    cfg.scene.robot.actuators = new_actuators


# ─────────────────────────────────────────────────────────────────────────────
# the FULL-DR baseline: all five terms ON
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class Go2AblationFullDrCfg(UnitreeGo2FlatEnvCfg):
    """friction + mass + motor gains + sensor noise + action latency."""

    def __post_init__(self):
        super().__post_init__()

        # 1. friction DR — stock was frozen (0.8, 0.8); widen to the
        #    Colab-proven range. dynamic keeps the 0.75 ratio.
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.75)

        # 2. mass DR — stock add U(-1, +3) kg: keep as-is (no change needed)

        # 3. motor-strength DR — NEW: scale PD gains per env
        self.events.actuator_gains = EventTerm(                # V2
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # 4. sensor-noise DR — stock ON (+/-0.1 / +/-0.2 / +/-0.05): keep

        # 5. action-latency DR — NEW: delay ~ U{0..4} physics steps (0-20 ms),
        #    resampled per env. Actuator model identical across ALL configs.
        _delayed_pd_actuators(self, min_delay=0, max_delay=4)


# ─────────────────────────────────────────────────────────────────────────────
# five one-knob-out ablations (each undoes exactly ONE term of FullDr)
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class Go2AblationNoFrictionCfg(Go2AblationFullDrCfg):
    def __post_init__(self):
        super().__post_init__()
        # back to frozen nominal (what the stock baseline actually had)
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)


@configclass
class Go2AblationNoMassCfg(Go2AblationFullDrCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.add_base_mass = None          # no payload randomization


@configclass
class Go2AblationNoMotorStrengthCfg(Go2AblationFullDrCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.actuator_gains = None         # gains fixed at nominal


@configclass
class Go2AblationNoSensorNoiseCfg(Go2AblationFullDrCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class Go2AblationNoActionLatencyCfg(Go2AblationFullDrCfg):
    def __post_init__(self):
        super().__post_init__()
        # SAME actuator model, delay frozen at zero — no confound
        _delayed_pd_actuators(self, min_delay=0, max_delay=0)


# ─────────────────────────────────────────────────────────────────────────────
# task registration — names consumed by run_ablation.sh and eval_sweep POLICIES
# ─────────────────────────────────────────────────────────────────────────────

_ABLATION_TASKS = {
    "Isaac-Go2-Ablation-FullDr-v0":          Go2AblationFullDrCfg,
    "Isaac-Go2-Ablation-NoFriction-v0":      Go2AblationNoFrictionCfg,
    "Isaac-Go2-Ablation-NoMass-v0":          Go2AblationNoMassCfg,
    "Isaac-Go2-Ablation-NoMotorStrength-v0": Go2AblationNoMotorStrengthCfg,
    "Isaac-Go2-Ablation-NoSensorNoise-v0":   Go2AblationNoSensorNoiseCfg,
    "Isaac-Go2-Ablation-NoActionLatency-v0": Go2AblationNoActionLatencyCfg,
}

for _task_name, _cfg_class in _ABLATION_TASKS.items():
    gym.register(                                              # V4: crib exact
        id=_task_name,                                         # kwargs from this
        entry_point="isaaclab.envs:ManagerBasedRLEnv",         # folder's
        disable_env_checker=True,                              # __init__.py
        kwargs={
            "env_cfg_entry_point": _cfg_class,
            "rsl_rl_cfg_entry_point":
                f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        },
    )
