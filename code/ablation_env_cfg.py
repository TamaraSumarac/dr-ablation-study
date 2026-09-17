"""
ablation_env_cfg.py — domain-randomization ablation configs for the Unitree Go2

Defines a full-DR baseline environment and five one-term-out ablations, and
registers each as a gym task so the stock Isaac Lab / RSL-RL train.py can
launch them by name (see run_ablation.sh).

INSTALL
  Copy this file into the Isaac Lab checkout at
    source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/
  and append to that folder's __init__.py:
    from . import ablation_env_cfg

WHAT THE STOCK GO2 FLAT ENV RANDOMIZES
  Isaac Lab's default Isaac-Velocity-Flat-Unitree-Go2-v0 config randomizes
  only base mass (add U(-1, +3) kg) and per-step sensor noise on the policy
  observations (base_lin_vel +/-0.1, base_ang_vel +/-0.2, projected_gravity
  +/-0.05, joint_pos +/-0.01, joint_vel +/-1.5). Its friction term is present
  but degenerate (static 0.8, dynamic 0.6, i.e. frozen). There is no motor-gain
  or action-latency randomization and no push disturbance.
  The full-DR baseline below therefore widens friction and ADDS motor-gain and
  action-latency randomization on top of the stock mass + sensor-noise terms.

DESIGN INVARIANTS
  - every run: same seed, same iterations, same reward and curriculum
  - all six configs use DelayedPDActuator with identical gains and limits, so
    the actuator model is constant across runs; only the delay range differs
  - each ablation differs from the full-DR baseline by exactly ONE term
"""

import gymnasium as gym

import isaaclab.envs.mdp as mdp
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass

from .flat_env_cfg import UnitreeGo2FlatEnvCfg
from . import agents


# ─────────────────────────────────────────────────────────────────────────────
# helper: replace every actuator group with a DelayedPDActuator that keeps the
# stock gains and limits and adds a per-env command delay.
# Delay is in PHYSICS steps: sim.dt = 0.005 s -> 1 step = 5 ms; with
# decimation 4, one control step = 4 physics steps = 20 ms.
# ─────────────────────────────────────────────────────────────────────────────

def _delayed_pd_actuators(cfg, min_delay, max_delay):
    new_actuators = {}
    for name, act in cfg.scene.robot.actuators.items():
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
# full-DR baseline: all five terms ON
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class Go2AblationFullDrCfg(UnitreeGo2FlatEnvCfg):
    """friction + mass + motor gains + sensor noise + action latency."""

    def __post_init__(self):
        super().__post_init__()

        # 1. friction DR — widen the stock frozen (0.8, 0.8) to U(0.4, 1.0);
        #    dynamic friction keeps the stock 0.75 ratio.
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.75)

        # 2. mass DR — stock add U(-1, +3) kg: kept as-is.

        # 3. motor-strength DR — new: scale PD stiffness and damping per env.
        self.events.actuator_gains = EventTerm(
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

        # 4. sensor-noise DR — stock per-step observation noise: kept as-is.

        # 5. action-latency DR — new: command delay ~ U{0..4} physics steps
        #    (0-20 ms), resampled per env at reset.
        _delayed_pd_actuators(self, min_delay=0, max_delay=4)


# ─────────────────────────────────────────────────────────────────────────────
# five one-term-out ablations (each undoes exactly ONE term of the baseline)
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class Go2AblationNoFrictionCfg(Go2AblationFullDrCfg):
    def __post_init__(self):
        super().__post_init__()
        # back to the stock frozen nominal
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
        # same actuator model, delay frozen at zero
        _delayed_pd_actuators(self, min_delay=0, max_delay=0)


# ─────────────────────────────────────────────────────────────────────────────
# task registration — names consumed by run_ablation.sh
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
    gym.register(
        id=_task_name,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _cfg_class,
            "rsl_rl_cfg_entry_point":
                f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        },
    )
