# Which domain randomizations actually matter? A one-knob-at-a-time study on a quadruped

Looking at robot demonstrations, it was pretty funny to see how violent some of them have become - robots getting kicked from every direction, pushed over, and even having their legs or arms damaged. At the same time, these demonstrations are incredibly impressive because they highlight just how robust modern locomotion policies are, allowing robots to recover from very large perturbations. They also made me wonder: which perturbations do robots actually need to experience during training to become robust in the real world? To explore that question, I decided to train the Unitree Go2 quadruped in simulation and test whether training with certain perturbations improves robustness to others.

[watch a robot get bullied but still adapt](https://www.youtube.com/watch?v=JQAfxp-FB0I)

In this project, I trained six locomotion policies for the Unitree Go2 in Isaac Lab, identical in every respect except that each policy omitted exactly one domain randomization during training. I then evaluated all six in eleven frozen test environments with physics the robots had never encountered before. **TL;DR: All six policies solve the nominal task equally well. Under unseen conditions, however, removing a single training randomization can dramatically reduce robustness, even to perturbations it was never designed to model.**

**Key takeaways:**
- Randomizing action latency by just 0–20 ms during training made the policy robust to delays of up to 100 ms at evaluation - 5× beyond the training range. The policy trained without it failed under the same perturbation, with over 60% of robots collapsing, often showcasing "jumping" behavior.
- The worst-performing case in the grid revealed that friction randomization and motor strength are tightly coupled: the policy trained without friction variation was *the only one that failed 100% of the time* under weak motors (low torque). Every step is a force chain: motor torque → joints → foot–ground contact, so weak motors limit force generation at the joints, while low friction limits force transmission at the ground. Training on one can therefore improve robustness to the other.
- *Mass perturbations fail gracefully.* The policy trained without mass randomization carries a +30% payload without a single fall, instead exhibiting a persistent tracking bias. This looks like a calibration error, not an instability: consistent with the policy having learned a fixed "force-to-acceleration" mapping during training, so that with the true mass changed underneath it (a = F/m with the wrong m), every command produces systematically less acceleration than expected. The failure gets uglier only in combination: when the no mass randomization policy faces weak motors, the videos show robots sagging and freezing in place, but still upright - a paralysis mode the fall-rate metric can't see.
- *Weak motors and latency fail catastrophically, but for different reasons.* Weak motors reduce the robot's ability to generate the forces needed for locomotion: under severe torque limits, the robots visibly sag, struggle to support and propel themselves, and eventually fail. Latency, by contrast, leaves the robot physically capable but delays its response to changes in the world. At large delays (100 ms in experiments), the policy is forced to act on increasingly stale observations, making it much harder to coordinate timely corrections. One perturbation limits force generation; the other limits how effectively those forces can be coordinated.
- *Sensor-noise randomization had the smallest effect* of any training randomization, even when evaluated at 1.5× the training noise amplitude. Notably, Isaac Lab's observation corruption already models state-estimation uncertainty rather than raw sensor noise, injecting up to ±0.1 m/s error on estimated base linear velocity and ±0.2 rad/s on estimated base angular velocity - magnitudes that already exceed the intrinsic noise of modern MEMS IMUs. This project therefore probes robustness to observation errors substantially beyond the underlying sensor noise, although not necessarily beyond the full range of real-world state-estimation errors. One important limitation is that only the amplitude of the injected noise was increased; the structure stayed the same. Real-world state-estimation failures often introduce biases, drift, correlated errors rather than simply larger white noise, so these results speak to amplitude robustness, not necessarily robustness to these more structured failure modes.

| Episode return (↑ better) | Tracking error (↓ better) | Fall rate (↓ better) |
|---|---|---|
| ![Episode return](figures/heatmap_ep_return.png) | ![Tracking error](figures/heatmap_track_err.png) | ![Fall rate](figures/heatmap_fell.png) |

*Rows: perturbation × severity (nominal first, then moderate → hard per type). Columns: policy — each trained missing exactly one randomization. Return and tracking error are shown as fractions of the full-DR baseline in the same row; fall rate is absolute (baseline's fall rate is zero in most rows, so a ratio would be undefined). Red borders: gap vs baseline exceeds 2σ (SEMs combined in quadrature; binomial for fall rate). All results can be found in [`figures/`](figures/).*

<!-- ![Episode return heatmap](figures/heatmap_ep_return.png)

*Rows: perturbation × severity. Columns: policy (each trained missing one randomization). Values: episode return as a fraction of the full-DR baseline in the same row. Red borders: gap vs baseline exceeds 2σ. Three more heatmaps (tracking error, fall rate, episode length) in [`figures/`](figures/).* -->

---

## **What this is**

The goal of this project is to understand how much each individual domain randomization contributes to a robot's ability to handle conditions it never trained on. Domain randomization (DR) is the standard recipe for sim-to-real transfer: by varying the training environment, the policy learns to avoid overfitting to a single simulator. However, modern DR recipes often combine many randomizations, making it difficult to tell which ones meaningfully improve robustness and which provide little benefit. To isolate their effects, this project removes one randomization at a time while keeping everything else fixed (same PPO, hyperparameters, random seed, and training iterations), then evaluates where each resulting policy succeeds or fails under a wide range of perturbations.

## **Method**

**Six policies**, trained using rsl-rl PPO on `Isaac-Velocity-Flat-Unitree-Go2` (1000 iterations, seed 42, all identical except one config term):

| Policy | Trained without |
|---|---|
| `baseline_full_dr` | nothing — all five randomizations on |
| `no_friction` | friction randomization (frozen at 0.8) |
| `no_mass` | base-mass randomization |
| `no_motor_strength` | PD-gain (motor strength) randomization |
| `no_sensor_noise` | observation noise |
| `no_action_latency` | action-delay randomization |

The full-DR baseline extends Isaac Lab default Go2 locomotion environment, which randomizes only base mass and sensor noise by default. Friction randomization (U(0.4, 1.0)), motor-gain scaling (U(0.9, 1.1)), and action latency (0–4 physics steps, or 0–20 ms, implemented through a delayed-PD actuator) were added so that each ablation removes a genuine source of variability. All six policies use the same delayed-PD actuator model; the no-latency policy simply fixes the delay at zero, ensuring that any observed differences can be attributed to latency rather than differences in the actuator model implementation.

**Eleven eval worlds:** Nominal (the training environment with all randomization off), plus five perturbation types × two severities. Two things matter here:

1. Eval values are **fixed, not randomized**. Every heatmap cell corresponds to a single frozen, deterministic world. During training, physics parameters are sampled from each policy's training distribution every episode; during evaluation, each parameter is pinned to one value and held fixed throughout the rollout.
2. Severities were chosen relative to the **edge of the training range, in the harmful direction per knob**: moderate = 10% beyond the edge, hard = 50% beyond. So friction goes *down* (0.36 / 0.20), mass goes *up* (+3.3 / +4.5 kg — about 22% / 30% of the Go2's body mass), motor gains go *down* (0.81× / 0.45×), and noise scales *up* (1.1× / 1.5×). Latency is the one exception to the percentage rule: delays are quantized to 5 ms physics steps, so 10% beyond the 20 ms training edge rounds back *into* the trained range, and even 50% beyond (30 ms) didn't look like it would produce a measurable effect — so its severities instead use 2× and 5× the trained maximum (40 ms / 100 ms). Every heatmap cell therefore lies outside the training distribution seen by any policy.

**Three metrics:** 30 independent robot rollouts (episodes) were collected for every heatmap cell (6 policies × 11 worlds × 30 = 1,980 episodes): episode return, tracking error (mean per-step velocity-command error, over lived steps), and fall rate (fraction of rollouts ending in a base-contact termination). No single metric tells the story - return conflates task success with style penalties, tracking error is blind to death, fall rate is blind to everything except death.

**Significance:** For each heatmap cell, I compare the policy mean against the full-DR baseline in the same evaluation world (same perturbation), combining uncertainty from both estimates using their standard errors (SEM = std/√30, added in quadrature; binomial SEM for fall rate). Gaps larger than 2σ receive a red border in the heatmaps. This is intended as a visual screening heuristic rather than a formal hypothesis test (see caveats).

---
## **Results**

### **All policies look the same in nominal environment**

The top row of each heatmap shows that all six policies perform nearly identically under nominal conditions: return ratios range from 0.99 to 1.01, with no differences exceeding the 2σ threshold. The effect of domain randomization is therefore not visible in the environment used for training; it emerges only when the policy is evaluated under out-of-distribution perturbations.

### **The diagonal: removing X hurts under X — but differently per knob**

| Randomization removed | What happens under its own perturbation | Failure mode |
|---|---|---|
| action latency | return ratio 0.05 (hard mode), tracking error ~1.1-1.4x baseline, 63% falls at 100 ms | catastrophic |
| motor strength | return ratio 0.71 (hard mode), tracking error within 10% of baseline, 63% falls under 0.45× gains | catastrophic |
| mass | return barely moves (0.97-0.98 both severities), tracking error 1.6× baseline, zero falls, even at +30% body mass | graceful |
| friction | return barely moves (0.96, both severities); tracking error runs 1.23× baseline; zero falls even at 0.20 friction | mild |
| sensor noise | return barely moves (0.96-0.97 both severities), tracking error in line with baseline, zero falls | negligible |

As a physicist, I found the mass perturbation particularly satisfying. A policy trained without mass randomization carries a +30% payload without falling, instead exhibiting a systematic tracking bias. If the policy has learned a fixed force-to-acceleration mapping during training, increasing the mass simply makes every command produce less acceleration than expected (a=F/m with the wrong m). Weak motors and latency, by contrast, directly degrade locomotion stability and ultimately lead to falls.

For the motor knob, "weak motors" has a precise meaning. Each joint runs a PD controller - torque proportional to position error and its rate - so a joint of inertia $J$ with error $e = q - q_\text{target}$ obeys a damped-oscillator equation:

$$J\ddot{e} + K_d\dot{e} + K_p e = 0
\quad\Longrightarrow\quad
\omega_0 = \sqrt{K_p/J},\qquad
\zeta = \frac{K_d}{2\sqrt{K_p J}}$$
<!-- 
A surprise for me here, coming from classical control: I expected well-designed joints to sit near critical damping ($\zeta \approx 1$) — reach the target, no oscillation. Go2's gains ($K_p = 25$, $K_d = 0.5$) put the joints far from that: for plausible leg inertias, $\zeta \approx 1$ lands well below 1, heavily underdamped. This turns out to be the deliberate convention in legged RL — low, compliant gains are favored for sim-to-real transfer and for surviving contact — and it works because the PD loop is not the whole controller: the RL policy sits above it, retargeting joints at 50 Hz (dt = 5ms, 4 physics steps per policy), effectively acting as the outer damper/corrector that a classical servo would build in via $K_d$. Soft springs below, intelligence above. (Which also foreshadows the latency result: delay poisons exactly the loop that stability was delegated to.) -->

One aspect I found particularly interesting from a classical controls perspective is that these gains are nowhere near critical damping ($\zeta \approx 1$), where one would normally expect a servo to settle as quickly as possible without oscillation. With Go2's default gains ($K_p = 25$, $K_d = 0.5$), plausible leg inertias ($J \approx 0.01\text{--}0.05~\mathrm{kg\,m^2}$) place the joints well into the underdamped regime. This seems to be intentional: legged RL typically favors compliant, low-gain joint controllers for improved contact robustness and sim-to-real transfer. The PD controller is therefore not expected to stabilize the robot on its own. Instead, the learned policy runs above it at 50 Hz (one policy step every four 5 ms physics steps), continually updating joint targets and effectively providing the higher-level damping and correction that a classical controller would otherwise achieve through larger derivative gains. Soft springs below, intelligence above—a design choice that also suggests a possible explanation for the latency results: if the policy provides the continual corrections that compensate for the compliant PD controller, then delaying those corrections would be expected to reduce stability.

For $\zeta < 1$ (the underdamped case — Go2's joints), a disturbed joint rings back toward its target as a decaying oscillation:

$$e(t) = A\,e^{-\zeta\omega_0 t}\cos\!\big(\omega_d t + \phi\big),
\qquad \omega_d = \omega_0\sqrt{1-\zeta^2}$$

where $A$ and $\phi$ are set by the initial disturbance. The two parameters play distinct roles: $\omega_0$ sets the overall timescale of the response, while $\zeta$ determines how quickly oscillations decay relative to how quickly they occur.

The perturbation scales both gains by $s$ ($K_p \to sK_p$, $K_d \to sK_d$), and every term in that motion degrades at once:

$$\omega_0 \to \sqrt{s}\,\omega_0, \qquad
\zeta \to \sqrt{s}\,\zeta, \qquad
\zeta\omega_0 \to s\,\zeta\omega_0, \qquad
e_\text{sag} = \frac{\tau_\text{gravity}}{sK_p} \;\propto\; \frac{1}{s}$$

A single scaling factor therefore degrades every aspect of the joint dynamics simultaneously. At $s = 0.45$, the natural frequency falls to roughly two-thirds of its original value, reducing tracking bandwidth. The damping ratio decreases by the same factor, so oscillations persist longer, while the exponential decay rate drops in proportion to s, causing disturbances to take more than twice as long to settle. Finally, the static position error required to balance gravity more than doubles, producing the visible joint sag in the evaluation videos. It is a remarkably compact perturbation: slower response, weaker damping, and larger steady-state error, all resulting from scaling one parameter.

### **Motor strength and friction are coupled**

The worst cell in the entire grid is not on the diagonal. It's `no_friction` × weak motors: return ratio 0.12 and a **100% fall rate** — every single robot falls, the only cell where that happens.

The coupling is intuitive from the mechanics of locomotion. Every step is a force chain:

$
\text{motor torque}
\;\longrightarrow\;
\text{joint forces}
\;\longrightarrow\;
\text{foot--ground contact forces}
$
Weak motors limit how much force can be generated at the joints, while low friction limits how much of that force can be transmitted to the ground. Although these perturbations act at different points in the chain, both ultimately constrain the forces available to stabilize and propel the robot. That picture matches the experiments remarkably well. The policy trained without friction randomization is the only one that never experienced limited force transmission during training, and it is also the only policy that completely collapses when motor strength is reduced. In contrast, every policy trained with friction randomization retains at least partial robustness despite the weak motors. The experiments therefore reveal an unexpected transfer: experience with friction variation improves robustness to weak actuators, even though the perturbations act on different parts of the locomotion system.


### **A third failure mode hidden by the metrics**

The videos also revealed a behavior that the quantitative metrics cannot distinguish. In `no_mass` × weak motors, the robots neither fall nor walk - they freeze, remaining sagged and effectively motionless until the episode times out. From the metrics' perspective, this looks identical to any other case of poor tracking: no fall, but large tracking error. The videos show that it is qualitatively different. Paralysis emerges as a third failure mode, distinct from both falling and degraded locomotion, and one that only visual inspection reveals. If I were to extend the evaluation, a simple "distance traveled" metric would separate it from ordinary tracking failures.

### **What the videos showed** ([`videos/`](videos/))

- **Baseline, nominal:** constant tiny, quick leg adjustments with continuous high-frequency micro-corrections.
- **Baseline, 100 ms latency:** slower leg motions, larger corrective steps, and noticeable body tilting. Most robots remain upright, though some fall.
- **No-latency policy, 100 ms latency:** robots appear to hop in place, struggling to follow the commanded velocity.
- **No-motor-strength policy on low friction:** robots sag, slip, and ultimately fall.
- **No-mass policy on weak motor:** robots sag and freeze in place, remaining largely motionless until the episode ends.

## **Caveats**

- **One robot, one task, flat terrain.** This is a sim-to-sim study, not sim-to-real - no real robot was harmed or, unfortunately, used for the experiment.
- **Limited statistical power.** Each policy was trained from a single random seed and evaluated with a single evaluation seed (30 episodes per heatmap cell). The 2σ threshold is therefore an informal screening tool rather than a formal significance test; a more rigorous analysis would train multiple independent policies per configuration.
- **Tracking error is conditioned on survival.** Once a robot falls, it stops accumulating tracking error, so high-fall-rate conditions can appear artificially better than they are. Comparisons involving the latency perturbation should therefore be interpreted with this censoring effect in mind.
- **Perturbation severities are design choices.** The null result for sensor noise, for example, may simply indicate that even 1.5× Isaac Lab's default observation corruption is not a sufficiently severe probe.
- **Not all failures look the same.** Some conditions produce paralysis rather than falls: the robot remains upright but effectively motionless until the episode ends. Fall rate alone therefore understates the severity of these failures.

## **Things to explore next**

**Testing mechanisms this study proposed**:

- **Log joint trajectories under increasing latency** to distinguish delayed-feedback instability from general policy degradation. True delayed-loop instability should produce a dominant oscillation frequency and a critical delay beyond which oscillations emerge.
- **Independent Kp/Kd scaling under latency**, to separate two competing hypotheses about delay tolerance: scaling both gains up makes joints execute stale commands more faithfully and should erode phase margin (worse), while raising damping alone adds velocity resistance independent of command staleness (plausibly better). Four eval cells would settle it.

**Sharpening the measurements:**

- **Multiple training seeds per config**, to turn the informal 2σ screen into real error bars.
- **Evaluate structured observation errors**, such as constant biases, slowly drifting biases, delayed state estimates, or correlated noise, to determine whether the weak sensor-noise result reflects the specific corruption model rather than observation robustness in general.
- **A distance-traveled metric**, to separate the paralysis failure mode (visible only on video) from degraded-but-walking tracking.

**Extending the setting:**

- **Rough terrain**, where friction and contact effects should couple more strongly.
- **The real thing**: the same one-knob methodology on physical hardware (an SO-101 arm is the planned sequel) — sim-to-real instead of sim-to-sim.

---
## The infrastructure detour

Two infrastructure issues were unexpectedly time-consuming. First, Isaac Sim consistently crashed inside containerized cloud deployments due to a glibc/container-runtime incompatibility (`The futex facility returned an unexpected error code`); running on a full virtual machine resolved the problem. Second, rendering required a complete graphics stack rather than CUDA alone: the correct NVIDIA userspace libraries, a Vulkan ICD, and a recent Vulkan loader. The full setup and debugging notes are documented in [`docs/cloud_setup_notes.md`](docs/cloud_setup_notes.md).

<!-- Running Isaac Sim on cloud infrastructure turned out to have a few surprises. Containerized deployments consistently crashed on the first physics step (`The futex facility returned an unexpected error code`) due to a glibc/container-runtime incompatibility that could not be fixed from inside the container. Moving to a full virtual machine resolved the issue.

Rendering the evaluation videos exposed a second class of issues. Many cloud GPU images provide CUDA but not a complete graphics stack, and Isaac Sim requires more than compute alone. Rendering ultimately depended on aligning three pieces: the correct NVIDIA userspace libraries, a Vulkan ICD, and a sufficiently recent Vulkan loader. The complete setup, along with the debugging steps that made it work, is documented in [`docs/cloud_setup_notes.md`](docs/cloud_setup_notes.md). -->

## Repo map

```
code/      eval_sweep.py (the 66-cell evaluation), ablation_env_cfg.py (the six
           training configs), run_ablation.sh (training launcher),
           record_cells.py (video recorder), analysis notebook
figures/   four heatmaps (return, tracking error, fall rate, episode length),
           significance-bordered
results/   results_week3_sweep.csv — the raw dataset: 1,980 episodes
videos/    selected clips 
docs/      cloud setup notes: full rebuild recipe + the Vulkan fix
```