
# 6-DOF Satellite Proximity Operations & Docking under Realistic Disturbances

A high-fidelity **Gymnasium** environment and **Deep Reinforcement Learning (DRL)** framework designed for Autonomous Satellite Proximity Operations, Inspection, and Docking (On-Orbit Servicing, Assembly, and Manufacturing — **OSAM**).

This project replaces oversimplified linearized equations of motion with dual-spacecraft numerical orbital propagation in the **Earth-Centered Inertial (ECI)** frame under **Earth oblateness ($J_2$) perturbations**, fully coupled with **6-DOF rigid-body rotational dynamics** and actuator/sensor imperfections. It incorporates an automated **Curriculum Learning pipeline** and **Potential-Based Reward Shaping** to bypass the sparse-reward exploration trap in orbital mechanics.

---

## Key Features

- **Nonlinear ECI Propagation + $J_2$ Perturbations**: Simulates the Chief and Deputy spacecraft independently in the inertial frame via numerical integration, capturing non-spherical gravity perturbations without Clohessy-Wiltshire (CW) linearization errors.
- **RSW / Hill (LVLH) Exact Kinematics**: Projects relative states into the standard RSW frame ($x$: Radial Outward, $y$: Along-Track, $z$: Cross-Track Orbit-Normal) using the exact analytical derivative of the transformation matrix $\dot{R}_{ECI \to LVLH}$.
- **Fully Coupled 6-DOF Dynamics**: 
  - Rigid-body attitude modeled via Euler's rotational equations and scalar-first unit quaternion kinematics.
  - Thrust applied in the Deputy **body frame** is rotated into the inertial frame via the instantaneous attitude quaternion, forcing the agent to steer its attitude to direct translational maneuvers.
- **Realistic Imperfections & Disturbances**:
  - Sensor Gaussian noise on relative positions, velocities, and orientation.
  - Rate gyro bias random-walk and angle random-walk.
  - Actuator uncertainties: thruster scale factor errors and angular misalignments.
- **Automatic Curriculum Learning**: Progressively scales initial standoff distances (from $5\text{ m}$ to $200\text{ m}$) and tumble rates as the policy exceeds success thresholds.
- **Flight Constraints**: Line-of-sight (LOS) approach cone corridor, spherical Keep-Out Zone (KOZ), actuator saturation, and dwell-time terminal docking verification.

---

## Mathematical & Physical Modeling

### 1. Translational Dynamics ($J_2$ Perturbed Orbit)
Both spacecraft are propagated in the Earth-Centered Inertial (ECI, J2000) frame:

$$\ddot{\mathbf{r}} = -\frac{\mu}{\Vert{}\mathbf{r}\Vert{}^3}\mathbf{r} + \mathbf{a}_{J_2} + \frac{\mathbf{F}_{\text{thrust}}}{m}$$

The Earth oblateness ($J_2$) perturbation acceleration is evaluated as:

$$\mathbf{a}_{J_2} = -\frac{3}{2}\frac{J_2 \mu R_{\oplus}^2}{\Vert{}\mathbf{r}\Vert{}^5} \begin{bmatrix} x \left(1 - 5\frac{z^2}{\Vert{}\mathbf{r}\Vert{}^2}\right) \\ y \left(1 - 5\frac{z^2}{\Vert{}\mathbf{r}\Vert{}^2}\right) \\ z \left(3 - 5\frac{z^2}{\Vert{}\mathbf{r}\Vert{}^2}\right) \end{bmatrix}$$

### 2. Relative Kinematics (RSW / Hill Frame)
The relative position and velocity in the Chief-centered rotating RSW frame are given by:

$$\mathbf{r}_{\text{rel}}^{\text{LVLH}} = R_{ECI \to LVLH} (\mathbf{r}_{\text{dep}} - \mathbf{r}_{\text{chief}})$$

$$\mathbf{v}_{\text{rel}}^{\text{LVLH}} = R_{ECI \to LVLH} (\mathbf{v}_{\text{dep}} - \mathbf{v}_{\text{chief}}) + \dot{R}_{ECI \to LVLH} (\mathbf{r}_{\text{dep}} - \mathbf{r}_{\text{chief}})$$

where the RSW basis triad is dynamically constructed as:

$$\hat{\mathbf{x}} = \frac{\mathbf{r}_c}{\Vert{}\mathbf{r}_c\Vert{}}, \quad \hat{\mathbf{z}} = \frac{\mathbf{r}_c \times \mathbf{v}_c}{\Vert{}\mathbf{r}_c \times \mathbf{v}_c\Vert{}}, \quad \hat{\mathbf{y}} = \hat{\mathbf{z}} \times \hat{\mathbf{x}}$$

### 3. Rotational Kinematics & Rigid-Body Dynamics
Attitude kinematics are propagated using unit quaternions $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$:

$$\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega} \end{bmatrix}$$

The Deputy's body angular rates $\boldsymbol{\omega}$ follow Euler's equations:

$$I \dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times (I \boldsymbol{\omega}) = \boldsymbol{\tau}_{\text{ctrl}} + \boldsymbol{\tau}_{\text{ext}}$$

### 4. Translation-Attitude Coupling
Thrust forces $\mathbf{F}_b$ commanded in the Deputy body frame rotate into inertial coordinates via the direction cosine matrix $C_{b \to ECI}(\mathbf{q})$:

$$\mathbf{F}_{ECI} = C_{b \to ECI}(\mathbf{q}) \cdot \mathbf{F}_b$$

---

## Environment Specifications

### Observation Space (`Box(15,)`)
| Index | Variable | Units | Description |
|:---:|:---:|:---:|:---|
| `0..2` | $\mathbf{r}_{\text{rel}}$ | $\text{m}$ | Relative position in Chief RSW frame $[x, y, z]$ |
| `3..5` | $\mathbf{v}_{\text{rel}}$ | $\text{m/s}$ | Relative velocity in Chief RSW frame $[v_x, v_y, v_z]$ |
| `6..9` | $\mathbf{q}_{\text{rel}}$ | - | Relative quaternion $[q_w, q_x, q_y, q_z]$ (RSW to Body) |
| `10..12`| $\boldsymbol{\omega}_{\text{rel}}$ | $\text{rad/s}$ | Relative angular rate in body frame $[\omega_x, \omega_y, \omega_z]$ |
| `13` | $m_{\text{fuel}} / m_0$ | - | Normalized remaining propellant fraction |
| `14` | $1 - (t / t_{\max})$ | - | Normalized remaining mission time |

### Action Space (`Box(6,)` in $[-1, 1]$)
| Index | Variable | Scale | Description |
|:---:|:---:|:---:|:---|
| `0..2` | $\mathbf{u}_F$ | $\pm 10.0\text{ N}$ | Commanded 3-axis thrust in Deputy body frame |
| `3..5` | $\mathbf{u}_\tau$ | $\pm 1.0\text{ N}\cdot\text{m}$ | Commanded 3-axis torque in Deputy body frame |

### Curriculum Progression
| Stage | Standoff Range ($y$) | Cross-Track / Radial Bounds ($x, z$) | Max Initial Tumble |
|:---|:---:|:---:|:---:|
| **Level 1 (Terminal)** | $5 - 15\text{ m}$ | $\pm 1\text{ m}$ | $\pm 15^\circ,\; 0.005\text{ rad/s}$ |
| **Level 2 (Mid-Range)** | $20 - 60\text{ m}$ | $\pm 3\text{ m}$ | $\pm 45^\circ,\; 0.020\text{ rad/s}$ |
| **Level 3 (Far-Range)** | $80 - 200\text{ m}$ | $\pm 10\text{ m}$ | $\pm 90^\circ,\; 0.050\text{ rad/s}$ |

---

## Repository Structure

```text
├── satellite_env.py          # Gymnasium environment (Orbital physics, 6-DOF dynamics, RK4)
├── train.py                  # PPO training loop with vectorized envs & curriculum callback
├── evaluate.py               # Deterministic evaluation and docking metric validation
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

```

---

## Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/your-username/XAI_DRL_for_OSAM_Spacecraft_Inspection.git](https://github.com/your-username/XAI_DRL_for_OSAM_Spacecraft_Inspection.git)
cd XAI_DRL_for_OSAM_Spacecraft_Inspection

```


2. **Create and activate a virtual environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

```


3. **Install required packages:**
```bash
pip install gymnasium stable-baselines3[extra] numpy

```



---

## Usage

### 1. Training the DRL Agent

Run the distributed training script using Proximal Policy Optimization (PPO). The script leverages `SubprocVecEnv` for multi-core rollout collection, normalizes observations with `VecNormalize`, and activates the `AutoCurriculumCallback`:

```bash
python train.py

```

*Checkpoints and normalization statistics (`vec_normalize_stats.pkl` and `ppo_6dof_satellite_docking.zip`) will be saved in your project root.*

### 2. Live Monitoring with TensorBoard

Track episode reward curves, policy entropy, value loss, and curriculum milestones in real time:

```bash
tensorboard --logdir ./satellite_tensorboard/

```

*Open `http://localhost:6006` in your browser.*

### 3. Evaluating Policy Performance

Benchmark the trained policy across curriculum levels in deterministic mode:

```bash
python evaluate.py

```

Example evaluation console output:

```text
==========================================
  EVALUATING ON CURRICULUM LEVEL 1
==========================================

--- Episode 1 ---
T=  1.0s | Pos=[  0.82,  11.45,  -0.34]m (11.48m) | Vel=0.32m/s | AttErr=12.4° | Dwell=0.0s
...
T= 48.0s | Pos=[  0.04,   0.12,   0.02]m ( 0.13m) | Vel=0.01m/s | AttErr= 2.1° | Dwell=5.0s

Episode 1 Result:
  Docked:       True
  Final Dist:   0.128 m
  Attitude Err: 2.14°
  Fuel Mass:    14.62 kg
  Total Reward: 114.82

```

---

## Docking Tolerances & Success Criteria

An episode is flagged as a successful docking when the Deputy satisfies all the following constraints simultaneously for a continuous dwell time of 5.0 seconds:

* **Relative Position:** $\|\mathbf{r}_{\text{rel}}\| \le 0.30\text{ m}$
* **Relative Velocity:** $\|\mathbf{v}_{\text{rel}}\| \le 0.05\text{ m/s}$
* **Attitude Alignment Error:** $\theta_{\text{err}} = 2 \arccos(|q_w|) \le 5.7^\circ$
* **Relative Angular Velocity:** $\|\boldsymbol{\omega}_{\text{rel}}\| \le 0.02\text{ rad/s}$ ($\sim 1.1^\circ/\text{s}$)

---

## References

1. **Clohessy, W. H., & Wiltshire, R. S. (1960).** *Terminal Guidance System for Satellite Rendezvous.* Journal of the Aerospace Sciences, 27(9), 653-658.
2. **Vallado, D. A. (2013).** *Fundamentals of Astrodynamics and Applications.* Microcosm Press.
3. **Wie, B. (2008).** *Space Vehicle Dynamics and Control.* American Institute of Aeronautics and Astronautics.
4. **Schulman, J., et al. (2017).** *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.

---



```
