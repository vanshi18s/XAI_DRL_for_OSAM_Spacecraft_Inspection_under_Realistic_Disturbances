# OSAM: 6-DOF Spacecraft Inspection with PPO + Run-Time Assurance + XAI

Task: a deputy spacecraft flies to a 15 m standoff point around a chief satellite, points its
camera (+X body) at the chief, and holds still for 10 s, without entering the 4 m keep-out sphere
and without pointing the camera within 25 deg of the Sun.

## Files
| file | purpose |
|---|---|
| `osam_env.py` | Gymnasium env: CW + differential-J2 translation, Euler/quaternion rotation (RK4), sensor noise, actuator latency, thruster gain error, power cap, **RTA/CBF filter** |
| `controllers.py` | PD + CW-feed-forward **baseline**; `ResidualEnv` (PPO learns a correction on top of the baseline) |
| `train.py` | SB3 PPO + automatic 3-level curriculum (RTA is ON during training) |
| `evaluate.py` | Monte-Carlo comparison: baseline vs PPO, RTA on/off, levels 1-3 -> CSV + plot |
| `explain.py` | SHAP (or permutation) importance of each observation for each thruster/torque |
| `test_physics.py`, `test_rta.py` | validation of the dynamics and adversarial tests of the safety filter |

## Run (needs `pip install stable-baselines3 gymnasium numpy pandas matplotlib shap`)
```
python test_physics.py && python test_rta.py     # ~1 min, no torch needed
python train.py --mode residual --steps 600000   # recommended, roughly 10-20 min on 2 CPU cores
python evaluate.py --episodes 100                # fills results/
python explain.py --model models/ppo_residual.zip --mode residual
```
`--mode pure` trains PPO from scratch (needs ~2M+ steps, no guarantee within a short time).

## Verified results (run, numpy only)
* CW propagation vs closed-form solution: error 0.0000 mm after 600 s
* Torque-free angular-momentum drift 3e-12, quaternion norm preserved, rocket-equation fuel accounting matches
* Adversarial tests (RTA on): ramming the chief from 1 m/s inbound -> 0/40 violations (0 -> 40/40 without RTA);
  torquing the camera into the Sun -> 0/40 (40/40 without); random actions -> 0/40 (38/40 sun violations without)
* PD baseline, 50 Monte-Carlo episodes per cell, realistic disturbances on: 100% success on all 3 levels with RTA.
  Without RTA, level 3: 84% success, 8 keep-out violations, 1 sun violation.

## NOT verified here
PPO training itself was not run in the build session (no PyTorch available there). The residual setup is
designed so PPO starts from a policy that already succeeds (tested: 100% success under PPO's initial
exploration noise), but the final learned-policy numbers come from YOUR run of `train.py` + `evaluate.py`.

## Honest modelling notes
* Translation uses linear CW (valid for separations << orbit radius) plus a differential-J2 disturbance; the chief orbit is circular.
* "Power" is a proxy: a cap on total commanded thrust (L1 norm) and a duty-cycle metric, not a thruster-device power model.
* RTA uses the noisy on-board measurement, plus a 1 m / 6 deg buffer for discrete-time margin. Guarantees hold under the
  actuator limits modelled and bounded body rates (enforced by the rate envelope); they are not formal proofs for every state.
* `rta_trans_pct` counts any modification of the command, including tiny ones.