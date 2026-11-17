"""Physics validation: run `python test_physics.py` (numpy only)."""
import numpy as np
from osam_env import *

def cw_analytic(x0, y0, z0, vx0, vy0, vz0, n, t):
    s, c = np.sin(n*t), np.cos(n*t)
    x = (4-3*c)*x0 + s/n*vx0 + 2/n*(1-c)*vy0
    y = 6*(s-n*t)*x0 + y0 - 2/n*(1-c)*vx0 + (4*s-3*n*t)/n*vy0
    z = c*z0 + s/n*vz0
    return np.array([x, y, z])

e = OSAMInspectionEnv({"realistic": False, "rta": False, "substeps": 2})
e.reset(seed=1)
# 1) Translational CW vs closed-form (zero thrust, no J2)
e.r = np.array([10.0, 50.0, -5.0]); e.v = np.array([0.01, -0.02, 0.005])
r0, v0 = e.r.copy(), e.v.copy(); e.w = np.zeros(3)
T = 600
for _ in range(T): e.step(np.zeros(6))
ref = cw_analytic(*r0, *v0, e.n, T)
print(f"[1] CW propagation error after {T}s: {np.linalg.norm(e.r-ref)*1000:.4f} mm  (should be ~0)")

# 2) Rotational: torque-free angular momentum + quaternion norm
e.reset(seed=2); e.w = np.array([0.05, -0.03, 0.04]); L0 = np.linalg.norm(e.I @ e.w)
for _ in range(600): e.step(np.zeros(6))
print(f"[2] |I w| drift: {abs(np.linalg.norm(e.I@e.w)-L0)/L0:.2e} (rel), |q|-1 = {abs(np.linalg.norm(e.q)-1):.1e}")

# 3) Thrust sign/frame: +X body thrust with identity-ish attitude accelerates along body X in LVLH
e.reset(seed=3); e.q = np.array([1.0,0,0,0]); e.w = np.array([0,0,e.n]); e.r=np.array([0,50.,0]); e.v=np.zeros(3)
a_expected = e.Fmax/(e.m_dry+e.fuel)
v_before = e.v.copy(); e.step(np.array([1.,0,0,0,0,0]))
print(f"[3] thrust accel along +x_LVLH: {(e.v[0]-v_before[0]):.5f} m/s^2 vs expected {a_expected:.5f}")

# 4) J2 differential magnitude
e = OSAMInspectionEnv(); e.reset(seed=4); e.r = np.array([0,100.,0])
print(f"[4] differential J2 accel at 100 m: {np.linalg.norm(e._j2_diff()):.2e} m/s^2")

# 5) Fuel: rocket-equation consistency
e = OSAMInspectionEnv({"realistic": False, "rta": False}); e.reset(seed=5)
for _ in range(50): e.step(np.array([1.,0,0,0,0,0]))
dv_rkt = e.isp*G0*np.log((e.m_dry+e.m_fuel0)/(e.m_dry+e.fuel))
print(f"[5] fuel used {e.m_fuel0-e.fuel:.4f} kg; sum|dv| {e.dv:.3f} m/s vs rocket-eq {dv_rkt:.3f} m/s (axis-aligned, close)")
