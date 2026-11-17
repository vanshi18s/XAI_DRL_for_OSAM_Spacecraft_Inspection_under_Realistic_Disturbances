"""
OSAM 6-DOF spacecraft inspection environment (Gymnasium) with Run-Time Assurance.

Task   : deputy spacecraft flies to a standoff inspection point around a chief
         (target) satellite, points its camera (+X body axis) at the chief and holds.
Dynamics:
  * Translation : Clohessy-Wiltshire (CW) in the chief LVLH frame
                  (x radial, y along-track, z cross-track) + differential-J2 disturbance
  * Rotation    : full rigid-body Euler equations, quaternion kinematics, RK4
  * Attitude is expressed relative to LVLH; q rotates BODY -> LVLH (Hamilton, [w,x,y,z]).
Imperfections (cfg["realistic"]=True): sensor noise, 1-step actuator latency,
  thruster gain error + noise, differential J2.
Safety (cfg["rta"]=True): Control-Barrier-Function filter (ASIF, closed form)
  1) Keep-out sphere around chief          (2nd-order CBF on range)
  2) Sun exclusion cone for the camera     (2nd-order CBF on boresight/sun angle)
Power/actuator proxy: total commanded thrust (L1 norm) can be capped (power_cap).
"""
from typing import Any, Dict, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

MU = 3.986004418e14
RE = 6378137.0
J2 = 1.08262668e-3
G0 = 9.80665


# ----------------------------------------------------------------- math utils
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def qnorm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0, 0, 0])


def q2R(q):
    """Rotation matrix for q: maps body-frame vectors to the parent (LVLH) frame."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])


def R2q(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1) * 2
        q = [0.25*s, (R[2, 1]-R[1, 2])/s, (R[0, 2]-R[2, 0])/s, (R[1, 0]-R[0, 1])/s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1]-R[1, 2])/s, 0.25*s, (R[0, 1]+R[1, 0])/s, (R[0, 2]+R[2, 0])/s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2]-R[2, 0])/s, (R[0, 1]+R[1, 0])/s, 0.25*s, (R[1, 2]+R[2, 1])/s]
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0]-R[0, 1])/s, (R[0, 2]+R[2, 0])/s, (R[1, 2]+R[2, 1])/s, 0.25*s]
    return qnorm(np.array(q))


def axis_angle_R(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang)*K + (1 - np.cos(ang))*(K @ K)


def j2_accel(r):
    d = np.linalg.norm(r)
    k = 1.5 * J2 * MU * RE**2 / d**5
    z2 = (r[2]/d)**2
    return -k * np.array([r[0]*(1 - 5*z2), r[1]*(1 - 5*z2), r[2]*(3 - 5*z2)])


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


# ------------------------------------------------------------------ environment
class OSAMInspectionEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        c = config or {}
        # timing
        self.dt = float(c.get("dt_rl", 1.0))
        self.sub = int(c.get("substeps", 2))
        self.max_time = float(c.get("max_time", 400.0))
        self.level = int(c.get("curriculum_level", 1))
        # orbit (circular, 51.6 deg, 500 km)
        self.a_orb = RE + float(c.get("alt_m", 500e3))
        self.n = np.sqrt(MU / self.a_orb**3)
        self.inc = np.deg2rad(c.get("inc_deg", 51.6))
        self.raan = np.deg2rad(c.get("raan_deg", 30.0))
        # spacecraft
        self.m_dry = float(c.get("dry_mass", 100.0))
        self.m_fuel0 = float(c.get("fuel_mass", 15.0))
        self.isp = float(c.get("isp", 220.0))
        self.I = np.diag(c.get("inertia", [10.0, 12.0, 15.0])).astype(float)
        self.Iinv = np.linalg.inv(self.I)
        self.Fmax = float(c.get("max_thrust", 10.0))      # N per axis
        self.Tmax = float(c.get("max_torque", 1.0))       # N m per axis
        self.power_cap = float(c.get("power_cap", 3.0))   # max sum|u_F| (3.0 = no cap)
        # imperfections
        self.realistic = bool(c.get("realistic", True))
        self.pos_std, self.vel_std = 0.05, 0.005
        self.att_std, self.gyro_std = 0.002, 0.0005
        self.latency = int(c.get("latency_steps", 1)) if self.realistic else 0
        # mission geometry
        self.standoff = float(c.get("standoff", 15.0))
        self.koz = float(c.get("koz_radius", 4.0))         # hard keep-out (violation)
        self.koz_cbf = self.koz + 1.0                      # CBF buffer
        self.sun_excl = np.deg2rad(c.get("sun_excl_deg", 25.0))
        self.sun_cbf = self.sun_excl + np.deg2rad(c.get("sun_buffer_deg", 6.0))   # CBF buffer (discrete-time margin)
        self.max_range = float(c.get("max_range", 400.0))
        # success box
        self.tol_pos, self.tol_vel = 1.0, 0.04
        self.tol_point, self.tol_rate = np.deg2rad(5.0), 0.01
        self.hold_time = float(c.get("hold_time", 10.0))
        # RTA
        self.rta_on = bool(c.get("rta", True))
        self.a_tr = float(c.get("cbf_alpha_trans", 0.06))
        self.a_at = float(c.get("cbf_alpha_att", 0.15))
        self.w_lim = float(c.get("rate_limit", 0.10))          # RTA body-rate envelope [rad/s]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(30,), dtype=np.float32)
        self.obs_names = (["ex", "ey", "ez", "vx", "vy", "vz", "rx", "ry", "rz",
                           "los_bx", "los_by", "los_bz", "q0", "q1", "q2", "q3",
                           "wx", "wy", "wz", "sun_bx", "sun_by", "sun_bz", "fuel", "time_left"]
                          + [f"prev_a{i}" for i in range(6)])

    # ----------------------------------------------------------- orbit helpers
    def _lvlh2eci(self):
        p = np.array([np.cos(self.raan), np.sin(self.raan), 0.0])
        qv = np.array([-np.sin(self.raan)*np.cos(self.inc), np.cos(self.raan)*np.cos(self.inc), np.sin(self.inc)])
        xh = np.cos(self.nu)*p + np.sin(self.nu)*qv
        yh = -np.sin(self.nu)*p + np.cos(self.nu)*qv
        zh = np.cross(xh, yh)
        return np.column_stack([xh, yh, zh])

    def _j2_diff(self):
        Rl = self._lvlh2eci()
        rc = Rl[:, 0] * self.a_orb
        rd = rc + Rl @ self.r
        return Rl.T @ (j2_accel(rd) - j2_accel(rc))

    def _sun_lvlh(self):
        return self._lvlh2eci().T @ self.sun_eci

    def _a_cw(self, r, v):
        n = self.n
        return np.array([3*n*n*r[0] + 2*n*v[1], -2*n*v[0], -n*n*r[2]])

    # ---------------------------------------------------------------- dynamics
    def _deriv(self, s, F, tau, m, a_dist):
        r, v, q, w = s[0:3], s[3:6], s[6:10], s[10:13]
        R = q2R(q)
        a = R @ F / m + self._a_cw(r, v) + a_dist
        w_rel = w - R.T @ np.array([0, 0, self.n])
        qd = 0.5 * qmul(q, np.array([0.0, *w_rel]))
        wd = self.Iinv @ (tau - np.cross(w, self.I @ w))
        return np.concatenate([v, a, qd, wd])

    def _rk4(self, F, tau, m, a_dist, h):
        s = np.concatenate([self.r, self.v, self.q, self.w])
        k1 = self._deriv(s, F, tau, m, a_dist)
        k2 = self._deriv(s + 0.5*h*k1, F, tau, m, a_dist)
        k3 = self._deriv(s + 0.5*h*k2, F, tau, m, a_dist)
        k4 = self._deriv(s + h*k3, F, tau, m, a_dist)
        s = s + h/6*(k1 + 2*k2 + 2*k3 + k4)
        self.r, self.v, self.q, self.w = s[0:3], s[3:6], qnorm(s[6:10]), s[10:13]

    # --------------------------------------------------------------- reset
    def set_curriculum_level(self, level: int):
        self.level = int(np.clip(level, 1, 3))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.t, self.nu = 0.0, rng.uniform(0, 2*np.pi)
        self.sun_eci = unit(rng.normal(size=3))
        self.fuel = self.m_fuel0
        self.hold, self.dv, self.duty_acc, self.rta_trans_n, self.rta_att_n = 0.0, 0.0, 0.0, 0, 0
        self.min_range, self.min_sun_ang, self.nsteps = 1e9, np.pi, 0
        self.gain = 1.0 + (rng.uniform(-0.02, 0.02, 3) if self.realistic else 0.0)
        self._buf = [np.zeros(6) for _ in range(self.latency)]
        self.prev_a = np.zeros(6)

        gmax = {1: 60.0, 2: 120.0, 3: 180.0}[self.level]   # max goal-vs-start bearing difference (deg)
        sun_l = self._sun_lvlh()
        for _ in range(200):          # feasible geometry: chief must not lie near the sun direction (start or goal)
            sdir = unit(rng.normal(size=3))
            ang = np.deg2rad(rng.uniform(0, gmax))
            perp = unit(np.cross(sdir, rng.normal(size=3)))
            gdir = np.cos(ang)*sdir + np.sin(ang)*perp
            if (np.arccos(np.clip(np.dot(-gdir, sun_l), -1, 1)) > np.deg2rad(60) and
                    np.arccos(np.clip(np.dot(-sdir, sun_l), -1, 1)) > np.deg2rad(60)):
                break
        self.goal = gdir * self.standoff
        self.r = sdir * rng.uniform(*self._range_bounds())
        self.v = np.array([0.0, -2*self.n*self.r[0], 0.0]) + rng.uniform(-0.03, 0.03, 3)

        # initial attitude: boresight near LOS to chief, random roll, random pointing offset
        att_bound = {1: 20, 2: 45, 3: 90}[self.level]
        for _ in range(50):
            xb = unit(-self.r)
            yb = unit(np.cross(xb, rng.normal(size=3)))
            R0 = np.column_stack([xb, yb, np.cross(xb, yb)])
            R0 = R0 @ axis_angle_R(rng.normal(size=3), np.deg2rad(rng.uniform(0, att_bound)))
            self.q = R2q(R0)
            if np.arccos(np.clip(q2R(self.q)[:, 0] @ self._sun_lvlh(), -1, 1)) > self.sun_cbf + np.deg2rad(4):
                break
        rate_b = {1: 0.005, 2: 0.015, 3: 0.03}[self.level]
        w_rel0 = rng.uniform(-rate_b, rate_b, 3)
        self.w = w_rel0 + q2R(self.q).T @ np.array([0, 0, self.n])

        self._d_prev = np.linalg.norm(self.r - self.goal)
        self._th_prev = self._point_err(self.r, self.q)
        self._meas = self._measure()
        return self._obs(), {}

    def _range_bounds(self):
        return {1: (25.0, 50.0), 2: (50.0, 100.0), 3: (100.0, 180.0)}[self.level]

    # --------------------------------------------------------- measurement/obs
    def _point_err(self, r, q):
        los_b = q2R(q).T @ unit(-r)
        return float(np.arccos(np.clip(los_b[0], -1, 1)))

    def _measure(self):
        r, v, q, w = self.r.copy(), self.v.copy(), self.q.copy(), self.w.copy()
        if self.realistic:
            rng = self.np_random
            r = r + rng.normal(0, self.pos_std, 3)
            v = v + rng.normal(0, self.vel_std, 3)
            dq = qnorm(np.array([1.0, *(0.5*rng.normal(0, self.att_std, 3))]))
            q = qnorm(qmul(q, dq))
            w = w + rng.normal(0, self.gyro_std, 3)
        return r, v, q, w

    def _obs(self):
        r, v, q, w = self._meas
        R = q2R(q)
        w_rel = w - R.T @ np.array([0, 0, self.n])
        los_b = R.T @ unit(-r)
        sun_b = R.T @ self._sun_lvlh()
        o = np.concatenate([(self.goal - r)/50.0, v/0.5, r/50.0, los_b, q, w_rel/0.1, sun_b,
                            [self.fuel/self.m_fuel0, 1 - self.t/self.max_time], self.prev_a])
        return np.clip(o, -5, 5).astype(np.float32)

    def get_measurement(self):
        """(r, v, q, w_abs) as seen by the on-board navigation (noisy if realistic)."""
        return self._meas

    # ------------------------------------------------------------------- RTA
    def _rta(self, F, tau):
        """ASIF: minimally modify (F,tau) so both CBF constraints hold. Returns F,tau,flags."""
        r, v, q, w = self._meas
        m = self.m_dry + self.fuel
        R = q2R(q)
        f_trans = f_att = False

        # (1) keep-out sphere: h = |r| - R_koz_cbf, relative degree 2
        d = np.linalg.norm(r); rh = r / d
        h = d - self.koz_cbf
        hd = rh @ v
        a_nat = self._a_cw(r, v) + self._j2_diff()
        a1 = self.a_tr
        b = -((v @ v - hd*hd)/d + rh @ a_nat + 2*a1*hd + a1*a1*h)
        a_nom = R @ F / m
        if rh @ a_nom < b:
            f_trans = True
            a_safe = a_nom + (b - rh @ a_nom) * rh
            Fn = np.clip(m * R.T @ a_safe, -self.Fmax, self.Fmax)
            if rh @ (R @ Fn / m) < b - 1e-9:              # saturated: use max braking along r_hat
                Fn = self.Fmax * np.sign(R.T @ rh)
            F = Fn

        # (1b) body-rate envelope: keeps the sun-CBF feasible (bounded spin -> bounded stopping angle)
        lo = self.I.diagonal() * (-self.w_lim - w) / self.dt
        hi = self.I.diagonal() * (self.w_lim - w) / self.dt
        tau_c = np.clip(tau, np.minimum(lo, hi), np.maximum(lo, hi))
        if np.any(np.abs(tau_c - tau) > 1e-12):
            f_att = True
            tau = tau_c

        # (2) sun exclusion cone for boresight (+X body): h = cos(th_ex) - s_b[0]
        s_b = R.T @ self._sun_lvlh()
        h_s = np.cos(self.sun_cbf) - s_b[0]
        hd_s = w[1]*s_b[2] - w[2]*s_b[1]
        sd = -np.cross(w, s_b)
        wd0 = self.Iinv @ (-np.cross(w, self.I @ w))
        c0 = (wd0[1]*s_b[2] - wd0[2]*s_b[1]) + (w[1]*sd[2] - w[2]*sd[1])
        g = self.Iinv @ np.array([0.0, s_b[2], -s_b[1]])
        al = self.a_at
        b_s = -(c0 + 2*al*hd_s + al*al*h_s)
        if np.linalg.norm(g) > 1e-9 and g @ tau < b_s:
            f_att = True
            tn = np.clip(tau + (b_s - g @ tau) * g / (g @ g), -self.Tmax, self.Tmax)
            if g @ tn < b_s - 1e-9:
                tn = self.Tmax * np.sign(g)
            tau = tn
        return F, tau, f_trans, f_att

    # -------------------------------------------------------------------- step
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        self._buf.append(a)
        u = self._buf.pop(0)                                   # latency
        F, tau = u[:3]*self.Fmax, u[3:]*self.Tmax
        f_t = f_a = False
        if self.rta_on:
            F, tau, f_t, f_a = self._rta(F, tau)
        l1 = np.abs(F).sum() / self.Fmax
        if l1 > self.power_cap:                                # power / duty proxy
            F = F * self.power_cap / l1
        if self.fuel <= 0:
            F, tau = np.zeros(3), np.zeros(3)
        u_eff = np.concatenate([F/self.Fmax, tau/self.Tmax])
        rta_dev = float(np.linalg.norm(u_eff - u))
        F_true = F * self.gain * (1 + (self.np_random.normal(0, 0.01, 3) if self.realistic else 0))

        m = self.m_dry + self.fuel
        a_dist = self._j2_diff() if self.realistic else np.zeros(3)
        R_now = q2R(self.q)
        self.dv += float(np.linalg.norm(R_now @ F_true / m)) * self.dt
        h = self.dt / self.sub
        for _ in range(self.sub):
            self._rk4(F_true, tau, m, a_dist, h)
        self.fuel = max(0.0, self.fuel - np.abs(F_true).sum() / (self.isp * G0) * self.dt)
        self.nu += self.n * self.dt
        self.t += self.dt
        self.nsteps += 1
        self.prev_a = a
        self.duty_acc += l1 / 3.0
        self.rta_trans_n += f_t
        self.rta_att_n += f_a

        # ---- truth-based metrics
        d_goal = float(np.linalg.norm(self.r - self.goal))
        rng_now = float(np.linalg.norm(self.r))
        speed = float(np.linalg.norm(self.v))
        Rq = q2R(self.q)
        th = self._point_err(self.r, self.q)
        w_rel = float(np.linalg.norm(self.w - Rq.T @ np.array([0, 0, self.n])))
        sun_ang = float(np.arccos(np.clip(Rq[:, 0] @ self._sun_lvlh(), -1, 1)))
        self.min_range = min(self.min_range, rng_now)
        self.min_sun_ang = min(self.min_sun_ang, sun_ang)
        inside = d_goal < self.tol_pos and speed < self.tol_vel and th < self.tol_point and w_rel < self.tol_rate
        self.hold = self.hold + self.dt if inside else 0.0

        v_lim = 0.03 + 0.05 * d_goal
        rew = (0.5 * (self._d_prev - d_goal) + 2.0 * (self._th_prev - th)
               - 2.0 * max(0.0, speed - v_lim) - 5.0 * w_rel
               - 0.01 * (np.abs(a[:3]).sum() + 0.5 * np.abs(a[3:]).sum())
               - 0.3 * rta_dev - (1.0 if sun_ang < self.sun_excl else 0.0)
               - 0.01 + (0.5 if inside else 0.0))
        self._d_prev, self._th_prev = d_goal, th

        success = self.hold >= self.hold_time
        koz_breach = rng_now < self.koz
        out = rng_now > self.max_range
        terminated = bool(success or koz_breach or out)
        truncated = bool((not terminated) and self.t >= self.max_time)
        if success:
            rew += 50.0 + 20.0 * self.fuel / self.m_fuel0
        elif koz_breach:
            rew -= 100.0
        elif out:
            rew -= 50.0

        self._meas = self._measure()
        info = dict(is_success=bool(success), pos_err=d_goal, speed=speed, point_err_deg=float(np.rad2deg(th)),
                    rate=w_rel, fuel_used_kg=self.m_fuel0 - self.fuel, delta_v=self.dv,
                    koz_violation=bool(self.min_range < self.koz),
                    sun_violation=bool(self.min_sun_ang < self.sun_excl), min_range=self.min_range,
                    min_sun_angle_deg=float(np.rad2deg(self.min_sun_ang)),
                    rta_trans_frac=self.rta_trans_n / self.nsteps, rta_att_frac=self.rta_att_n / self.nsteps,
                    duty_cycle=self.duty_acc / self.nsteps, sim_time=self.t, level=self.level)
        return self._obs(), float(rew), terminated, truncated, info

    def render(self):
        print(f"t={self.t:6.1f}s r={np.round(self.r,1)} |e|={np.linalg.norm(self.r-self.goal):6.2f}m "
              f"|v|={np.linalg.norm(self.v):.3f} point={np.rad2deg(self._point_err(self.r,self.q)):5.1f}deg "
              f"fuel={self.fuel:.2f}kg")