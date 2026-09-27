"""
6-DOF Satellite Proximity Operations Gymnasium Environment with Curriculum Learning.
"""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np


# Orbital & Physical Constants (WGS-84)
MU_EARTH = 3.986004418e14  # m^3 / s^2
R_EARTH = 6378137.0         # m
J2_EARTH = 1.08262668e-3
G0 = 9.80665                # m/s^2


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1.0 - 2.0 * (y**2 + z**2), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x**2 + z**2), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x**2 + y**2)],
    ], dtype=np.float64)


def rot_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z], dtype=np.float64))


def euler_rates_to_quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    omega_quat = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]], dtype=np.float64)
    return 0.5 * quat_multiply(q, omega_quat)


def gravitational_accel_eci(r_eci: np.ndarray) -> np.ndarray:
    r_mag = np.linalg.norm(r_eci)
    x, y, z = r_eci
    r2 = r_mag * r_mag
    z2_r2 = (z * z) / r2
    a_2body = -MU_EARTH / (r_mag**3) * r_eci
    factor = 1.5 * J2_EARTH * MU_EARTH * (R_EARTH**2) / (r_mag**5)
    a_j2 = -factor * np.array([
        x * (1.0 - 5.0 * z2_r2),
        y * (1.0 - 5.0 * z2_r2),
        z * (3.0 - 5.0 * z2_r2),
    ], dtype=np.float64)
    return a_2body + a_j2


def compute_lvlh_basis_and_rates(
    r_chief: np.ndarray, v_chief: np.ndarray, a_chief: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    rc_mag = np.linalg.norm(r_chief)
    x_hat = r_chief / rc_mag
    x_dot = (v_chief / rc_mag) - (r_chief * np.dot(r_chief, v_chief) / (rc_mag**3))

    h_vec = np.cross(r_chief, v_chief)
    h_mag = np.linalg.norm(h_vec)
    z_hat = h_vec / h_mag
    h_dot = np.cross(r_chief, a_chief)
    z_dot = (h_dot / h_mag) - (h_vec * np.dot(h_vec, h_dot) / (h_mag**3))

    y_hat = np.cross(z_hat, x_hat)
    y_dot = np.cross(z_dot, x_hat) + np.cross(z_hat, x_dot)

    R_eci2lvlh = np.vstack([x_hat, y_hat, z_hat])
    R_dot = np.vstack([x_dot, y_dot, z_dot])
    return R_eci2lvlh, R_dot


class Satellite6DOFProximityEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = config or {}

        # Simulation timing
        self.dt_rl = float(cfg.get("dt_rl", 1.0))
        self.dt_sim = float(cfg.get("dt_sim", 0.1))
        self.substeps = max(1, int(round(self.dt_rl / self.dt_sim)))
        self.max_time = float(cfg.get("max_time", 600.0))
        self.sim_time = 0.0

        # Curriculum Level: 1 = Near (5-15m), 2 = Mid (20-60m), 3 = Far (80-200m)
        self.curriculum_level = int(cfg.get("curriculum_level", 1))

        # Spacecraft specifications
        self.dry_mass = float(cfg.get("dry_mass", 100.0))
        self.fuel_mass_init = float(cfg.get("fuel_mass", 15.0))
        self.isp = float(cfg.get("isp", 220.0))
        self.inertia = np.array(cfg.get("inertia", [
            [10.0, 0.0, 0.0],
            [0.0, 12.0, 0.0],
            [0.0, 0.0, 15.0]
        ]), dtype=np.float64)
        self.inv_inertia = np.linalg.inv(self.inertia)

        # Actuation limits
        self.max_thrust = float(cfg.get("max_thrust", 10.0))  # N
        self.max_torque = float(cfg.get("max_torque", 1.0))   # N*m

        # Imperfections
        self.enable_noise = bool(cfg.get("enable_noise", False))
        self.pos_sensor_std = float(cfg.get("pos_sensor_std", 0.02))
        self.vel_sensor_std = float(cfg.get("vel_sensor_std", 0.005))
        self.quat_sensor_std = float(cfg.get("quat_sensor_std", 0.002))
        self.gyro_sensor_std = float(cfg.get("gyro_sensor_std", 0.001))

        # Docking Tolerances
        self.docking_pos_tol = float(cfg.get("docking_pos_tol", 0.30))   # 30 cm
        self.docking_vel_tol = float(cfg.get("docking_vel_tol", 0.05))   # 5 cm/s
        self.docking_att_tol = float(cfg.get("docking_att_tol", 0.10))   # ~5.7 deg
        self.docking_rate_tol = float(cfg.get("docking_rate_tol", 0.02)) # ~1.1 deg/s
        self.required_dwell_time = float(cfg.get("required_dwell_time", 5.0))

        # Geometry Constraints
        self.koz_radius = float(cfg.get("koz_radius", 0.8))
        self.approach_cone_half_angle = np.deg2rad(cfg.get("cone_angle_deg", 35.0))
        self.max_range = float(cfg.get("max_range", 600.0))

        # Action Space: [Fx, Fy, Fz, Tx, Ty, Tz]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Observation Space: [pos(3), vel(3), quat(4), omega(3), fuel(1), time(1)]
        obs_high = np.array([
            1000.0, 1000.0, 1000.0,
            10.0, 10.0, 10.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0
        ], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        # State Variables
        self.r_chief_eci = np.zeros(3, dtype=np.float64)
        self.v_chief_eci = np.zeros(3, dtype=np.float64)
        self.r_deputy_eci = np.zeros(3, dtype=np.float64)
        self.v_deputy_eci = np.zeros(3, dtype=np.float64)
        self.q_deputy_eci = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.w_deputy_body = np.zeros(3, dtype=np.float64)
        self.fuel_mass = self.fuel_mass_init
        self.dwell_timer = 0.0
        self.prev_potential = 0.0

    def set_curriculum_level(self, level: int):
        self.curriculum_level = max(1, min(3, level))

    def _chief_initial_orbit(self) -> Tuple[np.ndarray, np.ndarray]:
        alt = 500e3
        radius = R_EARTH + alt
        v_mag = np.sqrt(MU_EARTH / radius)
        # Circular orbit in equatorial plane for reproducibility
        r_eci = np.array([radius, 0.0, 0.0], dtype=np.float64)
        v_eci = np.array([0.0, v_mag, 0.0], dtype=np.float64)
        return r_eci, v_eci

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.sim_time = 0.0
        self.dwell_timer = 0.0
        self.fuel_mass = self.fuel_mass_init

        # 1. Chief initial orbit
        self.r_chief_eci, self.v_chief_eci = self._chief_initial_orbit()
        a_c = gravitational_accel_eci(self.r_chief_eci)
        R_eci2lvlh, R_dot = compute_lvlh_basis_and_rates(self.r_chief_eci, self.v_chief_eci, a_c)
        n = np.sqrt(MU_EARTH / (np.linalg.norm(self.r_chief_eci)**3))

        # 2. Curriculum-based Spawn
        if self.curriculum_level == 1:
            init_y = self.np_random.uniform(5.0, 15.0)
            init_x = self.np_random.uniform(-1.0, 1.0)
            init_z = self.np_random.uniform(-1.0, 1.0)
            angle_bound = np.deg2rad(15.0)
            rate_bound = 0.005
        elif self.curriculum_level == 2:
            init_y = self.np_random.uniform(20.0, 60.0)
            init_x = self.np_random.uniform(-3.0, 3.0)
            init_z = self.np_random.uniform(-3.0, 3.0)
            angle_bound = np.deg2rad(45.0)
            rate_bound = 0.02
        else:
            init_y = self.np_random.uniform(80.0, 200.0)
            init_x = self.np_random.uniform(-10.0, 10.0)
            init_z = self.np_random.uniform(-10.0, 10.0)
            angle_bound = np.deg2rad(90.0)
            rate_bound = 0.05

        r_rel_lvlh = np.array([init_x, init_y, init_z], dtype=np.float64)

        # CW Drift-free condition: vy0 = -2 * n * x0
        init_vy = -2.0 * n * init_x + self.np_random.uniform(-0.02, 0.02)
        init_vx = self.np_random.uniform(-0.01, 0.01)
        init_vz = self.np_random.uniform(-0.01, 0.01)
        v_rel_lvlh = np.array([init_vx, init_vy, init_vz], dtype=np.float64)

        # Transform relative to ECI
        self.r_deputy_eci = self.r_chief_eci + R_eci2lvlh.T @ r_rel_lvlh
        self.v_deputy_eci = self.v_chief_eci + R_eci2lvlh.T @ (v_rel_lvlh - R_dot @ (self.r_deputy_eci - self.r_chief_eci))

        # Initial relative attitude
        rand_axis = self.np_random.normal(size=3)
        rand_axis /= np.linalg.norm(rand_axis)
        rand_angle = self.np_random.uniform(-angle_bound, angle_bound)
        q_init_rel = np.array([
            np.cos(rand_angle / 2.0),
            *(rand_axis * np.sin(rand_angle / 2.0))
        ], dtype=np.float64)

        q_eci2lvlh = rot_matrix_to_quat(R_eci2lvlh)
        self.q_deputy_eci = quat_multiply(q_init_rel, q_eci2lvlh)
        self.w_deputy_body = self.np_random.uniform(-rate_bound, rate_bound, size=3)

        r_rel, v_rel, q_rel, w_rel = self._get_relative_state()
        self.prev_potential = self._compute_potential(r_rel, v_rel, q_rel, w_rel)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _compute_potential(self, r_rel, v_rel, q_rel, w_rel) -> float:
        d = np.linalg.norm(r_rel)
        v = np.linalg.norm(v_rel)
        att_err = 2.0 * np.arccos(np.clip(abs(q_rel[0]), 0.0, 1.0))
        w = np.linalg.norm(w_rel)
        # Smooth potential: high when close, low when far
        return float(-1.0 * d - 10.0 * v - 2.0 * att_err - 5.0 * w)

    def _get_relative_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        a_c = gravitational_accel_eci(self.r_chief_eci)
        R_eci2lvlh, R_dot = compute_lvlh_basis_and_rates(self.r_chief_eci, self.v_chief_eci, a_c)

        dr_eci = self.r_deputy_eci - self.r_chief_eci
        dv_eci = self.v_deputy_eci - self.v_chief_eci
        r_rel_lvlh = R_eci2lvlh @ dr_eci
        v_rel_lvlh = R_eci2lvlh @ dv_eci + R_dot @ dr_eci

        q_eci2lvlh = rot_matrix_to_quat(R_eci2lvlh)
        q_rel = quat_multiply(self.q_deputy_eci, quat_conjugate(q_eci2lvlh))
        q_rel = quat_normalize(q_rel)
        if q_rel[0] < 0:
            q_rel = -q_rel

        h_vec = np.cross(self.r_chief_eci, self.v_chief_eci)
        omega_lvlh_eci = h_vec / (np.linalg.norm(self.r_chief_eci)**2)
        R_body2eci = quat_to_rot_matrix(self.q_deputy_eci).T
        omega_rel_body = self.w_deputy_body - (R_body2eci.T @ omega_lvlh_eci)

        return r_rel_lvlh, v_rel_lvlh, q_rel, omega_rel_body

    def _get_observation(self) -> np.ndarray:
        r_rel, v_rel, q_rel, w_rel = self._get_relative_state()

        if self.enable_noise:
            r_obs = r_rel + self.np_random.normal(0.0, self.pos_sensor_std, size=3)
            v_obs = v_rel + self.np_random.normal(0.0, self.vel_sensor_std, size=3)
            dq_rot = self.np_random.normal(0.0, self.quat_sensor_std, size=3)
            dq = quat_normalize(np.array([1.0, *(0.5 * dq_rot)]))
            q_obs = quat_multiply(dq, q_rel)
            if q_obs[0] < 0:
                q_obs = -q_obs
            w_obs = w_rel + self.np_random.normal(0.0, self.gyro_sensor_std, size=3)
        else:
            r_obs, v_obs, q_obs, w_obs = r_rel, v_rel, q_rel, w_rel

        fuel_frac = np.clip(self.fuel_mass / self.fuel_mass_init, 0.0, 1.0)
        time_frac = np.clip(1.0 - (self.sim_time / self.max_time), 0.0, 1.0)
        return np.hstack([r_obs, v_obs, q_obs, w_obs, fuel_frac, time_frac], dtype=np.float32)

    def _rk4_dynamics_step(self, F_body: np.ndarray, tau_body: np.ndarray, dt: float):
        total_mass = self.dry_mass + self.fuel_mass
        thrust_mag = float(np.linalg.norm(F_body))
        dm = (thrust_mag / (self.isp * G0)) * dt
        self.fuel_mass = max(0.0, self.fuel_mass - dm)

        def get_derivs(r_c, v_c, r_d, v_d, q_d, w_d):
            a_c = gravitational_accel_eci(r_c)
            a_d_grav = gravitational_accel_eci(r_d)
            R_body2eci = quat_to_rot_matrix(q_d).T
            a_d_thrust = R_body2eci @ F_body / total_mass
            a_d = a_d_grav + a_d_thrust
            q_dot = euler_rates_to_quat_derivative(q_d, w_d)
            Iw = self.inertia @ w_d
            w_dot = self.inv_inertia @ (tau_body - np.cross(w_d, Iw))
            return v_c, a_c, v_d, a_d, q_dot, w_dot

        k1 = get_derivs(self.r_chief_eci, self.v_chief_eci, self.r_deputy_eci, self.v_deputy_eci, self.q_deputy_eci, self.w_deputy_body)
        k2 = get_derivs(
            self.r_chief_eci + 0.5 * dt * k1[0], self.v_chief_eci + 0.5 * dt * k1[1],
            self.r_deputy_eci + 0.5 * dt * k1[2], self.v_deputy_eci + 0.5 * dt * k1[3],
            quat_normalize(self.q_deputy_eci + 0.5 * dt * k1[4]), self.w_deputy_body + 0.5 * dt * k1[5]
        )
        k3 = get_derivs(
            self.r_chief_eci + 0.5 * dt * k2[0], self.v_chief_eci + 0.5 * dt * k2[1],
            self.r_deputy_eci + 0.5 * dt * k2[2], self.v_deputy_eci + 0.5 * dt * k2[3],
            quat_normalize(self.q_deputy_eci + 0.5 * dt * k2[4]), self.w_deputy_body + 0.5 * dt * k2[5]
        )
        k4 = get_derivs(
            self.r_chief_eci + dt * k3[0], self.v_chief_eci + dt * k3[1],
            self.r_deputy_eci + dt * k3[2], self.v_deputy_eci + dt * k3[3],
            quat_normalize(self.q_deputy_eci + dt * k3[4]), self.w_deputy_body + dt * k3[5]
        )

        self.r_chief_eci += (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        self.v_chief_eci += (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        self.r_deputy_eci += (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        self.v_deputy_eci += (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
        self.q_deputy_eci = quat_normalize(self.q_deputy_eci + (dt / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]))
        self.w_deputy_body += (dt / 6.0) * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0)
        if self.fuel_mass <= 0.0:
            F_cmd = np.zeros(3)
            tau_cmd = np.zeros(3)
        else:
            F_cmd = action[0:3] * self.max_thrust
            tau_cmd = action[3:6] * self.max_torque

        for _ in range(self.substeps):
            self._rk4_dynamics_step(F_cmd, tau_cmd, self.dt_sim)

        self.sim_time += self.dt_rl
        r_rel, v_rel, q_rel, w_rel = self._get_relative_state()

        pos_dist = float(np.linalg.norm(r_rel))
        vel_norm = float(np.linalg.norm(v_rel))
        w_norm = float(np.linalg.norm(w_rel))
        att_error = float(2.0 * np.arccos(np.clip(abs(q_rel[0]), 0.0, 1.0)))

        is_inside_box = (
            pos_dist < self.docking_pos_tol and
            vel_norm < self.docking_vel_tol and
            att_error < self.docking_att_tol and
            w_norm < self.docking_rate_tol
        )
        if is_inside_box:
            self.dwell_timer += self.dt_rl
        else:
            self.dwell_timer = 0.0

        terminated = False
        truncated = False

        # Termination logic
        if self.dwell_timer >= self.required_dwell_time:
            reward = 100.0
            terminated = True
        elif pos_dist < self.koz_radius and not is_inside_box:
            reward = -50.0
            terminated = True
        elif pos_dist > self.max_range:
            reward = -50.0
            terminated = True
        elif self.sim_time >= self.max_time:
            reward = 0.0
            truncated = True
        else:
            # Potential-based continuous progress reward (gives gradient across hundreds of meters)
            current_potential = self._compute_potential(r_rel, v_rel, q_rel, w_rel)
            progress_reward = current_potential - self.prev_potential
            self.prev_potential = current_potential

            # Action effort penalty
            effort = -0.005 * (np.linalg.norm(action[0:3]) + np.linalg.norm(action[3:6]))
            reward = progress_reward + effort

        obs = self._get_observation()
        info = {
            "pos_error": pos_dist,
            "vel_error": vel_norm,
            "att_error_rad": att_error,
            "rate_error_rad_s": w_norm,
            "fuel_mass_kg": self.fuel_mass,
            "dwell_time": self.dwell_timer,
            "is_docked": self.dwell_timer >= self.required_dwell_time,
            "curriculum_level": self.curriculum_level,
        }
        return obs, float(reward), terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        return {"sim_time": self.sim_time, "curriculum_level": self.curriculum_level}

    def render(self):
        r_rel, v_rel, q_rel, w_rel = self._get_relative_state()
        print(f"T={self.sim_time:5.1f}s | "
              f"Pos=[{r_rel[0]:6.2f}, {r_rel[1]:6.2f}, {r_rel[2]:6.2f}]m ({np.linalg.norm(r_rel):5.2f}m) | "
              f"Vel={np.linalg.norm(v_rel):4.2f}m/s | AttErr={np.rad2deg(2.0*np.arccos(np.clip(abs(q_rel[0]),0,1))):4.1f}° | "
              f"Dwell={self.dwell_timer:3.1f}s")