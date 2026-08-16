#!/usr/bin/env python3
"""Convergence profile for a single gain set: is it settling or limit-cycling?"""
import copy, logging, sys, tempfile
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from mujoco_sim import MuJoCoSimulation

logging.basicConfig(level=logging.ERROR)
REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "openarm_mujoco" / "v1" / "openarm_bimanual.xml"
BASE = yaml.safe_load(open(Path(__file__).parent / "config" / "sim_config.yaml"))
TARGET = np.array([1.0, -2.0, 0.5, 0.5, 1.0, -0.4, 0.4])
SCALE = np.array([1.0, 1.0, 0.6, 0.6, 0.2, 0.2, 0.2])


def run(kp_pos, kp_vel, ki_vel, kd_vel=0.0, sim_t=20.0, vel_lim=None):
    cfg = copy.deepcopy(BASE)
    cfg["position_controller"]["kp"] = [float(kp_pos)] * 7
    cfg["velocity_controller"]["kp"] = (kp_vel * SCALE).tolist()
    cfg["velocity_controller"]["ki"] = (ki_vel * SCALE).tolist()
    cfg["velocity_controller"]["kd"] = (kd_vel * SCALE).tolist()
    if vel_lim is not None:
        cfg["position_controller"]["max_lim"] = [float(vel_lim)] * 7
        cfg["position_controller"]["min_lim"] = [-float(vel_lim)] * 7
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f); path = f.name

    sim = MuJoCoSimulation(str(MODEL), path, arm_side="left_arm")
    sim.go_to_motor_angles(TARGET, timeout=300.0, sim_duration=sim_t)
    d = sim.observer.get_data()
    t = d["time"]; err = d["pos_ref"] - d["pos_fb"]

    print(f"\n=== kp_pos={kp_pos} kp_vel={kp_vel} ki_vel={ki_vel} kd_vel={kd_vel} "
          f"vel_lim={vel_lim or BASE['position_controller']['max_lim'][0]} ===")
    print(f"{'t(s)':>7}{'max|err|':>10}{'mean|err|':>11}{'max|vel|':>10}{'sat%':>7}")
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        i = min(int(frac * len(t)) - 1, len(t) - 1)
        w0 = max(0, i - len(t) // 20)
        tau = d["torque_cmd"][w0:i + 1]
        lim = np.array(BASE["velocity_controller"]["max_lim"], dtype=float)
        sat = np.mean(np.abs(tau) >= 0.999 * lim) * 100 if tau.size else 0
        print(f"{t[i]:>7.2f}{np.abs(err[i]).max():>10.4f}{np.abs(err[i]).mean():>11.4f}"
              f"{np.abs(d['vel_fb'][i]).max():>10.3f}{sat:>7.1f}")
    print(f"per-joint final |err| : "
          f"{np.array2string(np.abs(err[-1]), precision=3, suppress_small=True)}")
    return np.abs(err[-1]).max()


if __name__ == "__main__":
    run(8.0, 160.0, 20.0, sim_t=20.0)
    run(8.0, 160.0, 20.0, sim_t=20.0, vel_lim=0.8)
    run(4.0, 300.0, 40.0, kd_vel=2.0, sim_t=20.0, vel_lim=0.8)
