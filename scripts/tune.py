#!/usr/bin/env python3
"""Coarse gain sweep for the cascaded controller.

Runs a fixed-duration step to a target pose for each gain combination and scores
it on steady-state error, residual velocity and torque saturation.
"""
import copy, itertools, logging, sys, tempfile
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
SIM_T = 6.0

SCALE = np.array([1.0, 1.0, 0.6, 0.6, 0.2, 0.2, 0.2])   # actuator size profile


def evaluate(kp_pos, kp_vel, ki_vel, sim_t=SIM_T):
    cfg = copy.deepcopy(BASE)
    cfg["position_controller"]["kp"] = [float(kp_pos)] * 7
    cfg["velocity_controller"]["kp"] = (kp_vel * SCALE).tolist()
    cfg["velocity_controller"]["ki"] = (ki_vel * SCALE).tolist()
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        path = f.name
    try:
        sim = MuJoCoSimulation(str(MODEL), path, arm_side="left_arm")
        sim.go_to_motor_angles(TARGET, timeout=120.0, sim_duration=sim_t)
        d = sim.observer.get_data()
        if len(d["time"]) == 0:
            return None
        err = np.abs(d["pos_ref"][-1] - d["pos_fb"][-1])
        vel = np.abs(d["vel_fb"][-1])
        tau = d["torque_cmd"]
        lim = np.array(cfg["velocity_controller"]["max_lim"], dtype=float)
        sat = float(np.mean(np.abs(tau) >= 0.999 * lim))
        # settling: worst-joint error over the final 10% of the run
        n = len(err)
        tail = np.abs(d["pos_ref"][int(0.9 * len(d["time"])):] -
                      d["pos_fb"][int(0.9 * len(d["time"])):])
        return dict(max_err=float(err.max()), mean_err=float(err.mean()),
                    max_vel=float(vel.max()), sat=sat,
                    tail_max=float(tail.max()))
    except Exception as e:
        return dict(error=str(e)[:60])


if __name__ == "__main__":
    grid = list(itertools.product(
        [2.0, 4.0, 8.0],            # kp_pos  (velocity setpoint per rad error)
        [40.0, 80.0, 160.0],        # kp_vel  (torque per rad/s error)
        [0.0, 20.0, 100.0],         # ki_vel
    ))
    print(f"{'kp_pos':>7}{'kp_vel':>8}{'ki_vel':>8} | "
          f"{'max_err':>9}{'tail':>9}{'max_vel':>9}{'sat%':>7}")
    print("-" * 62)
    best, best_key = None, None
    for kp_p, kp_v, ki_v in grid:
        r = evaluate(kp_p, kp_v, ki_v)
        if r is None or "error" in r:
            print(f"{kp_p:>7}{kp_v:>8}{ki_v:>8} | FAILED {r}")
            continue
        print(f"{kp_p:>7}{kp_v:>8}{ki_v:>8} | {r['max_err']:>9.4f}"
              f"{r['tail_max']:>9.4f}{r['max_vel']:>9.3f}{r['sat']*100:>7.1f}")
        score = r["max_err"] + 0.5 * r["tail_max"] + 0.1 * r["max_vel"]
        if best is None or score < best:
            best, best_key = score, (kp_p, kp_v, ki_v, r)
    print("-" * 62)
    print(f"BEST: kp_pos={best_key[0]} kp_vel={best_key[1]} ki_vel={best_key[2]}")
    print(f"      {best_key[3]}")
