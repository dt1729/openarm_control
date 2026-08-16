#!/usr/bin/env python3
"""Find damping that removes the settle ringing.

Error at a fixed dwell was non-monotonic across dwell lengths, which is an
underdamped settle rather than divergence. This measures the actual settling
time: the first instant after the profile ends beyond which the error stays
inside a band.
"""
import copy, itertools, logging, sys, tempfile
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from mujoco_sim import MuJoCoSimulation

logging.basicConfig(level=logging.ERROR)
REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "openarm_mujoco" / "v1" / "scene.xml"
CFGP = Path(__file__).parent / "config" / "sim_config.yaml"
BASE = yaml.safe_load(open(CFGP))
TARGET = np.array([1.0, -2.0, 0.5, 0.5, 1.0, -0.4, 0.4])
BAND = 0.02          # rad
SIM_T = 12.0


def evaluate(ki_scale, kd_scale):
    cfg = copy.deepcopy(BASE)
    kp = np.array(cfg["velocity_controller"]["kp"], dtype=float)
    ki0 = np.array(cfg["velocity_controller"]["ki"], dtype=float)
    cfg["velocity_controller"]["ki"] = (ki0 * ki_scale).tolist()
    cfg["velocity_controller"]["kd"] = (kp * kd_scale).tolist()
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f); path = f.name

    sim = MuJoCoSimulation(str(MODEL), path, arm_side="left_arm")
    sim.go_to_motor_angles(TARGET, timeout=300.0, sim_duration=SIM_T)
    d = sim.observer.get_data()
    t = d["time"]; err = np.abs(d["pos_ref"] - d["pos_fb"]).max(axis=1)

    # settling time: last moment the error leaves the band
    outside = np.where(err > BAND)[0]
    t_settle = t[outside[-1]] if len(outside) else 0.0
    # overshoot past the target, measured after the profile completes
    after = t > 3.5
    return dict(final=float(err[-1]), t_settle=float(t_settle),
                peak_after=float(err[after].max()) if after.any() else 0.0)


if __name__ == "__main__":
    print(f"{'ki_x':>6}{'kd_x':>7} | {'final':>9}{'t_settle':>10}{'peak>3.5s':>11}")
    print("-" * 46)
    best = None
    for ki_s, kd_s in itertools.product([1.0, 0.3, 0.1], [0.0, 0.05, 0.2, 0.5]):
        r = evaluate(ki_s, kd_s)
        flag = ""
        if best is None or (r["t_settle"], r["final"]) < (best[0]["t_settle"], best[0]["final"]):
            best = (r, ki_s, kd_s); flag = "  <--"
        print(f"{ki_s:>6}{kd_s:>7} | {r['final']:>9.5f}{r['t_settle']:>10.2f}"
              f"{r['peak_after']:>11.5f}{flag}")
    print("-" * 46)
    print(f"BEST ki_scale={best[1]} kd_scale={best[2]} -> {best[0]}")
