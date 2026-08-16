#!/usr/bin/env python3
"""Headless test harness: runs a step to a target pose and reports tracking metrics."""
import logging, sys, time
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).parent))
from mujoco_sim import MuJoCoSimulation

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "openarm_mujoco" / "v1" / "openarm_bimanual.xml"
CONFIG = Path(__file__).parent / "config" / "sim_config.yaml"
TARGET = np.array([1.0, -2.0, 0.5, 0.5, 1.0, -0.4, 0.4])


def model_info():
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    print(f"model: nq={m.nq} nv={m.nv} nu={m.nu} njnt={m.njnt}")
    print(f"{'joint name':<26}{'jid':>5}{'   ':3}{'actuator name':<26}{'aid':>5}")
    for i in range(1, 8):
        jn = f"openarm_left_joint{i}"
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
        an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jid) if jid < m.nu else "(out of range)"
        print(f"{jn:<26}{jid:>5}   {str(an):<26}{aid:>5}")
    print()


def run(timeout=20.0, label=""):
    sim = MuJoCoSimulation(str(MODEL), str(CONFIG), arm_side="left_arm")
    t0 = time.time()
    sim.go_to_motor_angles(TARGET, timeout=timeout)
    wall = time.time() - t0
    d = sim.observer.get_data()
    if len(d["time"]) == 0:
        print("NO DATA"); return
    pos_fb, pos_ref = d["pos_fb"], d["pos_ref"]
    err = pos_ref - pos_fb
    final = np.abs(err[-1])
    tau = d["torque_cmd"]
    sim_t = d["time"][-1]

    print(f"===== {label} =====")
    print(f"samples={len(d['time'])}  sim_time={sim_t:.4f}s  wall={wall:.1f}s")
    print(f"final |err| per joint : {np.array2string(final, precision=4, suppress_small=True)}")
    print(f"max final |err|       : {final.max():.4f} rad")
    print(f"mean final |err|      : {final.mean():.4f} rad")
    print(f"torque range          : [{tau.min():.2f}, {tau.max():.2f}] Nm")
    sat = np.mean(np.abs(tau) >= 0.999 * np.abs(tau).max()) if tau.size else 0
    print(f"frac at max |torque|  : {sat*100:.1f}%")
    print(f"final |qvel|          : {np.abs(d['vel_fb'][-1]).max():.4f} rad/s")
    # steady-state check: error change over last 20% of the run
    n = len(err)
    if n > 10:
        tail = np.abs(err[int(0.8 * n):])
        print(f"tail err drift        : {abs(tail[-1].max() - tail[0].max()):.5f} rad")
    print()
    return final.max()


if __name__ == "__main__":
    model_info()
    run(timeout=float(sys.argv[1]) if len(sys.argv) > 1 else 20.0,
        label=sys.argv[2] if len(sys.argv) > 2 else "RUN")
