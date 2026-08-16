#!/usr/bin/env python3
"""Targeted retune of the one limit-cycling joint.

Six of seven joints settle exactly; joint 5 (index 4, lowest inertia at
0.00104 kg m^2) cycles with a ~3-4 s period. Two stages:

  Stage 1 - velocity loop P/I: speed up the inner loop so it stops lagging.
  Stage 2 - position loop P/D: D on POSITION error is well conditioned (the
            signal is smooth), unlike D on velocity error which amplifies the
            step-to-step difference by 1/dt = 1000.

Only joint 5's gains move; the other six keep their inertia-derived values.
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
J = 4                 # joint 5, zero-indexed
BAND = 0.01           # rad
SIM_T = 12.0


def evaluate(vel_kp_s=1.0, vel_ki_s=1.0, pos_kp=None, pos_kd=0.0):
    cfg = copy.deepcopy(BASE)
    cfg["velocity_controller"]["kp"][J] *= vel_kp_s
    cfg["velocity_controller"]["ki"][J] *= vel_ki_s
    if pos_kp is not None:
        cfg["position_controller"]["kp"][J] = pos_kp
    cfg["position_controller"]["kd"][J] = pos_kd

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f); path = f.name

    sim = MuJoCoSimulation(str(MODEL), path, arm_side="left_arm")
    sim.go_to_motor_angles(TARGET, timeout=300.0, sim_duration=SIM_T)
    d = sim.observer.get_data()
    t = d["time"]
    e5 = np.abs(d["pos_ref"][:, J] - d["pos_fb"][:, J])

    outside = np.where(e5 > BAND)[0]
    t_settle = float(t[outside[-1]]) if len(outside) else 0.0
    after = t > 4.0
    return dict(final=float(e5[-1]), t_settle=t_settle,
                ripple=float(e5[after].max() - e5[after].min()) if after.any() else 0.0,
                allmax=float(np.abs(d["pos_ref"][-1] - d["pos_fb"][-1]).max()))


def show(label, params, r, flag=""):
    print(f"{label:<26} | final={r['final']:.5f}  settle={r['t_settle']:>5.2f}s  "
          f"ripple={r['ripple']:.5f}  allmax={r['allmax']:.5f}{flag}")


if __name__ == "__main__":
    print("baseline")
    base = evaluate()
    show("as-is", None, base)

    print("\n--- stage 1: velocity loop P/I on joint 5 ---")
    best1, bp1 = None, None
    for kp_s, ki_s in itertools.product([1, 3, 10, 30], [0.3, 1.0, 3.0]):
        r = evaluate(vel_kp_s=kp_s, vel_ki_s=ki_s)
        score = r["t_settle"] + 50 * r["final"]
        f = ""
        if best1 is None or score < best1:
            best1, bp1 = score, (kp_s, ki_s, r); f = "  <--"
        show(f"vel kp x{kp_s}  ki x{ki_s}", None, r, f)
    kp_s, ki_s, _ = bp1
    print(f"stage 1 best: vel kp x{kp_s}, ki x{ki_s}")

    print("\n--- stage 2: position loop P/D on joint 5 ---")
    best2, bp2 = None, None
    for pkp, pkd in itertools.product([3.3333, 6.0, 10.0], [0.0, 0.1, 0.3, 1.0]):
        r = evaluate(vel_kp_s=kp_s, vel_ki_s=ki_s, pos_kp=pkp, pos_kd=pkd)
        score = r["t_settle"] + 50 * r["final"]
        f = ""
        if best2 is None or score < best2:
            best2, bp2 = score, (pkp, pkd, r); f = "  <--"
        show(f"pos kp={pkp}  kd={pkd}", None, r, f)

    pkp, pkd, rbest = bp2
    print(f"\nBEST: vel kp x{kp_s}, vel ki x{ki_s}, pos kp={pkp}, pos kd={pkd}")
    print(f"      {rbest}")

    cfg = copy.deepcopy(BASE)
    cfg["velocity_controller"]["kp"][J] *= kp_s
    cfg["velocity_controller"]["ki"][J] *= ki_s
    cfg["position_controller"]["kp"][J] = pkp
    cfg["position_controller"]["kd"][J] = pkd
    print("\nproposed joint-5 gains:")
    print(f"  position kp={cfg['position_controller']['kp'][J]:.4f} "
          f"kd={cfg['position_controller']['kd'][J]:.4f}")
    print(f"  velocity kp={cfg['velocity_controller']['kp'][J]:.4f} "
          f"ki={cfg['velocity_controller']['ki'][J]:.4f}")
    yaml.safe_dump(cfg, open(REPO / "docs" / "_proposed_gains.yaml", "w"), sort_keys=False)
    print(f"written to docs/_proposed_gains.yaml (not applied)")
