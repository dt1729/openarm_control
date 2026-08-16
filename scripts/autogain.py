#!/usr/bin/env python3
"""Size cascaded-loop gains from the joint-space inertia matrix.

For a velocity loop on M q̈ = τ, a proportional gain kp gives a first-order
response with time constant M/kp. So choosing a target closed-loop time constant
tau_v fixes kp = M_ii / tau_v per joint, which automatically respects the fact
that a wrist joint carries orders of magnitude less inertia than a shoulder.
"""
import copy, logging, sys, tempfile
from pathlib import Path
import numpy as np
import mujoco, yaml

sys.path.insert(0, str(Path(__file__).parent))
from mujoco_sim import MuJoCoSimulation

logging.basicConfig(level=logging.ERROR)
REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "openarm_mujoco" / "v1" / "openarm_bimanual.xml"
CFGP = Path(__file__).parent / "config" / "sim_config.yaml"
BASE = yaml.safe_load(open(CFGP))
TARGET = np.array([1.0, -2.0, 0.5, 0.5, 1.0, -0.4, 0.4])


def inertia_diag():
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_left_joint{i}")
            for i in range(1, 8)]
    dofs = [m.jnt_dofadr[j] for j in jids]
    out = []
    for q in (np.zeros(7), TARGET, TARGET * 0.5):
        d.qpos[:] = 0
        for i, j in enumerate(jids):
            d.qpos[m.jnt_qposadr[j]] = q[i]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        # Measure effective inertia directly instead of unpacking the sparse
        # mass matrix (whose binding differs across MuJoCo versions): apply a
        # unit torque to one DOF and read the resulting acceleration, having
        # subtracted the gravity-only baseline. M_eff = 1 / (a_torque - a_free).
        d.qfrc_applied[:] = 0
        mujoco.mj_forward(m, d)
        a0 = np.array(d.qacc)
        row = []
        for k in dofs:
            d.qfrc_applied[:] = 0
            d.qfrc_applied[k] = 1.0
            mujoco.mj_forward(m, d)
            da = d.qacc[k] - a0[k]
            row.append(1.0 / da if abs(da) > 1e-12 else np.inf)
        d.qfrc_applied[:] = 0
        out.append(np.array(row))
    return np.max(np.stack(out), axis=0)      # worst-case over sampled poses


def run(tau_v, tau_p, ki_ratio, vel_lim, sim_t=20.0, verbose=True):
    M = inertia_diag()
    kp_vel = M / tau_v
    ki_vel = kp_vel / ki_ratio
    kp_pos = np.full(7, 1.0 / tau_p)

    cfg = copy.deepcopy(BASE)
    cfg["position_controller"]["kp"] = kp_pos.tolist()
    cfg["position_controller"]["max_lim"] = [float(vel_lim)] * 7
    cfg["position_controller"]["min_lim"] = [-float(vel_lim)] * 7
    cfg["velocity_controller"]["kp"] = kp_vel.tolist()
    cfg["velocity_controller"]["ki"] = ki_vel.tolist()
    cfg["velocity_controller"]["kd"] = [0.0] * 7
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f); path = f.name

    sim = MuJoCoSimulation(str(MODEL), path, arm_side="left_arm")
    sim.go_to_motor_angles(TARGET, timeout=300.0, sim_duration=sim_t)
    d = sim.observer.get_data()
    err = d["pos_ref"] - d["pos_fb"]; t = d["time"]
    lim = np.array(BASE["velocity_controller"]["max_lim"], dtype=float)
    tail = slice(int(0.9 * len(t)), None)
    res = dict(
        max_err=float(np.abs(err[-1]).max()),
        tail_max=float(np.abs(err[tail]).max()),
        max_vel=float(np.abs(d["vel_fb"][-1]).max()),
        sat=float(np.mean(np.abs(d["torque_cmd"][tail]) >= 0.999 * lim)),
    )
    if verbose:
        print(f"tau_v={tau_v:<6} tau_p={tau_p:<5} ki_ratio={ki_ratio:<5} vlim={vel_lim:<4} | "
              f"final={res['max_err']:.4f}  tail={res['tail_max']:.4f}  "
              f"vel={res['max_vel']:.3f}  sat={res['sat']*100:.1f}%")
    return res, cfg


if __name__ == "__main__":
    M = inertia_diag()
    print("worst-case joint inertia diag (kg m^2):")
    print("  " + np.array2string(M, precision=5, suppress_small=False))
    print(f"  ratio shoulder/wrist = {M.max()/M.min():.1f}x\n")

    best = None
    for tau_v in (0.02, 0.05, 0.1):
        for tau_p in (0.3, 0.6):
            for ki_ratio in (0.2, 1.0):
                for vlim in (1.0, 2.0):
                    r, cfg = run(tau_v, tau_p, ki_ratio, vlim, sim_t=12.0)
                    score = r["tail_max"] + 0.05 * r["max_vel"]
                    if best is None or score < best[0]:
                        best = (score, (tau_v, tau_p, ki_ratio, vlim), r, cfg)
    print(f"\nBEST {best[1]} -> {best[2]}")
    yaml.safe_dump(best[3], open(CFGP, "w"), sort_keys=False)
    print(f"written to {CFGP}")
