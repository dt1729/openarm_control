#!/usr/bin/env python3
"""Print joint ranges for both arms so waypoints can be made legal for each."""
from pathlib import Path
import numpy as np, mujoco

REPO = Path(__file__).resolve().parent.parent
m = mujoco.MjModel.from_xml_path(str(REPO / "models" / "openarm_mujoco" / "v1" / "scene.xml"))

print(f"{'joint':<10}{'LEFT min':>10}{'LEFT max':>10}   {'RIGHT min':>11}{'RIGHT max':>11}   mirrored?")
for i in range(1, 8):
    lid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_left_joint{i}")
    rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_right_joint{i}")
    lo_l, hi_l = m.jnt_range[lid]
    lo_r, hi_r = m.jnt_range[rid]
    mir = "YES" if (abs(lo_l + hi_r) < 1e-3 and abs(hi_l + lo_r) < 1e-3
                    and abs(lo_l - lo_r) > 1e-3) else "no"
    print(f"joint{i:<5}{lo_l:>10.4f}{hi_l:>10.4f}   {lo_r:>11.4f}{hi_r:>11.4f}   {mir}")

POSES = [
    [ 0.0,  -0.6,  0.0,  0.8,  0.0,  0.0,  0.0],
    [ 1.0,  -2.0,  0.5,  0.5,  1.0, -0.4,  0.4],
    [-0.8,  -1.2, -0.9,  1.6, -1.0,  0.5, -0.6],
    [ 0.6,  -2.4,  1.0,  0.3,  0.8,  0.6,  1.0],
    [ 0.0,   0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
]
print("\nleft poses legal on the RIGHT arm as-is?")
for k, p in enumerate(POSES):
    bad = []
    for i, v in enumerate(p, start=1):
        rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_right_joint{i}")
        lo, hi = m.jnt_range[rid]
        if v < lo or v > hi:
            bad.append(f"j{i}={v:+.2f} not in [{lo:.2f},{hi:.2f}]")
    print(f"  pose {k}: {'OK' if not bad else '; '.join(bad)}")

print("\nsign-flipped (negate all) legal on the RIGHT arm?")
for k, p in enumerate(POSES):
    bad = []
    for i, v in enumerate(p, start=1):
        rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_right_joint{i}")
        lo, hi = m.jnt_range[rid]
        if -v < lo or -v > hi:
            bad.append(f"j{i}={-v:+.2f} not in [{lo:.2f},{hi:.2f}]")
    print(f"  pose {k}: {'OK' if not bad else '; '.join(bad)}")
