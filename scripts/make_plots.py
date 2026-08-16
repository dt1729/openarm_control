#!/usr/bin/env python3
"""Run the demo move and save SignalPlotter figures headlessly, for inspection."""
import logging, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")          # no display needed
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mujoco_sim import MuJoCoSimulation

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "openarm_mujoco" / "v1" / "scene.xml"
CONFIG = Path(__file__).parent / "config" / "sim_config.yaml"
TARGET = np.array([1.0, -2.0, 0.5, 0.5, 1.0, -0.4, 0.4])

sim = MuJoCoSimulation(str(MODEL), str(CONFIG), arm_side="left_arm")
sim.go_to_motor_angles(TARGET, timeout=300.0, sim_duration=12.0)

d = sim.observer.get_data()
err = np.abs(d["pos_ref"] - d["pos_fb"]).max(axis=1)
t = d["time"]
print("\nmax |err| over time:")
for frac in (0.25, 0.35, 0.45, 0.55, 0.7, 0.85, 1.0):
    i = min(int(frac * len(t)) - 1, len(t) - 1)
    print(f"  t={t[i]:6.2f}s  max|err|={err[i]:.5f}  "
          f"max|vel|={np.abs(d['vel_fb'][i]).max():.4f}")
print(f"\nper-joint final |err|: "
      f"{np.array2string(np.abs(d['pos_ref'][-1]-d['pos_fb'][-1]), precision=4, suppress_small=True)}")

out = REPO / "docs" / "plots"
sim.visualize(save_dir=str(out), show=False)
print(f"\nwrote figures to {out}")
