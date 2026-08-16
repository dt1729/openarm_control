#!/usr/bin/env python3
"""Record a multi-waypoint demo to docs/demo.gif via the existing replay path."""
import argparse, logging, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mujoco_sim import MuJoCoSimulation

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

REPO = Path(__file__).resolve().parent.parent
# scene.xml includes openarm_bimanual.xml and adds the headlight, floor, skybox
# and <statistic> framing hints — same joints, but renderable.
MODEL = REPO / "models" / "openarm_mujoco" / "v1" / "scene.xml"
CONFIG = Path(__file__).parent / "config" / "sim_config.yaml"

# Joint-space tour: reach out, lift, swing across, fold back, home.
POSES = [
    [ 0.0,  -0.6,  0.0,  0.8,  0.0,  0.0,  0.0],
    [ 1.0,  -2.0,  0.5,  0.5,  1.0, -0.4,  0.4],
    [-0.8,  -1.2, -0.9,  1.6, -1.0,  0.5, -0.6],
    [ 0.6,  -2.4,  1.0,  0.3,  0.8,  0.6,  1.0],
    [ 0.0,   0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
]

# Both arms move. Only joints 1 and 2 are mirrored between the arms — their
# ranges are negated ([-3.49,1.40] vs [-1.40,3.49] and [-3.32,0.17] vs
# [-0.17,3.32]) — while joints 3-7 share identical ranges. Negating everything
# would drive joint 4 negative, and its range is [0, 2.44].
MIRROR = np.array([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# The right arm also runs the list phase-shifted, so the arms are never in the
# same configuration — livelier than a pure mirror.
SHIFT = 2
WAYPOINTS = [
    list(POSES[i]) + list(MIRROR * np.array(POSES[(i + SHIFT) % len(POSES)]))
    for i in range(len(POSES))
]

p = argparse.ArgumentParser()
p.add_argument("--azimuth", type=float, default=180.0)   # front-on, both arms visible
p.add_argument("--elevation", type=float, default=-12.0)
p.add_argument("--distance", type=float, default=1.5)
p.add_argument("--lookat", type=float, nargs=3, default=[0.0, 0.0, 0.70])
p.add_argument("--fps", type=int, default=20)
p.add_argument("--speed", type=float, default=None)
p.add_argument("--width", type=int, default=560)
p.add_argument("--height", type=int, default=420)
p.add_argument("--dwell", type=float, default=1.2)
args = p.parse_args()

sim = MuJoCoSimulation(str(MODEL), str(CONFIG), arm_side="both")
errors = sim.follow_waypoints(WAYPOINTS, dwell=args.dwell)

print("\nper-waypoint max |err| (rad): " +
      "  ".join(f"{e.max():.4f}" for e in errors))

sim.record_gif(
    str(REPO / "docs" / "demo.gif"),
    playback_speed=args.speed, fps=args.fps,
    width=args.width, height=args.height,
    azimuth=args.azimuth, elevation=args.elevation,
    distance=args.distance, lookat=args.lookat,
)
