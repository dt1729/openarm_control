#!/usr/bin/env python3
"""Observer class for recording control system signals."""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Observer:
    """Records control signals and state feedback for analysis.

    Usage:
        obs = Observer(num_joints=7, dt=0.002)

        # In simulation loop:
        obs.record(
            time=elapsed,
            pos_ref=pos_state._ref,
            pos_fb=pos_state._fb,
            vel_ref=vel_state._ref,
            vel_fb=vel_state._fb,
            torque_cmd=control_signal
        )

        # After simulation:
        data = obs.get_data()
    """
    num_joints: int = 7
    dt: float = 0.002  # Sampling time in seconds

    # Internal storage (lists for efficient appending)
    _time: list = field(default_factory=list)
    _pos_ref: list = field(default_factory=list)
    _pos_fb: list = field(default_factory=list)
    _vel_ref: list = field(default_factory=list)
    _vel_fb: list = field(default_factory=list)
    _torque_cmd: list = field(default_factory=list)

    # Cached numpy arrays
    _arrays_cached: bool = field(default=False)
    _cached_data: Dict[str, np.ndarray] = field(default_factory=dict)

    def record(self,
               time: float,
               pos_ref: np.ndarray,
               pos_fb: np.ndarray,
               vel_ref: np.ndarray,
               vel_fb: np.ndarray,
               torque_cmd: np.ndarray) -> None:
        """Record one timestep of data.

        Args:
            time: Elapsed simulation time (seconds)
            pos_ref: Position reference/setpoint (rad)
            pos_fb: Position feedback (rad)
            vel_ref: Velocity reference (rad/s)
            vel_fb: Velocity feedback (rad/s)
            torque_cmd: Torque command (Nm)
        """
        self._time.append(time)
        self._pos_ref.append(pos_ref.copy())
        self._pos_fb.append(pos_fb.copy())
        self._vel_ref.append(vel_ref.copy())
        self._vel_fb.append(vel_fb.copy())
        self._torque_cmd.append(torque_cmd.copy())

        # Invalidate cache
        self._arrays_cached = False

    def to_arrays(self) -> None:
        """Convert internal lists to numpy arrays and cache them."""
        if self._arrays_cached:
            return

        self._cached_data = {
            'time': np.array(self._time),
            'pos_ref': np.array(self._pos_ref),
            'pos_fb': np.array(self._pos_fb),
            'vel_ref': np.array(self._vel_ref),
            'vel_fb': np.array(self._vel_fb),
            'torque_cmd': np.array(self._torque_cmd),
        }
        self._arrays_cached = True

    def get_data(self) -> Dict[str, np.ndarray]:
        """Get all recorded data as numpy arrays.

        Returns:
            Dictionary with keys: dt, time, pos_ref, pos_fb, vel_ref, vel_fb, torque_cmd
        """
        self.to_arrays()
        data = self._cached_data.copy()
        data['dt'] = self.dt
        return data

    def clear(self) -> None:
        """Clear all recorded data."""
        self._time.clear()
        self._pos_ref.clear()
        self._pos_fb.clear()
        self._vel_ref.clear()
        self._vel_fb.clear()
        self._torque_cmd.clear()
        self._arrays_cached = False
        self._cached_data.clear()

    def save(self, filepath: str) -> None:
        """Save recorded data to a .npz file.

        Args:
            filepath: Path to save file (will add .npz extension if not present)
        """
        self.to_arrays()
        path = Path(filepath)
        if path.suffix != '.npz':
            path = path.with_suffix('.npz')

        np.savez(path,
                 time=self._cached_data['time'],
                 pos_ref=self._cached_data['pos_ref'],
                 pos_fb=self._cached_data['pos_fb'],
                 vel_ref=self._cached_data['vel_ref'],
                 vel_fb=self._cached_data['vel_fb'],
                 torque_cmd=self._cached_data['torque_cmd'],
                 num_joints=self.num_joints,
                 dt=self.dt)
        logger.info(f"Data saved to {path}")

    def load(self, filepath: str) -> None:
        """Load recorded data from a .npz file.

        Args:
            filepath: Path to load file
        """
        path = Path(filepath)
        if path.suffix != '.npz':
            path = path.with_suffix('.npz')

        data = np.load(path)

        self.clear()
        self.num_joints = int(data['num_joints'])
        self.dt = float(data['dt'])

        # Load into lists
        self._time = data['time'].tolist()
        self._pos_ref = [row for row in data['pos_ref']]
        self._pos_fb = [row for row in data['pos_fb']]
        self._vel_ref = [row for row in data['vel_ref']]
        self._vel_fb = [row for row in data['vel_fb']]
        self._torque_cmd = [row for row in data['torque_cmd']]

        # Cache arrays
        self._cached_data = {
            'time': data['time'],
            'pos_ref': data['pos_ref'],
            'pos_fb': data['pos_fb'],
            'vel_ref': data['vel_ref'],
            'vel_fb': data['vel_fb'],
            'torque_cmd': data['torque_cmd'],
        }
        self._arrays_cached = True
        logger.info(f"Data loaded from {path} ({len(self._time)} samples, dt={self.dt}s)")

    @property
    def num_samples(self) -> int:
        """Return number of recorded samples."""
        return len(self._time)

    def __len__(self) -> int:
        return self.num_samples
