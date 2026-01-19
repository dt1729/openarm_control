#!/usr/bin/env python3
"""Signal plotter for control system analysis."""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List

from observer import Observer

logger = logging.getLogger(__name__)


class SignalPlotter:
    """Plots recorded control system data.

    Usage:
        from observer import Observer
        from plotter import SignalPlotter

        # After simulation with recorded data:
        plotter = SignalPlotter(observer)
        plotter.plot_all()
        plotter.show()
    """

    def __init__(self,
                 observer: Observer,
                 joint_names: Optional[List[str]] = None):
        """Initialize plotter with observer data.

        Args:
            observer: Observer instance with recorded data
            joint_names: Optional list of joint names for legends
        """
        self.observer = observer
        self.data = observer.get_data()

        if joint_names is None:
            self.joint_names = [f"Joint {i+1}" for i in range(observer.num_joints)]
        else:
            self.joint_names = joint_names

        self._figures = []

    def plot_by_joint(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create subplots for each joint showing all signals.

        Each joint gets a subplot with:
        - Position: reference (dashed) vs feedback (solid)
        - Velocity feedback
        - Torque command

        Args:
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        time = self.data['time']
        num_joints = self.observer.num_joints

        fig, axes = plt.subplots(num_joints, 3, figsize=(15, 2.5 * num_joints))
        fig.suptitle('Control Signals by Joint', fontsize=14, fontweight='bold')

        for j in range(num_joints):
            # Position tracking
            ax_pos = axes[j, 0]
            ax_pos.plot(time, self.data['pos_ref'][:, j], 'r--', label='Reference', linewidth=1.5)
            ax_pos.plot(time, self.data['pos_fb'][:, j], 'b-', label='Feedback', linewidth=1)
            ax_pos.set_ylabel(f'{self.joint_names[j]}\nPosition (rad)')
            ax_pos.legend(loc='upper right', fontsize=8)
            ax_pos.grid(True, alpha=0.3)
            if j == 0:
                ax_pos.set_title('Position Tracking')

            # Velocity
            ax_vel = axes[j, 1]
            ax_vel.plot(time, self.data['vel_ref'][:, j], 'r--', label='Reference', linewidth=1.5)
            ax_vel.plot(time, self.data['vel_fb'][:, j], 'g-', label='Feedback', linewidth=1)
            ax_vel.set_ylabel('Velocity (rad/s)')
            ax_vel.legend(loc='upper right', fontsize=8)
            ax_vel.grid(True, alpha=0.3)
            if j == 0:
                ax_vel.set_title('Velocity')

            # Torque
            ax_torque = axes[j, 2]
            ax_torque.plot(time, self.data['torque_cmd'][:, j], 'm-', label='Command', linewidth=1)
            ax_torque.set_ylabel('Torque (Nm)')
            ax_torque.legend(loc='upper right', fontsize=8)
            ax_torque.grid(True, alpha=0.3)
            if j == 0:
                ax_torque.set_title('Torque Command')

            # X-axis labels only on bottom row
            if j == num_joints - 1:
                ax_pos.set_xlabel('Time (s)')
                ax_vel.set_xlabel('Time (s)')
                ax_torque.set_xlabel('Time (s)')

        plt.tight_layout()

        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {path}")

        self._figures.append(fig)
        return fig

    def plot_by_signal(self, save_dir: Optional[str] = None) -> List[plt.Figure]:
        """Create separate figures for each signal type.

        Creates 3 figures:
        - Position tracking (reference vs feedback for all joints)
        - Velocity (reference vs feedback for all joints)
        - Torque commands (all joints)

        Args:
            save_dir: Optional directory to save figures

        Returns:
            List of matplotlib Figure objects
        """
        time = self.data['time']
        colors = plt.cm.tab10(np.linspace(0, 1, self.observer.num_joints))

        figures = []

        # Position tracking figure
        fig_pos, (ax_ref, ax_fb, ax_err) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig_pos.suptitle('Position Signals', fontsize=14, fontweight='bold')

        for j in range(self.observer.num_joints):
            ax_ref.plot(time, self.data['pos_ref'][:, j], color=colors[j],
                       linestyle='--', label=self.joint_names[j], linewidth=1)
            ax_fb.plot(time, self.data['pos_fb'][:, j], color=colors[j],
                      label=self.joint_names[j], linewidth=1)
            error = self.data['pos_ref'][:, j] - self.data['pos_fb'][:, j]
            ax_err.plot(time, error, color=colors[j], label=self.joint_names[j], linewidth=1)

        ax_ref.set_ylabel('Reference (rad)')
        ax_ref.legend(loc='upper right', fontsize=8, ncol=4)
        ax_ref.grid(True, alpha=0.3)
        ax_ref.set_title('Position Reference')

        ax_fb.set_ylabel('Feedback (rad)')
        ax_fb.legend(loc='upper right', fontsize=8, ncol=4)
        ax_fb.grid(True, alpha=0.3)
        ax_fb.set_title('Position Feedback')

        ax_err.set_ylabel('Error (rad)')
        ax_err.set_xlabel('Time (s)')
        ax_err.legend(loc='upper right', fontsize=8, ncol=4)
        ax_err.grid(True, alpha=0.3)
        ax_err.set_title('Position Error')

        plt.tight_layout()
        figures.append(fig_pos)

        # Velocity figure
        fig_vel, (ax_vref, ax_vfb, ax_verr) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig_vel.suptitle('Velocity Signals', fontsize=14, fontweight='bold')

        for j in range(self.observer.num_joints):
            ax_vref.plot(time, self.data['vel_ref'][:, j], color=colors[j],
                        linestyle='--', label=self.joint_names[j], linewidth=1)
            ax_vfb.plot(time, self.data['vel_fb'][:, j], color=colors[j],
                       label=self.joint_names[j], linewidth=1)
            verror = self.data['vel_ref'][:, j] - self.data['vel_fb'][:, j]
            ax_verr.plot(time, verror, color=colors[j], label=self.joint_names[j], linewidth=1)

        ax_vref.set_ylabel('Reference (rad/s)')
        ax_vref.legend(loc='upper right', fontsize=8, ncol=4)
        ax_vref.grid(True, alpha=0.3)
        ax_vref.set_title('Velocity Reference')

        ax_vfb.set_ylabel('Feedback (rad/s)')
        ax_vfb.legend(loc='upper right', fontsize=8, ncol=4)
        ax_vfb.grid(True, alpha=0.3)
        ax_vfb.set_title('Velocity Feedback')

        ax_verr.set_ylabel('Error (rad/s)')
        ax_verr.set_xlabel('Time (s)')
        ax_verr.legend(loc='upper right', fontsize=8, ncol=4)
        ax_verr.grid(True, alpha=0.3)
        ax_verr.set_title('Velocity Error')

        plt.tight_layout()
        figures.append(fig_vel)

        # Torque figure
        fig_torque, ax_torque = plt.subplots(1, 1, figsize=(12, 5))
        fig_torque.suptitle('Torque Commands', fontsize=14, fontweight='bold')

        for j in range(self.observer.num_joints):
            ax_torque.plot(time, self.data['torque_cmd'][:, j], color=colors[j],
                          label=self.joint_names[j], linewidth=1)

        ax_torque.set_ylabel('Torque (Nm)')
        ax_torque.set_xlabel('Time (s)')
        ax_torque.legend(loc='upper right', fontsize=8, ncol=4)
        ax_torque.grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig_torque)

        # Save if requested
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            fig_pos.savefig(save_path / 'position_signals.png', dpi=150, bbox_inches='tight')
            fig_vel.savefig(save_path / 'velocity_signals.png', dpi=150, bbox_inches='tight')
            fig_torque.savefig(save_path / 'torque_signals.png', dpi=150, bbox_inches='tight')
            logger.info(f"Saved figures to {save_path}")

        self._figures.extend(figures)
        return figures

    def plot_all(self, save_dir: Optional[str] = None) -> None:
        """Generate all plot layouts.

        Args:
            save_dir: Optional directory to save all figures
        """
        if save_dir:
            save_path = Path(save_dir)
            self.plot_by_joint(save_path=save_path / 'by_joint.png')
            self.plot_by_signal(save_dir=save_path)
        else:
            self.plot_by_joint()
            self.plot_by_signal()

    def show(self) -> None:
        """Display all created plots."""
        plt.show()

    def close_all(self) -> None:
        """Close all created figures."""
        for fig in self._figures:
            plt.close(fig)
        self._figures.clear()
