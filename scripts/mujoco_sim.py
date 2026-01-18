#!/usr/bin/env python3
import argparse
import signal
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from controller_impl import FF_Controllers, ControllerData
from gravity_compensation import GravityCompensationSim
from observer import Observer
from plotter import SignalPlotter

class MuJoCoSimulation:
    """MuJoCo simulation for OpenArm with cascaded PID control."""

    def __init__(self, model_path: str, config_path: str, arm_side: str = "left_arm"):
        """
        Initialize MuJoCo simulation.

        Args:
            model_path: Path to the MJCF model file
            config_path: Path to the YAML config file for controller gains
            arm_side: Either 'left_arm' or 'right_arm'
        """
        if arm_side not in ['left_arm', 'right_arm']:
            raise ValueError(f"Invalid arm_side: {arm_side}. Must be 'left_arm' or 'right_arm'.")

        self.arm_side = arm_side
        self.running = True

        # Load MuJoCo model
        print(f"Loading MuJoCo model from: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity = (0, 0, -9.81)

        # Load configuration
        self.config = self._load_config(config_path)

        # Setup arm joints
        self._setup_arm(arm_side)

        # Setup controllers
        self._setup_controllers(self.config)

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

        print("=== MuJoCo Simulation Initialized ===")
        print(f"Arm side       : {arm_side}")
        print(f"Model path     : {model_path}")
        print(f"Config path    : {config_path}")
        print(f"Number of DOFs : {len(self.joint_ids)}")
        print(f"Timestep       : {self.model.opt.timestep}")

    def _load_config(self, config_path: str) -> dict:
        """
        Load controller configuration from YAML file.

        Args:
            config_path: Path to the YAML config file

        Returns:
            Parsed configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_arm(self, arm_side: str):
        """
        Setup joint names and IDs for the specified arm.

        Args:
            arm_side: Either 'left_arm' or 'right_arm'
        """
        prefix = "openarm_left" if arm_side == "left_arm" else "openarm_right"
        self.joint_names = [f"{prefix}_joint{i}" for i in range(1, 8)]

        # Get joint IDs
        self.joint_ids = []
        for joint_name in self.joint_names:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                self.joint_ids.append(joint_id)
            except KeyError:
                raise ValueError(f"Joint {joint_name} not found in model")
        self.joint_ids = np.array(self.joint_ids)

        print(f"Found {len(self.joint_ids)} joints for {arm_side}")

        # Get actuator IDs for this arm
        self.actuator_names = [f"{prefix}_actuator{i}" for i in range(1, 8)]
        self.actuator_ids = []
        for actuator_name in self.actuator_names:
            try:
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                self.actuator_ids.append(actuator_id)
            except KeyError:
                raise ValueError(f"Actuator {actuator_name} not found in model")
        self.actuator_ids = np.array(self.actuator_ids)

        print(f"Found {len(self.actuator_ids)} actuators for {arm_side}")

        # Print actuator info
        num_actuators = self.model.nu
        print(f"Detected {num_actuators} total actuators in the OpenArm model.")
        for i in range(num_actuators):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  Index {i}: {name}")

    def _setup_controllers(self, config: dict):
        """
        Initialize PID controllers and state from config.

        Args:
            config: Configuration dictionary with controller gains
        """
        dt = config.get('dt', self.model.opt.timestep)
        num_joints = len(self.joint_ids)

        # Position controller
        pos_cfg = config['position_controller']
        self.pos_controller = FF_Controllers(
            _kp=np.array(pos_cfg['kp']),
            _ki=np.array(pos_cfg['ki']),
            _kd=np.array(pos_cfg['kd']),
            _dt=dt,
            _kgc=np.array(pos_cfg.get('kgc', [0.0] * num_joints)),
            _use_ff=[False] * num_joints, # Not implemented
            _max_lim=np.array(pos_cfg['max_lim']),
            _min_lim=np.array(pos_cfg['min_lim']),
            model=self.model,
            data=self.data
        )

        # Velocity controller
        vel_cfg = config['velocity_controller']
        self.vel_controller = FF_Controllers(
            _kp=np.array(vel_cfg['kp']),
            _ki=np.array(vel_cfg['ki']),
            _kd=np.array(vel_cfg['kd']),
            _dt=dt,
            _kgc=np.array(vel_cfg.get('kgc', [0.0] * num_joints)),
            _use_ff=[False] * num_joints,
            _max_lim=np.array(vel_cfg['max_lim']),
            _min_lim=np.array(vel_cfg['min_lim']),
            model=self.model,
            data=self.data
        )

        # Controller state
        self.pos_state = ControllerData(
            _ref=np.zeros(num_joints),
            _fb=np.zeros(num_joints),
            _prev_int=np.zeros(num_joints),
            _prev_err=np.zeros(num_joints)
        )

        self.vel_state = ControllerData(
            _ref=np.zeros(num_joints),
            _fb=np.zeros(num_joints),
            _prev_int=np.zeros(num_joints),
            _prev_err=np.zeros(num_joints)
        )

        self._gravity_comp = GravityCompensationSim(self.arm_side, self.model, self.data)

        # Setup observer for data recording
        self.observer = Observer(num_joints=num_joints, dt=dt)

    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C signal for graceful shutdown."""
        print("\nCtrl+C detected. Shutting down...")
        self.running = False

    def _keyboard_callback(self, keycode: int):
        """
        Handle keyboard input from viewer.

        Args:
            keycode: Key code from MuJoCo viewer
        """
        # Q key to quit
        if keycode == 81:
            self.running = False

    def _cascaded_controller_call(self) -> np.ndarray:
        """
        Execute cascaded position -> velocity control.

        Returns:
            Control signal (torque command)
        """
        # Position controller: outputs velocity setpoint
        pos_signal, self.pos_state._prev_int, self.pos_state._prev_err = \
            self.pos_controller.FF_PID_controller(
                np.zeros(len(self.joint_ids)),  # No feedforward for position
                self.pos_state._ref,
                self.pos_state._fb,
                self.pos_state._prev_int,
                self.pos_state._prev_err
            )
        vel_signal, self.vel_state._prev_int, self.vel_state._prev_err = \
            self.vel_controller.FF_PID_controller(
                self._gravity_comp.compute_gravity_torques(self.pos_state._fb),  # Gravity compensation can go here
                self.vel_state._ref + pos_signal,
                self.vel_state._fb,
                self.vel_state._prev_int,
                self.vel_state._prev_err
            )
        return vel_signal

    def run(self):
        """Run the main simulation loop with viewer."""
        print("Starting simulation...")

        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self._keyboard_callback
        ) as viewer:
            start_time = time.time()

            while viewer.is_running() and self.running:
                step_start = time.time()
                elapsed = time.time() - start_time

                # Read feedback from simulation
                self.pos_state._fb = self.data.actuator_length[self.actuator_ids]
                self.vel_state._fb = self.data.actuator_velocity[self.actuator_ids]

                # Generate test trajectory (sine wave for now)
                # TODO: Replace with actual trajectory generator
                for i in range(len(self.joint_ids)):
                    amplitude = 0.5
                    frequency = 0.5
                self.pos_state._ref = np.array([-2., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                # Compute control
                control_signal = self._cascaded_controller_call()
                # Apply control to actuators
                self.data.ctrl[self.actuator_ids] = control_signal

                # Record data
                self.observer.record(
                    time=elapsed,
                    pos_ref=self.pos_state._ref,
                    pos_fb=self.pos_state._fb,
                    vel_ref=self.vel_state._ref,
                    vel_fb=self.vel_state._fb,
                    torque_cmd=control_signal
                )

                # Step physics
                mujoco.mj_step(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)

                # Sync viewer
                viewer.sync()

                # Maintain real-time
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        self.cleanup()

    def cleanup(self):
        """Perform cleanup on shutdown."""
        print(f"Simulation shutdown complete. Recorded {len(self.observer)} samples.")

        # Plot recorded data
        if len(self.observer) > 0:
            print("Generating plots...")
            plotter = SignalPlotter(self.observer, joint_names=self.joint_names)
            plotter.plot_all()
            plotter.show()


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description='MuJoCo simulation for OpenArm')
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to MJCF model file (default: models/openarm_mujoco/v1/openarm_bimanual.xml)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML config file (default: config/sim_config.yaml)'
    )
    parser.add_argument(
        '--arm_side',
        type=str,
        choices=['left_arm', 'right_arm'],
        default='left_arm',
        help='Which arm to control (default: left_arm)'
    )

    args = parser.parse_args()

    # Determine paths relative to script location
    script_dir = Path.cwd()
    repo_root = script_dir.parent

    if args.model_path is None:
        model_path = repo_root / 'models' / 'openarm_mujoco' / 'v1' / 'openarm_bimanual.xml'
    else:
        model_path = Path(args.model_path)

    if args.config is None:
        config_path = script_dir / 'config' / 'sim_config.yaml'
    else:
        config_path = Path(args.config)

    # Validate paths
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    # Create and run simulation
    sim = MuJoCoSimulation(
        model_path=str(model_path),
        config_path=str(config_path),
        arm_side=args.arm_side
    )
    sim.run()

    return 0

if __name__ == "__main__":
    exit(main())
