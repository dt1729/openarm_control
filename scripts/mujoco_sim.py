#!/usr/bin/env python3
import argparse
import logging
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
from trajectory import TrapezoidalProfile

# Setup logger
logger = logging.getLogger(__name__)

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
        logger.info(f"Loading MuJoCo model from: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity = (0, 0, -9.81)
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

        # Load configuration (before the timestep, which the config now owns)
        self.config = self._load_config(config_path)
        self.model.opt.timestep = float(self.config.get('dt', 1e-3))

        # Setup arm joints
        self._setup_arm(arm_side)

        # Setup controllers
        self._setup_controllers(self.config)

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info("=== MuJoCo Simulation Initialized ===")
        logger.info(f"Arm side       : {arm_side}")
        logger.info(f"Model path     : {model_path}")
        logger.info(f"Config path    : {config_path}")
        logger.info(f"Number of DOFs : {len(self.joint_ids)}")
        logger.info(f"Timestep       : {self.model.opt.timestep}")

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
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint {joint_name} not found in model")
            self.joint_ids.append(joint_id)
        self.joint_ids = np.array(self.joint_ids)

        logger.info(f"Found {len(self.joint_ids)} joints for {arm_side}")

    def _setup_controllers(self, config: dict):
        """
        Initialize PID controllers and state from config.

        Args:
            config: Configuration dictionary with controller gains
        """
        # The controller sampling interval MUST be the physics step: the loop
        # runs the control law once per mj_step. A config value that disagrees
        # silently scales every integral and derivative term.
        dt = float(self.model.opt.timestep)
        cfg_dt = config.get('dt', None)
        if cfg_dt is not None and abs(float(cfg_dt) - dt) > 1e-12:
            logger.warning(
                f"config dt={cfg_dt} disagrees with model timestep={dt}; "
                f"using the model timestep."
            )
        num_joints = len(self.joint_ids)

        # Motion-profile limits for the trajectory generator
        self.motion_cfg = config.get('motion', {
            'enabled': True,
            'max_vel': [1.0] * num_joints,
            'max_acc': [1.0] * num_joints,
            'settle_time': 0.5,
        })

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
        logger.warning("Ctrl+C detected. Shutting down...")
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

        gravity_torques = self._gravity_comp.compute_gravity_torques(self.pos_state._fb)
        vel_ref_total = self.vel_state._ref + pos_signal

        vel_signal, self.vel_state._prev_int, self.vel_state._prev_err = \
            self.vel_controller.FF_PID_controller(
                gravity_torques,
                vel_ref_total,
                self.vel_state._fb,
                self.vel_state._prev_int,
                self.vel_state._prev_err
            )

        # Detailed debug logging
        pos_error = self.pos_state._ref - self.pos_state._fb
        vel_error = vel_ref_total - self.vel_state._fb
        logger.debug(f"pos_ref:        {self.pos_state._ref}")
        logger.debug(f"pos_fb:         {self.pos_state._fb}")
        logger.debug(f"pos_error:      {pos_error}")
        logger.debug(f"vel_ref:        {vel_ref_total}")
        logger.debug(f"vel_fb:         {self.vel_state._fb}")
        logger.debug(f"vel_error:      {vel_error}")
        logger.debug(f"gravity_torque: {gravity_torques}")
        logger.debug(f"torque_cmd:     {vel_signal}")

        return vel_signal

    def go_to_motor_angles(
        self,
        motor_angles: np.ndarray,
        timeout: float = 30.0,
        sim_duration: float = None
    ):
        """
        Drive the arm to target motor angles headless.

        Args:
            motor_angles: Target joint angles (rad), shape (num_joints,)
            timeout: Wall-clock safety timeout in seconds (default 30.0)
            sim_duration: Simulated seconds to run. When given, this bounds the
                run instead of wall time, so results are reproducible across
                machines; `timeout` remains only as a safety stop.
        """
        motor_angles = np.asarray(motor_angles)
        if motor_angles.shape[0] != len(self.joint_ids):
            raise ValueError(
                f"motor_angles has {motor_angles.shape[0]} elements, "
                f"expected {len(self.joint_ids)}"
            )

        logger.info(f"Target motor angles: {motor_angles}")
        logger.info(f"Running headless (wall timeout: {timeout}s)...")

        # Reset observer for fresh recording
        self.observer.clear()

        # Reset controller states
        self._reset_controller_states()

        # Build a time-synchronised trapezoidal reference from the current pose.
        profile = None
        if self.motion_cfg.get('enabled', True):
            q0 = np.array(self.data.actuator_length[self.joint_ids])
            profile = TrapezoidalProfile(
                q0=q0,
                qf=motor_angles,
                max_vel=np.array(self.motion_cfg['max_vel'], dtype=float),
                max_acc=np.array(self.motion_cfg['max_acc'], dtype=float),
                settle_time=float(self.motion_cfg.get('settle_time', 0.0)),
            )
            logger.info(
                f"Trajectory: {profile.duration:.2f}s motion "
                f"(+{profile.settle_time:.2f}s settle), "
                f"cruise vel {np.array2string(profile.v, precision=3)}"
            )

        sim_time = 0.0
        start_wall_time = time.time()
        next_log_time = 5.0  # Log every 5s wall time

        while self.running:
            wall_elapsed = time.time() - start_wall_time
            if sim_duration is not None:
                if sim_time >= sim_duration:
                    break
                if wall_elapsed >= timeout:
                    logger.warning(
                        f"Wall-clock safety timeout hit at sim_time={sim_time:.3f}s "
                        f"of {sim_duration}s requested."
                    )
                    break
            elif wall_elapsed >= timeout:
                break
            # Read feedback from simulation
            self.pos_state._fb = self.data.actuator_length[self.joint_ids]
            self.vel_state._fb = self.data.actuator_velocity[self.joint_ids]

            # Reference comes from the trajectory, not the raw target: a step
            # setpoint saturates the actuators regardless of gains. qd_ref is
            # injected as velocity feedforward into the inner loop.
            if profile is not None:
                q_ref, qd_ref = profile.at(sim_time)
                self.pos_state._ref = q_ref
                self.vel_state._ref = qd_ref
            else:
                self.pos_state._ref = motor_angles

            # Compute control
            control_signal = self._cascaded_controller_call()

            # Apply control to actuators
            self.data.ctrl[self.joint_ids] = control_signal

            # Record data
            self.observer.record(
                time=sim_time,
                pos_ref=self.pos_state._ref,
                pos_fb=self.pos_state._fb,
                vel_ref=self.vel_state._ref,
                vel_fb=self.vel_state._fb,
                torque_cmd=control_signal
            )

            # Step physics. mj_forward refreshes actuator_length/actuator_velocity
            # after integration so the next iteration reads current feedback.
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)

            sim_time += self.model.opt.timestep

            # Log progress every 5s wall time
            if wall_elapsed >= next_log_time:
                pos_error = np.abs(motor_angles - self.pos_state._fb)
                logger.info(f"Wall: {wall_elapsed:.1f}s | Sim: {sim_time:.2f}s | Max error: {np.max(pos_error):.4f} rad")
                next_log_time += 5.0

        wall_elapsed = time.time() - start_wall_time
        logger.info(f"Simulation complete. Wall time: {wall_elapsed:.1f}s, Sim time: {sim_time:.3f}s")
        final_error = np.abs(motor_angles - self.pos_state._fb)
        logger.info(f"Final position error: {final_error}")

    def _reset_controller_states(self):
        """Reset controller integral and error states."""
        num_joints = len(self.joint_ids)
        self.pos_state._prev_int = np.zeros(num_joints)
        self.pos_state._prev_err = np.zeros(num_joints)
        self.vel_state._prev_int = np.zeros(num_joints)
        self.vel_state._prev_err = np.zeros(num_joints)

    def run(self, replay: bool = True, playback_speed: float = None):
        """Run a demo moving to fixed target motor angles.

        Args:
            replay: Whether to replay in 3D viewer after simulation (default True)
            playback_speed: Playback speed multiplier for 3D viewer. If None,
                            auto-calculated to complete replay in ~5 seconds.
        """
        logger.info("Starting simulation...")

        # Demo target angles
        target_angles = np.array([1.0, -2., 0.5, 0.5, 1.0, -0.4, 0.4])

        self.go_to_motor_angles(target_angles, timeout=30.0)

        self.cleanup()

        if replay:
            if playback_speed is None:
                # Auto-calculate: aim for ~5 second replay
                sim_duration = len(self.observer) * self.observer.dt
                playback_speed = max(1.0, sim_duration / 5.0)
                logger.info(f"Auto playback speed: {playback_speed:.1f}x")
            self.replay_in_viewer(playback_speed=playback_speed)

        self.visualize()

    def visualize(self, save_dir: str = None, show: bool = True):
        """
        Visualize recorded data from observer.

        Args:
            save_dir: Optional directory to save plot images
            show: Whether to display plots interactively (default True)
        """
        if len(self.observer) == 0:
            logger.warning("No data to visualize. Run go_to_motor_angles() first.")
            return

        logger.info(f"Generating plots for {len(self.observer)} samples...")
        plotter = SignalPlotter(self.observer, joint_names=self.joint_names)
        plotter.plot_all(save_dir=save_dir)
        if show:
            plotter.show()

    def save_data(self, filepath: str):
        """
        Save recorded observer data to file for later visualization.

        Args:
            filepath: Path to save .npz file
        """
        self.observer.save(filepath)

    def load_data(self, filepath: str):
        """
        Load previously saved observer data.

        Args:
            filepath: Path to .npz file
        """
        self.observer.load(filepath)

    def cleanup(self):
        """Perform cleanup on shutdown."""
        logger.info(f"Simulation shutdown complete. Recorded {len(self.observer)} samples.")

    def replay_in_viewer(self, playback_speed: float = 1.0) -> None:
        """
        Replay recorded simulation in MuJoCo 3D viewer.

        Args:
            playback_speed: Playback multiplier (1.0 = real-time, 2.0 = 2x speed)
        """
        if len(self.observer) == 0:
            logger.warning("No data to replay. Run go_to_motor_angles() first.")
            return

        # Get recorded data
        data = self.observer.get_data()
        pos_fb = data['pos_fb']  # Shape: (num_samples, num_joints)
        dt = data['dt']
        num_frames = len(pos_fb)

        logger.info(f"Replaying {num_frames} frames in 3D viewer (playback_speed={playback_speed}x)")

        # Initialize model to first recorded position
        self.data.qpos[self.joint_ids] = pos_fb[0]
        mujoco.mj_forward(self.model, self.data)

        # Launch passive viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()

            while viewer.is_running():
                # Calculate target frame from wall-clock time
                elapsed_wall = time.time() - start_time
                sim_time = elapsed_wall * playback_speed
                frame_idx = int(sim_time / dt)

                # Check if replay is complete
                if frame_idx >= num_frames:
                    logger.info("Replay complete.")
                    break

                # Set joint positions to recorded values
                self.data.qpos[self.joint_ids] = pos_fb[frame_idx]

                # Update kinematics (no physics step, just forward kinematics)
                mujoco.mj_forward(self.model, self.data)

                # Sync viewer
                viewer.sync()

                # Sleep to prevent CPU spinning (1ms)
                time.sleep(0.001)

        logger.info("Viewer closed.")


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
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO). Use DEBUG for detailed control loop output.'
    )
    parser.add_argument(
        '--no-replay',
        action='store_true',
        help='Skip 3D viewer replay after simulation'
    )
    parser.add_argument(
        '--playback-speed',
        type=float,
        default=None,
        help='Playback speed multiplier for 3D viewer (default: auto-calculated for ~5s replay)'
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

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
        logger.error(f"Model file not found: {model_path}")
        return 1

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    # Create and run simulation
    sim = MuJoCoSimulation(
        model_path=str(model_path),
        config_path=str(config_path),
        arm_side=args.arm_side
    )
    sim.run(replay=not args.no_replay, playback_speed=args.playback_speed)

    return 0

if __name__ == "__main__":
    exit(main())
