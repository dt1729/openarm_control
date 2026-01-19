#!/usr/bin/env python3
# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import signal
import sys
import time
import pdb
from pathlib import Path

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

# try:
#     import openarm_can as oa
# except ImportError:
#     print("Error: openarm_can module not found. Please install it first.")
#     sys.exit(1)


class GravityCompensation:
    """Gravity compensation controller for OpenArm using MuJoCo dynamics."""

    def __init__(self, arm_side: str, can_interface: str, mjcf_path: str):
        """
        Initialize gravity compensation controller.

        Args:
            arm_side: Either 'left_arm' or 'right_arm'
            can_interface: CAN interface name (e.g., 'can0')
            mjcf_path: Path to the MJCF model file
        """
        if arm_side not in ['left_arm', 'right_arm']:
            raise ValueError(f"Invalid arm_side: {arm_side}. Must be 'left_arm' or 'right_arm'.")

        if not os.path.exists(mjcf_path):
            raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")

        self.arm_side = arm_side
        self.can_interface = can_interface
        self.mjcf_path = mjcf_path
        self.keep_running = True

        # Initialize MuJoCo model and data
        print(f"Loading MuJoCo model from: {mjcf_path}")
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        # Determine joint names based on arm side
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

        print(f"Found {len(self.joint_ids)} joints for {arm_side}")

        # Initialize OpenArm hardware
        print(f"Initializing OpenArm on {can_interface}")
        self.openarm = oa.OpenArm(can_interface, True)

        # Initialize arm motors based on the model (DM8009, DM8009, DM4340, DM4340, DM4310, DM4310, DM4310)
        motor_types = [
            oa.MotorType.DM8009,
            oa.MotorType.DM8009,
            oa.MotorType.DM4340,
            oa.MotorType.DM4340,
            oa.MotorType.DM4310,
            oa.MotorType.DM4310,
            oa.MotorType.DM4310,
        ]

        # Configure motor IDs (adjust these based on your hardware setup)
        send_ids = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
        recv_ids = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]

        self.openarm.init_arm_motors(motor_types, send_ids, recv_ids)
        self.openarm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.openarm.enable_all()
        self.openarm.recv_all()

        # Set up signal handler
        signal.signal(signal.SIGINT, self._signal_handler)

        print("=== OpenArm Gravity Compensation ===")
        print(f"Arm side       : {arm_side}")
        print(f"CAN interface  : {can_interface}")
        print(f"MJCF path      : {mjcf_path}")
        print(f"Number of DOFs : {len(self.joint_ids)}")

    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C signal."""
        print("\nCtrl+C detected. Exiting loop...")
        self.keep_running = False

    def compute_gravity_torques(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Compute gravity compensation torques using MuJoCo.

        Args:
            joint_positions: Array of joint positions (radians)

        Returns:
            Array of gravity compensation torques
        """
        # pdb.set_trace()
        # Set joint positions in MuJoCo data
        for i, joint_id in enumerate(self.joint_ids):
            self.data.qpos[joint_id] = joint_positions[i]

        # Set velocities to zero for gravity computation
        self.data.qvel[:] = 0

        # Forward kinematics and dynamics
        mujoco.mj_forward(self.model, self.data)

        # Get gravity compensation torques (negative of bias forces)
        # qfrc_bias contains gravitational, centrifugal and Coriolis forces
        # Since velocity is zero, it's just gravity
        grav_torques = np.zeros(len(self.joint_ids))
        for i, joint_id in enumerate(self.joint_ids):
            grav_torques[i] = self.data.qfrc_bias[joint_id]

        return grav_torques

    def run(self):
        """Run the gravity compensation control loop."""
        start_time = time.time()
        last_hz_display = start_time
        frame_count = 0

        print("Starting control loop...")

        while self.keep_running:
            frame_count += 1
            current_time = time.time()

            # Calculate and display Hz every second
            time_since_last_display = (current_time - last_hz_display) * 1000
            if time_since_last_display >= 1000:  # Every 1000ms (1 second)
                total_time = (current_time - start_time) * 1000
                hz = (frame_count * 1000.0) / total_time
                print(f"=== Loop Frequency: {hz:.2f} Hz ===")
                last_hz_display = current_time

            # Read joint positions from hardware
            motors = self.openarm.get_arm().get_motors()
            joint_positions = np.zeros(len(motors))

            for i, motor in enumerate(motors):
                joint_positions[i] = motor.get_position()

            # Compute gravity compensation torques
            grav_torques = self.compute_gravity_torques(joint_positions)

            # Send torque commands to motors
            cmds = [oa.MITParam(0, 0, 0, 0, float(t)) for t in grav_torques]
            self.openarm.get_arm().mit_control_all(cmds)

            # Receive feedback
            self.openarm.recv_all()

        # Cleanup
        print("Disabling motors...")
        time.sleep(0.1)
        self.openarm.disable_all()
        self.openarm.recv_all()
        print("Shutdown complete.")

class GravityCompensationSim:
    def __init__(self, arm_side: str, _model : mujoco.MjModel, _data : mujoco.MjData):
        """
        Initialize gravity compensation controller.

        Args:
            arm_side: Either 'left_arm' or 'right_arm'
            can_interface: CAN interface name (e.g., 'can0')
            mjcf_path: Path to the MJCF model file
        """
        if arm_side not in ['left_arm', 'right_arm']:
            raise ValueError(f"Invalid arm_side: {arm_side}. Must be 'left_arm' or 'right_arm'.")


        self.arm_side = arm_side
        self.keep_running = True

        # Initialize MuJoCo model and data
        self.model = _model
        self.data = _data

        # Determine joint names based on arm side
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

        logger.info(f"Found {len(self.joint_ids)} joints for {arm_side}")
        logger.info("=== OpenArm Gravity Compensation ===")
        logger.info(f"Arm side       : {arm_side}")
        logger.info(f"Number of DOFs : {len(self.joint_ids)}")

    def compute_gravity_torques(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Compute gravity compensation torques using MuJoCo.

        Args:
            joint_positions: Array of joint positions (radians)

        Returns:
            Array of gravity compensation torques
        """
        # pdb.set_trace()
        # Set joint positions in MuJoCo data
        for i, joint_id in enumerate(self.joint_ids):
            self.data.qpos[joint_id] = joint_positions[i]

        # Set velocities to zero for gravity computation
        self.data.qvel[:] = 0

        # Forward kinematics and dynamics
        mujoco.mj_forward(self.model, self.data)

        # Get gravity compensation torques (negative of bias forces)
        # qfrc_bias contains gravitational, centrifugal and Coriolis forces
        # Since velocity is zero, it's just gravity
        grav_torques = np.zeros(len(self.joint_ids))
        for i, joint_id in enumerate(self.joint_ids):
            grav_torques[i] = self.data.qfrc_bias[joint_id]

        return grav_torques