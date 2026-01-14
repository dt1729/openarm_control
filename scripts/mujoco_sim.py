import mujoco
import mujoco.viewer
import time
import numpy as np

from typing import TypedDict
from pathlib import Path
from controller_impl import FF_Controllers, ControllerData

# --- GLOBAL CONTROL STATE ---
# This dictionary tracks which keys are being pressed
key_state = {
    "quit": False,
    "reset": False,
    "active_actuator": 0,
    "direction": 0  # 1 for forward, -1 for backward, 0 for static
}

def keyboard_callback(keycode):
    """
    Only used to update our internal state. 
    Logic happens in the main loop.
    """
    global key_state
    
    # Q to Quit
    if keycode == 81: 
        key_state["quit"] = True

def cascaded_controller_call(position_control : FF_Controllers, velocity_control : FF_Controllers, model : mujoco.MjModel, data :mujoco.MjData, pos_controller_data : ControllerData, vel_controller_data : ControllerData):
    pos_signal, pos_controller_data._prev_int, pos_controller_data._prev_err = position_control.FF_PID_controller(np.array([0 for i in range(7)]),\
                                                                                    pos_controller_data._ref,\
                                                                                    pos_controller_data._fb,\
                                                                                    pos_controller_data._prev_int,\
                                                                                    pos_controller_data._prev_err)

    # for gravity compensation update the first term with gravity compensation controller and required torque setpoint for min jerk.
    vel_signal, vel_controller_data._prev_int, vel_controller_data._prev_err = velocity_control.FF_PID_controller(np.array([0 for i in range(7)]),\
                                                                                    vel_controller_data._ref + pos_signal,\
                                                                                    vel_controller_data._fb,\
                                                                                    vel_controller_data._prev_int,\
                                                                                    vel_controller_data._prev_err)
    
    return vel_signal, pos_controller_data, vel_controller_data                               

# 1. Load the model
# Ensure 'openarm.xml' and its mesh folder are in the same directory as this script.
try:
    _parent_dir = Path.cwd().parent
    model = mujoco.MjModel.from_xml_path(str(_parent_dir) + '/models/openarm_mujoco/v1/openarm_bimanual.xml')
    data = mujoco.MjData(model)
    model.opt.gravity = (0, 0, -9.81)
except ValueError as e:
    print(f"Error loading XML: {e}")
    exit()

# 2. Setup control parameters
num_actuators = model.nu
print(f"Detected {num_actuators} actuators in the OpenArm model.")

# Print actuator names for debugging
for i in range(num_actuators):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"Index {i}: Actuator Name: {name}")

prefix = "openarm_left" if arm_side == "left_arm" else "openarm_right"
joint_names = [f"{prefix}_joint{i}" for i in range(1, 8)]

# Get joint IDs
joint_ids = []
for joint_name in joint_names:
    try:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        joint_ids.append(joint_id)
    except KeyError:
        raise ValueError(f"Joint {joint_name} not found in model")


pos_cntrlr, vel_cntrlr  = FF_Controllers(), FF_Controllers()
pos_state = ControllerData(np.zeros(model.nu), np.zeros(model.nu), np.zeros(model.nu), np.zeros(model.nu))
vel_state = ControllerData(np.zeros(model.nu), np.zeros(model.nu), np.zeros(model.nu), np.zeros(model.nu))

# 3. Launch the simulation
with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_callback) as viewer:
    start_time = time.time()    
    while viewer.is_running()and not key_state["quit"]:
        step_start = time.time()
        elapsed = time.time() - start_time

        vel_state._fb = np.array([data.qvel[joint_id] for joint_id in joint_ids])
        pos_state._fb = np.array([data.qpos[joint_id] for joint_id in joint_ids])
        # --- Controls Algorithm goes here --- 
        # 1. Get feedback of all actuators.
        # 2. Bin the feedbacks as per left, right arm.
        # 3. For each arm 
        #   i. For each actuator
        #       a. Call PID controller for Position -> Velocity control based on position set point
        #           - Call PID controller for Velocity -> Torque control based on d(position)/dt set point
        #               4. Send cmd to data.ctrl for each 
        # Block Diagram for the same
        #                               Vel Feed-forward       Cur Feed-forward
        #                                      |                       |   
        #                                      v                       v
        #                                     (+)                     (+)
        #  Pos     +---+   +------------+   +---+   +------------+   +---+   +------------+   +-------+
        #  Cmd --->| Σ |-->|  Position  |-->| Σ |-->|  Velocity  |-->| Σ |-->|  Current   |-->| Power |--+--> ( M ) -- Load
        #   (+)    +-^-+   | Controller |   +-^-+   | Controller |   +-^-+   | Controller |   | Stage |  |      |
        #            | (-) +------------+     | (-) +------------+     | (-) +------------+   +-------+  |      |
        #            |                        |                        |                                 |      |
        #            |                      +----+                     +-------- Current Feedback -------+      |
        #            |                      |d/dt|                                                              |
        #            |                      +--^-+                          Position Feedback                   |
        #            |                         |                                (Encoder)                       |
        #            +-------------------------+----------------------------------------------------------------+



        # --- SEND COMMANDS TO ALL ACTUATORS ---
        # We generate a unique sine wave for each joint to see them all moving
        for i in range(num_actuators):
            # Amplitude and frequency for the motion
            amplitude = 0.5 
            frequency = 1.0
            
            cascaded_controller_call(pos_cntrlr, vel_cntrlr, model, data, pos_controller_data=pos_data, vel_controller_data=vel_data)
            # Calculate command: ctrl = sin(t + offset)
            # data.ctrl maps directly to the 'actuator' tags in your XML
            data.ctrl[i] = amplitude * np.sin(frequency * elapsed + (i * 0.5))
        # 4. Step the physics
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        # 5. Sync viewer with the new state
        viewer.sync()

        # 6. Maintain real-time frequency (e.g., 60Hz or based on model timestep)
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)