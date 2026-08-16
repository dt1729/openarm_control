import mujoco
import numpy as np

from dataclasses import dataclass, field

@dataclass
class PID_Controller:
    _kp : np.ndarray
    _ki : np.ndarray
    _kd : np.ndarray
    _kgc : np.ndarray # Feed forward gravity compensation gain
    _use_ff : list
    _dt : float
    _max_lim : np.ndarray
    _min_lim : np.ndarray
    model : mujoco.MjModel
    data: mujoco.MjData
    active_joint: int = 0  # Default value

@dataclass
class ControllerData:
    _ref     : np.ndarray
    _fb      : np.ndarray
    _prev_int: np.ndarray   # accumulated RAW error integral (not gain-scaled)
    _prev_err: np.ndarray


class FF_Controllers(PID_Controller):
    """Feedforward PID with trapezoidal integration and back-calculated anti-windup.

    The integral state carried between calls is the *raw* error integral. It is
    scaled by ``ki`` only when forming the output, so the accumulator is never
    multiplied by the gain more than once.
    """

    def FF_PID_controller(self, _FF_signal : np.ndarray, _set_point : np.ndarray,
                          _state_fb : np.ndarray, _prev_int : np.ndarray,
                          _prev_err : np.ndarray):
        """A feedforward PID controller; the caller owns the FF signal entirely
        (intended for gravity compensation).

        Args:
            _FF_signal (np.ndarray): Feedforward term, added directly to the output
            _set_point (np.ndarray): Set point for every motor in the kinematic chain
            _state_fb  (np.ndarray): State feedback
            _prev_int  (np.ndarray): Previous RAW error integral
            _prev_err  (np.ndarray): Previous error

        Returns:
            tuple: (control signal, updated raw error integral, current error)
        """
        _err = _set_point - _state_fb

        _prop = np.multiply(self._kp, _err)
        _der = np.multiply(self._kd, (_err - _prev_err) / self._dt)

        # Trapezoidal integration of the raw error: I += (e[k] + e[k-1]) * dt/2
        _integ = _prev_int + (_err + _prev_err) * self._dt / 2.0
        _int = np.multiply(self._ki, _integ)

        # Symmetric dynamic anti-windup: the integral may only use the headroom
        # left between the non-integral terms and the output limits.
        _rest = _prop + _der + _FF_signal
        _head_max = np.maximum(self._max_lim - _rest, 0.0)
        _head_min = np.minimum(self._min_lim - _rest, 0.0)
        _int_clamped = np.clip(_int, _head_min, _head_max)

        # Back-calculate the accumulator from the clamped term so a saturated
        # output cannot keep charging the integrator.
        _ki_safe = np.where(np.abs(self._ki) > 1e-12, self._ki, 1.0)
        _integ = np.where(np.abs(self._ki) > 1e-12, _int_clamped / _ki_safe, 0.0)

        _out = np.clip(_rest + _int_clamped, self._min_lim, self._max_lim)
        return _out, _integ, _err
