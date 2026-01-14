import mujoco
import numpy as np

from dataclasses import dataclass, field

@dataclass
class PID_Controller:
    _kp : np.ndarray
    _ki : np.ndarray
    _kd : np.ndarray
    _dt : np.ndarray
    _kgc : np.ndarray # Feed forward gravity compensation gain
    _use_ff : list
    _dt : float
    _max_lim : np.ndarray
    _min_lim : np.ndarray
    model : mujoco.Mjmodel
    data: mujoco.MjData
    active_joint: int = 0  # Default value

@dataclass
class ControllerData:
    _ref    : np.ndarray
    _fb     : np.ndarray 
    _prev_int: np.ndarray 
    _prev_err: np.ndarray


#TODO rewrite this with controllerData, simpler arguments and everything
class FF_Controllers(PID_Controller):
    def FF_PID_controller(self, _FF_signal : np.ndarray, _set_point : np.ndarray, _state_fb : np.ndarray, _prev_int : np.ndarray, _prev_err : np.ndarray) -> tuple:
        """A Feed forward PID controller, user controls FF signal completely, intended for Gravity compensation

        Args:
            _FF_signal(np.ndarray) : Feedforward signal from user
            _set_point(np.ndarray) : Set point values for all the motors in the kinematic chain
            _state_fb (np.ndarray) : State Feedback for velocity
            _prev_int (np.ndarray) : Previous value of integrator term.  
            _prev_err (np.ndarray) : Previous value of the calculated error
        Returns:
            tuple: tuple of np.ndarrays of the following values(PID control signal, integrator signal, error signal)
        """
        _err = _set_point - _state_fb
        _prop = np.multiply(self._kp, _err)

        # Dynamic anti-windup circuitry, turns off the integrand term if the signal will exceed the limits
        self._lim_max_integ = np.array([0.0 if self._max_lim[i] <= _prop[i] \
                                        else self._max_lim[i] - _prop[i] \
                                        for i in range(len(self._max_lim))])
        self._lim_min_integ = np.array([0.0 if self._max_lim[i] > _prop[i]\
                                        else self._min_lim[i] - _prop[i]\
                                        for i in range(len(self._min_lim))])   

        # Backwards Trapezoidal integration
        _int = np.multiply(self._ki, (_prev_int + (_err + self._prev_err)/self._dt))
        _int = np.clip(_int, self._lim_min_integ, self._lim_max_integ)
        _der = np.multiply(self._kd, (_err - _prev_err)/self._dt)
        
        return tuple(np.clip(_prop + _int + _der + _FF_signal, self._min_lim, self._max_lim), _int, _err)
                                                    
