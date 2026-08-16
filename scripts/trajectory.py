#!/usr/bin/env python3
"""Time-synchronised trapezoidal joint-space trajectory generation.

Commanding a setpoint as a step creates an instantaneous position error that
saturates the actuators and excites oscillation no matter how the loops are
tuned. This module turns a target pose into a continuous reference q_ref(t)
together with its derivative qd_ref(t), which the cascaded controller consumes
as a velocity feedforward.

All joints are time-synchronised: the joint with the longest travel sets the
total duration, and every other joint's cruise velocity is scaled down so it
arrives at the same instant. The joint-space path is therefore a straight line
and no joint is ever asked to move faster than its own limit.
"""
import numpy as np


class TrapezoidalProfile:
    """Time-synchronised trapezoidal velocity profile over N joints."""

    def __init__(self, q0, qf, max_vel, max_acc, settle_time: float = 0.0):
        self.q0 = np.asarray(q0, dtype=float)
        self.qf = np.asarray(qf, dtype=float)
        self.max_vel = np.asarray(max_vel, dtype=float)
        self.max_acc = np.asarray(max_acc, dtype=float)
        self.settle_time = float(settle_time)

        self.delta = self.qf - self.q0
        self.dist = np.abs(self.delta)
        self.sign = np.sign(self.delta)
        n = len(self.q0)

        # Per-joint minimum duration: trapezoidal if the joint reaches cruise
        # velocity, triangular otherwise.
        t_min = np.zeros(n)
        for i in range(n):
            d, v, a = self.dist[i], self.max_vel[i], self.max_acc[i]
            if d <= 1e-12:
                t_min[i] = 0.0
            elif d >= v * v / a:
                t_min[i] = d / v + v / a          # accelerate, cruise, decelerate
            else:
                t_min[i] = 2.0 * np.sqrt(d / a)   # triangular: never reaches v
        self.duration = float(t_min.max())

        # Re-solve each joint's cruise velocity so it takes exactly `duration`,
        # holding its acceleration limit: d = v*T - v^2/a  =>  v^2 - aTv + ad = 0.
        # The smaller root is the feasible (slower) branch; the discriminant is
        # non-negative because T >= 2*sqrt(d/a) for every joint by construction.
        self.v = np.zeros(n)
        self.a = np.array(self.max_acc, dtype=float)
        T = self.duration
        for i in range(n):
            d, a = self.dist[i], self.max_acc[i]
            if d <= 1e-12 or T <= 0.0:
                self.v[i] = 0.0
                continue
            disc = max(a * a * T * T - 4.0 * a * d, 0.0)
            self.v[i] = (a * T - np.sqrt(disc)) / 2.0
        self.t_acc = np.where(self.v > 1e-12, self.v / self.a, 0.0)

    @property
    def total_time(self) -> float:
        """Settle hold plus motion duration."""
        return self.settle_time + self.duration

    def at(self, t: float):
        """Reference position and velocity at time ``t``.

        Returns:
            (q_ref, qd_ref) — both shape (num_joints,)
        """
        n = len(self.q0)
        if t < self.settle_time:
            return self.q0.copy(), np.zeros(n)

        tm = t - self.settle_time
        q = np.empty(n)
        qd = np.zeros(n)
        T = self.duration

        for i in range(n):
            d = self.dist[i]
            if d <= 1e-12 or T <= 0.0:
                q[i] = self.qf[i]
                continue
            a, v, ta = self.a[i], self.v[i], self.t_acc[i]
            if tm >= T:
                q[i] = self.qf[i]
            elif tm < ta:                                   # accelerating
                q[i] = self.q0[i] + self.sign[i] * 0.5 * a * tm * tm
                qd[i] = self.sign[i] * a * tm
            elif tm < T - ta:                               # cruising
                d_acc = 0.5 * a * ta * ta
                q[i] = self.q0[i] + self.sign[i] * (d_acc + v * (tm - ta))
                qd[i] = self.sign[i] * v
            else:                                           # decelerating
                td = T - tm
                q[i] = self.qf[i] - self.sign[i] * 0.5 * a * td * td
                qd[i] = self.sign[i] * a * td
        return q, qd
