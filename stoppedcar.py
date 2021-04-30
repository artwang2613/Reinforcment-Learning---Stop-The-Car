import numpy as np

from typing import Tuple


class StoppedCar(object):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.

    Actions: left (0) and right (1)
    Reward: 1 always
    """

    def __init__(self):
        self._x = 0.  # position of the front of the car
        self._v = 0.  # velocity of the car

        # dynamics
        self._kcoef = -2.5  # breaking coefficient
        self._xc = 10.0  # position of the back of the car
        self._xg = 8.5  # goal position
        self._epsilon = 1.0
        self.dt = 0.05
        self.C = 0.1  # static coefficient of friction

        ranges = np.zeros((2,2))
        ranges[0, :] = [-1.0, self._xc]
        ranges[1, :] = [-0.5, 11.0]
        self.obs_ranges = ranges
        self.num_actions = 2

    def step(self, action: int)->Tuple[np.ndarray, float, bool]:
        x,v = self._x, self._v
        xc, xg, epsilon, dt, C = self._xc, self._xg, self._epsilon, self.dt, self.C

        xdot = v
        vdot = 0 + (action == 1) * self._kcoef * v

        x = x + dt * xdot
        v = v + dt * vdot

        if abs(v) < self.C:
            v = 0.0

        reward = 0.0
        collision = False
        if x >= xc - 1e-8:
            v = 0.0
            collision = True

        done = False

        if v == 0.0:
            done = True
            if abs(x - xg) <= epsilon:
                reward = 10.0  # landed in goal region
            elif collision:
                reward = -10.0  # hit car
            else:
                reward = -1.0  # not inside goal region
        self._x = x
        self._v = v
        obs = np.array([xc - x, v])
        return obs, reward, done

    def reset(self):
        """
        resets the state of the environment to the initial configuration
        """
        self._x = 0. + np.random.uniform(0.0, 1.0)
        self._v = 10. + np.random.uniform(-2.0, 0.0)
        return np.array([self._xc - self._x, self._v])