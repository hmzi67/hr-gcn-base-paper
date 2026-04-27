"""
One-Euro filter for temporal smoothing of joint positions.
Per-joint, per-coordinate — stateful, never recreate between frames.
Reference: Casiez et al., 2012.
"""
import math


class OneEuroFilter:
    def __init__(self, freq: float, min_cutoff: float = 1.0,
                 beta: float = 0.007, d_cutoff: float = 1.0):
        self.freq       = freq
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x         = None
        self._dx        = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * self.freq)

    def __call__(self, x: float) -> float:
        if self._x is None:
            self._x = x
            return x
        dx = (x - self._x) * self.freq
        a_d = self._alpha(self.d_cutoff)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff)
        self._x = a * x + (1 - a) * self._x
        return self._x


class SkeletonFilter:
    """One-Euro filter applied independently to every joint coordinate."""
    def __init__(self, n_joints: int, n_coords: int = 3, freq: float = 30.0):
        self.filters = [
            [OneEuroFilter(freq=freq) for _ in range(n_coords)]
            for _ in range(n_joints)
        ]

    def __call__(self, joints):
        """joints: numpy (n_joints, n_coords) → smoothed numpy array"""
        import numpy as np
        out = np.zeros_like(joints)
        for j in range(joints.shape[0]):
            for c in range(joints.shape[1]):
                out[j, c] = self.filters[j][c](joints[j, c])
        return out
