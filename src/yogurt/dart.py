"""DART (Laskey et al. 2017, "Disturbances for Augmenting Robot Trajectories") for grasp demonstrations.

During a perturbed take the follower is sent leader + noise, while the dataset stores the clean leader
pose as the action. The wrist camera and observation.state therefore show slightly off-track states,
labelled with what the human meant, and with the human's correction as they react to the drift. That is
exactly the recovery behaviour an ACT policy needs and never sees in clean demonstrations.

The noise is an Ornstein-Uhlenbeck process per joint (time constant `tau_s`), passed through a
low-pass filter (`smooth_s`) so it wanders smoothly instead of stepping every frame, scaled so its
spread is `sigma_deg` and capped at `max_sigmas` of that: the drift of an imperfect policy, not
per-frame jitter that the servos would just filter out. It ramps in over `ramp_s` at the start of a
take so the follower does not jump away from the leader the moment recording starts. The gripper is
left alone by default: random gripper noise drops cups and teaches the wrong thing.
"""

import math
import random

from .arm import JOINTS


class Perturbation:
    """Smooth joint-space noise for one take. `offset(dt)` advances it one tick and returns the offsets."""

    def __init__(self, cfg, rng: random.Random | None = None):
        self.sigma = {j: float(cfg.sigma_deg.get(j, 0.0)) for j in JOINTS}
        self.tau, self.ramp = float(cfg.tau_s), float(cfg.ramp_s)
        self.smooth, self.cap = float(cfg.get("smooth_s", 0.3)), float(cfg.get("max_sigmas", 2.0))
        # low-pass filtering an OU process of time constant tau shrinks its variance by tau / (tau + smooth)
        self.gain = math.sqrt((self.tau + self.smooth) / self.tau)
        self.rng = rng or random.Random()
        self.x = dict.fromkeys(JOINTS, 0.0)  # the OU state, unit spread
        self.y = dict.fromkeys(JOINTS, 0.0)  # its smoothed version
        self.t = 0.0

    def offset(self, dt: float) -> dict[str, float]:
        self.t += dt
        decay = math.exp(-dt / self.tau)
        spread = math.sqrt(1.0 - decay * decay)  # exact OU update: stationary std stays sigma
        follow = 1.0 - math.exp(-dt / self.smooth) if self.smooth > 0 else 1.0
        scale = min(1.0, self.t / self.ramp) if self.ramp > 0 else 1.0
        out = {}
        for j, sigma in self.sigma.items():
            if sigma:
                self.x[j] = self.x[j] * decay + spread * self.rng.gauss(0.0, 1.0)
                self.y[j] += (self.x[j] - self.y[j]) * follow
            v = sigma * min(max(self.y[j] * self.gain, -self.cap), self.cap)
            out[j] = scale * v
        return out

    def apply(self, clean: dict[str, float], dt: float) -> dict[str, float]:
        """The pose to send the follower: clean + noise. The caller clamps it to the joint limits."""
        noise = self.offset(dt)
        return {j: v + noise.get(j, 0.0) for j, v in clean.items()}
