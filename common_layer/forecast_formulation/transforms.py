"""
transforms.py — the variance-stabilising transform for prices that can be negative.

z = asinh(x / s) is linear near zero (so small negative prices stay small and keep their sign) and grows
like the logarithm for large |x| (so a 400 EUR/MWh spike does not dominate a loss function the way it does in
levels). It is odd, strictly increasing and exactly invertible: x = s * sinh(z). A plain logarithm cannot be
used because day-ahead prices have been negative since 2024.

The scale s sets where the behaviour changes from linear to logarithmic; `scale` is chosen from data in
evidence.heteroscedasticity_by_scale and written in config/forecast_formulation.yaml, never fitted silently.

Averaging in the transformed space and transforming back gives a median-type forecast, not a mean: for a point
forecast that must be the expected price, the back-transformed value needs a bias correction (or the model must
be trained on the level). `retransform_mean` does the correction by smearing (Duan) from held-out residuals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Asinh:
    scale: float

    def __post_init__(self):
        if not self.scale > 0:
            raise ValueError("scale must be positive")

    def forward(self, x):
        return np.arcsinh(np.asarray(x, dtype=float) / self.scale)

    def inverse(self, z):
        return self.scale * np.sinh(np.asarray(z, dtype=float))


class Identity:
    scale = None

    def forward(self, x):
        return np.asarray(x, dtype=float)

    def inverse(self, z):
        return np.asarray(z, dtype=float)


def make_transform(name: str, scale: Optional[float] = None):
    if name in (None, "none", "identity"):
        return Identity()
    if name == "asinh":
        if scale is None:
            raise ValueError("asinh needs a scale")
        return Asinh(float(scale))
    raise ValueError(f"unknown transform {name!r}")


def retransform_mean(transform, z_forecast, z_residuals) -> np.ndarray:
    """Expected price from a forecast made in transformed space: the mean of inverse(z_hat + e) over held-out
    residuals e (Duan's smearing estimate). Equals inverse(z_hat) only when the residuals are all zero."""
    z = np.atleast_1d(np.asarray(z_forecast, dtype=float))
    e = np.asarray(z_residuals, dtype=float).ravel()
    return transform.inverse(z[:, None] + e[None, :]).mean(axis=1)
