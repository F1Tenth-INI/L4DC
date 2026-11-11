"""Settings and car parameters, read from the ``settings`` folder."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(ROOT, "settings", "settings.yml")
CAR_FILE = os.path.join(ROOT, "settings", "car_parameters.yml")
MAPS_DIR = os.path.join(ROOT, "maps")

TIMESTEP_SIM = 0.01  # physics step [s]
GLOBAL_SPEED_LIMIT = 15.0  # the simulator clips the car's speed to this [m/s]


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in value.items()})
    return value


def load_settings(path: str = SETTINGS_FILE) -> SimpleNamespace:
    with open(path) as f:
        cfg = _namespace(yaml.safe_load(f))
    sim = cfg.simulation
    sim.map_dir = os.path.join(MAPS_DIR, sim.map)
    if not os.path.isdir(sim.map_dir):
        raise FileNotFoundError(f"no map '{sim.map}' in {MAPS_DIR}")
    if sim.waypoints not in ("raceline", "midline"):
        raise ValueError(f"simulation.waypoints must be raceline or midline, got {sim.waypoints!r}")
    if cfg.controller not in ("atmppi", "mppi"):
        raise ValueError(f"controller must be atmppi or mppi, got {cfg.controller!r}")
    sim.delay_steps = int(round(sim.control_delay / sim.timestep_control))  # in control steps
    sim.substeps = int(round(sim.timestep_control / TIMESTEP_SIM))
    return cfg


class CarParameters:
    """``settings/car_parameters.yml``, with the friction coefficient of the run."""

    ARRAY_ORDER = (  # layout of the flat array the dynamics take
        "mu", "lf", "lr", "h", "m", "I_z", "g", "C_Pf", "C_Pr", "servo_p", "s_min", "s_max", "sv_min", "sv_max",
        "a_min", "a_max", "v_min", "v_max", "v_switch", "steering_diff_low", "c_rr", "v_dead",
        "curve_resistance_factor", "brake_multiplier", "accel_multiplier", "steering_multiplier",
    )

    def __init__(self, friction: float | None = None, path: str = CAR_FILE):
        with open(path) as f:
            for key, value in yaml.safe_load(f).items():
                setattr(self, key, value)
        if friction is not None:
            self.mu = float(friction)

    def to_array(self) -> np.ndarray:
        values = []
        for name in self.ARRAY_ORDER:
            value = getattr(self, name)
            values.extend(value if isinstance(value, list) else [value])
        return np.array(values, dtype=np.float32)
