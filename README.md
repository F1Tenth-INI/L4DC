# Auto-Tuning MPPI for Autonomous Racing

Simulation code for the paper *Auto-Tuning MPPI for Autonomous Racing* (Bolli, Ferraro, Delbrück).
It contains the proposed controller (Auto-Tuning MPPI), the MPPI baseline it is compared with, the car
simulator and the four tracks of the experiments.

## Quick start

```bash
git clone https://github.com/F1Tenth-INI/L4DC.git && cd L4DC
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: smaller CPU-only torch
pip install -r requirements.txt
python run.py
```

A window opens and Auto-Tuning MPPI drives icra2022; lap times appear in the terminal.
To change what runs, edit [settings/settings.yml](settings/settings.yml) and start `python run.py` again:

- MPPI instead: `controller: mppi` (number of rollouts: `mppi.rollouts`)
- another track: `simulation.map: RCA1` (or `IPZ50`, `hangar3`), centre line: `simulation.waypoints: midline`
- friction, noise, seed, run length: `simulation.friction`, `state_noise`, `seed`, `duration`
- no window, as fast as possible: `simulation.render: false`

Results (lap times, crashes, settings) are saved in `results/`.

Over ssh there is no screen for the window: connect with X forwarding (`ssh -X user@host`; macOS needs
XQuartz, Windows an X server such as VcXsrv or MobaXterm), or set `render: false` and read the lap times.

## Setup

Python 3.11:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Everything runs on the CPU. For a smaller download, install the CPU build of torch first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Run

```bash
python run.py
```

All settings are in [settings/settings.yml](settings/settings.yml):

- `controller`: `atmppi` (Auto-Tuning MPPI) or `mppi` (baseline)
- `simulation`: track, raceline or midline, friction, observation noise, seed, duration, window on/off
- `mppi`: number of rollouts
- `atmppi`: the parameters of the paper's Table 1, set to the tuned values. The symbol of each is in the comment.

The car is described in [settings/car_parameters.yml](settings/car_parameters.yml); `simulation.friction`
replaces its friction coefficient. Both controllers predict with the same car parameters as the simulator.

A run prints every lap time and writes them, the crashes and the settings to `results/<date>_<controller>_<map>_seed<k>.json`.
Laps after the memory froze are marked `"frozen": true`.

### Viewer

With `render: true` a pygame window shows the run (mouse wheel: zoom, drag: pan, F: follow the car, Esc: stop).
For Auto-Tuning MPPI each waypoint of the memory is an arrow along its stored heading, 0.08 m long per m/s of
stored speed and colored from green (slow) to red; a number marks how often the car left the corridor there.
The orange line is the executed plan. For MPPI, red dots are 20 of its rollouts, the orange line its plan and
the green dots the waypoints its cost uses. `speed` caps the simulation at a multiple of real time while the
window is open; without the window it runs as fast as it can.

### Paper experiments

- Lap times on icra2022: `map: icra2022`, `waypoints: raceline`, `friction: 0.5`, `state_noise: 0.0`, `seed: 1`,
  once with `controller: atmppi` (50 rollouts) and once with `controller: mppi` (2000 rollouts).
- All tracks: every combination of the four maps, `raceline` / `midline` and friction 0.5 / 0.9, with
  `state_noise: 0.025`, 1000 s per run and several seeds; Auto-Tuning MPPI with 50 rollouts, MPPI with 50 and 2000.

## Tracks

`icra2022`, `RCA1`, `IPZ50` and `hangar3`. Each folder in [maps/](maps/) holds the occupancy image and its `.yaml`, the minimum-curvature raceline
(`<map>_wp.csv`) and the track centre (`centerline.csv`). Every 4th point is used as a waypoint. On the
midline, and before the first lap of Auto-Tuning MPPI, the target speed is 1.5 m/s.

## Simulation

- Single-track Pacejka tire model blended with a kinematic model below about 1.75 m/s, Euler steps of 0.01 s
  ([race/dynamics.py](race/dynamics.py)).
- Control period 0.04 s, commands reach the car 0.08 s later.
- Optional uniform noise on every observed state variable (`state_noise`).
- A crash is a body corner within 5 mm of a wall. The car then restarts from rest near a random waypoint, and
  the lap timer starts over. The car also starts the run at a random waypoint (seeded).

## Code

| file | |
|---|---|
| [run.py](run.py) | entry point |
| [race/simulation.py](race/simulation.py) | simulation loop, crashes, lap times, results |
| [race/track.py](race/track.py) | map, waypoints, lap timer, collision check |
| [race/dynamics.py](race/dynamics.py) | car model (JAX) |
| [race/controllers/atmppi.py](race/controllers/atmppi.py) | Auto-Tuning MPPI; the docstring maps the paper's sections to the code |
| [race/controllers/drive_costs_jax.py](race/controllers/drive_costs_jax.py) | its driving cost |
| [race/controllers/walls.py](race/controllers/walls.py) | its wall distances and free-space paths |
| [race/controllers/mppi.py](race/controllers/mppi.py) | MPPI baseline |
| [race/render.py](race/render.py) | pygame viewer |
