# Auto-Tuning MPPI for Autonomous Racing

A small simulator for a 1:10 scale race car (F1TENTH), with two controllers that drive it around a track:

- **MPPI**, a standard sampling-based controller, as the baseline.
- **Auto-Tuning MPPI**, which remembers what worked at each point of the track and improves it while it
  drives, so it gets faster lap after lap with only 50 samples per step.

This is the code of the paper *Auto-Tuning MPPI for Autonomous Racing* (Bolli, Ferraro, Delbrück), but you
don't need the paper to use it.

## Quick start

Needs Python 3.10 to 3.13 (3.11 recommended). Linux / macOS:

```bash
git clone https://github.com/F1Tenth-INI/L4DC.git && cd L4DC
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python run.py
```

Windows (cmd; Python from python.org or `winget install Python.Python.3.11`):

```bat
git clone https://github.com/F1Tenth-INI/L4DC.git && cd L4DC
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python run.py
```

The torch line is optional: it installs the smaller CPU-only build.

A window opens and Auto-Tuning MPPI drives the `icra2022` track. Every lap time is printed in the terminal,
and at the end all lap times and crashes are saved to `results/`.

To change what runs, edit [settings/settings.yml](settings/settings.yml) and start `python run.py` again:

| to | set |
|---|---|
| drive with the MPPI baseline | `controller: mppi` (samples per step: `mppi.rollouts`) |
| use another track | `simulation.map: RCA1` (or `IPZ50`, `hangar3`) |
| follow the track centre instead of the racing line | `simulation.waypoints: midline` |
| change grip, sensor noise, random seed, run length | `simulation.friction`, `state_noise`, `seed`, `duration` |
| run without a window, as fast as possible | `simulation.render: false` |

You can also pass another settings file: `python run.py my_settings.yml`.

Everything runs on the CPU.

## How it works

**The simulation.** The car is a single-track model with Pacejka tires, simulated in steps of 0.01 s
([race/dynamics.py](race/dynamics.py)). The car parameters are in
[settings/car_parameters.yml](settings/car_parameters.yml). The controller picks a steering angle and an
acceleration every 0.04 s, and each command reaches the car 0.08 s later, like on the real car. Optionally,
random noise is added to everything the controller observes. If a corner of the car touches a wall, that
counts as a crash: the car restarts from rest at a random point of the track, and the lap timer starts over.

Each track ([maps/](maps/)) comes with a racing line and a centre line, each a list of points with a target
speed. Both controllers use one of them as their guide.

**MPPI (baseline).** At every step MPPI tries many random command sequences (for example 2000). It predicts
1.4 s of driving for each with the car model and scores it: stay near the racing line, point along it, make
progress, keep off the walls. Then it averages the sequences, giving the better ones more weight, applies the
first command, and starts over at the next step. With few samples (for example 50) the plans get much
worse.

**Auto-Tuning MPPI.** Works like MPPI, but keeps a memory along the track. For each point of the guide line, the
memory stores how the car should pass there (position, heading, speed, sideways slide) and the command
sequence that gets it there. The memory starts slow, at 1.5 m/s. Then, at every step:

1. Sampling starts from the commands stored for the car's position, so 50 samples are enough to drive well.
2. Behind the car (20 points back, where it is known how the car actually drove), it tries small changes of the
   stored commands in simulation. If a change is faster, keeps enough distance to the walls and still connects to
   the rest of the memory, it replaces the stored entry.
3. If the car leaves its corridor or crashes, the latest changes around that place are undone.
4. If the best lap has not improved for 5 laps, it goes back to the memory of its best lap and finally stops
   changing it ("frozen").

So the car learns its own speed profile and line by driving.

The settings of Auto-Tuning MPPI are in the `atmppi` section of [settings/settings.yml](settings/settings.yml),
already set to tuned values. Each has a short comment; the symbol in brackets is the one used in the paper.

## The window

With `render: true` a pygame window shows the run. Mouse wheel: zoom, drag: pan, F: follow the car, Esc: stop.

- **Auto-Tuning MPPI:** each arrow is one entry of the memory. It points in the stored heading, and its
  length and color (green = slow, red = fast) show the stored speed. A number next to an arrow counts how often
  the car left the corridor there. The orange line is the plan the car is following.
- **MPPI:** red dots are some of the sampled predictions, the orange line is the plan, and green dots are the
  part of the racing line the cost looks at.

`speed` limits the simulation to that multiple of real time while the window is open.

**Over ssh** there is no screen, so the run continues without a window. To see it, connect with X forwarding
(`ssh -X user@host`; macOS needs XQuartz, Windows an X server such as VcXsrv or MobaXterm), or set
`render: false`.

## Results

Each run writes `results/<date>_<controller>_<map>_seed<k>.json` with every lap time, the crash times and the
settings used. Laps driven after the memory froze are marked `"frozen": true`.

To repeat the paper's experiments:

- **Lap times on icra2022:** `map: icra2022`, `waypoints: raceline`, `friction: 0.5`, `state_noise: 0.0`,
  `seed: 1`. Run it once with `controller: atmppi` and once with `controller: mppi` (2000 rollouts).
- **All tracks:** every combination of the four maps, `raceline` / `midline` and friction 0.5 / 0.9, with
  `state_noise: 0.025`, `duration: 1000` and several seeds. Auto-Tuning MPPI uses 50 rollouts, MPPI 50 and 2000.

## Code

| file | |
|---|---|
| [run.py](run.py) | entry point |
| [race/simulation.py](race/simulation.py) | simulation loop, crashes, lap times, results |
| [race/track.py](race/track.py) | map, guide lines, lap timer, collision check |
| [race/dynamics.py](race/dynamics.py) | car model (JAX) |
| [race/controllers/mppi.py](race/controllers/mppi.py) | MPPI baseline |
| [race/controllers/atmppi.py](race/controllers/atmppi.py) | Auto-Tuning MPPI. Its docstring maps the paper's sections to the code |
| [race/controllers/drive_costs_jax.py](race/controllers/drive_costs_jax.py) | its driving cost |
| [race/controllers/walls.py](race/controllers/walls.py) | its wall distances |
| [race/render.py](race/render.py) | pygame window |
