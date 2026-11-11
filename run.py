"""Run one simulation with the settings in settings/settings.yml:

    python run.py
    python run.py path/to/other_settings.yml
"""

import os
import sys

# Before JAX is imported: CPU, few threads (the controllers are small; one thread is the fastest here).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 "
                                   "inter_op_parallelism_threads=1")
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from race.config import SETTINGS_FILE, load_settings  # noqa: E402
from race.simulation import Simulation  # noqa: E402

if __name__ == "__main__":
    cfg = load_settings(sys.argv[1] if len(sys.argv) > 1 else SETTINGS_FILE)
    Simulation(cfg).run()
