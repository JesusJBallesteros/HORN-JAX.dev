"""HORN JAX package.

Re-exports the two modules almost every caller needs, so notebooks and scripts can
`from horn import init_net, step, ...` without knowing the file layout. The heavier
or more specialised modules (tasks, training, data, paths, report) are imported
explicitly by the code that uses them, which keeps `import horn` cheap and free of
side effects on disk.
"""

from .core import *    # HORNParams, HORNState, init_params, init_state, step, run_sequence, energy
from .model import *   # NetParams, init_net, forward, loss_and_acc, usable_band, ...
