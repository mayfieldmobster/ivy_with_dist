from . import collective_communication_ops
from .collective_communication_ops import *
from . import initialize
from .initialize import *
from . import maps
from .maps import *
from . import p2p
from .p2p import *
from . import cli
from . import _context
from . import _group

import mpi4py.MPI as MPI

NativeGroup = MPI.Comm
