import pickle

import Pyro5
Pyro5.config.SERPENT_BYTES_REPR = True
import Pyro5.api
from .nameserver import NameServer
from .server import PyroServer
from .utils import get_service_proxy


