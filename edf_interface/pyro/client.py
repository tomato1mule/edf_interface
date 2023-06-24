import time
from typing import List, Iterable, Optional, Union
import logging
from beartype import beartype
import Pyro5.errors

from edf_interface.pyro.utils import get_service_proxy, wrap_remote, PYRO_PROXY

@beartype
class PyroClientBase():
    services: List[PYRO_PROXY] = []
    log: logging.Logger

    def __init__(self, service_names: Iterable[str], timeout: Optional[Union[int, float]] = None):
        self.log = logging.getLogger("PyroClientBase")
        for name in service_names:
            service = get_service_proxy(name)
            self._register_remote_methods(service, timeout, _service_name_debug=name)
            self.services.append(service)
            

    def _register_remote_methods(self, service, timeout: Optional[Union[int, float]] = None, _service_name_debug: Optional[str] = None):
        if _service_name_debug is None:
            _service_name_debug = service._pyroUri.object
        self.log.warn(f"Attempting connection to server '{_service_name_debug}' @ {service._pyroUri}")

        init_time = time.time()
        while True:
            try:
                service._pyroBind()
                self.log.warn(f"Successfully connected to server '{_service_name_debug}' @ {service._pyroUri}")
                break
            except Pyro5.errors.NamingError:
                time_spent = (time.time() - init_time)
                if timeout is None:
                    self.log.warn(f"Retrying connection to server '{_service_name_debug}' @ {service._pyroUri}")
                    continue
                elif  time_spent >= timeout:
                    raise TimeoutError(f"Cannot find nameserver in {time_spent} seconds.")
                else:
                    self.log.warn(f"Retrying connection to server '{_service_name_debug}' @ {service._pyroUri}")
                    continue
            except Pyro5.errors.CommunicationError:
                time_spent = (time.time() - init_time)
                time.sleep(1.)
                if timeout is None:
                    self.log.warn(f"Retrying connection to server '{_service_name_debug}' @ {service._pyroUri}")
                    continue
                elif  time_spent >= timeout:
                    raise TimeoutError(f"Cannot find nameserver in {time_spent} seconds.")
                else:
                    self.log.warn(f"Retrying connection to server '{_service_name_debug}' @ {service._pyroUri}")
                    continue
                
                

        for method in service._pyroMethods:
            if hasattr(self, method):
                setattr(self, method, wrap_remote(getattr(service, method)))