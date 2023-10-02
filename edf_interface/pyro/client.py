import time
from typing import List, Iterable, Optional, Union
import logging
from beartype import beartype
import Pyro5.errors
import inspect
import functools

import Pyro5.api
from edf_interface.pyro.utils import get_service_proxy, PYRO_PROXY, _wrap_pyro_remote, look_for_nameserver

@beartype
class PyroClientBase():
    services: List[PYRO_PROXY] = []
    log: logging.Logger

    def __init__(self, service_names: Union[Iterable[str], str], timeout: Optional[Union[int, float]] = None,
                 nameserver_host: Optional[str] = None, nameserver_port: Optional[int] = None):
        self.log = logging.getLogger("PyroClientBase")
        if isinstance(service_names, str):
            service_names = [service_names]
            
        if nameserver_host or nameserver_port:
            self.ns = look_for_nameserver(
                host=nameserver_host, port=nameserver_port
            )
        else:
            self.ns = None
            
        for name in service_names:
            if self.ns is not None:
                service = Pyro5.api.Proxy(self.ns.lookup(name))
            else:
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
                time.sleep(1.)
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
                
                

        for method_name in service._pyroMethods:
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                if hasattr(method, '_remote_method_registered'):
                    if not method._remote_method_registered:
                        default_values = {name: param.default for name, param in inspect.signature(method).parameters.items() if param.default is not param.empty}
                        method = functools.partial(_wrap_pyro_remote(getattr(service, method_name)), **default_values)
                        setattr(self, method_name, method)