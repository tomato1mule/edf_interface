from typing import List, Iterable
from beartype import beartype
from edf_interface.pyro.utils import get_service_proxy, wrap_remote, PYRO_PROXY

@beartype
class PyroClientBase():
    services: List[PYRO_PROXY] = []

    def __init__(self, service_names: Iterable[str]):
        for name in service_names:
            self.services.append(get_service_proxy(name))
        for service in self.services:
            self._register_remote_methods(service)

    def _register_remote_methods(self, service):
        service._pyroBind()
        for method in service._pyroMethods:
            if hasattr(self, method):
                setattr(self, method, wrap_remote(getattr(service, method)))