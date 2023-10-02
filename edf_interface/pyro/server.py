from typing import Optional, Union
import threading
import logging
import time

from beartype import beartype
import Pyro5.nameserver, Pyro5.core, Pyro5.server, Pyro5.client, Pyro5.errors, Pyro5.api

from edf_interface.pyro.nameserver import NameServer
from edf_interface.pyro.utils import look_for_nameserver

class ServiceNotRegisteredException(Exception):
    pass

@beartype
class PyroServer():
    nameserver: Optional[NameServer] = None
    nameserver_proxy: Pyro5.client.Proxy = None
    _loop_thread: Optional[threading.Thread] = None
    server_daemon: Optional[Pyro5.server.Daemon] = None
    uri: Optional[Pyro5.core.URI] = None
    server_name: str    
    log: logging.Logger

    def __init__(self, server_name: str, 
                 init_nameserver: Optional[bool] = None,
                 nameserver_timeout: Union[float, int, str] = 'default',
                 nameserver_host: Optional[str] = None, nameserver_port: Optional[int] = None
                 ):

        # assert service._pyroExposed is True # use '@Pyro5.api.expose' decorator on the class.
        self.service = None
        self.server_name = server_name
        self.log = logging.getLogger(self.server_name)
        self.nameserver_host = nameserver_host
        self.nameserver_port = nameserver_port
        
        ############ Initialize Nameserver Proxy #############
        if init_nameserver:
            self.log.warning(f"{self.server_name}: Initializing nameserver")
            self.init_nameserver(host=nameserver_host, port=nameserver_port)
        else:
            self.nameserver = None

        if isinstance(nameserver_timeout, str):
            assert nameserver_timeout == 'default'
            if init_nameserver is None:
                nameserver_timeout = 1. # Wait 1 seconds before creating new nameserver.
            else:
                nameserver_timeout = -1 # Infinitely wait for nameserver

        try:
            if init_nameserver:
                self.log.debug(f"{self.server_name}: Looking for a nameserver...")
            else:
                self.log.warning(f"{self.server_name}: Looking for a nameserver...")
            self.nameserver_proxy = look_for_nameserver(wait=True, timeout=nameserver_timeout, host=nameserver_host, port=nameserver_port) # find nameserver
            if init_nameserver:
                self.log.debug(f"{self.server_name}: Nameserver found!")
            else:
                self.log.warning(f"{self.server_name}: Nameserver found!")
            
        except Exception as e:
            assert self.nameserver is None, f"Unknown error"
            if init_nameserver is None: # initialize new nameserver if cannot find nameserver
                self.log.warning(f"{self.server_name}: Cannot find a nameserver. Creating a new one.")
                self.init_nameserver(host=nameserver_host, port=nameserver_port)
                self.nameserver_proxy = Pyro5.api.locate_ns(host=nameserver_host, port=nameserver_port) # find nameserver
            else:
                self.log.error(f"{self.server_name}: {e}")
                raise Pyro5.errors.NamingError(f"{self.server_name}: Cannot find nameserver.")
        ######################################################

    def init_nameserver(self, host: Optional[str] = None, port: Optional[int] = None):
        self.nameserver = NameServer(init=True, host=host, port=port)
        self.log.warning(f"{self.server_name}: Initialized nameserver @ \"{self.nameserver.nsUri}\"")


    def register_service(self, service):
        self.service = service

    def init_server(self):
        if self.service is None:
            raise ServiceNotRegisteredException(f"Service not registered! Call PyroServer.register_service(...) before running the server.")
        ############ Initialize Server #############
        if self.nameserver_host:
            self.server_daemon = Pyro5.server.Daemon(host=self.nameserver_host, port=self.nameserver_port + 1 if self.nameserver_port is not None else 9091)                       # make a Pyro daemon
        else:
            self.server_daemon = Pyro5.server.Daemon() 
            
        self.uri = self.server_daemon.register(self.service)                 # register the greeting maker as a Pyro object
        self.nameserver_proxy.register(self.server_name, self.uri)           # register the object with a name in the name server
        ######################################################

    def _request_loop(self):
        try:
            self.server_daemon.requestLoop()
            self.log.debug(f"{self.server_name}: Server daemon loop exit (Service: {self.server_name})")
        finally:
            self.server_daemon.close()

    def run(self, nonblocking: bool = False):
        self.init_server()
        if nonblocking:
            self.log.warning(f"{self.server_name}: Running '{self.server_name}' server in background...")
            self._loop_thread = threading.Thread(target=self._request_loop)
            self._loop_thread.start()
        else:
            self.log.warning(f"{self.server_name}: Running '{self.server_name}' server...")
            self._request_loop()

    def close(self, timeout: Union[float, int] = 5.) -> bool:
        self.log.warning(f"Closing server '{self.server_name}'... (Timeout: {timeout} sec)")
        init_time = time.time()
        self.server_daemon.close()
        try:
            self._loop_thread.join(timeout=timeout)
        except AttributeError:
            pass
        
        try:
            is_alive = self._loop_thread.is_alive()
        except AttributeError:
            is_alive = False

        if is_alive:
            self.log.error(f"Server '{self.server_name}' daemon loop process not cleanly closed in {time.time() - init_time} seconds.")
            return False
        else:
            self.server_daemon.close()
            self.log.warning(f"Server '{self.server_name}' daemon loop process cleanly closed in {time.time() - init_time} seconds.")
            return True