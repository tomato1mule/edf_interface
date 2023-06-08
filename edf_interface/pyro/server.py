from typing import Optional, Union
import threading
import logging
import time

from beartype import beartype
import Pyro5.nameserver, Pyro5.core, Pyro5.server, Pyro5.client, Pyro5.errors, Pyro5.api

from edf_interface.pyro.nameserver import NameServer
from edf_interface.pyro.utils import look_for_nameserver

class PyroServer():
    nameserver: Optional[NameServer] = None
    nameserver_proxy: Pyro5.client.Proxy = None
    _loop_thread: Optional[threading.Thread] = None
    server_name: str
    server_daemon: Pyro5.server.Daemon
    uri: Pyro5.core.URI
    log: logging.Logger

    @beartype
    def __init__(self, 
                 service,
                 server_name: Optional[str] = None, 
                 init_nameserver: Optional[bool] = None,
                 nameserver_timeout: Union[float, int, str] = 'default'):

        # assert service._pyroExposed is True # use '@Pyro5.api.expose' decorator on the class.

        if server_name is None:
            if isinstance(service, object):
                self.server_name = service.__class__.__name__
            else:
                self.server_name = service.__name__
        else:
            self.server_name = server_name
        self.log = logging.getLogger(self.server_name)
        #self.service = Pyro5.api.expose(service)
        self.service = service

        ############ Initialize Nameserver Proxy #############
        if init_nameserver:
            self.log.info("Initializing nameserver")
            self.nameserver = NameServer(init=True)
        else:
            self.nameserver = None

        if isinstance(nameserver_timeout, str):
            assert nameserver_timeout == 'default'
            if init_nameserver is None:
                nameserver_timeout = 1. # Wait 1 seconds before creating new nameserver.
            else:
                nameserver_timeout = -1 # Infinitely wait for nameserver

        try:
            self.log.info("Looking for a nameserver")
            self.nameserver_proxy = look_for_nameserver(wait=True, timeout=nameserver_timeout) # find nameserver
        except Exception as e:
            assert self.nameserver is None, f"Unknown error"
            if init_nameserver is None: # initialize new nameserver if cannot find nameserver
                self.log.info("Cannot find a nameserver. Creating a new one.")
                self.nameserver = NameServer(init=True)
                self.nameserver_proxy = Pyro5.api.locate_ns() # find nameserver
            else:
                self.log.error(f"{e}")
                raise Pyro5.errors.NamingError(f"Cannot find nameserver.")
        ######################################################

        ############ Initialize Server #############
        self.server_daemon = Pyro5.server.Daemon()                           # make a Pyro daemon
        self.uri = self.server_daemon.register(self.service)                 # register the greeting maker as a Pyro object
        self.nameserver_proxy.register(self.server_name, self.uri)           # register the object with a name in the name server
        ######################################################

    @beartype
    def _request_loop(self):
        try:
            self.server_daemon.requestLoop()
            self.log.debug(f"Server daemon loop exit (Service: {self.server_name})")
        finally:
            self.server_daemon.close()

    @beartype
    def run(self, nonblocking: bool = False):
        if nonblocking:
            self.log.warning(f"Running {self.server_name} server in background...")
            self._loop_thread = threading.Thread(target=self._request_loop)
            self._loop_thread.start()
        else:
            self.log.warning(f"Running {self.server_name} server...")
            self._request_loop()

    @beartype
    def close(self, timeout: Union[float, int] = 5.) -> bool:
        self.log.warning(f"Closing server ({self.server_name})... (Timeout: {timeout} sec)")
        init_time = time.time()
        self.server_daemon.close()
        self._loop_thread.join(timeout=timeout)
        
        if self._loop_thread.is_alive():
            self.log.error(f"Server ({self.server_name}) daemon loop process not cleanly closed in {time.time() - init_time} seconds.")
            return False
        else:
            self.server_daemon.close()
            self.log.warning(f"Server ({self.server_name}) daemon loop process cleanly closed in {time.time() - init_time} seconds.")
            return True