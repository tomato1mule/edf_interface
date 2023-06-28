from typing import Optional, Union
import threading
import logging
import time

from beartype import beartype
from Pyro5.nameserver import start_ns, NameServerDaemon, BroadcastServer
from Pyro5.core import URI


class NameServer():
    nsUri: Optional[URI] = None
    daemon: Optional[NameServerDaemon] = None
    bcserver: Optional[BroadcastServer] = None
    _ns_thread: Optional[threading.Thread] = None
    log: logging.Logger


    @beartype
    def __init__(self, init=True, host: Optional[str] = None, port: Optional[int] = None):
        self.log = logging.getLogger("NameServer")
        if init:
            self.init_nameserver(host=host, port=port)
        

    def init_nameserver(self, host=None, port=None, enableBroadcast=True, bchost=None, bcport=None,
                        unixsocket=None, nathost=None, natport=None, storage=None):
        self.nsUri, self.daemon, self.bcserver = start_ns(host=host, port=port, enableBroadcast=enableBroadcast, bchost=bchost, bcport=bcport, unixsocket=unixsocket, nathost=nathost, natport=natport, storage=storage)
        self._ns_thread = threading.Thread(target=self._ns_daemon_request_loop, daemon=True)
        self._ns_thread.start()

    def _ns_daemon_request_loop(self):
        try:
            self.daemon.requestLoop()
            self.log.debug(f"Nameserver daemon loop exit")
        finally:
            self.daemon.close()

    def close(self, timeout = 5.) -> bool:
        self.log.warning(f"Closing Nameserver... (Timeout: {timeout} sec)")
        init_time = time.time()
        self.daemon.close()
        if self.bcserver is not None:
            self.bcserver.close()
        self._ns_thread.join(timeout=timeout)
        
        if self._ns_thread.is_alive():
            self.log.error(f"Nameserver daemon loop process not cleanly closed in {time.time() - init_time} seconds.")
            return False
        else:
            self.daemon.close()
            if self.bcserver is not None:
                self.bcserver.close()
            self.log.warning(f"Nameserver daemon loop process cleanly closed in {time.time() - init_time} seconds.")
            return True
