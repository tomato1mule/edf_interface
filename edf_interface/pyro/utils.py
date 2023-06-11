import time
from typing import Union, Optional, Callable
import logging

from beartype import beartype

import Pyro5.api, Pyro5.errors, Pyro5.client

@beartype
def look_for_nameserver(wait: bool = True,
                        timeout: Union[float, int] = -1, # infty if negative
                        ) -> Pyro5.client.Proxy:
    if timeout > 0: 
        assert wait is True, f"wait must be True if timeout is set."
        use_timeout = True
        init_time = time.time()
    else:
        use_timeout = False

    while True:
        if use_timeout:
            time_spent = (time.time() - init_time)
            if  time_spent >= timeout:
                raise TimeoutError(f"Cannot find nameserver in {time_spent} seconds.")
            
        try:
            proxy = Pyro5.api.locate_ns()    
        except Pyro5.errors.NamingError as e:
            if wait:
                continue
            else:
                raise Pyro5.errors.NamingError(f"Cannot find nameserver")

        break

    return proxy

@beartype
def get_service_proxy(name: str) -> Pyro5.api.Proxy:
    uri = f"PYRONAME:{name}"
    proxy = Pyro5.api.Proxy(uri)    # use name server object lookup uri shortcut

    return proxy

def serialize(serializer):
    def _serialize(fn):
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            return serializer(out)
        return wrapped
    return _serialize

def serialize_if(condition: Callable, serializer: Callable):
    def _serializer(x):
        criteria = condition(x)
        assert isinstance(criteria, bool), f"{criteria}"
        if criteria:
            return serializer(x)
        else:
            return x
    return serialize(serializer=_serializer)

