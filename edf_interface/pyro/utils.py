import time
from typing import Union, Optional, Callable
import logging
import pickle

import gzip
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

def serialize_input(serializer: Callable, class_method: bool) -> Callable:
    def serialize_(fn):
        if class_method:
            def wrapped(self, *args, **kwargs):
                args = [serializer(arg) for arg in args]
                kwargs = {key: serializer(val) for key, val in kwargs.items()}
                out = fn(self, *args, **kwargs)
                return out
        else:
            def wrapped(*args, **kwargs):
                args = [serializer(arg) for arg in args]
                kwargs = {key: serializer(val) for key, val in kwargs.items()}
                out = fn(*args, **kwargs)
                return out
        return wrapped
    return serialize_

def serialize_output(serializer: Callable) -> Callable:
    def serialize_(fn):
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            return serializer(out)
        return wrapped
    return serialize_

def deserialize_input(deserializer: Callable, class_method: bool) -> Callable:
    def deserialize_(fn):
        if class_method:
            def wrapped(self, *args, **kwargs):
                if class_method:
                    args = args [1:]
                args = [deserializer(arg) for arg in args]
                kwargs = {key: deserializer(val) for key, val in kwargs.items()}
                out = fn(self, *args, **kwargs)
                return out
        else:
            def wrapped(*args, **kwargs):
                if class_method:
                    args = args [1:]
                args = [deserializer(arg) for arg in args]
                kwargs = {key: deserializer(val) for key, val in kwargs.items()}
                out = fn(*args, **kwargs)
                return out
        return wrapped
    return deserialize_

def deserialize_output(deserializer: Callable) -> Callable:
    def deserialize_(fn):
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            return deserializer(out)
        return wrapped
    return deserialize_

def pickle_serialize(x, compress: bool = False):
    x = pickle.dumps(x)
    if compress:
        return gzip.compress(x)
    else:
        return x

def pickle_deserialize(x, compressed: bool = False):
    if compressed:
        x = gzip.decompress(x)
    return pickle.loads(x)

def default_serializer(x):
    return pickle_serialize(x, compress=True)

def default_deserializer(x):
    return pickle_deserialize(x, compressed=True)


def _expose(serializer: Callable = default_serializer,
           deserializer: Callable = default_deserializer,
           class_method: bool = True) -> Callable:
    def wrapped(fn: Callable):
        return Pyro5.api.expose(
            deserialize_input(deserializer=deserializer, class_method=class_method)(
                serialize_output(serializer=serializer)(
                    fn
                )
            )
        )
    return wrapped

def _wrap_remote(serializer: Callable = default_serializer,
                deserializer: Callable = default_deserializer,
                class_method: bool = False) -> Callable:
    def wrapped(fn: Callable):
        return serialize_input(serializer=serializer, class_method=class_method)(
            deserialize_output(deserializer=deserializer)(
                fn
            )
        )
    return wrapped

expose = _expose()
wrap_remote = _wrap_remote()
