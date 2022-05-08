from typing import Any, Tuple
import logging
from rich.logging import RichHandler
from enum import Enum
import numpy as np
import torch

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["LOG",
           #"node",
           #"simul",
           #"utils",
           #"data",
           #"model",
           #"flow_control",
           "set_seed",
           "DuplicateFilter",
           "CreateModelMode",
           "AntiEntropyProtocol",
           "MessageType",
           "Message",
           "CacheKey",
           "CacheItem",
           "Sizeable",
           "EqualityMixin"]


class DuplicateFilter(object):
    def __init__(self):
        """Removes duplicate log messages."""
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv

logging.basicConfig(level=logging.INFO,
                    #format="[%(asctime)s]  %(message)s",
                    format="%(message)s",
                    datefmt='%d%m%y-%H:%M:%S',
                    handlers=[RichHandler()])


LOG = logging.getLogger("rich")
"""The logging handler that filters out duplicate messages."""

LOG.addFilter(DuplicateFilter())


def set_seed(seed=0) -> None:
    """Sets the seed for the random number generator."""
    np.random.seed(seed)
    torch.manual_seed(seed)


class CreateModelMode(Enum):
    """The mode for creating/updating the gossip model."""

    UPDATE = 1
    """Update the model with the local data."""
    MERGE_UPDATE = 2
    """Merge the models and then make an update."""
    UPDATE_MERGE = 3
    """Update the models with the local data and then merge the models."""
    PASS = 4
    """Do nothing."""


class AntiEntropyProtocol(Enum):
    """The overall protocol of the gossip algorithm."""

    PUSH = 1
    """Push the local model to the gossip node(s)."""
    PULL = 2
    """Pull the gossip model from the gossip node(s)."""
    PUSH_PULL = 3
    """Push the local model to the gossip node(s) and then pull the gossip model from the gossip node(s)."""


class MessageType(Enum):
    """The type of a message."""

    PUSH = 1
    """The message contains the model (and possibly additional information)"""
    PULL = 2
    """Asks for the model to the receiver."""
    REPLY = 3
    """The message is a response to a PULL message."""
    PUSH_PULL = 4
    """The message contains the model (and possibly additional information) and also asks for the model."""


class EqualityMixin(object):
    def __init__(self):
        """Mixin for equality comparison."""
        pass

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, self.__class__) and self.__dict__ == other.__dict__)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


class Sizeable():
    def __init__(self):
        """The interface for objects that can be sized.
        
        Each class that implements this interface must define the method `get_size()`.
        """
        pass
    
    def get_size(self) -> int:
        """Returns the size of the object."""
        raise NotImplementedError()


class CacheKey(Sizeable):
    def __init__(self, *args):
        """The key for a cache item."""
        self.key = tuple(args)
    
    def get(self):
        """Returns the value of the cache item.

        Returns
        -------
        Any
            The value of the cache item.
        """
        return self.key
    
    def get_size(self) -> int:
        from gossipy.model.handler import ModelHandler
        val = ModelHandler._CACHE[self].value
        if isinstance(val, (float, int, bool)): return 1
        elif isinstance(val, Sizeable): return val.get_size()
        else: 
            LOG.warning("Impossible to compute the size of %s. Set to 0." %val)
            return 0
    
    def __repr__(self):
        return str(self.key)
    
    def __hash__(self):
        return hash(self.key)
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, CacheKey):
            return self.key == other.key
        return False

    def __ne__(self, other: Any):
        return not (self == other)


class CacheItem(Sizeable):
    def __init__(self, value: Any):
        """The class of an item in the cache.

        The constructor initializes the cache item with the specified value and with a single reference.

        Parameters
        ----------
        value : Any
            The value of the item.
        """
        self.value = value
        self.refs = 1
    
    def add_ref(self) -> None:
        """Adds a reference to the item."""
        self.refs += 1
    
    def del_ref(self) -> Any:
        """Deletes a reference to the item.
        
        Returns
        -------
        Any
            The value of the unreferenced item.
        """
        self.refs -= 1
        return self.value
    
    def is_referenced(self) -> bool:
        """Returns True if the item is referenced, False otherwise.
        
        Returns
        -------
        bool
            `True` if the item is referenced, `False` otherwise.
        """
        return self.refs > 0
    
    def get_size(self) -> int:
        if isinstance(self.value, (tuple, list)):
            sz: int = 0
            for t in self.value:
                if t is None: continue
                if isinstance(t, (float, int, bool)): sz += 1
                elif isinstance(t, Sizeable): sz += t.get_size()
                else: 
                    LOG.warning("Impossible to compute the size of %s. Set to 0." %t)
            return max(sz, 1)
        elif isinstance(self.value, Sizeable):
            return self.value.get_size()
        elif isinstance(self.value, (float, int, bool)):
            return 1
        else:
            LOG.warning("Impossible to compute the size of %s. Set to 0." %self.value)
            return 0


class Message(Sizeable):
    def __init__(self,
                 timestamp: int,
                 sender: int,
                 receiver: int,
                 type: MessageType,
                 value: Tuple[Any, ...]):
        """A class representing a message.

        Parameters
        ----------
        timestamp : int
            The message's timestamp with the respect to the simulation time.
        sender : int
            The sender node id.
        receiver : int
            The receiver node id.
        type : MessageType
            The message type.
        value : tuple[Any, ...]
            The message's payload. The typical payload is a single item tuple containing the model (handler).
        """
        self.timestamp = timestamp
        self.sender = sender
        self.receiver = receiver
        self.type = type
        self.value = value
    
    def get_size(self) -> int:
        """Computes and returns the estimated size of the message.

        The size is expressed in number of "atomic" values stored in the message.
        Atomic values are integers, floats, and booleans. Currently strings are not supported.

        Returns
        -------
        int
            The estimated size of the message.

        Raises
        ------
        TypeError
            If the message's payload contains values that are not atomic.
        """

        if self.value is None: return 1
        if isinstance(self.value, (tuple, list)):
            sz: int = 0
            for t in self.value:
                if t is None: continue
                if isinstance(t, (float, int, bool)): sz += 1
                elif isinstance(t, Sizeable): sz += t.get_size()
                else: raise TypeError("Cannot compute the size of the payload!")
            return max(sz, 1)
        elif isinstance(self.value, Sizeable):
            return self.value.get_size()
        elif isinstance(self.value, (float, int, bool)):
            return 1
        else:
            raise TypeError("Cannot compute the size of the payload!")
        
    def __repr__(self) -> str:
        s: str = "T%d [%d -> %d] {%s}: " %(self.timestamp,
                                           self.sender,
                                           self.receiver,
                                           self.type.name)
        s += "ACK" if self.value is None else str(self.value)
        return s
