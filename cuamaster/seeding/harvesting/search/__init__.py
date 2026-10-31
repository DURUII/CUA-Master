"""Search package: pluggable YouTube search backends."""
from .base import SearchBackend
from .factory import create_backend

__all__ = ["SearchBackend", "create_backend"]
