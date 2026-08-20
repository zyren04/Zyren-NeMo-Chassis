"""
Routing Package — Centralized Model Router
"""

from .router import ModelTarget, SwitchyardRouter, get_router, set_router

__all__ = [
    "ModelTarget",
    "SwitchyardRouter",
    "get_router",
    "set_router",
]
