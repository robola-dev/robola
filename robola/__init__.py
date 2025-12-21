"""
Robola - MuJoCo MJCF Editor Local Library

A local Python library for working with the Robola Web Editor to edit MuJoCo MJCF model files.
"""

from .server import serve, RobolaServer

__version__ = "0.2.4"
__all__ = ["serve", "RobolaServer", "__version__"]
