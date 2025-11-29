"""
Robola - MuJoCo MJCF Editor Local Library

本地 Python 库，用于与 Robola Web Editor 配合编辑 MuJoCo MJCF 模型文件。
"""

from .server import serve, RobolaServer

__version__ = "0.1.0"
__all__ = ["serve", "RobolaServer", "__version__"]
