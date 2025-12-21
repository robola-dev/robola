"""
Robola CLI
"""

import argparse
import sys
from pathlib import Path

from . import __version__

DEFAULT_MJCF_TEMPLATE = """
    <mujoco model="scene">
  <!--robola compile-->
  <compiler angle="radian" meshdir="assets" texturedir="textures"/>
  <visual>
    <global azimuth="220" elevation="-10" />
    <quality shadowsize="8192" />
    <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0 0 0" />
    <rgba haze="0.15 0.25 0.35 1" />
  </visual>
  <statistic meansize="0.05" extent="0.8" center="0.15 0.1 0.38" />
  <asset>
    <texture type="skybox" colorspace="auto" name="texture_0_" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
    <texture type="2d" colorspace="auto" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
  </asset>
  <worldbody>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" />
    <light name="light_0_" pos="0 0 2" dir="0 0 -1" intensity="15" diffuse="0.5 0.5 0.5" />
  </worldbody>
</mujoco>
"""


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        prog="robola",
        description="Robola MJCF Editor - Local Python Library",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Robola {__version__}",
        help="Show version information and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve 命令
    serve_parser = subparsers.add_parser("serve", help="Start the local WebSocket server")
    serve_parser.add_argument(
        "mjcf_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the MJCF XML file to edit (optional, can be loaded via UI)",
    )
    serve_parser.add_argument(
        "--new",
        action="store_true",
        help="Treat the provided MJCF path as a new file to create before serving",
    )
    serve_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=9527,
        help="WebSocket server port (default: 9527)",
    )
    serve_parser.add_argument(
        "--origin",
        "-o",
        type=str,
        default="*",
        help="Allowed CORS origin (default: *)",
    )
    serve_parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Simulation streaming frequency in Hz (1-60, default: 60)",
    )

    # version 命令
    subparsers.add_parser("version", help="Show version information")

    args = parser.parse_args()

    if args.command == "serve":
        mjcf_path = None

        if args.new:
            if not args.mjcf_path:
                print("Error: --new requires a target MJCF path", file=sys.stderr)
                sys.exit(1)

            new_path = Path(args.mjcf_path).resolve()
            if new_path.exists():
                print(f"Error: Target file already exists: {new_path}, please remove args: --new", file=sys.stderr)
                sys.exit(1)
            if new_path.suffix.lower() != ".xml":
                print(f"Error: --new target must be an XML file: {new_path}", file=sys.stderr)
                sys.exit(1)
            try:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                new_path.write_text(DEFAULT_MJCF_TEMPLATE.strip() + "\n", encoding="utf-8")
            except OSError as exc:
                print(f"Error: Failed to create template file: {exc}", file=sys.stderr)
                sys.exit(1)
            mjcf_path = str(new_path)
            print(f"Created new MJCF template at {new_path}")
        elif args.mjcf_path:
            mjcf_candidate = Path(args.mjcf_path).resolve()

            if not mjcf_candidate.exists():
                print(f"Error: File not found: {mjcf_candidate}", file=sys.stderr)
                sys.exit(1)

            if mjcf_candidate.suffix.lower() != ".xml":
                print(f"Error: File must be an XML file: {mjcf_candidate}", file=sys.stderr)
                sys.exit(1)
            
            mjcf_path = str(mjcf_candidate)

        from .server import serve

        if args.fps < 1 or args.fps > 60:
            print("Error: --fps must be between 1 and 60", file=sys.stderr)
            sys.exit(1)

        serve(
            mjcf_path=mjcf_path,
            port=args.port,
            allowed_origin=args.origin,
            fps=args.fps,
        )

    elif args.command == "version":
        print(f"Robola {__version__}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
