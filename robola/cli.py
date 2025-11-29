"""
Robola CLI - 命令行接口
"""

import argparse
import sys
from pathlib import Path


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        prog="robola",
        description="Robola MJCF Editor - Local Python Library",
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

    # version 命令
    subparsers.add_parser("version", help="Show version information")

    args = parser.parse_args()

    if args.command == "serve":
        mjcf_path = None
        
        if args.mjcf_path:
            mjcf_path = Path(args.mjcf_path).resolve()

            if not mjcf_path.exists():
                print(f"Error: File not found: {mjcf_path}", file=sys.stderr)
                sys.exit(1)

            if not mjcf_path.suffix.lower() == ".xml":
                print(f"Error: File must be an XML file: {mjcf_path}", file=sys.stderr)
                sys.exit(1)
            
            mjcf_path = str(mjcf_path)

        from .server import serve

        serve(
            mjcf_path=mjcf_path,
            port=args.port,
            allowed_origin=args.origin,
        )

    elif args.command == "version":
        from . import __version__

        print(f"Robola {__version__}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
