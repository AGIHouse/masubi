"""Allow running autotrust as a module with subcommands.

Usage:
    python -m autotrust freeze [--run-id <id>] [--teacher-dir <dir>]
    python -m autotrust export --checkpoint <path> [--format gguf] [--output <path>]
"""

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m autotrust <command> [args...]")
        print("Commands: freeze, export")
        sys.exit(1)

    command = sys.argv[1]
    # Remove the subcommand from argv so argparse in each module sees clean args
    remaining_argv = sys.argv[2:]

    if command == "freeze":
        from autotrust.freeze import main as freeze_main
        freeze_main(remaining_argv)
    elif command == "export":
        from autotrust.export import main as export_main
        export_main(remaining_argv)
    else:
        print(f"Unknown command: {command}")
        print("Commands: freeze, export")
        sys.exit(1)


if __name__ == "__main__":
    main()
