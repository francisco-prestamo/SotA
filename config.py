import argparse
import sys

# Global variable to store the result
_INSPECT_QUERY = False

def inspect_query() -> bool:
    """
    Returns whether query inspection mode is enabled.
    """
    global _INSPECT_QUERY
    return _INSPECT_QUERY

def _parse_args():
    global _INSPECT_QUERY

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', '--inspect-query', action='store_true', help='Enable query inspection mode.')

    # Parse only known args to avoid interfering with other modules
    args, _ = parser.parse_known_args(sys.argv[1:])
    _INSPECT_QUERY = args.inspect_query

# Run argument parsing once at import time
_parse_args()

