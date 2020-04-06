# !/usr/bin/env python
# -*- coding: utf-8 -*- #
# XL8-precommit: no unittests.
"""alignment_utils_main module."""
import sys

from laser import AlignmentUtils
from laser import parse_args
from laser import start_use_batching


def main():
    """Run the main task."""
    args = parse_args(sys.argv[1:])
    alignment_utils = AlignmentUtils()
    start_use_batching(alignment_utils, args)


if __name__ == "__main__":
    main()
