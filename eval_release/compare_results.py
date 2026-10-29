#!/usr/bin/env python
"""Compare two result JSONs field by field.

``evaluate.py`` already checks every full run against ``expected_results/``;
use this to compare any two runs yourself, e.g. the compact tier against the
full one:

    python evaluate.py --benchmark rich --rich-root ...
    python evaluate.py --benchmark rich --rich-root ... --tier full
    python compare_results.py results/rich.json results/rich_full.json

Pose and contact metrics agree exactly between the two tiers. The
vertex-to-scene vector metric can move in its last few digits: where two scene
points are exactly equidistant from a body vertex, dropping one of them lets the
tie break the other way, which changes that vertex's direction vector without
changing its distance or its contact bit.
"""

import argparse
from pathlib import Path

from lib.report import compare_results, load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--tolerance', type=float, default=1e-6,
                        help='relative difference to treat as agreement (default 1e-6)')
    args = parser.parse_args()

    agree = compare_results(load_json(args.left), load_json(args.right),
                            args.left.stem, args.right.stem, args.tolerance)
    raise SystemExit(0 if agree else 1)


if __name__ == '__main__':
    main()
