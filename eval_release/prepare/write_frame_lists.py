#!/usr/bin/env python
"""Regenerate the frame lists in ``assets/`` from the released predictions.

The lists say which frames each benchmark covers. For RICH they double as the
recipe for rebuilding the evaluation sample from your own RICH download, since
the sample itself is not redistributed here.
"""

import argparse
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=RELEASE_ROOT / 'release_data')
    parser.add_argument('--assets-dir', type=Path, default=RELEASE_ROOT / 'assets')
    return parser.parse_args()


def main():
    args = parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)

    prox = sorted(f'{p.parent.name}/{p.stem}' for p in (args.data_dir / 'prox').glob('*/*.npz'))
    rich = sorted(p.stem for p in (args.data_dir / 'rich').glob('*.npz'))

    for name, frames, header in [
        ('prox_frames.txt', prox,
         '# PROX-quantitative frames evaluated, as <recording>/<frame>.'),
        ('rich_sample_frames.txt', rich,
         '# RICH frames evaluated, as <scene>_<subject>_<action>+<frame>_<camera>.\n'
         '# Rebuild the evaluation sample from your own RICH download; see README.md.'),
    ]:
        (args.assets_dir / name).write_text(header + '\n' + '\n'.join(frames) + '\n')
        print(f'{args.assets_dir / name}: {len(frames)} frames')


if __name__ == '__main__':
    main()
