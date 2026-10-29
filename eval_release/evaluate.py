#!/usr/bin/env python
"""Reproduce PhySIC's published PROX and RICH numbers from the released predictions.

    python evaluate.py --prox-root /data/PROX/quantitative --rich-root /data/rich-sample100
    python evaluate.py --benchmark prox --prox-root ... --limit 10     # quick check

The released predictions are fetched from Hugging Face on first use. Results are
printed as tables and written to ``<out-dir>/<benchmark>.json``; a full run is
then compared field by field against ``expected_results/`` and reports MATCH or
the fields that differ. Dataset roots can also come from ``$PHYSIC_PROX_ROOT``
and ``$PHYSIC_RICH_ROOT``.
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lib.report import compare_results, print_results  # noqa: E402

HF_REPO = 'ympradyumna/physic-eval-release'
TIER_DIRS = {'compact': 'release_data', 'full': 'release_data_full'}
FRAME_LISTS = {'prox': 'assets/prox_frames.txt', 'rich': 'assets/rich_sample_frames.txt'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--benchmark', action='append', choices=['prox', 'rich'],
                        help='benchmark to run; repeat for both (default: both)')
    parser.add_argument('--prox-root', type=Path, default=os.environ.get('PHYSIC_PROX_ROOT'),
                        help='PROX quantitative root ($PHYSIC_PROX_ROOT)')
    parser.add_argument('--rich-root', type=Path, default=os.environ.get('PHYSIC_RICH_ROOT'),
                        help='RICH 100-frame sample root, see README ($PHYSIC_RICH_ROOT)')
    parser.add_argument('--data-root', type=Path, default=HERE.parent / 'data',
                        help='root holding body_models/smplx/, as fetch_data.sh lays it out '
                             '(default: the repo data/ folder)')
    parser.add_argument('--tier', choices=list(TIER_DIRS), default='compact',
                        help='released data tier: compact (25 MB) or full (13 GB); same numbers')
    parser.add_argument('--data-dir', type=Path, default=None,
                        help='read predictions from <data-dir>/<benchmark>/ instead of the '
                             'downloaded tier (still checked against the --tier expected results)')
    parser.add_argument('--scan-suffix', default='',
                        help="RICH: suffix before .ply when picking a scene scan, e.g. '_simp' "
                             "for GRAFT's decimated scans; empty means the scans as RICH ships them")
    parser.add_argument('--out-dir', type=Path, default=HERE / 'results')
    parser.add_argument('--limit', type=int, default=None,
                        help='only evaluate the first N frames (skips the expected-results check)')
    parser.add_argument('--no-scene-cache', action='store_true',
                        help='rebuild the scene ray-casting structure for every frame; much '
                             'slower, and must give identical numbers')
    parser.add_argument('--device', default='auto', help="'auto', 'cuda' or 'cpu'")
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='relative difference from expected_results/ still counted as a match '
                             '(default 1e-3; see README on GPU-to-GPU differences)')
    return parser.parse_args()


def missing_frames(path, benchmark):
    """Frames of the evaluation set with no prediction file under ``path``."""
    lines = (HERE / FRAME_LISTS[benchmark]).read_text().splitlines()
    expected = {line.strip() for line in lines if line.strip() and not line.startswith('#')}
    present = {p.relative_to(path).with_suffix('').as_posix() for p in path.rglob('*.npz')}
    return sorted(expected - present)


def prediction_dir(args, benchmark):
    """Where this benchmark's predictions live, downloading them if any are missing."""
    if args.data_dir is not None:
        path = args.data_dir / benchmark
    else:
        tier_dir = TIER_DIRS[args.tier]
        path = HERE / tier_dir / benchmark
        if missing_frames(path, benchmark):
            from huggingface_hub import snapshot_download
            print(f'downloading {tier_dir}/{benchmark} from {HF_REPO} ...')
            snapshot_download(HF_REPO, repo_type='dataset', local_dir=HERE,
                              allow_patterns=f'{tier_dir}/{benchmark}/*')
    missing = missing_frames(path, benchmark)
    if missing:
        raise SystemExit(f'{len(missing)} of the evaluated {benchmark} frames have no prediction '
                         f'file under {path}, e.g. {missing[0]}')
    return path


def result_name(args, benchmark):
    """``prox``, ``rich_full``, ``rich_simp`` ... -- also the expected-results file name."""
    name = benchmark
    if args.tier == 'full':
        name += '_full'
    if benchmark == 'rich' and args.scan_suffix:
        name += args.scan_suffix
    return name


def main():
    args = parse_args()
    benchmarks = args.benchmark or ['prox', 'rich']
    limit = args.limit or None

    import torch
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    roots = {'prox': args.prox_root, 'rich': args.rich_root}
    missing = [b for b in benchmarks if roots[b] is None]
    if missing:
        raise SystemExit(f"no dataset root for {', '.join(missing)}: pass "
                         + ', '.join(f'--{b}-root' for b in missing) + ' (see README)')

    contact_weights = HERE / 'assets/contact_vertice_weights.pkl'
    checks = {}
    for benchmark in benchmarks:
        data_dir = prediction_dir(args, benchmark)
        if benchmark == 'prox':
            from lib.prox import evaluate_prox
            results = evaluate_prox(args.prox_root, data_dir, args.data_root / 'body_models/smplx',
                                    contact_weights, limit=limit,
                                    no_scene_cache=args.no_scene_cache, device=device)
        else:
            from lib.rich import evaluate_rich
            results = evaluate_rich(args.rich_root, data_dir, args.data_root, contact_weights,
                                    scan_suffix=args.scan_suffix, limit=limit,
                                    no_scene_cache=args.no_scene_cache, device=device)
        print_results(results)

        name = result_name(args, benchmark)
        out_path = args.out_dir / f'{name}.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f'\nwrote {out_path}')

        expected_path = HERE / f'expected_results/{name}.json'
        if limit is not None:
            print('\n--limit given: not checked against expected_results/')
        elif not expected_path.exists():
            print(f'\nno expected_results/{name}.json for this configuration: not checked')
        else:
            print(f'\nchecking against {expected_path.relative_to(HERE)}')
            checks[name] = compare_results(json.loads(expected_path.read_text()), results,
                                           'expected', 'this run', args.tolerance, verbose=False)

    all_match = all(checks.values())
    if checks:
        summary = ', '.join(f"{name} {'matches' if ok else 'DIFFERS'}" for name, ok in checks.items())
        print(f"\n{'PASS' if all_match else 'FAIL'}: {summary}")
    raise SystemExit(0 if all_match else 1)


if __name__ == '__main__':
    main()
