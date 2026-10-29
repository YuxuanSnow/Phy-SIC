"""Printing and comparing result dicts."""

import json
import math


def print_results(results):
    """Print one benchmark's result dict as tables."""
    print(f"\n{results['dataset']} -- {results['num_frames']} frames\n")
    print(f"{'':10s} {'PA-MPJPE':>9s} {'MPJPE':>9s} {'MPVPE':>9s}   (mm)")
    for name, row in results['pose'].items():
        print(f"{name:10s} {row['pa_mpjpe']:9.2f} {row['mpjpe']:9.2f} {row['mpvpe']:9.2f}")

    for split, rows in results['contact'].items():
        print(f"\ncontact -- {split} vertices")
        print(f"{'':10s} {'Acc':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s}")
        for name, row in rows.items():
            print(f"{name:10s} {row['accuracy']:8.3f} {row['precision']:8.3f} "
                  f"{row['recall']:8.3f} {row['f1']:8.3f}")

    print(f"\nvertex-to-scene vectors   L2 {results['vectors']['v2s_l2_m']:.4f} m   "
          f"angle {results['vectors']['v2s_angle_deg']:.2f} deg")


def flatten(node, prefix=''):
    """Walk a result dict into {'pose.ours.mpjpe': 230.26, ...}."""
    flat = {}
    for key, value in node.items():
        path = f'{prefix}{key}'
        if isinstance(value, dict):
            flat.update(flatten(value, f'{path}.'))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[path] = float(value)
    return flat


def compare_results(left, right, left_name='left', right_name='right', tolerance=1e-6,
                    verbose=True):
    """Compare two result dicts field by field; True if every field agrees within
    ``tolerance`` (relative) and neither has fields the other lacks."""
    left, right = flatten(left), flatten(right)

    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    for name, keys in [(left_name, only_right), (right_name, only_left)]:
        for key in keys:
            print(f'missing from {name}: {key}')

    if verbose:
        print(f"{'metric':38s} {left_name:>16s} {right_name:>16s} {'rel. diff':>12s}")
    worst = 0.0
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        if math.isnan(a) or math.isnan(b):
            rel = 0.0 if math.isnan(a) and math.isnan(b) else math.inf
        else:
            rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
        worst = max(worst, rel)
        if verbose or rel > tolerance:
            flag = '' if rel <= tolerance else '   <-- differs'
            print(f'{key:38s} {a:16.6f} {b:16.6f} {rel:12.2e}{flag}')

    agree = worst <= tolerance and not only_left and not only_right
    print(f'largest relative difference: {worst:.2e} -- '
          f"{'MATCH' if agree else 'DIFFERENCES ABOVE TOLERANCE'}")
    return agree


def load_json(path):
    return json.loads(path.read_text())
