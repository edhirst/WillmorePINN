"""Systematic hyperparameter search for genus 2 Willmore energy minimisation.

Runs short trials to compare hyperparameter configurations efficiently.
Can be used for local sequential search or batched HPC parallel submission.

Usage (local sequential):
    cd /path/to/willmore
    conda run -n willmore python scratch/tune_genus2.py --n-trials 20 --epochs 300

Usage (single trial for HPC parallel submission, 0-indexed):
    python scratch/tune_genus2.py --trial-idx 5 --n-trials 20 --epochs 300

Usage (print report from existing results without running more trials):
    python scratch/tune_genus2.py --report

The search space is defined in SEARCH_SPACE below.  Edit it to focus on the
parameters most relevant to your current investigation.

Scoring (lower is better):
    score = best_W_second_half * (1 + 0.5 * CV_second_half)
where best_W is the minimum Willmore energy in the second half of training and
CV is the coefficient of variation (std/mean) over the same window, penalising
unstable runs that happen to briefly dip low.
"""

import sys
import os
import glob
import json
import copy
import argparse
import random
import functools

import numpy as np
import yaml
print = functools.partial(print, flush=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
# Each entry: key_path -> (kind, spec)
#   'categorical'  : sample uniformly from list spec
#   'log_uniform'  : sample from log-uniform(spec[0], spec[1])
# key_path uses '.' to index nested YAML keys, e.g.
#   'training.adaptive_training.willmore_warmup_epochs'
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    'training.learning_rate': (
        'log_uniform', [3e-5, 3e-4]
    ),
    'training.gradient_clip': (
        'categorical', [0.1, 0.2, 0.5, 1.0]
    ),
    'loss.regularity_weight': (
        'categorical', [0.5, 1.0, 2.0, 5.0]
    ),
    'loss.regularity_conformal_weight': (
        'categorical', [0.0, 0.5, 1.0, 2.0, 4.0]
    ),
    'loss.h2_clip': (
        'categorical', [10, 25, 50, 100, None]
    ),
    'loss.regularity_min_area_element': (
        'categorical', [0.01, 0.03, 0.05, 0.1]
    ),
    'training.adaptive_training.willmore_warmup_epochs': (
        'categorical', [0, 50, 100, 150]
    ),
    'training.adaptive_training.willmore_warmup_start': (
        'categorical', [0.02, 0.05, 0.1, 0.2]
    ),
    # Self-avoidance: directly discourages self-intersection
    'loss.self_avoidance_weight': (
        'categorical', [0.0, 0.02, 0.05, 0.1, 0.2]
    ),
    # Frequency curriculum: coarse-to-fine Fourier feature activation
    'model.freq_curriculum.start_freqs': (
        'categorical', [1, 2, 3, 6]
    ),
    'model.freq_curriculum.warmup_epochs': (
        'categorical', [0, 50, 100, 150]
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_params(search_space: dict, rng: random.Random) -> dict:
    """Sample one hyperparameter configuration from the search space."""
    params = {}
    for key, (kind, spec) in search_space.items():
        if kind == 'categorical':
            params[key] = rng.choice(spec)
        elif kind == 'log_uniform':
            lo, hi = spec
            params[key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            raise ValueError(f"Unknown sampling kind: {kind}")
    return params


def set_nested(cfg: dict, key_path: str, value) -> None:
    """Set a value at a dotted key path inside a nested dict."""
    keys = key_path.split('.')
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def score_trial(history: dict) -> float:
    """Score a completed trial; lower is better.

    Uses the minimum Willmore energy in the second half of training,
    penalised by the coefficient of variation (stability proxy).
    """
    W = np.array(history.get('willmore_energy', []), dtype=float)
    W = W[np.isfinite(W)]
    if len(W) < 10:
        return float('inf')

    second_half = W[len(W) // 2:]
    best = float(np.min(second_half))
    if best > 1000:
        return float('inf')

    cv = float(np.std(second_half)) / (float(np.mean(second_half)) + 1e-8)
    return best * (1.0 + 0.5 * cv)


def run_trial(base_config: dict, params: dict, trial_dir: str, trial_epochs: int) -> dict:
    """Run one short training trial and return the training history dict."""
    from run import train

    cfg = copy.deepcopy(base_config)

    # Apply hyperparameter overrides
    for key_path, value in params.items():
        set_nested(cfg, key_path, value)

    # Override training length
    cfg['training']['num_epochs'] = trial_epochs
    # Keep cosine T_max consistent with the shortened run
    if cfg['training'].get('scheduler', 'cosine') == 'cosine':
        cfg['training'].setdefault('scheduler_params', {})['T_max'] = trial_epochs

    # Shorten supervised pretraining to ≤50 epochs for faster trials
    pre = cfg.get('model', {}).get('supervised_pretraining', {})
    if pre.get('enabled', False):
        pre['num_epochs'] = min(50, pre.get('num_epochs', 150))

    # Redirect output to an isolated trial directory, not the main run_* hierarchy
    os.makedirs(trial_dir, exist_ok=True)
    cfg['output'] = {
        'checkpoint_dir': os.path.join(trial_dir, 'checkpoints'),
        'log_dir': os.path.join(trial_dir, 'logs'),
        'save_best_only': False,
        'metric_to_monitor': 'willmore_energy',
    }

    # Reduce checkpoint frequency to save I/O during short trials
    cfg['training']['log_frequency'] = 50

    history = train(config_dict=cfg)

    return history


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _merge_results(output_path: str) -> None:
    """Merge per-trial tune_results_trial_NNN.json files into output_path.

    Each parallel PBS job writes its own file to avoid concurrent write races.
    This function combines them into the single results JSON used by _print_report.
    """
    results_dir = os.path.dirname(os.path.abspath(output_path))
    pattern = os.path.join(results_dir, 'tune_results_trial_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No per-trial result files found matching {pattern}")
        return
    results = []
    for fpath in files:
        with open(fpath) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            results.extend(data)
        else:
            results.append(data)
    results.sort(key=lambda r: r.get('trial_idx', 0))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"Merged {len(results)} trial(s) from {len(files)} file(s) → {output_path}")


def _print_report(results_path: str) -> None:
    """Print a human-readable summary table of completed trial results."""
    if not os.path.exists(results_path):
        print(f"No results file found at {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    if not results:
        print("No results yet.")
        return

    results_sorted = sorted(results, key=lambda r: r.get('score', float('inf')))

    print(f"\n{'='*90}")
    print(f"{'HYPERPARAMETER SEARCH RESULTS':^90}")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'Trial':<7} {'Score':<10} {'Best W':<10}  Key Parameters")
    print('-' * 90)

    for rank, r in enumerate(results_sorted):
        s = r.get('score', float('inf'))
        bw = r.get('best_willmore', float('inf'))
        idx = r.get('trial_idx', '?')
        p = r.get('params', {})
        short = {
            'lr':   p.get('training.learning_rate', '?'),
            'reg':  p.get('loss.regularity_weight', '?'),
            'conf': p.get('loss.regularity_conformal_weight', '?'),
            'wup':  p.get('training.adaptive_training.willmore_warmup_epochs', '?'),
            'h2':   p.get('loss.h2_clip', '?'),
            'clip': p.get('training.gradient_clip', '?'),
        }
        p_str = '  '.join(f'{k}={v}' for k, v in short.items())
        print(f"{rank+1:<5} {idx:<7} {s:<10.3f} {bw:<10.3f}  {p_str}")

    print(f"\nBest trial: #{results_sorted[0].get('trial_idx', '?')}")
    print("Best parameters:")
    for k, v in results_sorted[0].get('params', {}).items():
        print(f"  {k}: {v}")

    # Additionally report the most impactful parameters by average score
    _print_param_analysis(results_sorted)


def _print_param_analysis(results: list) -> None:
    """Group results by each hyperparameter value and print mean score per group."""
    if len(results) < 4:
        return

    print(f"\n{'Parameter influence (mean score by value)':}")
    print('-' * 60)

    all_keys = set()
    for r in results:
        all_keys.update(r.get('params', {}).keys())

    finite_results = [r for r in results if np.isfinite(r.get('score', float('inf')))]
    if not finite_results:
        return

    for key in sorted(all_keys):
        groups: dict = {}
        for r in finite_results:
            val = r.get('params', {}).get(key, None)
            val_str = str(val)
            groups.setdefault(val_str, []).append(r['score'])

        if len(groups) <= 1:
            continue

        print(f"\n  {key}:")
        sorted_groups = sorted(groups.items(), key=lambda kv: np.mean(kv[1]))
        for val_str, scores in sorted_groups:
            print(f"    {val_str:<12}  mean={np.mean(scores):.3f}  n={len(scores)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for genus 2 Willmore minimisation"
    )
    parser.add_argument(
        '--config',
        default=os.path.join(ROOT, 'hyperparameters.yaml'),
        help='Base YAML configuration file (default: hyperparameters.yaml)'
    )
    parser.add_argument(
        '--n-trials', type=int, default=20,
        help='Total number of trials in the search (default: 20)'
    )
    parser.add_argument(
        '--epochs', type=int, default=300,
        help='Training epochs per trial, excluding pretraining (default: 300)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for hyperparameter sampling (default: 42)'
    )
    parser.add_argument(
        '--output',
        default=os.path.join(ROOT, 'tuning', 'tune_results.json'),
        help='Path for results JSON (default: tuning/tune_results.json)'
    )
    parser.add_argument(
        '--trial-idx', type=int, default=None,
        help='Run only this single trial index (0-based); for HPC parallel submission'
    )
    parser.add_argument(
        '--report', action='store_true',
        help='Print report from existing results file without running new trials'
    )
    parser.add_argument(
        '--merge', action='store_true',
        help='Merge per-trial result files (tune_results_trial_NNN.json) then print report'
    )
    args = parser.parse_args()

    # When running a single PBS array job, write to a per-trial file to avoid
    # concurrent write races between parallel jobs.
    _DEFAULT_OUTPUT = os.path.join(ROOT, 'tuning', 'tune_results.json')
    if args.trial_idx is not None and args.output == _DEFAULT_OUTPUT:
        args.output = os.path.join(
            ROOT, 'tuning', f'tune_results_trial_{args.trial_idx:03d}.json'
        )

    if args.merge:
        _merge_results(_DEFAULT_OUTPUT)
        _print_report(_DEFAULT_OUTPUT)
        return

    if args.report:
        _print_report(args.output)
        return

    # Load base configuration
    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    # Pre-generate all trial parameter sets deterministically from the seed
    rng = random.Random(args.seed)
    all_params = [sample_params(SEARCH_SPACE, rng) for _ in range(args.n_trials)]

    base_tune_dir = os.path.join(ROOT, 'tuning', 'tune_output')
    trial_indices = [args.trial_idx] if args.trial_idx is not None else range(args.n_trials)

    # Load existing results for incremental appending
    results = []
    if os.path.exists(args.output):
        with open(args.output) as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []

    existing_indices = {r['trial_idx'] for r in results}

    for idx in trial_indices:
        if idx in existing_indices:
            print(f"Trial {idx} already completed, skipping.")
            continue

        params = all_params[idx]
        trial_dir = os.path.join(base_tune_dir, f'trial_{idx:03d}')

        print(f"\n{'='*60}")
        print(f"Trial {idx}/{args.n_trials - 1}  ({args.epochs} epochs)")
        print("Parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print()

        try:
            history = run_trial(base_config, params, trial_dir, args.epochs)
            s = score_trial(history)
            W_all = history.get('willmore_energy', [float('inf')])
            W_finite = [w for w in W_all if np.isfinite(w)]
            best_W = float(min(W_finite)) if W_finite else float('inf')
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            s, best_W, history = float('inf'), float('inf'), {}

        result = {
            'trial_idx': idx,
            'score': s,
            'best_willmore': best_W,
            'params': params,
            'n_epochs': len(history.get('epoch', [])),
        }
        results.append(result)
        print(f"  score={s:.3f}  best_W={best_W:.3f}")

        # Save incrementally so partial results are preserved
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

    _print_report(args.output)


if __name__ == '__main__':
    main()
