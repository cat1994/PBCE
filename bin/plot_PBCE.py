#!/usr/bin/env python3
"""Plot archived PBCE/PEFR fixed and adaptive diagnostics from SSV or SSV.GZ.

The three columns retain the historical experiment's metric names. See the
release's docs/IMPLEMENTATION_STATUS.md for differences from the current paper.
No pickled NumPy files or notebook execution are required.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
GAMES = ("kuhn_3p", "leduc", "random_goofspiel", "sheriff", "tiny_bridge")
ALGORITHMS = ("A-EFR_IN", "CFR", "CFR_IN", "CSPS-EFR", "CFPS-EFR")
LABELS = dict(zip(ALGORITHMS, ("ACT-IN", "CF", "CF-IN", "CSPS", "CFPS")))
GAME_LABELS = dict(zip(GAMES, (
    "Kuhn Poker (N=3)", "Leduc Poker (N=2)", "Goofspiel (4, R, N=2)",
    "Sheriff (N=2)", "Tiny Bridge (N=2)",
)))
EPSILONS = ("1e-1", "5e-2", "1e-2", "5e-3", "1e-3", "0")
COLORS = ("#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73", "#8B7A00")
METRICS = ("posr", "osr", "csr")
TITLES = (r"Perturbed OSR($\mu$) Gap", r"OSR($\nu$) Gap", r"CSR($\nu$) Gap")
HEADER = "algorithm  (ev, milliseconds, max_full_phi_regrets_fus, max_full_phi_regret, max_phi_regret, sum_phi_regret)"
ROW = re.compile(r"^(\S+)\s+\(([^()]*)\)$")


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def find_input(directory, game, parameter):
    base = directory / f"pefr.{game}.null.sim.gen.{parameter}.ssv"
    candidates = [p for p in (base, Path(str(base) + ".gz")) if p.is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one plain/compressed input for {base}; found {len(candidates)}")
    return candidates[0]


def sample_indices(count, maximum):
    if maximum == 0 or count <= maximum:
        return set(range(count))
    # Same endpoint-preserving, uniformly spaced indices as the paper notebook.
    return {i * (count - 1) // (maximum - 1) for i in range(maximum)}


def fixed_records(path, max_points=100, stride=100):
    """Validate every block; only retain uniformly sampled records for rendering.

    Full regrets are divided by t+1. CSR is already a normalized value in the
    file and must NOT be divided by the round count or the game's utility range.
    """
    count = 0
    with open_text(path) as stream:
        if stream.readline().strip() != HEADER:
            raise ValueError(f"{path}: unsupported fixed schema (six fields required)")
        for line in stream:
            if line.startswith("t = "):
                if int(line[4:]) != count:
                    raise ValueError(f"{path}: missing, duplicate or unordered fixed round")
                count += 1
    if not count:
        raise ValueError(f"{path}: no fixed records")
    candidates = range(0, count, stride)
    keep = {candidates[index] for index in sample_indices(len(candidates), max_points)}
    result = {algorithm: [] for algorithm in ALGORITHMS}
    current = -1
    seen = set()
    with open_text(path) as stream:
        next(stream)
        for number, line in enumerate(stream, 2):
            line = line.strip()
            if not line:
                continue
            if line.startswith("t = "):
                if current >= 0 and seen != set(ALGORITHMS):
                    raise ValueError(f"{path}:{number}: incomplete algorithm block {current}")
                current = int(line[4:])
                seen = set()
                continue
            match = ROW.fullmatch(line)
            if not match or current < 0:
                raise ValueError(f"{path}:{number}: malformed fixed record")
            algorithm, values = match.groups()
            if algorithm not in result or algorithm in seen:
                raise ValueError(f"{path}:{number}: unknown/duplicate algorithm {algorithm}")
            numbers = [float(value) for value in values.split(",")]
            if len(numbers) != 6 or not all(map(math.isfinite, numbers)):
                raise ValueError(f"{path}:{number}: invalid/nonfinite metric tuple")
            seen.add(algorithm)
            if current in keep:
                result[algorithm].append({
                    "x": current + 1, "posr": numbers[2] / (current + 1),
                    "osr": numbers[3] / (current + 1), "csr": numbers[4],
                })
    if seen != set(ALGORITHMS):
        raise ValueError(f"{path}: incomplete final algorithm block")
    return result, {"rounds": count, "points_per_curve": len(keep), "last_plotted_round": max(keep) + 1}


def adaptive_records(path):
    """Use CHECKPOINT rows only; RUN_END is used for completion metadata."""
    result = {algorithm: [] for algorithm in ALGORITHMS}
    endings = {}
    with open_text(path) as stream:
        reader = csv.DictReader(stream)
        required = {"record_type", "algorithm", "global_t", "epsilon_stage", "epsilon", *METRICS, "status"}
        if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
            raise ValueError(f"{path}: duplicate adaptive columns")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing adaptive columns {sorted(required - set(reader.fieldnames or []))}")
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{number}: truncated or oversized adaptive row")
            algorithm = row["algorithm"]
            if algorithm not in result:
                raise ValueError(f"{path}:{number}: unknown algorithm {algorithm}")
            if algorithm in endings:
                raise ValueError(f"{path}:{number}: record after RUN_END")
            if row["record_type"] not in {"CHECKPOINT", "STAGE_END", "EPSILON_UPDATE", "RUN_END"}:
                raise ValueError(f"{path}:{number}: unknown record type")
            iteration = int(row["global_t"])
            epsilon = float(row["epsilon"])
            if iteration < 0 or not math.isfinite(epsilon) or epsilon <= 0:
                raise ValueError(f"{path}:{number}: invalid iteration/epsilon")
            if result[algorithm] and iteration < result[algorithm][-1]["x"]:
                raise ValueError(f"{path}:{number}: record precedes the latest checkpoint")
            if row["record_type"] == "RUN_END":
                if row["status"] not in {"MAX_GLOBAL_ITER", "STAGE_TIMEOUT", "FINAL_NOT_CERTIFIED", "CERTIFIED_FINAL"}:
                    raise ValueError(f"{path}:{number}: duplicate or unsuccessful RUN_END")
                if not all(math.isfinite(float(row[k])) for k in METRICS):
                    raise ValueError(f"{path}:{number}: nonfinite terminal metrics")
                endings[algorithm] = {"status": row["status"], "epsilon": epsilon, "global_t": iteration}
            elif row["record_type"] == "CHECKPOINT":
                record = {"x": int(row["global_t"]), "stage": int(row["epsilon_stage"]),
                          "epsilon": float(row["epsilon"]), **{k: float(row[k]) for k in METRICS}}
                if not all(math.isfinite(record[k]) for k in (*METRICS, "epsilon")):
                    raise ValueError(f"{path}:{number}: nonfinite checkpoint")
                if record["epsilon"] <= 0:
                    raise ValueError(f"{path}:{number}: nonpositive adaptive epsilon")
                if result[algorithm] and record["x"] <= result[algorithm][-1]["x"]:
                    raise ValueError(f"{path}:{number}: non-increasing checkpoints")
                result[algorithm].append(record)
    if set(endings) != set(ALGORITHMS) or not all(result.values()):
        raise ValueError(f"{path}: incomplete adaptive result")
    for algorithm in ALGORITHMS:
        result[algorithm][-1]["terminal_epsilon"] = endings[algorithm]["epsilon"]
    return result, {"endings": endings, "checkpoints": {k: len(v) for k, v in result.items()}}


def positive(values):
    # Keep nonpositive values in source data, but break their log-scale segments.
    return [value if value > 0 else float("nan") for value in values]


def epsilon_label(value):
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"{float(mantissa):g}\times10^{{{int(exponent)}}}"


def draw_grid(series, games, algorithm, mode, output, formats, connect_stages):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter, ScalarFormatter

    rc = {"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
          "axes.titleweight": "semibold", "xtick.labelsize": 10, "ytick.labelsize": 10,
          "legend.fontsize": 11, "figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
          "axes.edgecolor": "#333333", "axes.linewidth": 0.8, "pdf.fonttype": 42,
          "savefig.dpi": 180}
    with plt.rc_context(rc):
        height = 2.55 * len(games) + 1.2
        fig, axes = plt.subplots(len(games), 3, figsize=(13, height), squeeze=False)
        fig.subplots_adjust(left=0.12, right=0.985, bottom=0.6 / height,
                            top=1 - (0.9 if mode == "fixed" else 0.4) / height,
                            wspace=0.28, hspace=0.36)
        for row_index, game in enumerate(games):
            for col, metric in enumerate(METRICS):
                ax = axes[row_index, col]
                if mode == "fixed":
                    for epsilon, color in zip(EPSILONS, COLORS):
                        records = series[(game, epsilon)][algorithm]
                        ax.plot([r["x"] for r in records], positive([r[metric] for r in records]), color=color, lw=1.8)
                else:
                    records = series[game][algorithm]
                    stages = []
                    for record in records:
                        if not stages or record["stage"] != stages[-1][0]["stage"]:
                            stages.append([])
                        stages[-1].append(record)
                    for group in ([records] if connect_stages else stages):
                        ax.plot([r["x"] for r in group], positive([r[metric] for r in group]), color=COLORS[0], lw=1.7)
                    for group in stages[1:]:
                        ax.axvline(group[0]["x"], color="#8A8A8A", ls="--", lw=0.65, alpha=0.7)
                    ax.axvspan(stages[-1][0]["x"], stages[-1][-1]["x"], color="#E69F00", alpha=0.08, lw=0)
                    ax.text(0.98, 0.96, rf"$\epsilon_{{\mathrm{{final}}}}={epsilon_label(records[-1]['terminal_epsilon'])}$",
                            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#8A4B00",
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
                ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
                ax.yaxis.set_minor_formatter(NullFormatter())
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((4, 4))
                ax.xaxis.set_major_formatter(formatter)
                ax.grid(True, which="major", color="#B8B8B8", lw=0.65, alpha=0.55)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if row_index == 0:
                    ax.set_title(TITLES[col], pad=8)
                if col == 0:
                    ax.set_ylabel(GAME_LABELS[game], fontweight="semibold", labelpad=9)
                if row_index == len(games) - 1:
                    ax.set_xlabel("Round" if mode == "fixed" else "Global iteration")
        if mode == "fixed":
            handles = [Line2D([0], [0], color=c, lw=2, label=rf"$\epsilon={float(e):g}$") for e, c in zip(EPSILONS, COLORS)]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 1 - 0.05 / height), ncol=6, frameon=True)
        name = f"pefr.{algorithm}.round.gaps.alt.cfrplus" if mode == "fixed" else f"pefr.{algorithm}.adaptive.gaps"
        paths = []
        for extension in formats:
            path = output / mode / f"{name}.{extension}"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            paths.append(str(path))
            print(path, flush=True)
        plt.close(fig)
    return paths


def resolve(path):
    return path if path.is_absolute() else ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixed", "adaptive", "both"), default="both")
    parser.add_argument("--data-dir", type=Path, help="Root containing fixed/ and adaptive/ (plain or gzip SSV)")
    parser.add_argument("--fixed-data", type=Path, help="Override the fixed input directory")
    parser.add_argument("--adaptive-data", type=Path, help="Override the adaptive input directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--games", nargs="+", choices=GAMES, default=list(GAMES))
    parser.add_argument("--algorithms", nargs="+", choices=(*ALGORITHMS, *LABELS.values()), default=list(ALGORITHMS))
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png", "svg"), default=["pdf"])
    parser.add_argument("--max-points", type=int, default=100, help="Uniform fixed-curve sampling, 0 for all; first/last retained")
    parser.add_argument("--fixed-stride", type=int, default=100, help="Read every Nth round before curve sampling; 100 matches the paper notebook, 1 uses all rounds")
    parser.add_argument("--disconnect-stages", action="store_true", help="Break adaptive curves at epsilon transitions")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs and write metadata without requiring Matplotlib")
    args = parser.parse_args(argv)
    if args.max_points < 0 or args.max_points == 1:
        parser.error("--max-points must be 0 or at least 2")
    if args.fixed_stride < 1:
        parser.error("--fixed-stride must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    reverse_labels = {value: key for key, value in LABELS.items()}
    algorithms = list(dict.fromkeys(reverse_labels.get(value, value) for value in args.algorithms))
    games = list(dict.fromkeys(args.games))
    if args.data_dir:
        fixed_dir, adaptive_dir = resolve(args.data_dir) / "fixed", resolve(args.data_dir) / "adaptive"
    elif (ROOT / "data/raw/fixed").is_dir():
        fixed_dir, adaptive_dir = ROOT / "data/raw/fixed", ROOT / "data/raw/adaptive"
    else:
        # Source workspace: paper adaptive runs use decay 0.5, not root's 0.75.
        fixed_dir = ROOT / "data_pbce"
        adaptive_dir = fixed_dir / "data_pbce_gamma0d5"
    fixed_dir = resolve(args.fixed_data) if args.fixed_data else fixed_dir
    adaptive_dir = resolve(args.adaptive_data) if args.adaptive_data else adaptive_dir
    output = resolve(args.output_dir)
    report = {"schema_version": 1, "metric_interpretation": "historical implementation diagnostics; see implementation audit",
              "fixed_normalization": "posr=fus/(t+1); osr=full/(t+1); csr=max_phi_regret",
              "adaptive_normalization": "use raw posr, osr, csr checkpoint columns",
              "max_fixed_points": args.max_points, "fixed_stride": args.fixed_stride, "connect_adaptive_stages": not args.disconnect_stages,
              "inputs": [], "figures": []}
    for mode in ("fixed", "adaptive"):
        if args.mode not in (mode, "both"):
            continue
        series = {}
        for game in games:
            for parameter in (EPSILONS if mode == "fixed" else ("adaptive",)):
                path = find_input(fixed_dir if mode == "fixed" else adaptive_dir, game, parameter)
                records, metadata = fixed_records(path, args.max_points, args.fixed_stride) if mode == "fixed" else adaptive_records(path)
                series[(game, parameter) if mode == "fixed" else game] = records
                report["inputs"].append({"mode": mode, "game": game, "parameter": parameter,
                    "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **metadata})
                print(f"Validated {path.name}", flush=True)
        if not args.validate_only:
            for algorithm in algorithms:
                report["figures"].extend(draw_grid(series, games, algorithm, mode, output, args.formats, not args.disconnect_stages))
    output.mkdir(parents=True, exist_ok=True)
    (output / "plot_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        sys.exit(f"PBCE plot error: {error}")
