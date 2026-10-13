#!/usr/bin/env python3
"""Plot PBCE/PEFR fixed and adaptive diagnostics from SSV or SSV.GZ.

Raw algorithm labels and numerical values are preserved. Display labels follow
the paper, with visible provenance: historical CFR/CFR_IN are not the paper's
CF/CF-IN families. See docs/IMPLEMENTATION_STATUS.md for the metric audit.
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
PAPER_CLASSES = {
    "A-EFR_IN": "InformedActionSequencePredecessors",
    "CFR": "ImmediateExternalSequencePredecessors",
    "CFR_IN": "ImmediateInternalSequencePredecessors",
    "CSPS-EFR": "CausalPartialSequencePredecessors",
    "CFPS-EFR": "CounterfactualPartialSequencePredecessors",
}
HISTORICAL_CLASSES = {
    **PAPER_CLASSES,
    "CFR": "CounterfactualPartialSequenceExInPredecessors",
    "CFR_IN": "TwiceInformedPartialSequencePredecessors",
}
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


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_provenance(path, label_mode="auto", manifest_path=None):
    """Resolve family provenance without changing any raw parser keys.

    Archive hashes establish historical identity, while a launch sidecar records
    the selected family mode for a fresh run. Unrecognized external files remain
    unspecified unless the caller explicitly declares a mode.
    """
    digest = sha256(path)
    manifest_path = manifest_path or ROOT / "data/manifest.json"
    evidence = []
    detected = set()
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("files", []):
            if digest in (entry.get("gzip_sha256"), entry.get("source_sha256")):
                detected.add("historical")
                evidence.append("bundled_archive_sha256")
                break
    uncompressed = path.with_suffix("") if path.suffix == ".gz" else path
    sidecar = uncompressed.with_suffix(".metadata.json")
    if sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        legacy = metadata.get("legacy_algorithm_mapping")
        if isinstance(legacy, bool):
            detected.add("historical" if legacy else "paper")
            evidence.append("launcher_metadata_mode")
    if len(detected) > 1:
        raise ValueError(f"{path.name}: archive hash and launch metadata disagree on algorithm families")
    inferred = next(iter(detected), "unspecified")
    if label_mode != "auto":
        if inferred != "unspecified" and inferred != label_mode:
            raise ValueError(f"{path.name}: --label-mode {label_mode} contradicts {inferred} source provenance")
        inferred = label_mode
        evidence.append("explicit_label_mode")
    classes = PAPER_CLASSES if inferred == "paper" else HISTORICAL_CLASSES if inferred == "historical" else {}
    return {
        "sha256": digest,
        "label_mode": inferred,
        "evidence": evidence or ["unverified_external_input"],
        "algorithm_mapping": [
            {"raw_label": raw, "display_label": LABELS[raw],
             "class": classes.get(raw),
             "paper_family_equivalence": (
                 None if inferred == "unspecified" else
                 not (inferred == "historical" and raw in ("CFR", "CFR_IN"))) }
            for raw in ALGORITHMS
        ],
    }


def figure_provenance(algorithm, contexts):
    modes = {context["label_mode"] for context in contexts}
    if len(modes) != 1:
        if algorithm in ("CFR", "CFR_IN"):
            raise ValueError(
                f"{LABELS[algorithm]}: mixed or unknown family provenance across inputs; "
                "plot historical and paper-family runs separately")
        return "mixed", "Mixed label provenance; see plot_manifest.json"
    mode = next(iter(modes))
    if mode == "historical":
        if algorithm in ("CFR", "CFR_IN"):
            note = f"Historical mapping: raw {algorithm} is not the paper's {LABELS[algorithm]} family"
        else:
            note = f"Historical-family label mapping (raw label: {algorithm}); source values unchanged"
    elif mode == "paper":
        note = f"Paper-family mapping (raw label: {algorithm})"
    else:
        note = f"Unverified label mapping from {algorithm}; family identity unspecified"
    return mode, note


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


def draw_grid(series, games, algorithm, mode, output, formats, connect_stages, contexts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter, ScalarFormatter

    label_mode, provenance_note = figure_provenance(algorithm, contexts)
    display_label = LABELS[algorithm]
    rc = {"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
          "axes.titleweight": "semibold", "xtick.labelsize": 10, "ytick.labelsize": 10,
          "legend.fontsize": 11, "figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
          "axes.edgecolor": "#333333", "axes.linewidth": 0.8, "pdf.fonttype": 42,
          "savefig.dpi": 180, "svg.hashsalt": "pbce-anonymous"}
    with plt.rc_context(rc):
        height = 2.55 * len(games) + 1.8
        fig, axes = plt.subplots(len(games), 3, figsize=(13, height), squeeze=False)
        fig.subplots_adjust(left=0.12, right=0.985, bottom=0.6 / height,
                            top=1 - (1.5 if mode == "fixed" else 1.0) / height,
                            wspace=0.28, hspace=0.36)
        fig.suptitle(f"{display_label} — {mode.capitalize()} perturbation", y=1 - 0.08 / height,
                     fontsize=15, fontweight="semibold")
        fig.text(0.5, 1 - 0.43 / height, provenance_note, ha="center", va="top", fontsize=10)
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
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 1 - 0.73 / height), ncol=6, frameon=True)
        # Raw filenames remain a stable processing interface; only visible
        # labels and the sidecar manifest carry the display/provenance mapping.
        name = f"pefr.{algorithm}.round.gaps.alt.cfrplus" if mode == "fixed" else f"pefr.{algorithm}.adaptive.gaps"
        paths = []
        for extension in formats:
            path = output / mode / f"{name}.{extension}"
            path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "pdf": {"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None},
                "svg": {"Creator": None, "Date": None},
                "png": {"Software": None},
            }[extension]
            fig.savefig(path, bbox_inches="tight", facecolor="white", metadata=metadata)
            paths.append(path.relative_to(output).as_posix())
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
    parser.add_argument("--label-mode", choices=("auto", "historical", "paper"), default="auto",
                        help="Display provenance only; raw labels stay unchanged. auto uses bundled archive hashes or launch metadata, otherwise marks the mapping unverified. Explicit modes declare external family provenance, but cannot override contradictory evidence.")
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
    else:
        fixed_dir, adaptive_dir = ROOT / "data/raw/fixed", ROOT / "data/raw/adaptive"
    fixed_dir = resolve(args.fixed_data) if args.fixed_data else fixed_dir
    adaptive_dir = resolve(args.adaptive_data) if args.adaptive_data else adaptive_dir
    output = resolve(args.output_dir)
    report = {"schema_version": 2, "metric_interpretation": "implementation diagnostics; family provenance does not certify full-continuation metric equivalence; see implementation audit",
              "fixed_normalization": "posr=fus/(t+1); osr=full/(t+1); csr=max_phi_regret",
              "adaptive_normalization": "use raw posr, osr, csr checkpoint columns",
              "label_mode_request": args.label_mode,
              "label_policy": "Raw data labels are unchanged; paper display names are not a claim that historical CFR/CFR_IN implement CF/CF-IN.",
              "max_fixed_points": args.max_points, "fixed_stride": args.fixed_stride, "connect_adaptive_stages": not args.disconnect_stages,
              "inputs": [], "figures": []}
    for mode in ("fixed", "adaptive"):
        if args.mode not in (mode, "both"):
            continue
        series = {}
        contexts = []
        for game in games:
            for parameter in (EPSILONS if mode == "fixed" else ("adaptive",)):
                path = find_input(fixed_dir if mode == "fixed" else adaptive_dir, game, parameter)
                records, metadata = fixed_records(path, args.max_points, args.fixed_stride) if mode == "fixed" else adaptive_records(path)
                context = label_provenance(path, args.label_mode)
                contexts.append(context)
                series[(game, parameter) if mode == "fixed" else game] = records
                report["inputs"].append({"mode": mode, "game": game, "parameter": parameter,
                    "path": path.name, **context, **metadata})
                print(f"Validated {path.name}", flush=True)
        for algorithm in algorithms:
            label_mode, note = figure_provenance(algorithm, contexts)
            figure = {"mode": mode, "raw_label": algorithm, "display_label": LABELS[algorithm],
                      "label_mode": label_mode, "provenance_note": note, "files": []}
            if not args.validate_only:
                figure["files"] = draw_grid(series, games, algorithm, mode, output, args.formats,
                                            not args.disconnect_stages, contexts)
            report["figures"].append(figure)
    output.mkdir(parents=True, exist_ok=True)
    (output / "plot_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        sys.exit(f"PBCE plot error: {error}")
