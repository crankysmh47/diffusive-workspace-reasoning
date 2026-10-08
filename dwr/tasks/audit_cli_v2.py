"""Command-line entry point for the independent v2 audit."""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .audit_v2 import AUDIT_VERSION, GATE_FIELDS, run_audit_v2
from .config_v2 import load_config_v2
from .dataset_v2 import _producer_identity, registered_limits, source_identity
from .manifest_v2 import validate_manifest_v2
from .run_support import (ResourceGuard, atomic_json, remaining_global_seconds,
                          sha256_file, stable_json, strict_json_loads)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independently audit a v2 task generation run.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("feasibility", "full"), required=True)
    parser.add_argument("--data", required=True, type=Path)
    return parser


def _known_artifacts(data_dir: Path) -> tuple[Path, ...]:
    names = ("audit_manifest.json", "audit_report.json", "audit_gate.json",
             "preflight_report.json", "audit_manifest.json.tmp", "audit_report.json.tmp",
             "audit_gate.json.tmp", "preflight_report.json.tmp")
    return tuple(data_dir / name for name in names)


def _exclusive_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(stable_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_terminal_failure(data_dir: Path, primary_error: dict,
                            finalization_errors: list[dict], snapshot: dict | None) -> None:
    path = data_dir / "terminal_failure.json"
    payload = {"failure_version": "dwr-task-terminal-failure-v1", "status": "incomplete",
               "primary_error": primary_error, "finalization_errors": finalization_errors,
               "manifest_snapshot": snapshot, "failure_scope": "audit"}
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(stable_json(payload) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as exc:
        print(f"audit terminal failure sidecar could not be written: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"primary audit failure: {stable_json(primary_error)}", file=sys.stderr)
        for item in finalization_errors:
            print(f"audit finalization failure: {stable_json(item)}", file=sys.stderr)


def _mark_incomplete(data_dir: Path, report: dict, gate: dict | None,
                     manifest: dict, primary_error: dict,
                     finalization_errors: list[dict]) -> None:
    report["status"] = "incomplete"
    report["g2_status"] = "not_assessed"
    report["error"] = primary_error
    report["finalization_errors"] = list(finalization_errors)
    if gate is not None:
        gate["status"] = "incomplete"
        gate["g2_status"] = "not_assessed"
    manifest["status"] = "incomplete"
    manifest["error"] = primary_error
    manifest["finalization_errors"] = list(finalization_errors)
    manifest["finished_utc"] = _utc()
    try:
        atomic_json(data_dir / "audit_report.json", report)
        if gate is not None:
            atomic_json(data_dir / "audit_gate.json", gate)
        preflight = data_dir / "preflight_report.json"
        if preflight.exists():
            preflight.unlink()
        atomic_json(data_dir / "audit_manifest.json", manifest)
    except BaseException as exc:
        finalization_errors.append({"type": type(exc).__name__, "message": str(exc)})
        _write_terminal_failure(data_dir, primary_error, finalization_errors, manifest)


def _finalize_manifest(data_dir: Path, manifest: dict, report: dict, gate: dict | None,
                       guard: ResourceGuard, raw_generation: dict | None,
                       cpu_started: float, primary_error: dict | None,
                       finalization_errors: list[dict], *, charge_guard: bool) -> dict | None:
    generation = raw_generation or {}
    def snapshot() -> None:
        elapsed = guard.elapsed
        audit_cpu = max(0.0, time.process_time() - cpu_started)
        manifest["finished_utc"] = _utc()
        manifest["audit_elapsed_seconds"] = round(elapsed, 6)
        manifest["combined_generation_audit_elapsed_seconds"] = round(
            generation.get("generation_elapsed_seconds", 0.0) + elapsed, 6)
        manifest["audit_cpu_seconds"] = round(audit_cpu, 6)
        manifest["combined_generation_audit_cpu_seconds"] = round(
            generation.get("generation_cpu_seconds", 0.0) + audit_cpu, 6)
        manifest["peak_resident_bytes"] = guard.peak_bytes
        manifest["combined_peak_resident_bytes"] = max(
            [n for n in (guard.peak_bytes, generation.get("peak_resident_bytes")) if type(n) is int] or [None])
        manifest["accounting_end_boundary"] = "post_first_terminal_manifest_write_resource_check"
        manifest["accounting_note"] = "one bounded final metadata flush follows the terminal resource check"
        if type(report.get("resource")) is dict:
            report["resource"].update({
                "audit_elapsed_seconds": manifest["audit_elapsed_seconds"],
                "combined_generation_audit_elapsed_seconds": manifest["combined_generation_audit_elapsed_seconds"],
                "audit_cpu_seconds": manifest["audit_cpu_seconds"],
                "combined_generation_audit_cpu_seconds": manifest["combined_generation_audit_cpu_seconds"],
                "peak_resident_bytes": manifest["peak_resident_bytes"],
                "combined_peak_resident_bytes": manifest["combined_peak_resident_bytes"]})
    snapshot()
    try:
        atomic_json(data_dir / "audit_manifest.json", manifest)
        if charge_guard:
            guard.check()
        snapshot()
        atomic_json(data_dir / "audit_report.json", report)
        atomic_json(data_dir / "audit_manifest.json", manifest)
    except BaseException as exc:
        finalization_errors.append({"type": type(exc).__name__, "message": str(exc)})
        if primary_error is None:
            primary_error = {"type": type(exc).__name__, "message": str(exc), "phase": "audit_finalization"}
        snapshot()
        _mark_incomplete(data_dir, report, gate, manifest, primary_error, finalization_errors)
    return primary_error


def _exit_code(report: dict) -> int:
    return 0 if (report.get("status") == "complete" and
                 report.get("g2_status") in ("PASS_FEASIBILITY", "measured_pass")) else 1


def _preflight(config, source: dict, data_dir: Path, generation_hash: str,
               artifacts: dict, gate_path: Path) -> dict:
    gate_hash = sha256_file(gate_path)
    return {"report_version": "dwr-task-v2-feasibility-preflight-v1",
            "status": "PASS_FEASIBILITY", "mode": "feasibility",
            "config_sha256": config.sha256, "source": source,
            "generation_dir": str(data_dir.resolve()),
            "generation_manifest_sha256": generation_hash,
            "generation_artifacts": artifacts,
            "audit_artifact_path": str(gate_path.resolve()),
            "audit_artifact_sha256": gate_hash, "audit_source": source}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Start the process-wide guard before config, data, source, and manifest I/O.
    command_started = time.monotonic()
    cpu_started = time.process_time()
    guard = ResourceGuard(600.0, 1024)
    args = _parser().parse_args(argv)
    data_dir = args.data.resolve()
    manifest_path = data_dir / "manifest.json"
    raw_generation = None
    generation_hash = None
    remaining = 0.0
    config = None
    source = None
    primary_error = None
    audit_started = False
    gate = None
    collision_refused = False
    finalization_errors: list[dict] = []
    report: dict
    manifest: dict
    started_utc = _utc()
    try:
        if not data_dir.is_dir():
            raise ValueError("generation data directory does not exist")
        collisions = [str(path.name) for path in _known_artifacts(data_dir) if path.exists()]
        if collisions:
            collision_refused = True
            raise FileExistsError("audit evidence already exists; refusing a second audit: " + ", ".join(collisions))
        config = load_config_v2(args.config)
        limits = registered_limits(config.resolved, args.mode)
        source = source_identity()
        if (data_dir / "terminal_failure.json").exists():
            raise ValueError("generation has a terminal_failure.json sidecar; audit refused")
        raw_generation = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest_v2(raw_generation, config_sha256=config.sha256, config=config.resolved,
                             mode=args.mode, limits=limits, expected_producer=_producer_identity())
        generation_hash = sha256_file(manifest_path)
        remaining = remaining_global_seconds(raw_generation, limits.wall_seconds)
        guard.wall_seconds = remaining
        guard.memory_bytes = limits.memory_mib * 1024 * 1024
        producer = raw_generation["producer"]
        audit_producer = {"audit_v2_sha256": sha256_file(Path(__file__).with_name("audit_v2.py")),
                          "audit_cli_v2_sha256": sha256_file(Path(__file__))}
        manifest = {"audit_manifest_version": "dwr-task-v2-audit-run-v1", "status": "running",
                    "created_utc": started_utc, "command": [sys.executable, "-m", "dwr.tasks.audit_cli_v2", *argv],
                    "cwd": str(Path.cwd()), "audit_source": source,
                    "generation_source": raw_generation["source"], "audit_producer": audit_producer,
                    "generation_producer": producer, "config_path": str(config.path),
                    "config_sha256": config.sha256, "resolved_config": config.resolved,
                    "generation_dir": str(data_dir), "generation_manifest_sha256": generation_hash,
                    "generation_artifacts": raw_generation["artifacts"], "mode": args.mode,
                    "os": platform.platform(), "python": sys.version,
                    "cpu_count": os.cpu_count(), "rng": "none; audit is deterministic and uses no RNG",
                    "precision": "not applicable; CPU integer/rational audit only",
                    "caps": {"generation_plus_audit_seconds": limits.wall_seconds,
                             "resident_memory_mib": limits.memory_mib, "cpu_workers": 1,
                             "attempts_per_partition": limits.attempts_per_partition},
                    "generation_elapsed_seconds": raw_generation["generation_elapsed_seconds"],
                    "remaining_seconds_at_audit_start": remaining,
                    "created_monotonic_seconds": command_started, "argv": argv,
                    "audit_cpu_started_seconds": cpu_started,
                    "started_utc": started_utc, "finished_utc": None,
                    "error": None, "finalization_errors": []}
        _exclusive_json(data_dir / "audit_manifest.json", manifest)
        audit_started = True
        if remaining <= 0:
            report = {"status": "not_assessed", "mode": args.mode, "g2_status": "not_assessed",
                      "config_sha256": config.sha256, "generation_manifest_sha256": generation_hash,
                      "generation_source": raw_generation["source"], "audit_source": source,
                      "reason": "global generation-plus-audit budget was exhausted before replay",
                      "generation_complete": raw_generation["status"] == "complete",
                      "replay_pass": False, "integrity_pass": False, "metrics": None}
        else:
            result = run_audit_v2(data_dir, config, source, args.mode, guard=guard)
            report = {"status": "complete", **result,
                      "resource": {"audit_elapsed_seconds": round(guard.elapsed, 6),
                                   "combined_generation_audit_elapsed_seconds": round(raw_generation["generation_elapsed_seconds"] + guard.elapsed, 6),
                                   "audit_cpu_seconds": round(time.process_time() - cpu_started, 6),
                                   "combined_generation_audit_cpu_seconds": round(raw_generation["generation_cpu_seconds"] + time.process_time() - cpu_started, 6),
                                   "remaining_global_seconds": max(0.0, remaining - guard.elapsed),
                                   "peak_resident_bytes": guard.peak_bytes,
                                   "combined_peak_resident_bytes": max(
                                       [n for n in (guard.peak_bytes, raw_generation["peak_resident_bytes"])
                                        if type(n) is int] or [None])}}
        if report.get("resource") is None:
            report["resource"] = {"audit_elapsed_seconds": round(guard.elapsed, 6),
                "combined_generation_audit_elapsed_seconds": round(raw_generation["generation_elapsed_seconds"] + guard.elapsed, 6),
                "audit_cpu_seconds": round(time.process_time() - cpu_started, 6),
                "combined_generation_audit_cpu_seconds": round(raw_generation["generation_cpu_seconds"] + time.process_time() - cpu_started, 6),
                "remaining_global_seconds": max(0.0, remaining - guard.elapsed),
                "peak_resident_bytes": guard.peak_bytes,
                "combined_peak_resident_bytes": max(
                    [n for n in (guard.peak_bytes, raw_generation["peak_resident_bytes"])
                     if type(n) is int] or [None])}
        gate = {"audit_version": AUDIT_VERSION, "status": "complete" if report["status"] == "complete" else "incomplete",
                "mode": args.mode, "g2_status": report["g2_status"],
                "config_sha256": config.sha256, "source": source,
                "generation_manifest_sha256": generation_hash,
                "generation_artifacts": raw_generation["artifacts"]}
        if set(gate) != GATE_FIELDS:
            raise RuntimeError("internal audit gate shape error")
        if remaining > 0:
            guard.check()
        atomic_json(data_dir / "audit_report.json", report)
        atomic_json(data_dir / "audit_gate.json", gate)
        if (args.mode == "feasibility" and gate["status"] == "complete" and
                gate["g2_status"] == "PASS_FEASIBILITY" and source == raw_generation["source"] and
                source.get("dirty") is False):
            wrapper = _preflight(config, source, data_dir, generation_hash,
                                 raw_generation["artifacts"], data_dir / "audit_gate.json")
            atomic_json(data_dir / "preflight_report.json", wrapper)
        manifest["status"] = "complete" if report["status"] == "complete" else "incomplete"
        if remaining > 0:
            guard.check()
    except BaseException as exc:
        primary_error = {"type": type(exc).__name__, "message": str(exc), "phase": "audit"}
        report = {"status": "incomplete", "mode": getattr(locals().get("args"), "mode", None),
                  "g2_status": "not_assessed", "error": primary_error,
                  "generation_manifest_sha256": generation_hash,
                  "config_sha256": None if config is None else config.sha256}
        if audit_started and source is not None and raw_generation is not None and generation_hash is not None and config is not None:
            gate = {"audit_version": AUDIT_VERSION, "status": "incomplete",
                    "mode": args.mode, "g2_status": "not_assessed", "config_sha256": config.sha256,
                    "source": source, "generation_manifest_sha256": generation_hash,
                    "generation_artifacts": raw_generation.get("artifacts", {})}
            try:
                atomic_json(data_dir / "audit_report.json", report)
                atomic_json(data_dir / "audit_gate.json", gate)
                preflight = data_dir / "preflight_report.json"
                if preflight.exists(): preflight.unlink()
            except BaseException as final_exc:
                finalization_errors.append({"type": type(final_exc).__name__, "message": str(final_exc)})
                manifest.setdefault("finalization_errors", []).extend(finalization_errors)
                _write_terminal_failure(data_dir, primary_error, finalization_errors, manifest)
        elif "manifest" in locals() and not collision_refused:
            _write_terminal_failure(data_dir, primary_error, finalization_errors, manifest)
        if "manifest" in locals():
            manifest["status"] = "incomplete"
            manifest["error"] = primary_error
    finally:
        if "manifest" in locals() and audit_started:
            primary_error = _finalize_manifest(data_dir, manifest, report, gate, guard,
                raw_generation, cpu_started, primary_error, finalization_errors,
                charge_guard=remaining > 0)
    print(stable_json({"status": report.get("status"), "g2_status": report.get("g2_status"),
                       "report": str(data_dir / "audit_report.json")}))
    return _exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
