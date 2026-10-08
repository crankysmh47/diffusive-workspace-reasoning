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
             "preflight_report.json", "preflight_report.invalidated.json",
             "audit_manifest.json.tmp", "audit_report.json.tmp",
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
                     finalization_errors: list[dict], *, write_report: bool = True) -> None:
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
        if write_report:
            atomic_json(data_dir / "audit_report.json", report)
        if gate is not None:
            atomic_json(data_dir / "audit_gate.json", gate)
        preflight = data_dir / "preflight_report.json"
        if preflight.exists():
            invalidated = data_dir / "preflight_report.invalidated.json"
            if invalidated.exists():
                raise FileExistsError("invalidated preflight evidence already exists")
            preflight.replace(invalidated)
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
    detailed_report_written = False
    try:
        # Detailed per-example output can be large; its serialization and I/O remain charged.
        atomic_json(data_dir / "audit_report.json", report)
        detailed_report_written = True
        if gate is not None:
            atomic_json(data_dir / "audit_gate.json", gate)
        snapshot()
        atomic_json(data_dir / "audit_manifest.json", manifest)
        if charge_guard:
            guard.check()
        snapshot()
        # Exactly one bounded metadata flush follows the charged terminal check.
        atomic_json(data_dir / "audit_manifest.json", manifest)
    except BaseException as exc:
        finalization_errors.append({"type": type(exc).__name__, "message": str(exc)})
        if primary_error is None:
            primary_error = {"type": type(exc).__name__, "message": str(exc), "phase": "audit_finalization"}
        snapshot()
        _mark_incomplete(data_dir, report, gate, manifest, primary_error, finalization_errors,
                         write_report=not detailed_report_written)
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
    command_started = time.monotonic()
    cpu_started = time.process_time()
    guard = ResourceGuard(600.0, 1024)
    args = _parser().parse_args(argv)
    data_dir = args.data.resolve()
    manifest_path = data_dir / "manifest.json"
    raw_generation = generation_hash = config = source = None
    remaining = 0.0
    primary_error = None
    audit_started = False
    attempt_recorded = False
    gate = None
    finalization_errors: list[dict] = []
    report: dict = {"status": "not_assessed", "mode": args.mode, "g2_status": "not_assessed"}
    started_utc = _utc()
    manifest = {"audit_manifest_version": "dwr-task-v2-audit-run-v1", "status": "running",
                "created_utc": started_utc,
                "command": [sys.executable, "-m", "dwr.tasks.audit_cli_v2", *argv],
                "argv": argv, "cwd": str(Path.cwd()), "generation_dir": str(data_dir),
                "mode": args.mode, "config_path": str(args.config.resolve()),
                "os": platform.platform(), "python": sys.version, "cpu_count": os.cpu_count(),
                "rng": "none; audit is deterministic and uses no RNG",
                "precision": "not applicable; CPU integer/rational audit only",
                "created_monotonic_seconds": command_started, "audit_cpu_started_seconds": cpu_started,
                "started_utc": started_utc, "finished_utc": None, "error": None,
                "finalization_errors": []}
    try:
        if not data_dir.is_dir():
            raise FileNotFoundError("generation data directory does not exist; no report was written")
        collisions = [str(path.name) for path in _known_artifacts(data_dir) if path.exists()]
        if collisions:
            raise FileExistsError("audit evidence already exists; refusing a second audit: " + ", ".join(collisions))
        # Claim this audit attempt before loading any external identity or metadata.
        _exclusive_json(data_dir / "audit_manifest.json", manifest)
        attempt_recorded = True
        config = load_config_v2(args.config)
        manifest.update({"config_path": str(config.path), "config_sha256": config.sha256,
                         "resolved_config": config.resolved})
        limits = registered_limits(config.resolved, args.mode)
        source = source_identity()
        manifest["audit_source"] = source
        if (data_dir / "terminal_failure.json").exists():
            sidecar_path = data_dir / "terminal_failure.json"
            try:
                sidecar = strict_json_loads(sidecar_path.read_text(encoding="utf-8"))
            except BaseException as sidecar_exc:
                sidecar = {"read_error": {"type": type(sidecar_exc).__name__, "message": str(sidecar_exc)}}
            manifest["generation_terminal_failure"] = {
                "sha256": sha256_file(sidecar_path), "content": sidecar}
            raise ValueError("generation has a terminal_failure.json sidecar; audit refused")
        raw_generation = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
        generation_hash = sha256_file(manifest_path)
        if type(raw_generation) is dict:
            manifest["generation_manifest_sha256"] = generation_hash
            manifest["generation_source"] = raw_generation.get("source")
            manifest["generation_artifacts"] = raw_generation.get("artifacts")
        validate_manifest_v2(raw_generation, config_sha256=config.sha256, config=config.resolved,
                             mode=args.mode, limits=limits, expected_producer=_producer_identity())
        remaining = remaining_global_seconds(raw_generation, limits.wall_seconds)
        guard.wall_seconds = remaining
        guard.memory_bytes = limits.memory_mib * 1024 * 1024
        producer = raw_generation["producer"]
        audit_producer = {"audit_v2_sha256": sha256_file(Path(__file__).with_name("audit_v2.py")),
                          "audit_cli_v2_sha256": sha256_file(Path(__file__))}
        manifest.update({"audit_producer": audit_producer, "generation_producer": producer,
                    "caps": {"generation_plus_audit_seconds": limits.wall_seconds,
                             "resident_memory_mib": limits.memory_mib, "cpu_workers": 1,
                             "attempts_per_partition": limits.attempts_per_partition},
                    "generation_elapsed_seconds": raw_generation["generation_elapsed_seconds"],
                    "remaining_seconds_at_audit_start": remaining})
        atomic_json(data_dir / "audit_manifest.json", manifest)
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
        manifest["status"] = "complete" if report["status"] == "complete" else "incomplete"
        # The wrapper hashes this provisional gate; both remain charged and are
        # invalidated if report finalization or the terminal guard later fails.
        atomic_json(data_dir / "audit_gate.json", gate)
        if (args.mode == "feasibility" and gate["status"] == "complete" and
                gate["g2_status"] == "PASS_FEASIBILITY" and source == raw_generation["source"] and
                source.get("dirty") is False):
            wrapper = _preflight(config, source, data_dir, generation_hash,
                                 raw_generation["artifacts"], data_dir / "audit_gate.json")
            atomic_json(data_dir / "preflight_report.json", wrapper)
    except BaseException as exc:
        primary_error = {"type": type(exc).__name__, "message": str(exc), "phase": "audit"}
        report = {"status": "not_assessed" if not audit_started else "incomplete",
                  "mode": args.mode, "g2_status": "not_assessed", "error": primary_error,
                  "command": manifest["command"], "argv": argv,
                  "audit_source": source,
                  "generation_manifest_sha256": generation_hash,
                  "config_sha256": None if config is None else config.sha256,
                  "generation_terminal_failure": manifest.get("generation_terminal_failure")}
        if audit_started and source is not None and raw_generation is not None and generation_hash is not None and config is not None:
            gate = {"audit_version": AUDIT_VERSION, "status": "incomplete",
                    "mode": args.mode, "g2_status": "not_assessed", "config_sha256": config.sha256,
                    "source": source, "generation_manifest_sha256": generation_hash,
                    "generation_artifacts": raw_generation.get("artifacts", {})}
        manifest["status"] = "incomplete"
        manifest["error"] = primary_error
        if (not attempt_recorded and data_dir.is_dir() and
                not isinstance(exc, FileExistsError)):
            _write_terminal_failure(data_dir, primary_error, finalization_errors, manifest)
    finally:
        if audit_started:
            primary_error = _finalize_manifest(data_dir, manifest, report, gate, guard,
                raw_generation, cpu_started, primary_error, finalization_errors,
                charge_guard=remaining > 0)
        elif attempt_recorded:
            manifest["finished_utc"] = _utc()
            report["status"] = "not_assessed"
            try:
                atomic_json(data_dir / "audit_report.json", report)
                atomic_json(data_dir / "audit_manifest.json", manifest)
            except BaseException as persist_exc:
                finalization_errors.append({"type": type(persist_exc).__name__, "message": str(persist_exc)})
                _write_terminal_failure(data_dir, primary_error or {"type": "PersistenceError", "message": "setup evidence persistence failed"}, finalization_errors, manifest)
    if primary_error is not None:
        print(f"audit failed: {stable_json(primary_error)}", file=sys.stderr)
        for item in finalization_errors:
            print(f"audit finalization failed: {stable_json(item)}", file=sys.stderr)
    output = {"status": report.get("status"), "g2_status": report.get("g2_status")}
    report_path = data_dir / "audit_report.json"
    if report_path.exists():
        output["report"] = str(report_path)
    print(stable_json(output))
    return _exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
