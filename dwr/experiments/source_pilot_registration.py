"""Closed, metadata-first registration and source readers for the validation pilot."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping

from dwr.tasks.run_support import stable_json


CONFIG_PATH = "configs/experiments/source_validation_pilot_v1.json"
CONFIG_SHA256 = "125ebe2ee9672b58f07f6511f29be28b5eef04382ee8a71cc7a9951e30503a31"
STARTUP_RETRY_CONFIG_PATH = "configs/experiments/source_validation_pilot_startup_retry_v2.json"
STARTUP_RETRY_CONFIG_SHA256 = "e91739eb52cf3716c15c85710c563de2572f14ca1cc10838419c3e6e529b7e94"
REGISTRATION_REPAIR_CONFIG_PATH = "configs/experiments/source_validation_pilot_registration_repair_v3.json"
REGISTRATION_REPAIR_CONFIG_SHA256 = "7e535a95dfb21bd7f45286f3cff586adbbd5c78a0a89c348a5f5160a2976070b"
COHORT_PATH = "configs/splits/source_validation_pilot_cohort_v1.json"
COHORT_SHA256 = "615e357ceff2a7b0a32e9509d68d1be2e881a233f6387088746b9f58bf32afe7"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_STRATA = ("A2", "A3", "B2", "C2", "D2")
_PROFILE = {"eq", "neq", "add_mod"}

_TOP_KEYS = {"config_version", "source", "data", "dependencies", "model", "host", "arms",
    "decoding", "seeds", "training", "evaluation", "diagnostics", "limits", "output_dir",
    "study_type", "result_policy"}
_SECTION_KEYS = {
    "source": {"ancestry_minimum", "first_launch_identity", "source_policy"},
    "data": {"source_split", "source_train_records", "source_train_order", "source_validation_split",
        "source_validation_records", "validation_examples", "validation_selection",
        "validation_selection_prefix", "validation_strata_order", "validation_semantics",
        "source_export_dir", "source_export_manifest_sha256", "source_export_data_sha256",
        "source_token_audit_dir", "source_token_audit_manifest_sha256", "token_lengths_sha256",
        "cohort_path", "cohort_sha256", "frozen_reference_overlap_count", "protected_semantics"},
    "dependencies": {"feasibility_config_path", "feasibility_config_sha256", "feasibility_run_dir",
        "feasibility_manifest_sha256", "feasibility_outcome_sha256", "feasibility_module_sha256",
        "reference_config_path", "reference_config_sha256", "reference_cohort_path",
        "reference_cohort_sha256", "reference_run_dir", "reference_manifest_sha256",
        "reference_summary_sha256", "reference_module_sha256", "source_overfit_config_path",
        "source_overfit_config_sha256", "source_overfit_result_path", "source_overfit_result_sha256",
        "source_overfit_terminal_job_manifest_sha256", "source_overfit_terminal_worker_manifest_sha256",
        "source_overfit_independent_audit_path", "source_overfit_independent_audit_sha256"},
    "model": {"repo_id", "revision", "weight_sha256", "artifact_manifest_path",
        "artifact_manifest_sha256", "artifact_inventory_sha256", "model_config_sha256",
        "feasibility_outcome_sha256"},
    "host": {"layout_policy", "slot_identity", "num_slots", "latent_dim", "max_steps", "prompt_layer",
        "adapter_layers", "reader_heads", "workspace_heads", "context_dim", "gate_init",
        "original_dtype", "new_dtype", "autocast", "scaler", "gradient_checkpointing", "kv_cache",
        "prompt_cache", "offload", "quantization", "sdpa", "numerics"},
    "decoding": {"batch_size", "embedding_vocab_size", "eos_token_id", "pad_token_id",
        "max_prompt_tokens", "max_answer_tokens_including_eos", "max_new_tokens_including_eos",
        "max_total_tokens", "prompt_serialization", "tokenizer_add_special_tokens", "method",
        "truncate_or_skip", "kv_cache", "prompt_cache"},
    "seeds": {"python_global", "numpy_global", "torch_cpu_global", "torch_cuda_global",
        "workspace_initialization", "reader_initialization", "source_sampler", "training_noise",
        "evaluation_noise_base"},
    "training": {"batch_size", "optimizer_steps", "gradient_accumulation", "gradient_clip_norm",
        "optimizer", "schedule", "noise_policy", "sampling_policy", "initialization_policy",
        "checkpoint_rule", "diagnose_state_gradient_steps"},
    "evaluation": {"steps", "source", "checkpoint_rule", "draws_per_example", "noise_seed_formula",
        "recreate_generator_for_ce_and_generation", "state_preservation", "CE", "vsr",
        "learning_gate", "workspace_screen", "terminal_results"},
    "diagnostics": {"steps", "noise_rule", "shuffle", "donor_map_path", "donor_map_key",
        "learned_workspace_bank_shape", "learned_workspace_bank_dtype", "learned_workspace_bank_mib_approx",
        "labels_used_for_bank_or_donors", "diagnostic_metric", "reader_gate_metrics"},
    "limits": {"scientific_invocations", "training_seeds_per_arm", "tuning_trials", "scientific_retries",
        "arms_sequential", "arm_wall_seconds", "arm_work_seconds", "arm_finalization_reserve_seconds",
        "arm_external_watchdog_seconds", "watchdog_receipt_hash_seconds", "total_wall_seconds",
        "maximum_interarm_overhead_seconds", "interarm_overhead_inside_total_wall_cap",
        "effective_arm_deadline", "matrix_external_watchdog_seconds", "maximum_operational_recoveries_per_arm",
        "maximum_workers_per_arm", "maximum_physical_updates_per_arm", "maximum_peak_allocated_mib",
        "maximum_peak_reserved_mib", "minimum_initial_free_mib", "resident_memory_mib",
        "gpu_hours_wall_cap", "per_arm_physical_predictions", "per_arm_greedy_decoder_calls",
        "per_arm_teacher_forced_ce_answer_forwards", "combined_evaluation_answer_forward_caps",
        "count_greedy_answer_decoder_forwards_only", "count_started_ce_before_model_call",
        "track_prompt_encodes_workspace_updates_training_answer_forwards_separately",
        "include_all_computation_in_wall_and_memory", "decoder_calls_per_prediction_bound"},
    "result_policy": {"failed_arm_stops_matrix", "operational_fault_separate_from_numerical_divergence",
        "partial_learning_outcome_retained_if_later_diagnostics_interrupt", "no_automatic_transfer_training"},
}
_NESTED_KEYS = {
    "host.sdpa": {"implementation", "cudnn", "flash", "math", "memory_efficient"},
    "host.numerics": {"cudnn_benchmark", "default_float_dtype", "deterministic_algorithms",
        "float32_matmul_precision", "tf32_cudnn", "tf32_matmul"},
    "training.optimizer": {"type", "learning_rate", "betas", "epsilon", "weight_decay", "amsgrad",
        "capturable", "differentiable", "foreach", "fused", "maximize"},
    "training.schedule": {"type", "function", "warmup_steps", "first_update_factor", "step_order"},
    "evaluation.learning_gate": {"K4_final_vsr_min", "K4_final_minus_initial_vsr_min",
        "K4_final_valid_each_represented_primitive_min", "K4_final_ce_max_ratio"},
    "evaluation.workspace_screen": {"K4_minus_trained_K0_final_vsr_min",
        "K4_unablated_minus_shuffled_final_vsr_min"},
}
_EXPECTED_FIXED = {
    "config_version": "dwr-source-validation-pilot-v1",
    "source.ancestry_minimum": "d4d3a6664f245869604a01be91dfdd67175417d1",
    "data.source_export_manifest_sha256": "7db0b61ba40bea60bf17bc5faf615bb2c8ae21320739c3bea589e4d62aea0069",
    "data.source_export_data_sha256": "73502736c17b9867f723d22710e4a643a54c7ca8543fb47d122ef7d3d6e7a0fc",
    "data.source_token_audit_manifest_sha256": "aec131295af1d015d9ef7619ad35d0ad03792a7f147cdf5164bc4c8e4e800826",
    "data.token_lengths_sha256": "b26a54dc18099601d146c1c2a53609a72b848b282a5761a063972e72851c30de",
    "data.cohort_sha256": COHORT_SHA256,
    "dependencies.feasibility_config_sha256": "840615e265a427946627f3320f667154e17bc56c53c15b6f90a453c10693144f",
    "dependencies.feasibility_manifest_sha256": "1bf5cb31b37c3ea3c4f72b7fa4270b1a4657f2cb8680d8ffd7edf864747b8864",
    "dependencies.feasibility_outcome_sha256": "013f015c73b294e98f1bc4ce02d65a08c727ffd9195a699b18309b59bdabaf6a",
    "dependencies.feasibility_module_sha256": "5c8c8f36f198f08ab906be86e88c2dbac04471f7e6f5585b93d1fb402a3c5139",
    "dependencies.reference_config_sha256": "e3a67600b9bada872bd4523f319378a2088e25824df82d90c44d2fb00b42144f",
    "dependencies.reference_cohort_sha256": "bca9505b817a9b75b9d7aba33710ddc17b47596bdb9a9772924348d15b4b64bd",
    "dependencies.reference_manifest_sha256": "4cd5b8f754ee9b7861a536a6a819a36b9ea64771ae1d244a74774aebba644667",
    "dependencies.reference_summary_sha256": "9b5391e88d3f7f00dded943ba82195545a1ee51a1918902f3a116961eca6770a",
    "dependencies.reference_module_sha256": "729431dd17b9e7b4c612d314f0f9dea1a266945a04f68818ce194ec181f17cad",
    "dependencies.source_overfit_config_sha256": "e6c3de7e59e58c494b89052086c0840dcd3421d17396fd5a55cac40611049594",
    "dependencies.source_overfit_result_sha256": "494fb4e65a3da0350fa35d726aed4aacac4860fa6a3e52b7be89439f10e8bcfc",
    "dependencies.source_overfit_terminal_job_manifest_sha256": "2411b8bc9798c75cd7fb851963b6f0b5f95024a88d13389bdc0ad9c29491b7d0",
    "dependencies.source_overfit_terminal_worker_manifest_sha256": "28d9850eb843d3df018f5ee934be6063f12c3902d029694e0e934aa64088c90e",
    "dependencies.source_overfit_independent_audit_sha256": "5f735f94bc626203d4f162fd260c85f4a61d191e29a69542198efa53f538ff72",
    "model.repo_id": "Qwen/Qwen2.5-1.5B",
    "model.revision": "8faed761d45a263340a0528343f099c05c9a4323",
    "model.weight_sha256": "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796",
    "model.artifact_manifest_sha256": "166492a8c0ae52d5b0a766fa281528a2348eca43f2b128f86da70936a4289606",
    "model.artifact_inventory_sha256": "4d5c001069a7c1e225b72a5723fc4a23e6ab316ec940454bbd42b02440ad2dea",
    "model.model_config_sha256": "0e8c8aa86468aba09c9d32157ff4bc2301c7e6c50e4398960425b2ea71e66f77",
    "host.num_slots": 64, "host.latent_dim": 768, "host.max_steps": 4,
    "training.optimizer_steps": 512,
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_links(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for item in reversed((absolute, *absolute.parents)):
        if not item.exists() and not item.is_symlink():
            continue
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
            raise ValueError("pilot paths cannot contain symlinks or reparse points")


def _read(path: Path, label: str) -> bytes:
    _reject_links(path)
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.read_bytes()


def _json(raw: bytes, label: str, *, allow_null: bool = False) -> Any:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=_unique,
                           parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be strict UTF-8 JSON without duplicate/nonfinite values") from exc
    if not allow_null:
        _no_null(value, label)
    return value


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _no_null(value: Any, where: str) -> None:
    if value is None:
        raise ValueError(f"null is forbidden: {where}")
    if isinstance(value, dict):
        for key, child in value.items():
            _no_null(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _no_null(child, f"{where}[{index}]")


def _keys(value: Any, expected: set[str], label: str) -> None:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} has unknown or missing keys")


def _field(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        current = current[part]
    return current


def _validate_config(config: Any) -> dict[str, Any]:
    _keys(config, _TOP_KEYS, "pilot config")
    for section, keys in _SECTION_KEYS.items():
        _keys(config[section], keys, f"pilot config {section}")
    for section, keys in _NESTED_KEYS.items():
        parent_name, child_name = section.split(".")
        _keys(config[parent_name][child_name], keys, f"pilot config {section}")
    arms = config["arms"]
    if type(arms) is not list or len(arms) != 3:
        raise ValueError("pilot config must contain the three registered arms")
    if any(type(arm) is not dict or set(arm) != {"arm_id", "method", "steps"} for arm in arms):
        raise ValueError("pilot arm entries have unknown or missing keys")
    if [(x["arm_id"], x["method"], x["steps"]) for x in arms] != [
            ("K0", "noise_memory", 0), ("K1", "EX", 1), ("K4", "EX", 4)]:
        raise ValueError("pilot arms differ from the registered K0/K1/K4 ordering")
    if config["host"]["sdpa"].keys() != _NESTED_KEYS["host.sdpa"]:
        raise ValueError("host SDPA fields are malformed")
    for dotted, expected in _EXPECTED_FIXED.items():
        if _field(config, dotted) != expected:
            raise ValueError(f"pilot pin or fixed setting differs: {dotted}")
    data = config["data"]
    fixed_data = {"source_split": "source_train", "source_train_records": 2436,
        "source_train_order": "lexicographic_example_id", "source_validation_split": "source_validation",
        "source_validation_records": 265, "validation_examples": 45,
        "validation_selection": "9_minimum_sha256_prefix_plus_example_id_per_pooled_fine_stratum_no_refill",
        "validation_selection_prefix": "dwr-source-pilot-validation-v1:",
        "validation_strata_order": list(_STRATA), "validation_semantics": True,
        "protected_semantics": False, "frozen_reference_overlap_count": 1}
    if any(data.get(key) != value for key, value in fixed_data.items()):
        raise ValueError("pilot data scope or selection policy differs from registration")
    if config["training"]["batch_size"] != 1 or config["training"]["gradient_accumulation"] != 1:
        raise ValueError("pilot must use batch size one without accumulation")
    if config["evaluation"]["steps"] != [0, 512] or config["evaluation"]["draws_per_example"] != 1:
        raise ValueError("pilot official evaluation steps/draw count differ")
    if config["diagnostics"]["learned_workspace_bank_shape"] != [45, 64, 768] or \
            config["diagnostics"]["learned_workspace_bank_dtype"] != "float32":
        raise ValueError("registered final workspace bank shape/dtype differs")
    if config["output_dir"] != "runs/source-validation-pilot-20261017-v1":
        raise ValueError("pilot output path differs from registration")
    limits = config["limits"]
    if limits["maximum_interarm_overhead_seconds"] != 600 or \
            limits["interarm_overhead_inside_total_wall_cap"] is not True or \
            limits["effective_arm_deadline"] != "min(arm_start_plus_5400_seconds,matrix_start_plus_16200_seconds)" or \
            limits["matrix_external_watchdog_seconds"] != 16260:
        raise ValueError("pilot matrix wall-cap/deadline rule differs from registration")
    for dotted, value in _EXPECTED_FIXED.items():
        if dotted.endswith("sha256") and (not isinstance(value, str) or _HASH.fullmatch(value) is None):
            raise ValueError(f"registered digest is malformed: {dotted}")
    return config


def _safe_rel(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise ValueError("registered paths must be forward-slash relative paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("registered path is unsafe")
    return path


def _resolve(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    target = root.joinpath(*_safe_rel(relative).parts)
    _reject_links(target)
    resolved_root = root.resolve(strict=True)
    resolved = target.resolve(strict=must_exist)
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError("registered path escapes repository root")
    if must_exist and not resolved.is_file() and not resolved.is_dir():
        raise ValueError("registered path does not exist")
    return resolved


def _manifest(root: Path, run_dir: str, digest: str, label: str) -> dict[str, Any]:
    run = _resolve(root, run_dir)
    manifest_raw = _read(run / "manifest.json", f"{label} terminal manifest")
    if _sha(manifest_raw) != digest:
        raise ValueError(f"{label} manifest pin mismatch")
    manifest = _json(manifest_raw, f"{label} terminal manifest", allow_null=True)
    if manifest.get("status") != "completed":
        raise ValueError(f"{label} is not a completed run")
    return manifest


def _validate_feasibility_outcome(manifest: Mapping[str, Any], outcome_raw: bytes,
                                  expected_sha256: str) -> dict[str, Any]:
    """Verify both the manifest link and the actual pinned feasibility outcome bytes."""
    if not isinstance(outcome_raw, bytes) or _sha(outcome_raw) != expected_sha256:
        raise ValueError("feasibility outcome bytes differ from the registered pin")
    artifacts = manifest.get("artifacts")
    linked = [item for item in artifacts if type(item) is dict and item.get("path") == "outcome.json"] \
        if type(artifacts) is list else []
    if len(linked) != 1 or linked[0].get("sha256") != expected_sha256:
        raise ValueError("feasibility outcome artifact link differs from its pin")
    outcome = _json(outcome_raw, "feasibility outcome", allow_null=True)
    if type(outcome) is not dict or outcome.get("status") != "PASS_FEASIBILITY":
        raise ValueError("pinned model feasibility proof is not PASS")
    return outcome


def _read_token_metadata(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    token_path = _resolve(root,
        f"{config['data']['source_token_audit_dir']}/token_lengths.jsonl")
    raw = _read(token_path, "token-length metadata")
    if _sha(raw) != config["data"]["token_lengths_sha256"]:
        raise ValueError("token-length metadata bytes differ from registration")
    fields = {"answer_tokens_including_eos", "example_id", "fine_stratum", "profile",
              "prompt_tokens", "split", "total_tokens", "within_caps"}
    rows = []
    seen: set[str] = set()
    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.endswith(b"\n"):
            raise ValueError(f"token metadata line {number} is unterminated")
        row = _json(line[:-1], f"token metadata line {number}")
        _keys(row, fields, f"token metadata line {number}")
        eid = row["example_id"]
        if not isinstance(eid, str) or _HASH.fullmatch(eid) is None or eid in seen:
            raise ValueError("token metadata IDs must be unique lowercase SHA-256 strings")
        if row["split"] not in {"source_train", "source_validation"} or row["profile"] not in _PROFILE or \
                row["fine_stratum"] not in _STRATA:
            raise ValueError("token metadata split/profile/stratum is invalid")
        for key in ("answer_tokens_including_eos", "prompt_tokens", "total_tokens"):
            if type(row[key]) is not int or row[key] <= 0:
                raise ValueError("token metadata lengths must be positive non-boolean integers")
        if (row["answer_tokens_including_eos"] > config["decoding"]["max_answer_tokens_including_eos"] or
                row["prompt_tokens"] > config["decoding"]["max_prompt_tokens"] or
                row["total_tokens"] != row["answer_tokens_including_eos"] + row["prompt_tokens"] or
                row["total_tokens"] > config["decoding"]["max_total_tokens"] or row["within_caps"] is not True):
            raise ValueError("token metadata does not satisfy the registered sequence caps")
        if line != (stable_json(row) + "\n").encode("utf-8"):
            raise ValueError(f"token metadata line {number} is not canonical JSONL")
        seen.add(eid)
        rows.append(row)
    expected = {"source_train": 2436, "source_validation": 265}
    if len(rows) != sum(expected.values()) or {split: sum(x["split"] == split for x in rows)
            for split in expected} != expected:
        raise ValueError("token metadata counts differ from the pinned source populations")
    return tuple(rows)


def _validation_selection(token_rows: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped = {fine: [] for fine in _STRATA}
    for row in token_rows:
        if row.get("split") == "source_validation":
            grouped[row["fine_stratum"]].append(row)
    selected = []
    for fine in _STRATA:
        if len(grouped[fine]) < 9:
            raise ValueError(f"validation fine stratum {fine} has fewer than nine metadata rows")
        ranked = sorted(grouped[fine], key=lambda row: (
            hashlib.sha256(("dwr-source-pilot-validation-v1:" + row["example_id"]).encode("utf-8")).hexdigest(),
            row["example_id"]))
        selected.extend(ranked[:9])
    return tuple(selected)


def _validate_cohort(cohort: Any, token_rows: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> dict[str, Any]:
    _keys(cohort, {"frozen_reference_overlap_ids", "metadata_only_selection", "schema_version",
        "selected", "selection", "shuffle_donor_map", "source_train_count",
        "source_validation_count", "study", "token_lengths_path", "token_lengths_sha256",
        "validation_primitive_fine_inventory"}, "pilot cohort")
    if cohort["schema_version"] != "dwr-source-pilot-cohort-v1" or cohort["metadata_only_selection"] is not True:
        raise ValueError("pilot cohort schema/metadata-only guardrail is invalid")
    if (cohort["study"] != "dwr-source-validation-pilot-v1" or cohort["source_train_count"] != 2436 or
            cohort["source_validation_count"] != 265 or
            cohort["token_lengths_path"] != "runs/source-token-cap-audit-20261015-v1/token_lengths.jsonl" or
            cohort["token_lengths_sha256"] != "b26a54dc18099601d146c1c2a53609a72b848b282a5761a063972e72851c30de"):
        raise ValueError("pilot cohort metadata-only input guardrail differs from registration")
    expected_inventory = [{"count": sum(row["split"] == "source_validation" and
            row["profile"] == profile and row["fine_stratum"] == fine for row in token_rows),
        "fine_stratum": fine, "profile": profile}
        for profile in ("eq", "neq", "add_mod") for fine in _STRATA]
    if cohort["validation_primitive_fine_inventory"] != expected_inventory:
        raise ValueError("validation primitive/fine metadata inventory changed")
    selected = cohort["selected"]
    expected = _validation_selection(token_rows)
    if type(selected) is not list or selected != list(expected) or len(selected) != 45:
        raise ValueError("pilot cohort differs from the frozen metadata-hash selection")
    if cohort["selection"] != {"hash": "SHA256 UTF8 prefix+example_id",
            "ordering": "fine A2 A3 B2 C2 D2 then digest then example_id",
            "per_pooled_fine_stratum": 9, "prefix": "dwr-source-pilot-validation-v1:", "refill": False}:
        raise ValueError("pilot cohort selection recipe is malformed")
    ids = [row["example_id"] for row in selected]
    if len(set(ids)) != 45 or [row["fine_stratum"] for row in selected] != [fine for fine in _STRATA for _ in range(9)]:
        raise ValueError("pilot cohort needs exactly nine ordered IDs from every pooled fine stratum")
    if type(cohort["frozen_reference_overlap_ids"]) is not list or \
            len(cohort["frozen_reference_overlap_ids"]) != 1 or \
            not set(cohort["frozen_reference_overlap_ids"]) <= set(ids):
        raise ValueError("pilot cohort overlap record is malformed")
    sorted_ids = sorted(ids)
    expected_donors = {eid: sorted_ids[(i + 1) % len(sorted_ids)] for i, eid in enumerate(sorted_ids)}
    if cohort["shuffle_donor_map"] != expected_donors:
        raise ValueError("pilot cohort shuffle map is not the frozen cyclic ID mapping")
    return cohort


def _select_pilot_rows(rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
                       selected: tuple[dict[str, Any], ...] | list[dict[str, Any]], *,
                       split: str) -> tuple[dict[str, Any], ...]:
    if split not in {"source_train", "source_validation"} or not isinstance(rows, (tuple, list)):
        raise ValueError("source rows and split are invalid")
    if not isinstance(selected, (tuple, list)) or not selected:
        raise ValueError("registered selection must be nonempty")
    expected = {item.get("example_id"): item for item in selected}
    if len(expected) != len(selected) or any(not isinstance(key, str) or _HASH.fullmatch(key) is None
                                               for key in expected):
        raise ValueError("registered source IDs must be unique SHA-256 strings")
    found: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for row in rows:
        if type(row) is not dict or row.get("split") != split:
            raise ValueError("source reader returned a row from an unexpected split")
        eid = row.get("example_id")
        if not isinstance(eid, str) or _HASH.fullmatch(eid) is None or eid in seen:
            raise ValueError("source reader returned malformed or duplicate IDs")
        seen.add(eid)
        if eid in expected:
            for key in ("split", "profile", "fine_stratum"):
                if row.get(key) != expected[eid].get(key):
                    raise ValueError(f"source reader metadata differs from cohort: {key}")
            found[eid] = row
    if set(found) != set(expected):
        raise ValueError("source reader did not return every selected ID exactly once; no refill is allowed")
    return tuple(found[item["example_id"]] for item in selected)


def _verify_dependencies(root: Path, config: Mapping[str, Any], token_rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    from dwr.experiments import gpu_feasibility

    dependencies = config["dependencies"]
    data = config["data"]
    export = _manifest(root, data["source_export_dir"], data["source_export_manifest_sha256"], "source export")
    if export.get("counts") != {"exported_records": 2701, "source_train": 2436, "source_validation": 265}:
        raise ValueError("source export manifest counts differ from the registration")
    export_summary_raw = _read(_resolve(root, f"{data['source_export_dir']}/summary.json"), "source export summary")
    export_summary = _json(export_summary_raw, "source export summary", allow_null=True)
    if _sha(export_summary_raw) != "8d7d318837010f26db263080feea2e326260a08f4d9a7b42bc5f71a0f554b542" or \
            export_summary.get("output_sha256") != data["source_export_data_sha256"]:
        raise ValueError("source export summary/data proof differs from registration")
    output_raw = _read(_resolve(root, f"{data['source_export_dir']}/source_examples.jsonl"), "source export payload")
    if _sha(output_raw) != data["source_export_data_sha256"]:
        raise ValueError("source export payload bytes differ from the opaque data pin")

    token_manifest = _manifest(root, data["source_token_audit_dir"],
                               data["source_token_audit_manifest_sha256"], "token audit")
    if token_manifest.get("counts") != {"records": 2701, "source_train": 2436, "source_validation": 265}:
        raise ValueError("token-audit manifest counts differ from the registration")
    token_summary_raw = _read(_resolve(root, f"{data['source_token_audit_dir']}/summary.json"), "token audit summary")
    token_summary = _json(token_summary_raw, "token audit summary", allow_null=True)
    if _sha(token_summary_raw) != "6dfce99fcbff0393de23c390671d52b288c4042c0926356a7a4e7be3f4b255c8" or \
            token_summary.get("status") != "PASS_SOURCE_TOKEN_CAPS" or token_summary.get("protected_semantic_payload_access") is not False or \
            token_summary.get("gpu_operations") is not False:
        raise ValueError("token-audit summary does not prove the pinned metadata-only pass")
    if len(token_rows) != token_manifest["counts"]["records"]:
        raise ValueError("loaded token metadata count differs from its manifest")

    feasibility_path = _resolve(root, dependencies["feasibility_config_path"])
    feasibility_raw = _read(feasibility_path, "feasibility config")
    if _sha(feasibility_raw) != dependencies["feasibility_config_sha256"]:
        raise ValueError("feasibility config bytes differ from the pin")
    helper_path = Path(gpu_feasibility.__file__).resolve(strict=True)
    if _sha(_read(helper_path, "feasibility helper")) != dependencies["feasibility_module_sha256"]:
        raise ValueError("feasibility helper bytes differ from the pin")
    feasibility = gpu_feasibility.validate_config_bytes(feasibility_raw)
    feasibility_manifest = _manifest(root, dependencies["feasibility_run_dir"],
        dependencies["feasibility_manifest_sha256"], "feasibility proof")
    outcome_raw = _read(_resolve(root, f"{dependencies['feasibility_run_dir']}/outcome.json"),
                        "feasibility outcome")
    outcome = _validate_feasibility_outcome(feasibility_manifest, outcome_raw,
                                            dependencies["feasibility_outcome_sha256"])
    reference_config_raw = _read(_resolve(root, dependencies["reference_config_path"]), "reference config")
    if _sha(reference_config_raw) != dependencies["reference_config_sha256"]:
        raise ValueError("frozen-reference config bytes differ from pin")
    reference_config = _json(reference_config_raw, "reference config")
    reference_cohort_raw = _read(_resolve(root, dependencies["reference_cohort_path"]), "reference cohort")
    if _sha(reference_cohort_raw) != dependencies["reference_cohort_sha256"]:
        raise ValueError("frozen-reference cohort bytes differ from pin")
    reference_cohort = _json(reference_cohort_raw, "reference cohort")
    reference_ids = {row["example_id"] for row in reference_cohort.get("selected", [])}
    actual_overlap = sorted(reference_ids & {row["example_id"] for row in _validation_selection(token_rows)})
    cohort_raw = _read(_resolve(root, data["cohort_path"]), "pilot cohort")
    cohort = _json(cohort_raw, "pilot cohort")
    if cohort.get("frozen_reference_overlap_ids") != actual_overlap or len(actual_overlap) != 1:
        raise ValueError("pilot cohort overlap differs from pinned frozen-reference membership")
    if reference_config.get("dependencies", {}).get("cohort_sha256") != dependencies["reference_cohort_sha256"]:
        raise ValueError("reference config does not bind the registered reference cohort")
    _manifest(root, dependencies["reference_run_dir"], dependencies["reference_manifest_sha256"], "frozen reference")
    reference_summary_raw = _read(_resolve(root, f"{dependencies['reference_run_dir']}/summary.json"),
                                  "frozen-reference summary")
    reference_summary = _json(reference_summary_raw, "frozen-reference summary", allow_null=True)
    if _sha(reference_summary_raw) != dependencies["reference_summary_sha256"] or \
            reference_summary.get("status") != "PASS_FROZEN_REFERENCE_OPERATIONAL_V1":
        raise ValueError("frozen-reference summary identity/status mismatch")
    reference_module = Path(__import__("dwr.experiments.frozen_reference", fromlist=["__file__"]).__file__).resolve(strict=True)
    if _sha(_read(reference_module, "frozen-reference module")) != dependencies["reference_module_sha256"]:
        raise ValueError("frozen-reference module bytes differ from pin")

    overfit_config = _read(_resolve(root, dependencies["source_overfit_config_path"]), "overfit registration")
    if _sha(overfit_config) != dependencies["source_overfit_config_sha256"]:
        raise ValueError("passing source-overfit config bytes differ from pin")
    result_raw = _read(_resolve(root, dependencies["source_overfit_result_path"]), "source-overfit result")
    if _sha(result_raw) != dependencies["source_overfit_result_sha256"]:
        raise ValueError("passing source-overfit result bytes differ from pin")
    result = _json(result_raw, "source-overfit result", allow_null=True)
    if result.get("result") != "PASS_TINY_SOURCE_OVERFIT" or \
            result.get("artifact_sha256", {}).get("job_terminal_manifest") != dependencies["source_overfit_terminal_job_manifest_sha256"] or \
            result.get("artifact_sha256", {}).get("worker_terminal_manifest") != dependencies["source_overfit_terminal_worker_manifest_sha256"]:
        raise ValueError("passing source-overfit terminal identities differ from registration")
    audit_raw = _read(_resolve(root, dependencies["source_overfit_independent_audit_path"]),
                      "independent overfit audit")
    if _sha(audit_raw) != dependencies["source_overfit_independent_audit_sha256"] or \
            _json(audit_raw, "independent overfit audit").get("result") != "PASS_TINY_SOURCE_OVERFIT":
        raise ValueError("source-overfit independent terminal audit differs from pin")

    artifact_manifest_raw = _read(_resolve(root, config["model"]["artifact_manifest_path"]), "model artifact manifest")
    if _sha(artifact_manifest_raw) != config["model"]["artifact_manifest_sha256"]:
        raise ValueError("model artifact manifest bytes differ from pin")
    artifact_manifest = _json(artifact_manifest_raw, "model artifact manifest", allow_null=True)
    if artifact_manifest.get("status") != "completed":
        raise ValueError("pinned model artifact inventory is not completed")
    if feasibility["model"]["weight_sha256"] != config["model"]["weight_sha256"] or \
            feasibility["model"]["artifact_inventory_sha256"] != config["model"]["artifact_inventory_sha256"] or \
            feasibility["model"]["config_sha256"] != config["model"]["model_config_sha256"]:
        raise ValueError("pilot model pins differ from the full feasibility host policy")
    layout = feasibility["layout"]
    host = config["host"]
    layout_pairs = {"slot_identity": "slot_identity", "num_slots": "num_slots",
        "latent_dim": "latent_dim", "max_steps": "max_steps", "prompt_layer": "prompt_layer",
        "adapter_layers": "adapter_layers", "reader_heads": "reader_heads",
        "workspace_heads": "workspace_heads", "context_dim": "context_dim", "gate_init": "gate_init"}
    if any(layout.get(source) != host[target] for source, target in layout_pairs.items()):
        raise ValueError("pilot host layout differs from the inherited feasibility policy")
    precision = feasibility["precision"]
    precision_pairs = {"original": "original_dtype", "new": "new_dtype",
        "autocast_new": "autocast", "scaler": "scaler",
        "gradient_checkpointing": "gradient_checkpointing", "kv_cache": "kv_cache",
        "prompt_cache": "prompt_cache", "offload": "offload", "quantization": "quantization"}
    if any(precision.get(source) != host[target] for source, target in precision_pairs.items()):
        raise ValueError("pilot precision policy differs from the inherited feasibility policy")
    attention = feasibility["attention"]
    expected_sdpa = {"implementation": attention["implementation"],
        "cudnn": attention["builtin_cudnn"], "flash": attention["builtin_flash"],
        "math": attention["builtin_math"], "memory_efficient": attention["builtin_memory_efficient"]}
    if host["sdpa"] != expected_sdpa or host["numerics"] != feasibility["numerics"]:
        raise ValueError("pilot attention/numerics differ from the inherited feasibility policy")
    return feasibility


def _validate_startup_retry_config(config: Any) -> dict[str, Any]:
    """Validate the same scientific study with its separately registered run identity."""
    outputs = {"dwr-source-validation-pilot-startup-retry-v2":
        "runs/source-validation-pilot-20261019-startup-retry-v2",
        "dwr-source-validation-pilot-registration-repair-v3":
        "runs/source-validation-pilot-20261019-registration-repair-v3"}
    if not isinstance(config, dict) or config.get("config_version") not in outputs or \
            config.get("output_dir") != outputs[config["config_version"]]:
        raise ValueError("startup retry version or output path differs from registration")
    baseline = copy.deepcopy(config)
    baseline["config_version"] = "dwr-source-validation-pilot-v1"
    baseline["output_dir"] = "runs/source-validation-pilot-20261017-v1"
    original_raw = _read(Path(__file__).resolve().parents[2] / CONFIG_PATH, "original pilot registration")
    if _sha(original_raw) != CONFIG_SHA256 or baseline != _json(original_raw, "original pilot registration"):
        raise ValueError("startup retry must preserve every original scientific setting")
    _validate_config(baseline)
    return config


def _verify_startup_memory_failure(root: Path) -> None:
    from dwr.experiments.records import verify_artifacts
    segment = root / "runs/source-validation-pilot-20261017-v1/arms/K0/segment-000"
    if _sha(_read(segment / "manifest.json", "original startup failure")) != \
            "87e94e39bcbbb41dd67e6de321e6ae619252ada40731ea517f4c6fd5aba7acde":
        raise ValueError("startup retry requires the preserved original preflight failure")
    terminal = verify_artifacts(segment)
    rows = [_json(line.encode("utf-8"), "startup failure event")
        for line in (segment / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    if terminal["status"] != "failed" or [row["kind"] for row in rows] != [
            "worker_receipt_published", "source_examples_access_authorized",
            "source_examples_access_authorized", "known_fault"] or rows[-1]["payload"] != {
                "exception_type": "RuntimeError", "phase": "preflight",
                "message_sha256": "463078d1c0d20e1317adb085156762d451109190009861fc5c884a27be4b4c5a"}:
        raise ValueError("startup retry is limited to the zero-model-work memory preflight failure")


def load_pilot_registration(root: Path, config_path: Path) -> dict[str, Any]:
    """Load the only production registration and its metadata/proof identities."""
    root = Path(root).resolve(strict=True)
    requested = Path(os.path.abspath(os.fspath(config_path)))
    _reject_links(requested)
    relative = requested.resolve(strict=True).relative_to(root).as_posix()
    allowed = {CONFIG_PATH: (CONFIG_SHA256, False),
        STARTUP_RETRY_CONFIG_PATH: (STARTUP_RETRY_CONFIG_SHA256, True),
        REGISTRATION_REPAIR_CONFIG_PATH: (REGISTRATION_REPAIR_CONFIG_SHA256, True)}
    if relative not in allowed:
        raise ValueError("only the registered source-validation pilot config path is accepted")
    expected = root.joinpath(*PurePosixPath(relative).parts)
    config_sha256, startup_retry = allowed[relative]
    _reject_links(expected)
    config_raw = _read(expected, "pilot config")
    if _sha(config_raw) != config_sha256:
        raise ValueError("pilot config bytes differ from the closed registration")
    config = (_validate_startup_retry_config if startup_retry else _validate_config)(
        _json(config_raw, "pilot config"))
    if startup_retry:
        _verify_startup_memory_failure(root)
    if relative == REGISTRATION_REPAIR_CONFIG_PATH:
        previous = root / "runs/source-validation-pilot-20261019-startup-retry-v2"
        if _sha(_read(previous / "status.json", "v2 startup failure")) != \
                "656a95ee4eead66aec65116b9dc55e6b1a930463aeb7a4d48631e474e91905f3" or \
                _read(previous / "arms/K0/segment-000/events.jsonl", "v2 worker events") != b"" or \
                (previous / "arms/K0/segment-000/worker-receipt.json").exists():
            raise ValueError("registration repair requires the preserved zero-access v2 startup failure")
    cohort_raw = _read(_resolve(root, config["data"]["cohort_path"]), "pilot cohort")
    if _sha(cohort_raw) != COHORT_SHA256:
        raise ValueError("frozen pilot cohort bytes differ from registration")
    cohort_value = _json(cohort_raw, "pilot cohort")
    token_rows = _read_token_metadata(root, config)
    cohort = _validate_cohort(cohort_value, token_rows)
    feasibility_config = _verify_dependencies(root, config, token_rows)
    selected_ids = {row["example_id"] for row in cohort["selected"]}
    train_ids = {row["example_id"] for row in token_rows if row["split"] == "source_train"}
    if selected_ids & train_ids:
        raise ValueError("pilot validation selection overlaps source-training IDs")
    return {"config": config, "config_raw": config_raw, "config_path": expected,
        "config_sha256": config_sha256, "cohort": cohort, "cohort_raw": cohort_raw,
        "cohort_sha256": COHORT_SHA256, "token_rows": token_rows,
        "source_pins": {key: config["data"][key] for key in
            ("source_export_manifest_sha256", "source_export_data_sha256",
             "source_token_audit_manifest_sha256", "token_lengths_sha256")},
        "proof_pins": copy.deepcopy(config["dependencies"]),
        "model_pins": copy.deepcopy(config["model"]), "feasibility_config": feasibility_config}


def read_pilot_examples(root: Path, registration: Mapping[str, Any], *, split: str):
    """Read exactly the registered source population/cohort after caller access authorization."""
    if split not in {"source_train", "source_validation"}:
        raise ValueError("pilot readers permit only source_train or source_validation")
    config, cohort = registration["config"], registration["cohort"]
    from dwr.experiments.source_reader import read_source_export
    from dwr.tasks import schema
    from dwr.tasks.run_support import stable_json as canonical_json
    from dwr.tasks.verifier import verify_answer
    from dwr.training.batches import SourceExample
    from dwr.tasks.schema import Solution

    data = config["data"]
    expected_count = data["source_train_records"] if split == "source_train" else data["source_validation_records"]
    rows = read_source_export(Path(root).joinpath(*PurePosixPath(data["source_export_dir"]).parts),
        expected_manifest_sha256=data["source_export_manifest_sha256"], split=split)
    if len(rows) != expected_count:
        raise ValueError(f"source reader returned {len(rows)} rows, expected {expected_count}")
    token_by_id = {row["example_id"]: row for row in registration["token_rows"] if row["split"] == split}
    if len(token_by_id) != expected_count or {row["example_id"] for row in rows} != set(token_by_id):
        raise ValueError("source reader and token metadata do not cover exact split membership")
    for row in rows:
        meta = token_by_id[row["example_id"]]
        if any(row.get(key) != meta.get(key) for key in ("split", "profile", "fine_stratum")):
            raise ValueError("source row profile/stratum/split differs from token metadata")
    if split == "source_train":
        ordered = tuple(sorted(rows, key=lambda row: row["example_id"]))
    else:
        ordered = _select_pilot_rows(rows, cohort["selected"], split=split)
    output = []
    for row in ordered:
        task = schema.parse_task_json(row["task"])
        assignment = row["solution"]["assignment"]
        answer = canonical_json({"assignment": assignment})
        if not verify_answer(task, answer).valid:
            raise ValueError("source reader row failed the strict answer verifier")
        solution = Solution(tuple((name, assignment[name]) for name in task.variables))
        output.append(SourceExample(row["example_id"], split, task, solution))
    metadata = {row["example_id"]: {key: token_by_id[row["example_id"]][key]
        for key in ("profile", "fine_stratum", "prompt_tokens", "answer_tokens_including_eos", "total_tokens")}
        for row in ordered}
    if set(metadata) != {example.example_id for example in output}:
        raise ValueError("pilot examples and separate metadata map differ in membership")
    return tuple(output), metadata


def pilot_identities(registration: Mapping[str, Any], *, source_commit: str, arm_id: str) -> dict[str, str]:
    """Resolve the six complete checkpoint identities for one fixed arm."""
    if arm_id not in {item["arm_id"] for item in registration["config"]["arms"]}:
        raise ValueError("unknown pilot arm")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", source_commit):
        raise ValueError("source commit must be lowercase hexadecimal commit text")
    arm = next(item for item in registration["config"]["arms"] if item["arm_id"] == arm_id)
    arm_config = {"pilot_config_sha256": registration["config_sha256"],
                  "arm_id": arm_id, "steps": arm["steps"]}
    config_digest = _sha(json.dumps(arm_config, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8"))
    data = registration["config"]["data"]
    deps = registration["config"]["dependencies"]
    data_identity = {"source_export_manifest_sha256": data["source_export_manifest_sha256"],
        "source_export_data_sha256": data["source_export_data_sha256"],
        "source_token_audit_manifest_sha256": data["source_token_audit_manifest_sha256"],
        "token_lengths_sha256": data["token_lengths_sha256"],
        "cohort_sha256": registration["cohort_sha256"],
        "feasibility_manifest_sha256": deps["feasibility_manifest_sha256"],
        "feasibility_outcome_sha256": deps["feasibility_outcome_sha256"],
        "reference_manifest_sha256": deps["reference_manifest_sha256"],
        "reference_summary_sha256": deps["reference_summary_sha256"],
        "source_overfit_result_sha256": deps["source_overfit_result_sha256"],
        "source_overfit_terminal_job_manifest_sha256": deps["source_overfit_terminal_job_manifest_sha256"],
        "source_overfit_terminal_worker_manifest_sha256": deps["source_overfit_terminal_worker_manifest_sha256"],
        "source_overfit_independent_audit_sha256": deps["source_overfit_independent_audit_sha256"]}
    canonical = lambda value: _sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8"))
    result = {"source": _sha(source_commit.encode("ascii")), "config": config_digest,
        "data": canonical(data_identity), "model": registration["model_pins"]["weight_sha256"],
        "tokenizer": registration["model_pins"]["artifact_inventory_sha256"],
        "schedule": canonical(registration["config"]["training"]["schedule"])}
    if any(_HASH.fullmatch(value) is None for value in result.values()):
        raise ValueError("resolved pilot checkpoint identity is malformed")
    return result
