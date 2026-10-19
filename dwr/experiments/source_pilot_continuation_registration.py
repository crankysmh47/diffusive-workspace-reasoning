"""Closed operational continuation registration; no scientific retuning."""
from __future__ import annotations

import copy
from pathlib import Path

CONFIG_PATH = 'configs/experiments/source_validation_pilot_operational_continuation_v5.json'
CONFIG_SHA256 = '826004053ef194a6e817c9f660eef9ec36cacdb350e1e86a2ed6c063d03e0232'
MANIFEST_PATH = 'configs/models/source_validation_pilot_v4_continuation.json'
MANIFEST_SHA256 = '89e832a0ffcf48899035ba4c79a8053604f81049729523d2cf41e44961f990f6'
# Only supervision, registration and aggregation differ; tensors, decoding,
# training, tasks and scoring retain the exact producer bytes.
OPERATIONAL_FILES = {
    'dwr/experiments/source_pilot_job.py',
    'dwr/experiments/source_pilot_watchdog.py',
    'dwr/experiments/source_pilot_registration.py',
    'dwr/experiments/source_pilot.py',
    'dwr/experiments/source_pilot_continuation.py',
    'dwr/experiments/source_pilot_continuation_registration.py',
}


def validate_continuation_config(config: dict, baseline: dict) -> dict:
    expected = copy.deepcopy(baseline)
    expected.update(config_version='dwr-source-validation-pilot-operational-continuation-v5',
        output_dir='runs/source-validation-pilot-20261019-operational-continuation-v5',
        continuation_manifest_path=MANIFEST_PATH, continuation_manifest_sha256=MANIFEST_SHA256)
    expected['source']['source_policy'] = 'shared_scientific_code_explicit_mixed_operational_source_provenance'
    expected['limits'].update(minimum_initial_free_mib=13000, total_wall_seconds=5400,
        matrix_external_watchdog_seconds=5460, gpu_hours_wall_cap=1.5,
        effective_arm_deadline='min(arm_start_plus_5400_seconds,continuation_matrix_start_plus_5400_seconds)')
    if config != expected:
        raise ValueError('continuation must preserve all scientific settings and exact operational amendments')
    return config


def load_continuation_registration(root: Path, config_path: Path) -> dict:
    from dwr.experiments.source_pilot_registration import (
        _read, _sha, _json, TOKENIZER_REPAIR_CONFIG_PATH, load_pilot_registration)
    root = Path(root).resolve(strict=True)
    if Path(config_path).resolve(strict=True) != (root / CONFIG_PATH).resolve(strict=True):
        raise ValueError('only the closed continuation config path is accepted')
    raw = _read(root / CONFIG_PATH, 'continuation config')
    if _sha(raw) != CONFIG_SHA256:
        raise ValueError('continuation config bytes differ from registration')
    parent = load_pilot_registration(root, root / TOKENIZER_REPAIR_CONFIG_PATH)
    config = validate_continuation_config(_json(raw, 'continuation config'), parent['config'])
    manifest_raw = _read(root / MANIFEST_PATH, 'continuation manifest')
    if _sha(manifest_raw) != MANIFEST_SHA256:
        raise ValueError('continuation parent receipt bytes differ from registration')
    continuation = _json(manifest_raw, 'continuation manifest')
    initial_path = root / continuation['parent_matrix'] / 'manifest.initial.json'
    initial = _json(_read(initial_path, 'parent initial manifest'), 'parent initial manifest')
    scientific = {item['path']: item['sha256'] for item in initial['source']['files']
        if item['path'].startswith('dwr/') and item['path'].endswith('.py')
        and item['path'] not in OPERATIONAL_FILES}
    for relative, digest in scientific.items():
        if _sha(_read(root / relative, 'unchanged scientific source')) != digest:
            raise ValueError(f'continuation scientific source differs: {relative}')
    current = {p.relative_to(root).as_posix() for p in (root / 'dwr').rglob('*.py')}
    if current - set(scientific) - OPERATIONAL_FILES:
        raise ValueError('continuation has unregistered scientific source modules')
    registration = dict(parent, config=config, config_raw=raw, config_path=root / CONFIG_PATH,
        config_sha256=CONFIG_SHA256, continuation=continuation)
    from dwr.experiments.source_pilot_continuation import load_parent_imports
    load_parent_imports(root, registration)
    return registration
