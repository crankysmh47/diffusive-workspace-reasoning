import copy
import json
from pathlib import Path
import unittest


class ContinuationRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        base = self.root / 'configs/experiments'
        self.baseline = json.loads((base / 'source_validation_pilot_tokenizer_repair_v4.json').read_text(encoding='utf-8'))
        self.config = json.loads((base / 'source_validation_pilot_operational_continuation_v5.json').read_text(encoding='utf-8'))

    def test_only_declared_operational_changes_are_accepted(self):
        from dwr.experiments.source_pilot_continuation_registration import validate_continuation_config
        self.assertEqual(validate_continuation_config(self.config, self.baseline), self.config)
        for section, key, value in [('training', 'optimizer_steps', 513), ('training', 'batch_size', 2),
                                    ('seeds', 'training_noise', 1), ('limits', 'minimum_initial_free_mib', 12000),
                                    ('evaluation', 'draws_per_example', 2)]:
            changed = copy.deepcopy(self.config)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                validate_continuation_config(changed, self.baseline)

    def test_live_registration_verifies_parent_imports_without_model_access(self):
        if not (self.root / 'runs/source-validation-pilot-20261019-tokenizer-repair-v4/arms/K1/manifest.json').exists():
            self.skipTest('local parent artifacts are not distributed with Git')
        from dwr.experiments.source_pilot_continuation_registration import load_continuation_registration, CONFIG_PATH
        from dwr.experiments.source_pilot_continuation import load_parent_imports
        reg = load_continuation_registration(self.root, self.root / CONFIG_PATH)
        audits, sources = load_parent_imports(self.root, reg)
        self.assertEqual(set(audits), {'K0', 'K1'})
        self.assertEqual(set(sources), {'K0', 'K1'})


if __name__ == '__main__':
    unittest.main()
