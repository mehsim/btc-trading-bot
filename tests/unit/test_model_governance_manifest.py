import unittest
import os
import json
import pytest
import numpy as np
from xgboost import XGBClassifier
from ensemble import save_ensemble_classifier, load_ensemble_classifier, EnsembleClassifier

class TestModelGovernanceManifest(unittest.TestCase):
    def setUp(self):
        self.prefix = "test_gov_model"
        # Train dummy XGBoost classifier
        X = np.random.randn(50, 4)
        y = np.random.choice([0, 1, 2], size=50)
        xgb = XGBClassifier()
        xgb.fit(X, y)
        clf = EnsembleClassifier(xgb)
        save_ensemble_classifier(clf, self.prefix)

    def tearDown(self):
        for ext in ["_xgb.json", "_weights.json", "_manifest.json"]:
            path = f"{self.prefix}{ext}"
            if os.path.exists(path):
                os.remove(path)

    def test_feature_contract_hash_matching_passes(self):
        """Verify F-11: Valid feature contract hash matches and loads cleanly."""
        feats = ["feat_a", "feat_b", "feat_c", "feat_d"]
        # Save manifest
        from ensemble import write_model_manifest
        write_model_manifest(self.prefix, feature_names=feats)

        # Load with matching feature list -> Should pass cleanly
        loaded_clf = load_ensemble_classifier(self.prefix, n_features=4, feature_names=feats)
        self.assertIsNotNone(loaded_clf)

    def test_feature_contract_hash_mismatch_fails_closed(self):
        """Verify F-11: Mismatched feature contract hash raises RuntimeError (Fail-Closed)."""
        feats_train = ["feat_a", "feat_b", "feat_c", "feat_d"]
        from ensemble import write_model_manifest
        write_model_manifest(self.prefix, feature_names=feats_train)

        # Attempt to load with altered feature list -> Must raise RuntimeError
        feats_corrupted = ["feat_a", "feat_b", "feat_c", "feat_WRONG"]
        with self.assertRaises(RuntimeError) as cm:
            load_ensemble_classifier(self.prefix, n_features=4, feature_names=feats_corrupted)
        self.assertIn("Feature contract hash mismatch", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
