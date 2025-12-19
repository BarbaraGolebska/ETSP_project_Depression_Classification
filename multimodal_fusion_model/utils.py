import joblib
import json
import numpy as np
import lightgbm as lgb

class ModelPipeline:
    def __init__(self, scaler, booster, best_threshold, mask):
        self.scaler = scaler
        self.booster = booster
        self.best_threshold = best_threshold
        self.mask = mask

    @classmethod
    def load(cls, scaler_path, booster_path, metrics_path, mask_path):
        scaler = joblib.load(scaler_path)
        booster = lgb.Booster(model_file=booster_path)

        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        best_threshold = metrics["best_threshold"]

        mask = joblib.load(mask_path) 

        return cls(scaler, booster, best_threshold, mask)

    def _apply_mask(self, X):
        return X[:, self.mask]

    def predict_proba(self, X):
        X_masked = self._apply_mask(X)          
        X_scaled = self.scaler.transform(X_masked)
        y_proba = self.booster.predict(X_scaled)
        return np.vstack([1 - y_proba, y_proba]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.best_threshold).astype(int)
