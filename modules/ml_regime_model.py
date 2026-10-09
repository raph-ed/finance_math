import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

class WalkForwardRegimeDetector:
    """
    Implements a strict Walk-Forward cross-validation regime classifier with purging/embargo
    to eliminate look-ahead and overlap bias.
    """
    def __init__(self, model_type="rf", horizon=20, embargo=5):
        self.model_type = model_type
        self.horizon = horizon
        self.embargo = embargo
        self.trained_model = None
        self.feature_names = None
        self.feature_importances_ = None
        
    def _create_model(self):
        if self.model_type == "logistic":
            return LogisticRegression(penalty="l2", C=1.0, max_iter=500, class_weight="balanced")
        else:
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_split=15,
                class_weight="balanced_subsample",
                random_state=42
            )
            
    def run_walk_forward(self, dataset, feature_cols, train_window=504, test_step=63):
        """
        Runs anchored/rolling walk-forward training.
        Purges the last `horizon + embargo` days from the train set before testing.
        """
        self.feature_names = feature_cols
        X = dataset[feature_cols].values
        y = dataset["target"].values
        n = len(dataset)
        
        predictions = []
        probabilities = []
        pred_indices = []
        
        start_idx = train_window
        while start_idx + test_step <= n:
            train_end = start_idx - (self.horizon + self.embargo)
            if train_end < 150:
                start_idx += test_step
                continue
                
            X_train, y_train = X[:train_end], y[:train_end]
            test_end = min(start_idx + test_step, n)
            X_test = X[start_idx:test_end]
            
            clf = self._create_model()
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            
            predictions.extend(y_pred)
            probabilities.extend(y_prob)
            pred_indices.extend(dataset.index[start_idx:test_end])
            
            start_idx += test_step
            
        # Fit final model on all eligible data for live inference
        final_cutoff = n - (self.horizon + self.embargo)
        if final_cutoff > 200:
            self.trained_model = self._create_model()
            self.trained_model.fit(X[:final_cutoff], y[:final_cutoff])
            if hasattr(self.trained_model, "feature_importances_"):
                self.feature_importances_ = pd.Series(self.trained_model.feature_importances_, index=self.feature_names)
            elif hasattr(self.trained_model, "coef_"):
                self.feature_importances_ = pd.Series(np.abs(self.trained_model.coef_).mean(axis=0), index=self.feature_names)
                
        df_results = pd.DataFrame({
            "predicted_regime": predictions,
            "prob_bull": [p[0] if len(p) > 0 else 0 for p in probabilities],
            "prob_neutral": [p[1] if len(p) > 1 else 0 for p in probabilities],
            "prob_selloff": [p[2] if len(p) > 2 else 0 for p in probabilities],
        }, index=pred_indices)
        
        return df_results
