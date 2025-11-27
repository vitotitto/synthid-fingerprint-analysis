"""
Watermark Classifier

- Cross-validation with stratified group splits
- Feature importance analysis

"""

import numpy as np
import pandas as pd
import joblib
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, confusion_matrix,
    classification_report, precision_recall_curve,
    average_precision_score, f1_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns



# Global plot style for CLI visualisations
sns.set_style('whitegrid')


#################################################################################################
# CONFIGURATION
#################################################################################################

# Features list
# These features were selected through systematic evaluation for optimal
# discrimination between watermarked and real images from a slightly bigger subset
FEATURES_LIST = [
    # Lattice features (6) - frequency domain and autocorrelation patterns mainly work for high frequencies 
    'lat_diag_total_frac',                    # diagonal energy fraction in frequency domain
    'lat_axis_total_frac',                    # axis-aligned energy fraction in frequency domain
    'lat_line_axis_density',                  # density of harmonic peaks along axes
    'lat_diag_over_axis_total',               # ratio of diagonal to axis energy
    'lat_ac_var_axis_over_diag',              # autocorrelation variance ratio
    'lat_p2_fraction_of_smallperiod_energy',  # period-2 energy as fraction of small periods
    
    # Angular features (3) - directional energy distribution - medium and lower frequencies 
    'ang_entropy',                            # entropy of angular energy histogram
    'ang_spectral_flatness',                  # geometric/arithmetic mean ratio
    'ang_dip_contrast',                       # contrast at angular dips
    
    # Diagonal stripe feature (1)
    'ds_autocorr_diagonal',                   # autocorrelation along diagonal direction - medium and lower frequencies
    
    # Checkerboard confidence (1)
    'cb_confidence',                          # overall checkerboard pattern confidence - high frequencies
    
    # Quality metrics (2)
    'spatial_coherence',                      # spatial pattern regularity
    'snr',                                    # signal-to-noise ratio in dB - final check
]


#################################################################################################
# CONFIGURATION DATACLASS
#################################################################################################

@dataclass
class ClassifierConfig:
    """
    Configuration parameters for classifier training and evaluation

    - If optimise_threshold is False: use fixed decision_threshold for inference
    - If optimise_threshold is True: use CV-derived mean optimal threshold
      (or fall back to training data if CV isn't available)
    """

    positive_label: str = 'watermarked'  
    
    # Cross-validation settings
    n_splits: int = 5
    random_state: int = 42
    
    # Logistic regression hyperparameters
    C: float = 1.0              # inverse regularisation strength
    max_iter: int = 1000        # maximum iterations for solver convergence
    solver: str = 'lbfgs'       # optimisation algorithm (limited memory BFGS)
    
    # Threshold optimisation
    optimise_threshold: bool = False # If True: uses CV / training to choose inference threshold,  otherwise it uses fixed 0.5 
    optimisation_metric: str = 'f1'  # options: 'f1', 'youden', 'precision_recall'
    decision_threshold: float = 0.5 # fixed threshold
    
    # Output control
    save_plots: bool = True
    save_logs: bool = True
    verbose: bool = True


#################################################################################################
# DATA STRUCTURES FOR RESULTS
#################################################################################################

@dataclass
class CVFoldResult:
    """Results from a single cross-validation fold."""
    
    fold: int
    window_auc: float
    image_auc: float
    optimal_threshold: float
    n_train_images: int
    n_test_images: int
    test_image_ids: List[str]
    feature_importance: np.ndarray
    predictions: pd.DataFrame


@dataclass
class CVResults:
    """Aggregated cross-validation results across all folds."""
    
    fold_results: List[CVFoldResult]
    
    # AUC statistics
    mean_window_auc: float
    std_window_auc: float
    mean_image_auc: float
    std_image_auc: float
    
    # Threshold statistics
    mean_optimal_threshold: float
    std_optimal_threshold: float
    
    # Feature analysis
    avg_feature_importance: np.ndarray
    feature_names: List[str]
    
    # ROC curve data (for plotting)
    tprs: List[np.ndarray]      # true positive rates interpolated to common FPR grid
    base_fpr: np.ndarray        # common false positive rate grid


@dataclass
class EvaluationResults:
    """Results from model evaluation on holdout/test data."""
    
    window_auc: float
    image_auc: float
    confusion_matrix: np.ndarray
    classification_report: str
    optimal_threshold: float
    predictions: pd.DataFrame
    pr_auc: float  # precision-recall AUC
    
    def get_metrics_dict(self, threshold: float = None) -> Dict[str, float]:
        """
        Compute classification metrics at a specified threshold.
        
        Args:
            threshold: Classification threshold. Uses optimal if None.
        
        Returns:
            Dictionary containing accuracy, precision, recall, specificity,
            F1 score, and confusion matrix counts.
        """
        if threshold is None:
            threshold = self.optimal_threshold
        
        # The threshold to get binary predictions
        preds_binary = (self.predictions['pred_proba'] >= threshold).astype(int)
        
        # Confusion matrix: [[TN, FP], [FN, TP]]

        cm = confusion_matrix(self.predictions['label'], preds_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Derived metrics
        # accuracy = (TP + TN) / total
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # precision = TP / (TP + FP), proportion of positive predictions correct
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # recall = TP / (TP + FN), proportion of actual positives found
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # specificity = TN / (TN + FP), proportion of actual negatives correct
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # F1 = 2 * precision * recall / (precision + recall)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        return {
            'auc': self.image_auc,
            'pr_auc': self.pr_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'true_negatives': int(tn),
        }


#################################################################################################
# MAIN CLASSIFIER CLASS
#################################################################################################

class WatermarkClassifier:
    """
    Watermark classifier with cross-validation
    
    Uses logistic regression on a 13 features list extracted from
    image windows. Supports stratified group k-fold cross-validation to ensure
    no image appears in both train and test sets within the same fold (to prevent leakage).
    """
    
    def __init__(self, config: ClassifierConfig):
        """
        Initialise classifier with configuration.
        
        Args:
            config: ClassifierConfig instance with training parameters.
        """
        self.config = config
        self.feature_names = FEATURES_LIST
        
        # Model components populated during training
        self.scaler = None
        self.model = None
        self.optimal_threshold = 0.5
        
        # Training metadata
        self.cv_results = None
        self.training_timestamp = None
        
        if config.verbose:
            print(f"Initialised classifier with {len(self.feature_names)} features")
    
    
    #################################################################################################
    # DATA LOADING
    #################################################################################################
    
    def load_features(self, csv_path: Union[str, Path], 
                      label_col: str = 'label') -> pd.DataFrame:
        """
        Orchestrates loading, cleaning, and validating the dataset.
        
        Args:
            csv_path: Path to features CSV.
            label_col: Name of the label column.
        
        Returns:
            DataFrame with features and labels, cleaned and validated.
        """
        df = pd.read_csv(csv_path)
        
        # 1. Validate expected features exist
        self._validate_features(df)
        
        # 2. Standardise and validate labels
        df = self._process_labels(df, label_col)
        
        # 3. Ensure grouping IDs exist (for cross-validation)
        df = self._ensure_image_ids(df)
        
        # 4. Sanitise feature data (handle Inf/NaN)
        df = self._sanitise_features(df)
        
        if self.config.verbose:
            print(f"\nLoaded {len(df)} windows from {df['image_id'].nunique()} images.")
            print(f"Using {len(self.feature_names)} features.")
            print(f"Class balance: {df['label'].value_counts().to_dict()}")
            
        return df
    
    
    #################################################################################################
    # DATA LOADING HELPERS (Private)
    #################################################################################################
    
    def _process_labels(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """Renames label column and enforces explicit binary mapping."""
        
        # Rename column if necessary
        if label_col != 'label':
            if label_col not in df.columns:
                raise ValueError(f"Label column '{label_col}' not found.")
            df = df.rename(columns={label_col: 'label'})
        
        # If already numeric, assume it's correct
        if pd.api.types.is_numeric_dtype(df['label']):
            return df
        
        # Validation: ensure positive label exists in data
        unique_labels = df['label'].unique()
        if self.config.positive_label not in unique_labels:
            raise ValueError(
                f"Positive label '{self.config.positive_label}' not found. "
                f"Available labels: {unique_labels}"
            )
        
        # Apply explicit mapping: positive_label -> 1, everything else -> 0
        df['label'] = (df['label'] == self.config.positive_label).astype(int)
        
        if self.config.verbose:
            print(f"Mapped '{self.config.positive_label}' to 1, others to 0")
            
        return df
    
    
    def _ensure_image_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures image_id exists for GroupKFold, falling back to filename or index."""
        
        if 'image_id' in df.columns:
            return df
        
        if 'filename' in df.columns:
            df['image_id'] = df['filename']
        else:
            print("WARNING: 'image_id' missing. Using index (no group leakage protection).")
            df['image_id'] = df.index.astype(str)
            
        return df
    
    
    def _validate_features(self, df: pd.DataFrame) -> None:
        """Checks that all expected features from FEATURES_LIST are present."""
        
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise ValueError(f"Missing expected features: {missing}")
    
    
    def _sanitise_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replaces Infinity and NaN values with 0.0, with logging.
        
        Note: 0.0 imputation is used because Inf/NaN typically indicate degenerate
        windows (flat regions, no signal) where the feature computation failed.
        For these cases, 0.0 represents "no pattern detected" which is semantically
        appropriate for most features. Mean imputation would be worse as it would
        assign average pattern strength to windows with no valid signal.
        """
        
        features = df[self.feature_names]
        
        # Check for bad values (only calculate if verbose to save time)
        if self.config.verbose:
            n_inf = np.isinf(features).values.sum()
            n_nan = features.isna().sum().sum()
            if n_inf > 0 or n_nan > 0:
                print(f"WARNING: Imputing {n_inf} Inf and {n_nan} NaN values with 0.0")
        
        # Perform replacement
        df[self.feature_names] = features.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        
        return df
    
    
    #################################################################################################
    # CROSS-VALIDATION
    #################################################################################################
    
    def cross_validate(self, df: pd.DataFrame) -> CVResults:
        """
        Perform stratified group k-fold cross-validation.
        
        Ensures that all windows from a single image are kept together in either
        the train or test set, preventing data leakage.
        
        Args:
            df: DataFrame with features, labels, and image_id columns.
        
        Returns:
            CVResults object with aggregated metrics across all folds.
        """
        X = df[self.feature_names].values
        y = df['label'].values
        groups = df['image_id'].values

        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError(
                f"Cross-validation requires both classes in 'label' "
                f"Got only class {unique_labels[0]!r}"
            )

        
        # Shuffles images deterministically for reproducibility
        unique_images = np.unique(groups)
        np.random.seed(self.config.random_state)
        np.random.shuffle(unique_images)
        
        # Map each window to its image's shuffled position
        img_order = {img: i for i, img in enumerate(unique_images)}
        shuffle_order = np.array([img_order[img] for img in groups])
        
        # Stratified group k-fold - stratifies by label whilst keeping groups intact
        sgkf = StratifiedGroupKFold(
            n_splits=self.config.n_splits,
            shuffle=True,  
            random_state=self.config.random_state
        ) # in cross_validate the images are shuffled manually, shuffle=True shuffles the second time, can be False        
        fold_results = []
        tprs = []
        
        #FPR grid for ROC curve averaging (101 points from 0 to 1)
        base_fpr = np.linspace(0, 1, 101)
        
        if self.config.verbose:
            print(f"\n{'#'*60}")
            print(f"Cross-Validation: {self.config.n_splits}-Fold Stratified Group Split")
            print(f"{'#'*60}")
        
        for fold_idx, (train_idx, test_idx) in enumerate(
            sgkf.split(X, y, groups=shuffle_order)
        ):
            fold_num = fold_idx + 1
            
            # Splitting data by indices
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_image_ids = groups[test_idx]
            
            # Standardising features, fit on train, transform both
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Training logistic regression
            clf = LogisticRegression(
                C=self.config.C,
                solver=self.config.solver,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state
            )
            clf.fit(X_train_scaled, y_train)
            
            # Probabilities for positive class
            y_pred_proba_window = clf.predict_proba(X_test_scaled)[:, 1]
            
            # Aggregating window predictions to image level by averaging
            fold_df = pd.DataFrame({
                'image_id': test_image_ids,
                'label': y_test,
                'pred_proba': y_pred_proba_window
            })
            img_agg = fold_df.groupby('image_id').agg({
                'label': 'first',           # all windows share same label
                'pred_proba': 'mean'        # average probability across windows
            }).reset_index()
            
            # Computing AUC at window and image level
            window_auc = roc_auc_score(y_test, y_pred_proba_window)
            image_auc = roc_auc_score(img_agg['label'], img_agg['pred_proba'])
            
            # Optimal threshold for this fold
            optimal_threshold = self._find_optimal_threshold(
                img_agg['label'], 
                img_agg['pred_proba'],
                metric=self.config.optimisation_metric
            )
            
            # Getting ROC curve and interpolate TPR to common FPR grid
            fpr, tpr, _ = roc_curve(img_agg['label'], img_agg['pred_proba'])
            tprs.append(np.interp(base_fpr, fpr, tpr))
            tprs[-1][0] = 0.0  # ensures curve starts at origin
            
            # Fold results
            fold_result = CVFoldResult(
                fold=fold_num,
                window_auc=window_auc,
                image_auc=image_auc,
                optimal_threshold=optimal_threshold,
                n_train_images=len(np.unique(groups[train_idx])),
                n_test_images=len(np.unique(test_image_ids)),
                test_image_ids=list(np.unique(test_image_ids)),
                feature_importance=clf.coef_[0],  # logistic regression coefficients
                predictions=img_agg
            )
            fold_results.append(fold_result)
            
            if self.config.verbose:
                print(f"Fold {fold_num}/{self.config.n_splits} | "
                      f"Window AUC: {window_auc:.3f} | "
                      f"Image AUC: {image_auc:.3f} | "
                      f"Threshold: {optimal_threshold:.3f}")
        
        # Aggregating results across folds
        cv_results = CVResults(
            fold_results=fold_results,
            mean_window_auc=np.mean([f.window_auc for f in fold_results]),
            std_window_auc=np.std([f.window_auc for f in fold_results]),
            mean_image_auc=np.mean([f.image_auc for f in fold_results]),
            std_image_auc=np.std([f.image_auc for f in fold_results]),
            mean_optimal_threshold=np.mean([f.optimal_threshold for f in fold_results]),
            std_optimal_threshold=np.std([f.optimal_threshold for f in fold_results]),
            avg_feature_importance=np.mean(
                [f.feature_importance for f in fold_results], axis=0
            ),
            feature_names=self.feature_names,
            tprs=tprs,
            base_fpr=base_fpr
        )
        
        if self.config.verbose:
            print(f"{'#'*60}")
            print(f"CV Results:")
            print(f"  Window AUC: {cv_results.mean_window_auc:.3f} +/- {cv_results.std_window_auc:.3f}")
            print(f"  Image AUC:  {cv_results.mean_image_auc:.3f} +/- {cv_results.std_image_auc:.3f}")
            print(f"  Optimal Threshold: {cv_results.mean_optimal_threshold:.3f} +/- {cv_results.std_optimal_threshold:.3f}")
            print(f"{'#'*60}")
        
        self.cv_results = cv_results
        return cv_results
    
    
    def _find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                 metric: str = 'f1') -> float:
        """
        Find optimal classification threshold using specified metric.
        
        Args:
            y_true: True binary labels.
            y_pred_proba: Predicted probabilities for positive class.
            metric: Optimisation metric - 'f1', 'youden', or 'precision_recall'.
        
        Returns:
            Optimal threshold value between 0 and 1.
        """
        if metric == 'f1':
            # Maximises F1 score = 2 * precision * recall / (precision + recall)
            thresholds = np.linspace(0, 1, 101)
            f1_scores = [
                f1_score(y_true, y_pred_proba >= t, zero_division=0) 
                for t in thresholds
            ]
            return thresholds[np.argmax(f1_scores)]
        
        elif metric == 'youden':
            # Maximises Youdens J stats = sensitivity + specificity - 1 = TPR - FPR
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            return thresholds[optimal_idx]
        
        elif metric == 'precision_recall':
            # threshold where precision equals recall
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            # precision and recall arrays are one longer than thresholds
            diff = np.abs(precision[:-1] - recall[:-1])
            optimal_idx = np.argmin(diff)
            return thresholds[optimal_idx]
        
        else:
            raise ValueError(f"Unknown optimisation metric: {metric}")
    
    
    #################################################################################################
    # TRAINING
    #################################################################################################
    
    def train(self, df: pd.DataFrame, run_cv: bool = True) -> 'WatermarkClassifier':
        """
        Train final model on all available data.
        
        Args:
            df: DataFrame with features and labels.
            run_cv: Whether to run cross-validation first for metrics.
        
        Returns:
            Self for method chaining.
        """
        if run_cv:
            self.cross_validate(df)
        
        X = df[self.feature_names].values
        y = df['label'].values
        
        if self.config.verbose:
            print(f"\n{'#'*60}")
            print("Training Final Model on All Data")
            print(f"{'#'*60}")
        
        # Fitting scaler on all data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Training model on all data
        self.model = LogisticRegression(
            C=self.config.C,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state
        )
        self.model.fit(X_scaled, y)
        
        # Determining classification threshold for inference
        if not self.config.optimise_threshold:
            # Fixed decision threshold for production / inference
            self.optimal_threshold = self.config.decision_threshold
            if self.config.verbose:
                print(
                    f"Using fixed decision threshold = {self.optimal_threshold:.3f} "
                    "(no threshold optimisation for inference this time)."
                )

        elif self.cv_results:
            # Using CV-derived threshold ( may be preferred)
            self.optimal_threshold = self.cv_results.mean_optimal_threshold
            if self.config.verbose:
                print(
                    f"Using CV mean optimal threshold = {self.optimal_threshold:.3f} "
                    "for inference."
                )

        else:
            # Fallback part to calculate threshold on training data
            y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
            self.optimal_threshold = self._find_optimal_threshold(
                y, y_pred_proba, metric=self.config.optimisation_metric
            )
            if self.config.verbose:
                print(
                    "WARNING: Threshold calculated on training data (no CV). "
                    "May overfit."
                )


        
        self.training_timestamp = datetime.now().isoformat()
        
        if self.config.verbose:
            print(f"Training complete. Model ready for inference.")
            print(f"Optimal threshold: {self.optimal_threshold:.3f}")
            print(f"{'#'*60}")
        
        return self
    
    
    #################################################################################################
    # EVALUATION
    #################################################################################################
    
    def evaluate(self, df: pd.DataFrame, threshold: float = None) -> EvaluationResults:
        """
        Evaluate model on test/holdout data.
        
        Args:
            df: DataFrame with features and labels.
            threshold: Classification threshold. Uses optimal if None.
        
        Returns:
            EvaluationResults object with metrics and predictions.
        
        Raises:
            ValueError: If model has not been trained.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if threshold is None:
            threshold = self.optimal_threshold
        
        X = df[self.feature_names].values
        y = df['label'].values
        groups = df['image_id'].values
        
        # Scaling features using trained scaler
        X_scaled = self.scaler.transform(X)
        
        # Predicting probabilities
        y_pred_proba_window = self.model.predict_proba(X_scaled)[:, 1]

        # Window-level AUC
        if len(np.unique(y)) < 2:
            if self.config.verbose:
                print("WARNING: Evaluation data contains only one class "
                    "Window-level ROC AUC is undefined and set to NaN")
            window_auc = np.nan
        else:
            window_auc = roc_auc_score(y, y_pred_proba_window)

        
        # Aggregating to image level
        result_df = pd.DataFrame({
            'image_id': groups,
            'filename': df['filename'].values if 'filename' in df.columns else groups,
            'label': y,
            'pred_proba': y_pred_proba_window
        })
        
        img_agg = result_df.groupby(['image_id', 'filename']).agg({
            'label': 'first',
            'pred_proba': 'mean'
        }).reset_index()
        
        # Image-level metrics
        img_labels = img_agg['label'].values

        if len(np.unique(img_labels)) < 2:
            if self.config.verbose:
                print("WARNING: Image-level evaluation contains only one class "
                    "ROC AUC and PR AUC are undefined and set to NaN")
            image_auc = np.nan
            pr_auc = np.nan
        else:
            image_auc = roc_auc_score(img_labels, img_agg['pred_proba'])
            pr_auc = average_precision_score(img_labels, img_agg['pred_proba'])



        
        # Binary predictions at threshold
        img_agg['pred_class'] = (img_agg['pred_proba'] >= threshold).astype(int)

        # Confusion matrix
        cm = confusion_matrix(img_agg['label'], img_agg['pred_class'], labels=[0, 1])

        # Classification report (force both classes in the schema)
        clf_report = classification_report(
            img_agg['label'],
            img_agg['pred_class'],
            labels=[0, 1],
            target_names=['Real', 'Watermarked'],
            digits=3
        )

        
        results = EvaluationResults(
            window_auc=window_auc,
            image_auc=image_auc,
            confusion_matrix=cm,
            classification_report=clf_report,
            optimal_threshold=threshold,
            predictions=img_agg,
            pr_auc=pr_auc
        )
        
        if self.config.verbose:
            print(f"\n{'#'*60}")
            print("Evaluation Results")
            print(f"{'#'*60}")
            print(f"Window AUC: {window_auc:.3f}")
            print(f"Image AUC:  {image_auc:.3f}")
            print(f"PR AUC:     {pr_auc:.3f}")
            print(f"\nConfusion Matrix (threshold={threshold:.3f}):")
            print(f"              Predicted")
            print(f"            Real  Watermarked")
            print(f"Actual Real    {cm[0,0]:3d}      {cm[0,1]:3d}")
            print(f"     Watermark {cm[1,0]:3d}      {cm[1,1]:3d}")
            print(f"\n{clf_report}")
            print(f"{'#'*60}")
        
        return results
    
    
    #################################################################################################
    # MODEL PERSISTENCE
    #################################################################################################
    
    def save(self, output_dir: Union[str, Path], name: str = 'classifier') -> Path:
        """
        Save trained model to disk to use on new holdout images
        
        Args:
            output_dir: Output directory path.
            name: Base name for saved files.
        
        Returns:
            Path to saved model file.
        
        Raises:
            ValueError: If model has not been trained.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / f'{name}.joblib'
        
        # Packimg all model components
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'optimal_threshold': self.optimal_threshold,
            'config': asdict(self.config),
            'training_timestamp': self.training_timestamp,
            'cv_results': {
                'mean_image_auc': self.cv_results.mean_image_auc if self.cv_results else None,
                'std_image_auc': self.cv_results.std_image_auc if self.cv_results else None,
                'mean_optimal_threshold': self.cv_results.mean_optimal_threshold if self.cv_results else None,
            } if self.cv_results else None
        }
        
        joblib.dump(model_data, model_path)
        
        if self.config.verbose:
            print(f"\nModel saved: {model_path}")
        
        return model_path
    
    
    @classmethod
    def load(cls, model_path: Union[str, Path]) -> 'WatermarkClassifier':
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to saved model file.
        
        Returns:
            Loaded WatermarkClassifier instance ready for inference.
        """
        model_data = joblib.load(model_path)
        
        # Reconstructing config from saved data
        config_dict = model_data.get('config', {})
        config = ClassifierConfig(**config_dict)
        
        # Making instance and restore state
        classifier = cls(config)
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        classifier.feature_names = model_data['feature_names']
        classifier.optimal_threshold = model_data.get('optimal_threshold', 0.5)
        classifier.training_timestamp = model_data.get('training_timestamp')
        
        if classifier.config.verbose:
            print(f"Model loaded from {model_path}")
            print(f"  Features: {len(classifier.feature_names)}")
            print(f"  Threshold: {classifier.optimal_threshold:.3f}")
        
            if model_data.get('cv_results'):
                cv = model_data['cv_results']
                if cv['mean_image_auc']:
                    print(f"  CV AUC: {cv['mean_image_auc']:.3f} +/- {cv['std_image_auc']:.3f}")

        
        return classifier
    
    
    #################################################################################################
    # VISUALISATION
    #################################################################################################
    
    def visualise_cv_results(self, output_dir: Union[str, Path]):
        """
        Generate comprehensive visualisations of cross-validation results.
        
        Args:
            output_dir: Output directory for plot files.
        
        Raises:
            ValueError: If cross-validation has not been run.
        """
        if self.cv_results is None:
            raise ValueError("No CV results available. Run cross_validate() first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cv = self.cv_results
        
        self._plot_roc_curve(cv, output_dir / 'cv_roc_curve.png')
        self._plot_feature_importance(cv, output_dir / 'feature_importance.png')
        self._plot_fold_performance(cv, output_dir / 'fold_performance.png')
        self._analyse_feature_importance(cv, output_dir)
        
        if self.config.verbose:
            print(f"\nVisualisations saved to {output_dir}/")
    
    
    def _plot_roc_curve(self, cv: CVResults, save_path: Path):
        """Plot ROC curve with individual folds and mean."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plots individual fold curves
        for i, tpr in enumerate(cv.tprs):
            ax.plot(cv.base_fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i+1}')
        
        # Computes and plots mean ROC
        mean_tpr = np.mean(cv.tprs, axis=0)
        mean_tpr[-1] = 1.0  # ensure curve ends at (1, 1)
        
        # AUC of mean curve here
        mean_auc = auc(cv.base_fpr, mean_tpr)
        std_auc = cv.std_image_auc
        
        ax.plot(
            cv.base_fpr, mean_tpr,
            label=f'Mean ROC (AUC = {mean_auc:.3f} +/- {std_auc:.3f})',
            lw=3, color='navy'
        )
        
        # Diagonal reference line (random classifier ref)
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.3)
        
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Cross-Validation ROC Curves (Image-Level)', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    
    def _plot_feature_importance(self, cv: CVResults, save_path: Path):
        """Plot horizontal bar chart of feature importance (coefficients)."""
        # Dataframe sorted by absolute importance
        imp_df = pd.DataFrame({
            'feature': cv.feature_names,
            'coefficient': cv.avg_feature_importance,
            'abs_coefficient': np.abs(cv.avg_feature_importance)
        }).sort_values('abs_coefficient', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Colour bars by sign: red for negative, blue for positive
        colours = ['red' if x < 0 else 'blue' for x in imp_df['coefficient']]
        
        y_pos = np.arange(len(imp_df))
        ax.barh(y_pos, imp_df['coefficient'], color=colours, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(imp_df['feature'], fontsize=10)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Coefficient', fontsize=12)
        ax.set_title('Feature Importance (Logistic Regression Coefficients)', 
                     fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        ax.invert_yaxis()  # highest importance at top
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # also save as CSV
        csv_path = save_path.with_suffix('.csv')
        imp_df.to_csv(csv_path, index=False)
    
    
    def _plot_fold_performance(self, cv: CVResults, save_path: Path):
        """Plot performance metrics across folds."""
        fold_nums = [f.fold for f in cv.fold_results]
        window_aucs = [f.window_auc for f in cv.fold_results]
        image_aucs = [f.image_auc for f in cv.fold_results]
        thresholds = [f.optimal_threshold for f in cv.fold_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Window AUC across folds
        axes[0, 0].plot(fold_nums, window_aucs, 'o-', linewidth=2, markersize=8)
        axes[0, 0].axhline(cv.mean_window_auc, color='red', linestyle='--', 
                           label=f'Mean: {cv.mean_window_auc:.3f}')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('Window AUC')
        axes[0, 0].set_title('Window-Level AUC')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Image AUC across folds
        axes[0, 1].plot(fold_nums, image_aucs, 'o-', linewidth=2, markersize=8, color='green')
        axes[0, 1].axhline(cv.mean_image_auc, color='red', linestyle='--',
                           label=f'Mean: {cv.mean_image_auc:.3f}')
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('Image AUC')
        axes[0, 1].set_title('Image-Level AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Optimal threshold across folds
        axes[1, 0].plot(fold_nums, thresholds, 'o-', linewidth=2, markersize=8, color='purple')
        axes[1, 0].axhline(cv.mean_optimal_threshold, color='red', linestyle='--',
                           label=f'Mean: {cv.mean_optimal_threshold:.3f}')
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('Optimal Threshold')
        axes[1, 0].set_title('Optimal Classification Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Histogram of image AUCs
        axes[1, 1].hist(image_aucs, bins=5, edgecolor='black', alpha=0.7, color='green')
        axes[1, 1].axvline(cv.mean_image_auc, color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {cv.mean_image_auc:.3f}')
        axes[1, 1].set_xlabel('Image AUC')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Distribution of Image AUC Across Folds')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    
    def _analyse_feature_importance(self, cv: CVResults, output_dir: Path):
        """Perform statistical analysis of feature importance across folds."""
        # Collect coefficients from all folds: shape (n_folds, n_features)
        importance_matrix = np.array([f.feature_importance for f in cv.fold_results])
        
        # Compute statistics across folds
        mean_imp = np.mean(importance_matrix, axis=0)
        std_imp = np.std(importance_matrix, axis=0)
        
        # One-sample t-test: is each coefficient significantly different from zero?
        t_stats = []
        p_values = []
        for i in range(importance_matrix.shape[1]):
            # t-test against null hypothesis that mean = 0
            t, p = stats.ttest_1samp(importance_matrix[:, i], 0)
            t_stats.append(t)
            p_values.append(p)
        
        # Create analysis dataframe
        analysis_df = pd.DataFrame({
            'feature': cv.feature_names,
            'mean_coefficient': mean_imp,
            'std_coefficient': std_imp,
            'abs_mean_coefficient': np.abs(mean_imp),
            't_statistic': t_stats,
            'p_value': p_values,
            'significant': np.array(p_values) < 0.05  # significance at alpha=0.05
        }).sort_values('abs_mean_coefficient', ascending=False)
        
        # Save to CSV
        analysis_path = output_dir / 'feature_importance_analysis.csv'
        analysis_df.to_csv(analysis_path, index=False)
        
        if self.config.verbose:
            n_significant = analysis_df['significant'].sum()
            print(f"\nFeature Importance Analysis:")
            print(f"  Significant features (p<0.05): {n_significant}/{len(analysis_df)}")
            print(f"\nTop 5 Most Important Features:")
            print(analysis_df.head(5)[['feature', 'mean_coefficient', 'p_value']].to_string(index=False))
    
    
    def visualise_evaluation(self, results: EvaluationResults, output_dir: Union[str, Path]):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        labels = results.predictions['label'].values
        if len(np.unique(labels)) >= 2:
            self._plot_eval_curves(results, output_dir / 'eval_curves.png')
        else:
            if self.config.verbose:
                print("Skipping ROC / PR plots: evaluation set has a single class.")

        self._plot_confusion_matrix(results, output_dir / 'confusion_matrix.png')
        self._plot_prediction_distribution(results, output_dir / 'prediction_distribution.png')

    
    
    def _plot_eval_curves(self, results: EvaluationResults, save_path: Path):
        """Plot ROC and Precision-Recall curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        y_true = results.predictions['label'].values
        y_scores = results.predictions['pred_proba'].values

        # If only one class present, curves are undefined
        if len(np.unique(y_true)) < 2:
            if self.config.verbose:
                print("Skipping ROC/PR plots: evaluation data contains only one class.")
            
            titles = [
                "ROC Curve (undefined: single class in data)",
                "Precision-Recall (undefined: single class in data)"
            ]
            for ax, title in zip(axes, titles):
                ax.text(0.5, 0.5, title, ha='center', va='center')
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_label = (
            f'AUC = {results.image_auc:.3f}'
            if not np.isnan(results.image_auc)
            else 'AUC = N/A'
        )
        axes[0].plot(fpr, tpr, linewidth=2, label=roc_label, color='navy')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3)
        axes[0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0].set_ylabel('True Positive Rate', fontsize=12)
        axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right')
        axes[0].grid(alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_label = (
            f'PR AUC = {results.pr_auc:.3f}'
            if not np.isnan(results.pr_auc)
            else 'PR AUC = N/A'
        )
        axes[1].plot(recall, precision, linewidth=2, 
                     label=pr_label, color='green')
        axes[1].set_xlabel('Recall', fontsize=12)
        axes[1].set_ylabel('Precision', fontsize=12)
        axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1].legend(loc='lower left')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    
    
    def _plot_confusion_matrix(self, results: EvaluationResults, save_path: Path):
        """Plot confusion matrix as heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            results.confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Real', 'Watermarked'],
            yticklabels=['Real', 'Watermarked'],
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix (threshold={results.optimal_threshold:.3f})',
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    
    def _plot_prediction_distribution(self, results: EvaluationResults, save_path: Path):
        """Plot histogram of predicted probabilities by class."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate predictions by true label
        real_preds = results.predictions[results.predictions['label'] == 0]['pred_proba']
        wm_preds = results.predictions[results.predictions['label'] == 1]['pred_proba']
        
        # Fixed bins spanning 0-1 to prevent ValueError with zero-variance data
        fixed_bins = np.linspace(0, 1, 21)  # 20 bins
        
        if len(real_preds) > 0:
            ax.hist(real_preds, bins=fixed_bins, alpha=0.6, 
                    label=f'Real Images (n={len(real_preds)})', 
                    color='blue', edgecolor='black')
        
        if len(wm_preds) > 0:
            ax.hist(wm_preds, bins=fixed_bins, alpha=0.6, 
                    label=f'Watermarked Images (n={len(wm_preds)})',
                    color='red', edgecolor='black')
        
        # Threshold line
        ax.axvline(results.optimal_threshold, color='black', linestyle='--',
                   linewidth=2, label=f'Threshold = {results.optimal_threshold:.3f}')
        
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
        ax.legend(loc='upper center')
        ax.grid(alpha=0.3, axis='y')
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


##################################################################################################
#  FEATURE GENERATION (if run for the first time)
##################################################################################################

def generate_features_from_images(
    positive_dir: Union[str, Path],
    negative_dir: Union[str, Path],
    output_csv: Union[str, Path],
    channel: str = 'a_lab',
    grid_size: Tuple[int, int] = (4, 4),
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Generate features from image directories using pipeline module
    
    Args:
        positive_dir: Directory containing watermarked images
        negative_dir: Directory containing real images
        output_csv: Output CSV path
        channel: Colour channel to extract features from
        grid_size: Window grid dimensions (rows, cols)
        n_jobs: Number of parallel workers
    
    Returns:
        DataFrame with extracted features from all images
    """
    import tempfile
    
    # Importing the pipeline
    try:
        import parallelised_feature_pipeline as feature_pipeline
    except ImportError:
        raise ImportError(
            "Could not import 'parallelised_feature_pipeline'. "
            "Ensure the file is in the same directory and contains the "
            "'extract_features_to_csv' function."
        )

    output_csv = Path(output_csv)
    
    print(f"\n{'#'*60}")
    print("Generating Features from Images (Module Import)")
    print(f"{'#'*60}")
    
    # Using tempfile for safer temporary file handling
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        temp_pos = temp_dir / 'positive_features.csv'
        temp_neg = temp_dir / 'negative_features.csv'
        
        # Extracting watermarked images 
        print("\nProcessing watermarked images...")
        feature_pipeline.extract_features_to_csv(
            input_folder=positive_dir,
            output_path=temp_pos,
            channel=channel,
            label_name='watermarked',
            grid_rows=grid_size[0],
            grid_cols=grid_size[1],
            n_jobs=n_jobs
        )
        
        # Features from real images
        print("\nProcessing real images...")
        feature_pipeline.extract_features_to_csv(
            input_folder=negative_dir,
            output_path=temp_neg,
            channel=channel,
            label_name='real',
            grid_rows=grid_size[0],
            grid_cols=grid_size[1],
            n_jobs=n_jobs
        )
        
        # Combining results
        print("\nCombining features...")
        if not temp_pos.exists() or not temp_neg.exists():
            raise FileNotFoundError("Feature pipeline did not generate expected output files.")

        df_pos = pd.read_csv(temp_pos)
        df_neg = pd.read_csv(temp_neg)
        
        # Map labels to binary: watermarked=1, real=0
        df_pos['label'] = 1
        df_neg['label'] = 0
        
        df_combined = pd.concat([df_pos, df_neg], ignore_index=True)
        df_combined.to_csv(output_csv, index=False)
        
        print(f"Combined features saved: {output_csv}")
        print(f"  Total windows: {len(df_combined)}")
        
        # Temp directory and files automatically cleaned up when exiting 'with' block
        return df_combined


##################################################################################################
# COMMAND LINE INTERFACE
##################################################################################################

def main():
    parser = argparse.ArgumentParser(
        description='Production Watermark Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from existing CSV
  python balanced_classifier.py \\
      --input features.csv \\
      --output model_output \\
      --mode train
  
  # Train from images (generates features)
  python balanced_classifier.py \\
      --positive_dir ./images/watermarked \\
      --negative_dir ./images/real \\
      --output model_output \\
      --mode train_from_images
  
  # Evaluate on holdout set
  python balanced_classifier.py \\
      --model model_output/classifier.joblib \\
      --holdout holdout_features.csv \\
      --output holdout_results \\
      --mode evaluate
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'train_from_images', 'evaluate'],
                        help='Operation mode')
    
    # Input/output arguments
    parser.add_argument('--input', type=str,
                        help='Input features CSV (for train mode)')
    parser.add_argument('--positive_dir', type=str,
                        help='Watermarked images directory (for train_from_images)')
    parser.add_argument('--negative_dir', type=str,
                        help='Real images directory (for train_from_images)')
    parser.add_argument('--holdout', type=str,
                        help='Holdout features CSV (for evaluate mode)')
    parser.add_argument('--model', type=str,
                        help='Trained model path (for evaluate mode)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    
    # Feature extraction arguments (for train_from_images)
    parser.add_argument('--channel', type=str, default='a_lab',
                        help='Colour channel for feature extraction (default: a_lab)')
    parser.add_argument('--grid_rows', type=int, default=4,
                        help='Grid rows (default: 4)')
    parser.add_argument('--grid_cols', type=int, default=4,
                        help='Grid columns (default: 4)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Parallel workers for feature extraction (default: -1)')
    
    # Model configuration arguments
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Logistic regression regularisation (default: 1.0)')
    parser.add_argument('--threshold_metric', type=str, default='f1',
                        choices=['f1', 'youden', 'precision_recall'],
                        help='Threshold optimisation metric (default: f1)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--positive_label', type=str, default='watermarked',
                        help='String label treated as positive class (1). Default: watermarked')
    parser.add_argument('--optimise_threshold', action='store_true',
                        help='Use cross-validated optimal threshold for inference')
    parser.add_argument('--max_iter', type=int, default=1000,
                    help='Maximum iterations for solver (default: 1000)')
    parser.add_argument('--solver', type=str, default='lbfgs',
                    choices=['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
                    help='Solver for logistic regression (default: lbfgs)')

    
    # Output control arguments
    parser.add_argument('--no_plots', action='store_true',
                        help='Disable plot generation')
    parser.add_argument('--no_logs', action='store_true',
                        help='Disable log file generation')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = ClassifierConfig(
        positive_label=args.positive_label,
        n_splits=args.n_splits,
        C=args.C,
        max_iter=args.max_iter,
        solver=args.solver,
        random_state=args.random_seed,
        optimise_threshold=args.optimise_threshold,
        optimisation_metric=args.threshold_metric,
        save_plots=not args.no_plots,
        save_logs=not args.no_logs,
        verbose=not args.quiet
    )

    
    
    ##################################################################################################
    # MODE: TRAIN
    ##################################################################################################
    
    if args.mode == 'train':
        if not args.input:
            parser.error("--input required for train mode")
        
        classifier = WatermarkClassifier(config)
        df = classifier.load_features(args.input)
        classifier.train(df, run_cv=True)
        model_path = classifier.save(output_dir)
        
        if config.save_plots:
            classifier.visualise_cv_results(output_dir)
        
        if config.save_logs and classifier.cv_results:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'mode': 'train',
                'input_file': str(args.input),
                'config': asdict(config),
                'results': {
                    'mean_image_auc': classifier.cv_results.mean_image_auc,
                    'std_image_auc': classifier.cv_results.std_image_auc,
                    'mean_window_auc': classifier.cv_results.mean_window_auc,
                    'std_window_auc': classifier.cv_results.std_window_auc,
                    'mean_optimal_threshold': classifier.cv_results.mean_optimal_threshold,
                }
            }
            
            log_path = output_dir / 'training_log.json'
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"Training log saved: {log_path}")
    
    
    ##################################################################################################
    # MODE: TRAIN FROM IMAGES
    ##################################################################################################
    
    elif args.mode == 'train_from_images':
        if not args.positive_dir or not args.negative_dir:
            parser.error("--positive_dir and --negative_dir required for train_from_images mode")
        
        # Generating features from image directories
        features_csv = output_dir / 'extracted_features.csv'
        df = generate_features_from_images(
            positive_dir=args.positive_dir,
            negative_dir=args.negative_dir,
            output_csv=features_csv,
            channel=args.channel,
            grid_size=(args.grid_rows, args.grid_cols),
            n_jobs=args.n_jobs
        )
        
        classifier = WatermarkClassifier(config)
        classifier.train(df, run_cv=True)
        model_path = classifier.save(output_dir)
        
        if config.save_plots:
            classifier.visualise_cv_results(output_dir)
        
        if config.save_logs and classifier.cv_results:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'mode': 'train_from_images',
                'positive_dir': str(args.positive_dir),
                'negative_dir': str(args.negative_dir),
                'channel': args.channel,
                'grid_size': [args.grid_rows, args.grid_cols],
                'config': asdict(config),
                'results': {
                    'mean_image_auc': classifier.cv_results.mean_image_auc,
                    'std_image_auc': classifier.cv_results.std_image_auc,
                    'optimal_threshold': classifier.cv_results.mean_optimal_threshold,
                }
            }
            
            log_path = output_dir / 'training_log.json'
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"Training log saved: {log_path}")
    
    
    ##################################################################################################
    # MODE: EVALUATE
    ##################################################################################################
    
    elif args.mode == 'evaluate':
        if not args.model or not args.holdout:
            parser.error("--model and --holdout required for evaluate mode")
        
        classifier = WatermarkClassifier.load(args.model)

        # Override runtime behaviour with CLI flags
        classifier.config.verbose = not args.quiet
        classifier.config.save_plots = not args.no_plots
        classifier.config.save_logs = not args.no_logs

        df_holdout = classifier.load_features(args.holdout)
        results = classifier.evaluate(df_holdout)
        
        if classifier.config.save_plots:
            classifier.visualise_evaluation(results, output_dir)
        
        # Saving predictions
        predictions_path = output_dir / 'predictions.csv'
        results.predictions.to_csv(predictions_path, index=False)
        if classifier.config.verbose:
            print(f"Predictions saved: {predictions_path}")
        
        if classifier.config.save_logs:
            metrics = results.get_metrics_dict()
            
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'mode': 'evaluate',
                'model_path': str(args.model),
                'holdout_file': str(args.holdout),
                'results': {
                    'image_auc': results.image_auc,
                    'window_auc': results.window_auc,
                    'pr_auc': results.pr_auc,
                    'threshold': results.optimal_threshold,
                    'metrics': metrics,
                }
            }
            
            log_path = output_dir / 'evaluation_log.json'
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            if classifier.config.verbose:
                print(f"Evaluation log saved: {log_path}")



if __name__ == '__main__':
    main()