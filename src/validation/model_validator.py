import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import (
    cross_val_score, cross_validate, KFold, StratifiedKFold,
    TimeSeriesSplit, GroupKFold, LeaveOneOut, RepeatedKFold
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    explained_variance_score, max_error
)
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy import stats
import mlflow
import mlflow.sklearn
from loguru import logger
import json
import warnings
warnings.filterwarnings('ignore')


class ModelValidator:
    """Advanced model validation framework with comprehensive cross-validation and model comparison."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model validator.

        Args:
            config: Configuration dictionary containing validation settings
        """
        self.config = config
        self.cv_strategies = {
            'kfold': KFold,
            'stratified': StratifiedKFold,
            'timeseries': TimeSeriesSplit,
            'group': GroupKFold,
            'loo': LeaveOneOut,
            'repeated': RepeatedKFold
        }

        self.models = {
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'rf': RandomForestRegressor,
            'gb': GradientBoostingRegressor,
            'svr': SVR,
            'knn': KNeighborsRegressor,
            'dt': DecisionTreeRegressor,
            'xgb': xgb.XGBRegressor,
            'lgb': lgb.LGBMRegressor,
            'cat': CatBoostRegressor
        }

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.get('logging', {}).get('level', 'INFO')),
            format=self.config.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

    def get_cv_strategy(self, strategy_name: str, **kwargs) -> object:
        """
        Get cross-validation strategy instance.

        Args:
            strategy_name: Name of the CV strategy
            **kwargs: Additional parameters for the CV strategy

        Returns:
            CV strategy instance
        """
        if strategy_name not in self.cv_strategies:
            raise ValueError(f"Unknown CV strategy: {strategy_name}")

        strategy_class = self.cv_strategies[strategy_name]

        # Set default parameters based on strategy
        if strategy_name == 'kfold':
            kwargs.setdefault('n_splits', 5)
            kwargs.setdefault('shuffle', True)
            kwargs.setdefault('random_state', 42)
        elif strategy_name == 'stratified':
            kwargs.setdefault('n_splits', 5)
            kwargs.setdefault('shuffle', True)
            kwargs.setdefault('random_state', 42)
        elif strategy_name == 'timeseries':
            kwargs.setdefault('n_splits', 5)
        elif strategy_name == 'repeated':
            kwargs.setdefault('n_splits', 5)
            kwargs.setdefault('n_repeats', 3)
            kwargs.setdefault('random_state', 42)

        return strategy_class(**kwargs)

    def perform_cross_validation(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv_strategy: str = 'kfold',
        scoring: List[str] = None,
        return_train_score: bool = True,
        **cv_kwargs
    ) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation.

        Args:
            model: Model to validate
            X: Features
            y: Target
            cv_strategy: Cross-validation strategy
            scoring: List of scoring metrics
            return_train_score: Whether to return training scores
            **cv_kwargs: Additional CV parameters

        Returns:
            Dictionary with CV results
        """
        try:
            logger.info(f"Performing cross-validation with {cv_strategy} strategy")

            if scoring is None:
                scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

            cv = self.get_cv_strategy(cv_strategy, **cv_kwargs)

            # Perform cross-validation
            cv_results = cross_validate(
                model, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=return_train_score,
                n_jobs=-1
            )

            # Calculate additional metrics
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)

            # Process results
            results = {
                'cv_strategy': cv_strategy,
                'n_splits': cv.get_n_splits(X, y),
                'test_scores': {},
                'train_scores': {} if return_train_score else None
            }

            for scorer in scoring:
                if scorer.startswith('neg_'):
                    # Convert negative scores to positive
                    metric_name = scorer[4:]  # Remove 'neg_' prefix
                    test_key = f"test_{scorer}"
                    results['test_scores'][metric_name] = {
                        'mean': -cv_results[test_key].mean(),
                        'std': cv_results[test_key].std(),
                        'scores': -cv_results[test_key]
                    }
                    if return_train_score:
                        train_key = f"train_{scorer}"
                        results['train_scores'][metric_name] = {
                            'mean': -cv_results[train_key].mean(),
                            'std': cv_results[train_key].std(),
                            'scores': -cv_results[train_key]
                        }
                else:
                    test_key = f"test_{scorer}"
                    results['test_scores'][scorer] = {
                        'mean': cv_results[test_key].mean(),
                        'std': cv_results[test_key].std(),
                        'scores': cv_results[test_key]
                    }
                    if return_train_score:
                        train_key = f"train_{scorer}"
                        results['train_scores'][scorer] = {
                            'mean': cv_results[train_key].mean(),
                            'std': cv_results[train_key].std(),
                            'scores': cv_results[train_key]
                        }

            # Additional statistics
            results['fit_time'] = {
                'mean': cv_results['fit_time'].mean(),
                'std': cv_results['fit_time'].std()
            }
            results['score_time'] = {
                'mean': cv_results['score_time'].mean(),
                'std': cv_results['score_time'].std()
            }

            # Confidence intervals
            confidence_level = 0.95
            for metric in results['test_scores']:
                scores = results['test_scores'][metric]['scores']
                mean = np.mean(scores)
                sem = stats.sem(scores)
                ci = stats.t.interval(confidence_level, len(scores)-1, mean, sem)
                results['test_scores'][metric]['ci_lower'] = ci[0]
                results['test_scores'][metric]['ci_upper'] = ci[1]

            logger.info(f"Cross-validation completed. Mean R2: {results['test_scores']['r2']['mean']:.4f}")
            return results

        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise

    def compare_models(
        self,
        models: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        cv_strategy: str = 'kfold',
        **cv_kwargs
    ) -> Dict[str, Any]:
        """
        Compare multiple models using cross-validation.

        Args:
            models: List of model names to compare
            X: Features
            y: Target
            cv_strategy: Cross-validation strategy
            **cv_kwargs: Additional CV parameters

        Returns:
            Dictionary with model comparison results
        """
        try:
            logger.info(f"Comparing {len(models)} models")

            comparison_results = {}
            best_score = -np.inf
            best_model = None

            for model_name in models:
                if model_name not in self.models:
                    logger.warning(f"Unknown model: {model_name}. Skipping.")
                    continue

                logger.info(f"Evaluating {model_name}")

                # Initialize model with default parameters
                model_class = self.models[model_name]
                model = model_class()

                # Perform cross-validation
                cv_results = self.perform_cross_validation(
                    model, X, y, cv_strategy, **cv_kwargs
                )

                comparison_results[model_name] = cv_results

                # Track best model
                r2_score = cv_results['test_scores']['r2']['mean']
                if r2_score > best_score:
                    best_score = r2_score
                    best_model = model_name

                # Log to MLflow
                with mlflow.start_run(run_name=f"model_comparison_{model_name}", nested=True):
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("cv_strategy", cv_strategy)
                    mlflow.log_metric("cv_r2_mean", cv_results['test_scores']['r2']['mean'])
                    mlflow.log_metric("cv_r2_std", cv_results['test_scores']['r2']['std'])
                    mlflow.log_metric("cv_mae_mean", cv_results['test_scores']['mean_absolute_error']['mean'])
                    mlflow.log_metric("cv_rmse_mean", cv_results['test_scores']['mean_squared_error']['mean'])

            comparison_results['best_model'] = best_model
            comparison_results['best_score'] = best_score

            logger.info(f"Model comparison completed. Best model: {best_model} (R2: {best_score:.4f})")
            return comparison_results

        except Exception as e:
            logger.error(f"Error in model comparison: {str(e)}")
            raise

    def statistical_significance_test(
        self,
        model1_results: Dict[str, Any],
        model2_results: Dict[str, Any],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical significance test between two models.

        Args:
            model1_results: CV results for model 1
            model2_results: CV results for model 2
            alpha: Significance level

        Returns:
            Dictionary with statistical test results
        """
        try:
            logger.info("Performing statistical significance test")

            results = {}

            for metric in ['r2', 'mean_absolute_error', 'mean_squared_error']:
                scores1 = model1_results['test_scores'][metric]['scores']
                scores2 = model2_results['test_scores'][metric]['scores']

                # Perform t-test
                t_stat, p_value = stats.ttest_rel(scores1, scores2)

                # Determine if difference is significant
                significant = p_value < alpha

                results[metric] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': significant,
                    'alpha': alpha,
                    'model1_mean': np.mean(scores1),
                    'model2_mean': np.mean(scores2),
                    'difference': np.mean(scores1) - np.mean(scores2)
                }

            return results

        except Exception as e:
            logger.error(f"Error in statistical significance test: {str(e)}")
            raise

    def learning_curves_analysis(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        train_sizes: np.ndarray = None,
        cv_strategy: str = 'kfold',
        **cv_kwargs
    ) -> Dict[str, Any]:
        """
        Analyze learning curves for model diagnostics.

        Args:
            model: Model to analyze
            X: Features
            y: Target
            train_sizes: Array of training set sizes
            cv_strategy: Cross-validation strategy
            **cv_kwargs: Additional CV parameters

        Returns:
            Dictionary with learning curve results
        """
        try:
            logger.info("Analyzing learning curves")

            from sklearn.model_selection import learning_curve

            if train_sizes is None:
                train_sizes = np.linspace(0.1, 1.0, 10)

            cv = self.get_cv_strategy(cv_strategy, **cv_kwargs)

            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=cv,
                scoring='r2',
                n_jobs=-1
            )

            results = {
                'train_sizes': train_sizes_abs,
                'train_scores': {
                    'mean': np.mean(train_scores, axis=1),
                    'std': np.std(train_scores, axis=1),
                    'all': train_scores
                },
                'validation_scores': {
                    'mean': np.mean(val_scores, axis=1),
                    'std': np.std(val_scores, axis=1),
                    'all': val_scores
                }
            }

            # Check for overfitting/underfitting
            final_train_score = results['train_scores']['mean'][-1]
            final_val_score = results['validation_scores']['mean'][-1]

            convergence_threshold = 0.05
            converged = abs(final_train_score - final_val_score) < convergence_threshold

            results['diagnostics'] = {
                'final_train_score': final_train_score,
                'final_val_score': final_val_score,
                'score_gap': abs(final_train_score - final_val_score),
                'converged': converged,
                'overfitting': final_train_score > final_val_score + convergence_threshold,
                'underfitting': final_val_score < 0.7  # Arbitrary threshold
            }

            logger.info(f"Learning curves analysis completed. Converged: {converged}")
            return results

        except Exception as e:
            logger.error(f"Error in learning curves analysis: {str(e)}")
            raise

    def residual_analysis(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv_strategy: str = 'kfold',
        **cv_kwargs
    ) -> Dict[str, Any]:
        """
        Perform residual analysis for model diagnostics.

        Args:
            model: Trained model
            X: Features
            y: Target
            cv_strategy: Cross-validation strategy
            **cv_kwargs: Additional CV parameters

        Returns:
            Dictionary with residual analysis results
        """
        try:
            logger.info("Performing residual analysis")

            from sklearn.model_selection import cross_val_predict

            cv = self.get_cv_strategy(cv_strategy, **cv_kwargs)

            # Get cross-validated predictions
            y_pred = cross_val_predict(model, X, y, cv=cv)

            residuals = y - y_pred

            # Statistical tests
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            normal_distribution = shapiro_p > 0.05

            # Heteroscedasticity test (Breusch-Pagan)
            try:
                from statsmodels.stats.diagnostic import het_breuschpagan
                from statsmodels.regression.linear_model import OLS
                import statsmodels.api as sm

                X_with_const = sm.add_constant(X)
                ols_model = OLS(y, X_with_const).fit()
                bp_test = het_breuschpagan(ols_model.resid, X_with_const)
                heteroscedasticity_p = bp_test[1]
                homoscedastic = heteroscedasticity_p > 0.05
            except:
                heteroscedasticity_p = None
                homoscedastic = None

            results = {
                'residuals': residuals,
                'predictions': y_pred,
                'normality_test': {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'normal_distribution': normal_distribution
                },
                'heteroscedasticity_test': {
                    'p_value': heteroscedasticity_p,
                    'homoscedastic': homoscedastic
                },
                'residual_stats': {
                    'mean': np.mean(residuals),
                    'std': np.std(residuals),
                    'min': np.min(residuals),
                    'max': np.max(residuals),
                    'skewness': stats.skew(residuals),
                    'kurtosis': stats.kurtosis(residuals)
                }
            }

            logger.info(f"Residual analysis completed. Normal distribution: {normal_distribution}")
            return results

        except Exception as e:
            logger.error(f"Error in residual analysis: {str(e)}")
            raise

    def save_validation_report(self, results: Dict[str, Any], path: str) -> None:
        """
        Save validation results to file.

        Args:
            results: Validation results
            path: Path to save results
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Convert numpy arrays to lists for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj

            serializable_results = convert_to_serializable(results)

            with open(path, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"Validation report saved to {path}")

        except Exception as e:
            logger.error(f"Error saving validation report: {str(e)}")
            raise

    def log_to_mlflow(self, results: Dict[str, Any], run_name: str = None) -> None:
        """
        Log validation results to MLflow.

        Args:
            results: Validation results
            run_name: Name for the MLflow run
        """
        try:
            with mlflow.start_run(run_name=run_name):
                # Log key metrics
                if 'test_scores' in results:
                    for metric, values in results['test_scores'].items():
                        if isinstance(values, dict) and 'mean' in values:
                            mlflow.log_metric(f"cv_{metric}_mean", values['mean'])
                            mlflow.log_metric(f"cv_{metric}_std", values['std'])

                # Log diagnostics if available
                if 'diagnostics' in results:
                    for key, value in results['diagnostics'].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"diagnostic_{key}", value)
                        else:
                            mlflow.log_param(f"diagnostic_{key}", value)

                # Log best model if available
                if 'best_model' in results:
                    mlflow.log_param("best_model", results['best_model'])
                    mlflow.log_metric("best_score", results['best_score'])

            logger.info("Validation results logged to MLflow")

        except Exception as e:
            logger.error(f"Error logging to MLflow: {str(e)}")
            raise
