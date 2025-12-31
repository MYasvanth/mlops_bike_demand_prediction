import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class ModelExplainer:
    """Comprehensive model explainability and interpretability toolkit."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model explainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.shap_explainer = None
        self.lime_explainer = None
        self.is_initialized = False

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.get('logging', {}).get('level', 'INFO')),
            format=self.config.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

    def initialize_explainer(self, model, X_train: pd.DataFrame, X_background: Optional[pd.DataFrame] = None) -> None:
        """
        Initialize SHAP and LIME explainers.

        Args:
            model: Trained model
            X_train: Training data for initialization
            X_background: Background data for SHAP (optional, defaults to X_train sample)
        """
        try:
            logger.info("Initializing model explainers")

            # Initialize SHAP explainer
            if X_background is None:
                # Use a sample of training data as background
                background_size = min(100, len(X_train))
                X_background = X_train.sample(n=background_size, random_state=42)

            self.shap_explainer = shap.TreeExplainer(model, X_background)
            logger.info("SHAP TreeExplainer initialized")

            # Initialize LIME explainer
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=X_train.columns.tolist(),
                class_names=['bike_demand'],
                mode='regression',
                random_state=42
            )
            logger.info("LIME TabularExplainer initialized")

            self.is_initialized = True
            logger.info("Model explainers initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing explainers: {str(e)}")
            raise

    def explain_prediction_shap(self, X_instance: pd.DataFrame, max_evals: int = 1000) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP.

        Args:
            X_instance: Single instance to explain
            max_evals: Maximum evaluations for SHAP

        Returns:
            Dictionary with SHAP explanation results
        """
        if not self.is_initialized:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")

        try:
            logger.info("Generating SHAP explanation for prediction")

            # Calculate SHAP values (handle different SHAP versions)
            try:
                # Try with max_evals for newer SHAP versions
                shap_values = self.shap_explainer(X_instance, max_evals=max_evals)
            except TypeError:
                # Fallback for older SHAP versions without max_evals
                logger.info("Using SHAP without max_evals parameter (older version)")
                shap_values = self.shap_explainer(X_instance)

            # Extract values for the single instance
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) > 1:
                    shap_vals = shap_values.values[0]
                else:
                    shap_vals = shap_values.values
            else:
                shap_vals = shap_values[0]

            # Get base value
            base_value = float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') else 0.0

            # Create explanation dictionary
            explanation = {
                'shap_values': shap_vals.tolist(),
                'base_value': base_value,
                'feature_names': X_instance.columns.tolist(),
                'feature_values': X_instance.iloc[0].tolist(),
                'expected_value': base_value
            }

            # Calculate prediction
            explanation['prediction'] = base_value + sum(shap_vals)

            logger.info("SHAP explanation generated successfully")
            return explanation

        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {str(e)}")
            raise

    def explain_prediction_lime(self, X_instance: pd.DataFrame, num_features: int = 10) -> Dict[str, Any]:
        """
        Explain a single prediction using LIME.

        Args:
            X_instance: Single instance to explain
            num_features: Number of features to include in explanation

        Returns:
            Dictionary with LIME explanation results
        """
        if not self.is_initialized:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")

        try:
            logger.info("Generating LIME explanation for prediction")

            # Generate LIME explanation
            exp = self.lime_explainer.explain_instance(
                data_row=X_instance.iloc[0].values,
                predict_fn=lambda x: self.shap_explainer.model.predict(pd.DataFrame(x, columns=X_instance.columns)),
                num_features=num_features
            )

            # Extract explanation data
            explanation = {
                'intercept': exp.intercept[0],
                'prediction': exp.predicted_value,
                'feature_importance': dict(exp.as_list()),
                'feature_names': [feat[0] for feat in exp.as_list()],
                'feature_weights': [feat[1] for feat in exp.as_list()]
            }

            logger.info("LIME explanation generated successfully")
            return explanation

        except Exception as e:
            logger.error(f"Error generating LIME explanation: {str(e)}")
            raise

    def global_feature_importance_shap(self, X_test: pd.DataFrame, max_evals: int = 1000) -> Dict[str, Any]:
        """
        Calculate global feature importance using SHAP.

        Args:
            X_test: Test data for global explanation
            max_evals: Maximum evaluations for SHAP

        Returns:
            Dictionary with global SHAP feature importance
        """
        if not self.is_initialized:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")

        try:
            logger.info("Calculating global SHAP feature importance")

            # Calculate SHAP values for test set (handle different SHAP versions)
            try:
                # Try with max_evals for newer SHAP versions
                shap_values = self.shap_explainer(X_test, max_evals=max_evals)
            except TypeError:
                # Fallback for older SHAP versions without max_evals
                logger.info("Using SHAP without max_evals parameter (older version)")
                shap_values = self.shap_explainer(X_test)

            # Calculate mean absolute SHAP values
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) > 2:
                    mean_shap = np.mean(np.abs(shap_values.values[:, :, 0]), axis=0)
                else:
                    mean_shap = np.mean(np.abs(shap_values.values), axis=0)
            else:
                mean_shap = np.mean(np.abs(shap_values), axis=0)

            # Create feature importance dictionary
            feature_importance = dict(zip(X_test.columns, mean_shap))

            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

            result = {
                'feature_importance': sorted_importance,
                'mean_abs_shap_values': mean_shap.tolist(),
                'feature_names': X_test.columns.tolist()
            }

            logger.info("Global SHAP feature importance calculated successfully")
            return result

        except Exception as e:
            logger.error(f"Error calculating global SHAP importance: {str(e)}")
            raise

    def create_partial_dependence_plots(self, model, X_train: pd.DataFrame, features: List[str],
                                      save_path: Path, kind: str = 'average') -> None:
        """
        Create partial dependence plots for specified features.

        Args:
            model: Trained model
            X_train: Training data
            features: List of features to plot
            save_path: Path to save the plots
            kind: Type of PD plot ('average' or 'individual')
        """
        try:
            logger.info(f"Creating partial dependence plots for features: {features}")

            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Create partial dependence plots
            fig, ax = plt.subplots(figsize=(15, 10))

            try:
                PartialDependenceDisplay.from_estimator(
                    model, X_train, features, ax=ax, kind=kind,
                    subsample=1000, random_state=42
                )
            except Exception as e:
                logger.warning(f"PartialDependenceDisplay failed: {e}, using manual calculation")

                # Manual partial dependence calculation
                for i, feature in enumerate(features):
                    plt.subplot(len(features), 1, i+1)

                    try:
                        pd_results = partial_dependence(model, X_train, [feature])
                        plt.plot(pd_results['values'][0], pd_results['average'][0])
                        plt.xlabel(feature)
                        plt.ylabel('Partial Dependence')
                        plt.title(f'Partial Dependence of {feature}')
                        plt.grid(True, alpha=0.3)
                    except Exception as e2:
                        logger.error(f"Failed to create PD plot for {feature}: {e2}")

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Partial dependence plots saved to {save_path}")

        except Exception as e:
            logger.error(f"Error creating partial dependence plots: {str(e)}")
            raise

    def create_shap_summary_plot(self, shap_values, X_test: pd.DataFrame, save_path: Path) -> None:
        """
        Create SHAP summary plot.

        Args:
            shap_values: SHAP values
            X_test: Test data
            save_path: Path to save the plot
        """
        try:
            logger.info("Creating SHAP summary plot")

            save_path.parent.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, show=False)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"SHAP summary plot saved to {save_path}")

        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {str(e)}")
            raise

    def create_shap_waterfall_plot(self, shap_values, X_instance: pd.DataFrame,
                                 save_path: Path, max_display: int = 10) -> None:
        """
        Create SHAP waterfall plot for single prediction.

        Args:
            shap_values: SHAP values for single instance
            X_instance: Single instance data
            save_path: Path to save the plot
            max_display: Maximum features to display
        """
        try:
            logger.info("Creating SHAP waterfall plot")

            save_path.parent.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(10, 8))
            shap.plots.waterfall(shap_values[0], max_display=max_display, show=False)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"SHAP waterfall plot saved to {save_path}")

        except Exception as e:
            logger.error(f"Error creating SHAP waterfall plot: {str(e)}")
            raise

    def analyze_feature_interactions(self, model, X_test: pd.DataFrame,
                                   feature_pairs: List[Tuple[str, str]],
                                   save_path: Path) -> Dict[str, Any]:
        """
        Analyze feature interactions using SHAP.

        Args:
            model: Trained model
            X_test: Test data
            feature_pairs: List of feature pairs to analyze
            save_path: Path to save interaction plots

        Returns:
            Dictionary with interaction analysis results
        """
        try:
            logger.info("Analyzing feature interactions")

            save_path.parent.mkdir(parents=True, exist_ok=True)

            interaction_results = {}

            for feat1, feat2 in feature_pairs:
                try:
                    plt.figure(figsize=(10, 8))
                    shap.dependence_plot(feat1, self.shap_explainer.shap_values, X_test,
                                       interaction_index=feat2, show=False)
                    plt.tight_layout()
                    plt.savefig(save_path / f"interaction_{feat1}_{feat2}.png", dpi=300, bbox_inches='tight')
                    plt.close()

                    interaction_results[f"{feat1}_{feat2}"] = {
                        'plot_saved': True,
                        'path': str(save_path / f"interaction_{feat1}_{feat2}.png")
                    }

                except Exception as e:
                    logger.warning(f"Failed to create interaction plot for {feat1}-{feat2}: {e}")
                    interaction_results[f"{feat1}_{feat2}"] = {
                        'plot_saved': False,
                        'error': str(e)
                    }

            logger.info("Feature interaction analysis completed")
            return interaction_results

        except Exception as e:
            logger.error(f"Error in feature interaction analysis: {str(e)}")
            raise

    def generate_explainability_report(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                    y_test: pd.Series, reports_dir: Path) -> Dict[str, Any]:
        """
        Generate comprehensive explainability report.

        Args:
            model: Trained model
            X_train: Training data
            X_test: Test data
            y_test: Test target
            reports_dir: Directory to save reports

        Returns:
            Dictionary with report contents
        """
        try:
            logger.info("Generating comprehensive explainability report")

            reports_dir.mkdir(parents=True, exist_ok=True)

            # Initialize explainers
            self.initialize_explainer(model, X_train)

            # Global feature importance
            global_shap = self.global_feature_importance_shap(X_test)

            # Sample predictions for local explanations
            sample_indices = np.random.choice(len(X_test), size=min(5, len(X_test)), replace=False)
            local_explanations = []

            for idx in sample_indices:
                X_instance = X_test.iloc[[idx]]
                actual_value = y_test.iloc[idx]

                # SHAP explanation
                shap_exp = self.explain_prediction_shap(X_instance)

                # LIME explanation
                lime_exp = self.explain_prediction_lime(X_instance)

                local_explanations.append({
                    'instance_index': int(idx),
                    'actual_value': float(actual_value),
                    'shap_explanation': shap_exp,
                    'lime_explanation': lime_exp
                })

            # Create plots
            plots_dir = reports_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Partial dependence plots for top features
            top_features = list(global_shap['feature_importance'].keys())[:4]
            self.create_partial_dependence_plots(
                model, X_train, top_features,
                plots_dir / "partial_dependence.png"
            )

            # SHAP summary plot
            shap_values = self.shap_explainer(X_test.sample(min(100, len(X_test))))
            self.create_shap_summary_plot(shap_values, X_test, plots_dir / "shap_summary.png")

            # Feature interaction analysis
            feature_pairs = [('temp', 'hum'), ('temp', 'windspeed'), ('season', 'mnth')]
            interaction_results = self.analyze_feature_interactions(
                model, X_test, feature_pairs, plots_dir
            )

            # Compile report
            report = {
                'global_explanations': {
                    'shap_feature_importance': global_shap
                },
                'local_explanations': local_explanations,
                'plots_generated': {
                    'partial_dependence': str(plots_dir / "partial_dependence.png"),
                    'shap_summary': str(plots_dir / "shap_summary.png"),
                    'feature_interactions': interaction_results
                },
                'feature_insights': {
                    'top_features': top_features,
                    'most_important_feature': top_features[0] if top_features else None
                },
                'report_metadata': {
                    'generated_at': pd.Timestamp.now().isoformat(),
                    'test_samples': len(X_test),
                    'training_samples': len(X_train)
                }
            }

            # Save report
            report_path = reports_dir / "explainability_report.json"
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Explainability report generated and saved to {report_path}")
            return report

        except Exception as e:
            logger.error(f"Error generating explainability report: {str(e)}")
            raise
