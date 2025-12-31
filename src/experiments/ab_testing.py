import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.power import TTestPower
from statsmodels.stats.proportion import proportions_ztest
import mlflow
import mlflow.sklearn
from loguru import logger
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ABTestingFramework:
    """Advanced A/B testing framework for model comparison and experimentation."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the A/B testing framework.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.test_results = {}

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.get('logging', {}).get('level', 'INFO')),
            format=self.config.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

    def design_experiment(
        self,
        baseline_metric: float,
        expected_improvement: float,
        significance_level: float = 0.05,
        power: float = 0.8,
        metric_type: str = 'continuous'
    ) -> Dict[str, Any]:
        """
        Design A/B test experiment parameters.

        Args:
            baseline_metric: Current baseline metric value
            expected_improvement: Expected improvement percentage
            significance_level: Statistical significance level (alpha)
            power: Statistical power (1 - beta)
            metric_type: Type of metric ('continuous' or 'proportion')

        Returns:
            Dictionary with experiment design parameters
        """
        try:
            logger.info("Designing A/B test experiment")

            if metric_type == 'continuous':
                # Calculate effect size for continuous metrics
                expected_mean = baseline_metric * (1 + expected_improvement)
                effect_size = abs(expected_mean - baseline_metric) / baseline_metric

                # Calculate required sample size
                analysis = TTestPower()
                sample_size = analysis.solve_power(
                    effect_size=effect_size,
                    alpha=significance_level,
                    power=power,
                    alternative='two-sided'
                )

            elif metric_type == 'proportion':
                # For proportion metrics (e.g., conversion rates)
                p1 = baseline_metric
                p2 = baseline_metric * (1 + expected_improvement)

                # Calculate required sample size for proportion test
                # Using approximation formula
                z_alpha = stats.norm.ppf(1 - significance_level/2)
                z_beta = stats.norm.ppf(power)

                p_avg = (p1 + p2) / 2
                sample_size = ((z_alpha + z_beta) ** 2 * p_avg * (1 - p_avg) /
                             (p1 - p2) ** 2)

            else:
                raise ValueError(f"Unsupported metric type: {metric_type}")

            # Calculate minimum detectable effect
            mde = baseline_metric * expected_improvement

            design = {
                'baseline_metric': baseline_metric,
                'expected_improvement': expected_improvement,
                'significance_level': significance_level,
                'power': power,
                'metric_type': metric_type,
                'required_sample_size_per_group': int(np.ceil(sample_size)),
                'total_sample_size': int(np.ceil(sample_size * 2)),
                'minimum_detectable_effect': mde,
                'effect_size': effect_size if metric_type == 'continuous' else abs(p1 - p2),
                'design_date': datetime.now().isoformat()
            }

            logger.info(f"Experiment design completed. Required sample size: {design['total_sample_size']}")
            return design

        except Exception as e:
            logger.error(f"Error designing experiment: {str(e)}")
            raise

    def run_ab_test(
        self,
        control_data: pd.DataFrame,
        treatment_data: pd.DataFrame,
        metric_column: str,
        test_name: str,
        metric_type: str = 'continuous',
        alternative: str = 'two-sided'
    ) -> Dict[str, Any]:
        """
        Run A/B test analysis.

        Args:
            control_data: Data for control group
            treatment_data: Data for treatment group
            metric_column: Column name of the metric to test
            test_name: Name of the test
            metric_type: Type of metric ('continuous' or 'proportion')
            alternative: Alternative hypothesis ('two-sided', 'greater', 'less')

        Returns:
            Dictionary with test results
        """
        try:
            logger.info(f"Running A/B test: {test_name}")

            # Extract metrics
            control_metric = control_data[metric_column].values
            treatment_metric = treatment_data[metric_column].values

            # Basic statistics
            control_stats = {
                'mean': np.mean(control_metric),
                'std': np.std(control_metric),
                'count': len(control_metric),
                'median': np.median(control_metric)
            }

            treatment_stats = {
                'mean': np.mean(treatment_metric),
                'std': np.std(treatment_metric),
                'count': len(treatment_metric),
                'median': np.median(treatment_metric)
            }

            # Perform statistical test
            if metric_type == 'continuous':
                # t-test for continuous metrics
                t_stat, p_value = stats.ttest_ind(
                    control_metric, treatment_metric,
                    alternative=alternative,
                    equal_var=False  # Welch's t-test
                )
                test_type = 't-test'

            elif metric_type == 'proportion':
                # z-test for proportions
                control_successes = np.sum(control_metric)
                treatment_successes = np.sum(treatment_metric)

                z_stat, p_value = proportions_ztest(
                    [control_successes, treatment_successes],
                    [len(control_metric), len(treatment_metric)],
                    alternative=alternative
                )
                test_type = 'z-test'

            else:
                raise ValueError(f"Unsupported metric type: {metric_type}")

            # Calculate confidence intervals
            confidence_level = 0.95
            if metric_type == 'continuous':
                control_ci = stats.t.interval(
                    confidence_level, len(control_metric)-1,
                    loc=control_stats['mean'], scale=stats.sem(control_metric)
                )
                treatment_ci = stats.t.interval(
                    confidence_level, len(treatment_metric)-1,
                    loc=treatment_stats['mean'], scale=stats.sem(treatment_metric)
                )
            else:
                # For proportions, use Wilson score interval approximation
                def proportion_ci(successes, n, confidence=0.95):
                    p = successes / n
                    z = stats.norm.ppf(1 - (1 - confidence) / 2)
                    denominator = 1 + z**2 / n
                    center = (p + z**2 / (2 * n)) / denominator
                    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
                    return (center - spread, center + spread)

                control_ci = proportion_ci(control_successes, len(control_metric))
                treatment_ci = proportion_ci(treatment_successes, len(treatment_metric))

            # Determine statistical significance
            alpha = self.config.get('ab_testing', {}).get('significance_level', 0.05)
            significant = p_value < alpha

            # Calculate effect size
            if metric_type == 'continuous':
                # Cohen's d
                pooled_std = np.sqrt(
                    ((len(control_metric) - 1) * control_stats['std']**2 +
                     (len(treatment_metric) - 1) * treatment_stats['std']**2) /
                    (len(control_metric) + len(treatment_metric) - 2)
                )
                effect_size = abs(treatment_stats['mean'] - control_stats['mean']) / pooled_std
            else:
                # For proportions, use difference
                effect_size = abs(treatment_stats['mean'] - control_stats['mean'])

            # Calculate practical significance
            practical_threshold = self.config.get('ab_testing', {}).get('practical_significance_threshold', 0.01)
            practically_significant = effect_size > practical_threshold

            # Calculate lift
            lift = (treatment_stats['mean'] - control_stats['mean']) / control_stats['mean']

            results = {
                'test_name': test_name,
                'test_type': test_type,
                'metric_type': metric_type,
                'alternative': alternative,
                'control': {
                    'stats': control_stats,
                    'confidence_interval': control_ci
                },
                'treatment': {
                    'stats': treatment_stats,
                    'confidence_interval': treatment_ci
                },
                'statistical_test': {
                    'statistic': t_stat if metric_type == 'continuous' else z_stat,
                    'p_value': p_value,
                    'alpha': alpha,
                    'significant': significant
                },
                'effect_analysis': {
                    'effect_size': effect_size,
                    'lift': lift,
                    'practical_significance_threshold': practical_threshold,
                    'practically_significant': practically_significant
                },
                'sample_sizes': {
                    'control': len(control_metric),
                    'treatment': len(treatment_metric),
                    'total': len(control_metric) + len(treatment_metric)
                },
                'test_timestamp': datetime.now().isoformat()
            }

            # Store results
            self.test_results[test_name] = results

            # Log to MLflow
            with mlflow.start_run(run_name=f"ab_test_{test_name}", nested=True):
                mlflow.log_param("test_name", test_name)
                mlflow.log_param("test_type", test_type)
                mlflow.log_param("metric_type", metric_type)
                mlflow.log_metric("control_mean", control_stats['mean'])
                mlflow.log_metric("treatment_mean", treatment_stats['mean'])
                mlflow.log_metric("p_value", p_value)
                mlflow.log_metric("effect_size", effect_size)
                mlflow.log_metric("lift", lift)
                mlflow.log_param("statistically_significant", significant)
                mlflow.log_param("practically_significant", practically_significant)

            logger.info(f"A/B test completed. Significant: {significant}, Lift: {lift:.4f}")
            return results

        except Exception as e:
            logger.error(f"Error running A/B test: {str(e)}")
            raise

    def sequential_testing(
        self,
        control_data: pd.DataFrame,
        treatment_data: pd.DataFrame,
        metric_column: str,
        test_name: str,
        batch_size: int = 100,
        max_batches: int = 20
    ) -> Dict[str, Any]:
        """
        Perform sequential A/B testing with early stopping.

        Args:
            control_data: Data for control group
            treatment_data: Data for treatment group
            metric_column: Column name of the metric to test
            test_name: Name of the test
            batch_size: Size of each batch
            max_batches: Maximum number of batches

        Returns:
            Dictionary with sequential testing results
        """
        try:
            logger.info(f"Running sequential A/B test: {test_name}")

            results_over_time = []
            early_stop = False
            stop_reason = None

            for batch in range(1, max_batches + 1):
                # Sample data for current batch
                control_sample = control_data.sample(
                    n=min(batch_size, len(control_data)),
                    random_state=batch
                )
                treatment_sample = treatment_data.sample(
                    n=min(batch_size, len(treatment_data)),
                    random_state=batch
                )

                # Run test on current sample
                batch_result = self.run_ab_test(
                    control_sample, treatment_sample,
                    metric_column, f"{test_name}_batch_{batch}"
                )

                results_over_time.append({
                    'batch': batch,
                    'sample_size': batch * batch_size,
                    'p_value': batch_result['statistical_test']['p_value'],
                    'effect_size': batch_result['effect_analysis']['effect_size'],
                    'lift': batch_result['effect_analysis']['lift'],
                    'significant': batch_result['statistical_test']['significant']
                })

                # Check early stopping conditions
                alpha = self.config.get('ab_testing', {}).get('significance_level', 0.05)

                # Stop if we have strong evidence (very low p-value)
                if batch_result['statistical_test']['p_value'] < alpha / 10:
                    early_stop = True
                    stop_reason = "Strong statistical significance"
                    break

                # Stop if effect is clearly negligible
                if batch_result['effect_analysis']['effect_size'] < 0.01:
                    early_stop = True
                    stop_reason = "Negligible effect size"
                    break

            sequential_results = {
                'test_name': test_name,
                'sequential_results': results_over_time,
                'early_stop': early_stop,
                'stop_reason': stop_reason,
                'final_batch': len(results_over_time),
                'total_sample_size': len(results_over_time) * batch_size
            }

            logger.info(f"Sequential testing completed. Early stop: {early_stop}")
            return sequential_results

        except Exception as e:
            logger.error(f"Error in sequential testing: {str(e)}")
            raise

    def multi_arm_bandit_simulation(
        self,
        arms_data: Dict[str, pd.DataFrame],
        metric_column: str,
        n_rounds: int = 1000,
        epsilon: float = 0.1
    ) -> Dict[str, Any]:
        """
        Simulate multi-arm bandit for adaptive experimentation.

        Args:
            arms_data: Dictionary of arm names to DataFrames
            metric_column: Column name of the metric
            n_rounds: Number of simulation rounds
            epsilon: Exploration parameter for epsilon-greedy

        Returns:
            Dictionary with bandit simulation results
        """
        try:
            logger.info("Running multi-arm bandit simulation")

            arm_names = list(arms_data.keys())
            n_arms = len(arm_names)

            # Initialize bandit parameters
            rewards = np.zeros(n_arms)
            counts = np.zeros(n_arms)
            total_rewards = np.zeros(n_arms)

            results = []

            for round_num in range(n_rounds):
                # Epsilon-greedy selection
                if np.random.random() < epsilon:
                    # Explore: random arm
                    chosen_arm = np.random.randint(n_arms)
                else:
                    # Exploit: best arm so far
                    chosen_arm = np.argmax(rewards)

                arm_name = arm_names[chosen_arm]

                # Sample reward from arm's data
                reward = arms_data[arm_name][metric_column].sample(1).iloc[0]
                total_rewards[chosen_arm] += reward
                counts[chosen_arm] += 1
                rewards[chosen_arm] = total_rewards[chosen_arm] / counts[chosen_arm]

                results.append({
                    'round': round_num + 1,
                    'chosen_arm': arm_name,
                    'reward': reward,
                    'cumulative_reward': total_rewards[chosen_arm]
                })

            # Final statistics
            final_stats = {}
            for i, arm_name in enumerate(arm_names):
                final_stats[arm_name] = {
                    'total_reward': total_rewards[i],
                    'count': counts[i],
                    'average_reward': rewards[i]
                }

            best_arm = arm_names[np.argmax(rewards)]

            bandit_results = {
                'simulation_results': results,
                'final_statistics': final_stats,
                'best_arm': best_arm,
                'total_rounds': n_rounds,
                'epsilon': epsilon,
                'regret': np.max(rewards) - rewards  # Regret for each arm
            }

            logger.info(f"Multi-arm bandit simulation completed. Best arm: {best_arm}")
            return bandit_results

        except Exception as e:
            logger.error(f"Error in bandit simulation: {str(e)}")
            raise

    def save_test_results(self, path: str) -> None:
        """
        Save A/B test results to file.

        Args:
            path: Path to save results
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Convert numpy types to Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj

            serializable_results = convert_to_serializable(self.test_results)

            with open(path, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"A/B test results saved to {path}")

        except Exception as e:
            logger.error(f"Error saving test results: {str(e)}")
            raise

    def generate_experiment_report(self, experiment_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive experiment report.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary with experiment report
        """
        try:
            logger.info(f"Generating experiment report: {experiment_name}")

            report = {
                'experiment_name': experiment_name,
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_tests': len(self.test_results),
                    'significant_tests': sum(1 for r in self.test_results.values()
                                           if r['statistical_test']['significant']),
                    'practically_significant_tests': sum(1 for r in self.test_results.values()
                                                       if r['effect_analysis']['practically_significant'])
                },
                'test_results': self.test_results
            }

            # Calculate overall statistics
            if self.test_results:
                lifts = [r['effect_analysis']['lift'] for r in self.test_results.values()]
                p_values = [r['statistical_test']['p_value'] for r in self.test_results.values()]

                report['overall_statistics'] = {
                    'average_lift': np.mean(lifts),
                    'median_lift': np.median(lifts),
                    'max_lift': np.max(lifts),
                    'min_lift': np.min(lifts),
                    'average_p_value': np.mean(p_values),
                    'significant_tests_ratio': report['summary']['significant_tests'] / report['summary']['total_tests']
                }

            logger.info(f"Experiment report generated with {len(self.test_results)} tests")
            return report

        except Exception as e:
            logger.error(f"Error generating experiment report: {str(e)}")
            raise
