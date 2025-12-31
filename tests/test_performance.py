import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from memory_profiler import profile as memory_profile
from src.models.bike_demand_model import BikeDemandModel
from src.features.data_processor import DataProcessor
from src.pipelines.training_pipeline import TrainingPipeline


class TestPerformance:
    """Performance tests for ML pipeline components."""

    @pytest.fixture
    def perf_config(self):
        """Configuration optimized for performance testing."""
        return {
            'model': {
                'name': 'performance_test_model',
                'type': 'random_forest',
                'version': '1.0.0'
            },
            'hyperparameters': {
                'n_estimators': 10,  # Small for faster testing
                'max_depth': 5,
                'random_state': 42,
                'n_jobs': -1  # Use all cores
            },
            'features': {
                'numerical': ['temp', 'hum', 'windspeed'],
                'categorical': ['season', 'weathersit']
            },
            'target': 'cnt',
            'data': {
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1,
                'random_state': 42
            },
            'evaluation': {
                'metrics': ['mae', 'rmse'],
                'cv_folds': 3
            }
        }

    @pytest.fixture
    def large_dataset(self):
        """Generate a large dataset for performance testing."""
        np.random.seed(42)
        n_samples = 10000  # Large dataset

        data = {
            'temp': np.random.normal(0.5, 0.2, n_samples).clip(0, 1),
            'hum': np.random.normal(0.6, 0.15, n_samples).clip(0, 1),
            'windspeed': np.random.normal(0.2, 0.1, n_samples).clip(0, 1),
            'season': np.random.choice([1, 2, 3, 4], n_samples),
            'weathersit': np.random.choice([1, 2, 3], n_samples),
            'cnt': np.random.normal(4000, 1500, n_samples).clip(0, None)
        }

        return pd.DataFrame(data)

    def test_model_training_performance(self, perf_config, large_dataset):
        """Test model training performance with large dataset."""
        model = BikeDemandModel(perf_config)

        # Preprocess data
        processed_data = model.preprocess_data(large_dataset)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_data)

        # Time training
        start_time = time.time()
        metrics = model.train(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time

        # Performance assertions
        assert training_time < 30  # Should train within 30 seconds
        assert model.is_trained
        assert 'train_mae' in metrics

        print(f"Training time for {len(X_train)} samples: {training_time:.2f} seconds")

    def test_inference_performance(self, perf_config, large_dataset):
        """Test model inference performance."""
        model = BikeDemandModel(perf_config)

        # Train model
        processed_data = model.preprocess_data(large_dataset)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_data)
        model.train(X_train, y_train)

        # Test batch inference performance
        batch_sizes = [1, 10, 100, 1000]

        for batch_size in batch_sizes:
            test_batch = X_test.head(batch_size)

            start_time = time.time()
            predictions = model.predict(test_batch)
            inference_time = time.time() - start_time

            # Performance requirements (adjust based on hardware)
            max_time_per_sample = 0.01  # 10ms per sample
            assert inference_time / batch_size < max_time_per_sample, \
                f"Inference too slow: {inference_time / batch_size:.4f}s per sample for batch size {batch_size}"

            assert len(predictions) == batch_size
            print(f"Batch size {batch_size}: {inference_time:.4f}s total, {inference_time/batch_size:.6f}s per sample")

    def test_data_processing_performance(self, perf_config, large_dataset):
        """Test data processing performance."""
        processor = DataProcessor(perf_config)

        # Time preprocessing
        start_time = time.time()
        processed_data = processor.preprocess_features(large_dataset, fit=True)
        processing_time = time.time() - start_time

        # Performance assertions
        assert processing_time < 10  # Should process within 10 seconds
        assert processor.is_fitted
        assert len(processed_data) == len(large_dataset)

        print(f"Data processing time for {len(large_dataset)} samples: {processing_time:.2f} seconds")

    def test_memory_usage_during_training(self, perf_config, large_dataset):
        """Test memory usage during model training."""
        process = psutil.Process(os.getpid())

        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        model = BikeDemandModel(perf_config)
        processed_data = model.preprocess_data(large_dataset)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_data)

        # Monitor memory during training
        peak_memory = baseline_memory
        start_time = time.time()

        # Train model and monitor memory
        for i in range(0, len(X_train), 1000):  # Process in chunks to monitor
            if i == 0:
                model.train(X_train, y_train, X_val, y_val)
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)

        training_time = time.time() - start_time
        memory_increase = peak_memory - baseline_memory

        # Memory assertions (adjust based on system capabilities)
        assert memory_increase < 1000, f"Memory usage too high: {memory_increase:.1f}MB increase"
        assert model.is_trained

        print(f"Peak memory usage: {peak_memory:.1f}MB (increase: {memory_increase:.1f}MB)")
        print(f"Training completed in {training_time:.2f} seconds")

    def test_pipeline_end_to_end_performance(self, perf_config, large_dataset):
        """Test complete pipeline performance."""
        from src.pipelines.training_pipeline import TrainingPipeline

        start_time = time.time()
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Run complete pipeline
        pipeline = TrainingPipeline(perf_config)
        results = pipeline.run(large_dataset)

        end_time = time.time()
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        total_time = end_time - start_time
        memory_used = final_memory - initial_memory

        # Performance assertions
        assert total_time < 60  # Complete pipeline within 1 minute
        assert memory_used < 1500  # Memory increase within reasonable limits
        assert results['model'].is_trained
        assert 'metrics' in results

        print(f"End-to-end pipeline performance:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Memory used: {memory_used:.1f} MB")
        print(f"  Samples processed: {len(large_dataset)}")

    def test_concurrent_inference_performance(self, perf_config, large_dataset):
        """Test concurrent inference performance."""
        import concurrent.futures
        import threading

        model = BikeDemandModel(perf_config)

        # Train model
        processed_data = model.preprocess_data(large_dataset)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_data)
        model.train(X_train, y_train)

        # Test concurrent predictions
        def predict_batch(batch_data):
            return model.predict(batch_data)

        # Split test data into batches
        batch_size = 100
        batches = [X_test[i:i+batch_size] for i in range(0, len(X_test), batch_size)][:5]  # Test 5 batches

        start_time = time.time()

        # Run concurrent predictions
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(predict_batch, batch) for batch in batches]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        concurrent_time = time.time() - start_time

        # Sequential prediction for comparison
        start_time = time.time()
        sequential_results = [predict_batch(batch) for batch in batches]
        sequential_time = time.time() - start_time

        # Concurrent should be faster (though exact speedup depends on system)
        speedup = sequential_time / concurrent_time
        print(f"Concurrent inference speedup: {speedup:.2f}x")

        # All predictions should be completed
        total_predictions = sum(len(result) for result in results)
        assert total_predictions == sum(len(batch) for batch in batches)

    def test_scalability_with_data_size(self, perf_config):
        """Test how performance scales with data size."""
        data_sizes = [1000, 5000, 10000]
        training_times = []
        memory_usages = []

        for n_samples in data_sizes:
            # Generate dataset of specific size
            np.random.seed(42)
            data = {
                'temp': np.random.normal(0.5, 0.2, n_samples).clip(0, 1),
                'hum': np.random.normal(0.6, 0.15, n_samples).clip(0, 1),
                'windspeed': np.random.normal(0.2, 0.1, n_samples).clip(0, 1),
                'season': np.random.choice([1, 2, 3, 4], n_samples),
                'weathersit': np.random.choice([1, 2, 3], n_samples),
                'cnt': np.random.normal(4000, 1500, n_samples).clip(0, None)
            }
            df = pd.DataFrame(data)

            # Measure performance
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024

            start_time = time.time()
            model = BikeDemandModel(perf_config)
            processed_data = model.preprocess_data(df)
            X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_data)
            model.train(X_train, y_train)
            training_time = time.time() - start_time

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_usage = final_memory - initial_memory

            training_times.append(training_time)
            memory_usages.append(memory_usage)

            print(f"Data size {n_samples}: {training_time:.2f}s, {memory_usage:.1f}MB")

        # Check scaling (should be roughly linear or better)
        for i in range(1, len(data_sizes)):
            time_ratio = training_times[i] / training_times[i-1]
            size_ratio = data_sizes[i] / data_sizes[i-1]

            # Time should scale reasonably with data size (allowing for some overhead)
            assert time_ratio < size_ratio * 2, f"Poor scaling: {time_ratio:.2f} vs {size_ratio:.2f}"

    @pytest.mark.parametrize("n_jobs", [1, 2, -1])
    def test_parallel_training_performance(self, perf_config, large_dataset, n_jobs):
        """Test parallel training performance with different core counts."""
        # Modify config for parallel training
        config = perf_config.copy()
        config['hyperparameters']['n_jobs'] = n_jobs

        model = BikeDemandModel(config)
        processed_data = model.preprocess_data(large_dataset)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_data)

        start_time = time.time()
        metrics = model.train(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time

        assert model.is_trained
        assert training_time > 0

        print(f"Training with n_jobs={n_jobs}: {training_time:.2f} seconds")

    def test_model_serialization_performance(self, perf_config, large_dataset, tmp_path):
        """Test model serialization/deserialization performance."""
        model = BikeDemandModel(perf_config)

        # Train model
        processed_data = model.preprocess_data(large_dataset)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_data)
        model.train(X_train, y_train)

        # Test save performance
        model_path = tmp_path / "perf_test_model.joblib"
        start_time = time.time()
        model.save_model(str(model_path))
        save_time = time.time() - start_time

        # Test load performance
        start_time = time.time()
        new_model = BikeDemandModel(perf_config)
        new_model.load_model(str(model_path))
        load_time = time.time() - start_time

        # Performance assertions
        assert save_time < 5  # Save within 5 seconds
        assert load_time < 2  # Load within 2 seconds
        assert new_model.is_trained

        print(f"Model serialization: save={save_time:.2f}s, load={load_time:.2f}s")

    def test_data_quality_checks_performance(self, perf_config, large_dataset):
        """Test performance of data quality checks."""
        from src.features.data_processor import DataProcessor

        processor = DataProcessor(perf_config)

        # Time data validation
        start_time = time.time()
        processor.validate_data(large_dataset)
        validation_time = time.time() - start_time

        # Time preprocessing
        start_time = time.time()
        processed_data = processor.preprocess_features(large_dataset, fit=True)
        preprocessing_time = time.time() - start_time

        # Performance assertions
        assert validation_time < 1  # Validation should be fast
        assert preprocessing_time < 15  # Preprocessing within reasonable time

        print(f"Data quality checks: validation={validation_time:.3f}s, preprocessing={preprocessing_time:.2f}s")
