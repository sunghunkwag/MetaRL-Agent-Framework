"""
Integration Test for SSM-MetaRL Benchmarks

Verifies that:
1. All task distributions can be loaded
2. Benchmark runner handles new modes correctly
3. End-to-end pipeline runs without error
"""

import unittest
import sys
import os
import shutil
from pathlib import Path
import torch

# Add project root
sys.path.insert(0, os.getcwd())

from experiments.task_distributions import get_task_distribution, list_task_distributions, TASK_DISTRIBUTIONS
from experiments.serious_benchmark import BenchmarkRunner

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.test_output_dir = 'test_results'
        os.makedirs(self.test_output_dir, exist_ok=True)

        self.config = {
            'hidden_dim': 32,
            'inner_lr': 0.01,
            'outer_lr': 0.001,
            'adapt_lr': 0.01,
            'support_steps': 10, # Short for testing
            'query_steps': 10,
            'tasks_per_epoch': 2,
            'device': 'cpu',
            'epochs': 1
        }

    def tearDown(self):
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_all_distributions_load(self):
        """Test that all registered distributions can be instantiated."""
        print("\nTesting distribution loading...")
        distributions = list_task_distributions()
        self.assertGreaterEqual(len(distributions), 10)

        for name in distributions:
            print(f"  Loading {name}...")
            try:
                dist = get_task_distribution(name)
                # Check if we have tasks (except possibly for iMuJoCo if not installed/found)
                if 'imujoco' in name and dist.num_tasks == 0:
                    print(f"    Warning: No tasks for {name} (likely due to missing installation)")
                else:
                    self.assertGreater(dist.num_tasks, 0)
                    # Sample one task
                    env = dist.sample_task(0)
                    env.reset()
                    env.close()
            except ImportError:
                print(f"    Skipping {name} due to missing dependency")
            except Exception as e:
                self.fail(f"Failed to load {name}: {e}")

    def test_benchmark_modes(self):
        """Test few-shot and zero-shot modes end-to-end."""
        # Use a simple distribution for speed, e.g., halfcheetah-vel
        dist_name = 'halfcheetah-vel'

        print(f"\nTesting benchmark modes on {dist_name}...")

        # 1. Standard Run
        print("  Running standard benchmark...")
        runner = BenchmarkRunner(dist_name, 'ssm', self.config)
        metrics = runner.run(num_epochs=1, eval_interval=1)
        self.assertIn('train_loss', metrics)

        # 2. Few-Shot Test
        print("  Running few-shot test...")
        # Train briefly first (mocking by just running 1 epoch)
        runner.train(num_epochs=1)
        results = runner.few_shot_test(task_id=0, num_shots=[1, 2])
        self.assertIn(1, results)
        self.assertIn(2, results)
        print(f"    Results: {results}")

        # 3. Zero-Shot Test
        print("  Running zero-shot test...")
        # Define split
        train_tasks = [0, 1]
        test_tasks = [2]
        # Runner needs to be re-initialized or we reuse
        runner.zero_shot_test(train_tasks, test_tasks)

    def test_metaworld_integration(self):
        """Specific test for MetaWorld if available."""
        try:
            import metaworld
        except ImportError:
            print("\nMetaWorld not installed, skipping integration test")
            return

        print("\nTesting MetaWorld ML10...")
        runner = BenchmarkRunner('metaworld-ml10', 'ssm', self.config)

        # Check split
        self.assertEqual(len(runner.train_task_ids), 10)
        self.assertEqual(len(runner.test_task_ids), 5)

        # Run minimal training
        runner.train(num_epochs=1)

        # Run few-shot on a test task
        test_id = runner.test_task_ids[0]
        res = runner.few_shot_test(test_id, num_shots=[1])
        self.assertIn(1, res)

if __name__ == '__main__':
    unittest.main()
