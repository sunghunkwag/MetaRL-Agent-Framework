"""
Serious Benchmark Suite for SSM-MetaRL

This script runs comprehensive benchmarks on high-dimensional tasks with
SOTA baseline comparisons to prove the framework's effectiveness beyond
toy problems.

Usage:
    python experiments/serious_benchmark.py --task halfcheetah-vel --method ssm
    python experiments/serious_benchmark.py --task ant-dir --method all --epochs 100
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
from collections import defaultdict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from experiments.task_distributions import get_task_distribution, list_task_distributions
from experiments.baselines import get_baseline_policy, list_baselines
import gymnasium as gym


class BenchmarkRunner:
    """Runs meta-learning benchmarks and collects metrics."""
    
    def __init__(self, task_dist_name: str, method_name: str, config: Dict[str, Any]):
        self.task_dist_name = task_dist_name
        self.method_name = method_name
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Create task distribution
        self.task_dist = get_task_distribution(task_dist_name)
        
        # Determine train/test split
        # Priority: config > inherent split > all train (empty test)
        inherent_split = self.task_dist.get_train_test_split()

        if config.get('train_tasks') is not None:
            self.train_task_ids = config['train_tasks']
        elif inherent_split:
            self.train_task_ids = inherent_split[0]
        else:
            self.train_task_ids = list(range(self.task_dist.num_tasks))

        if config.get('test_tasks') is not None:
            self.test_task_ids = config['test_tasks']
        elif inherent_split:
            self.test_task_ids = inherent_split[1]
        else:
            # If no split and no config, use last task as test if multiple tasks
            if len(self.train_task_ids) > 1:
                self.test_task_ids = [self.train_task_ids.pop()]
            else:
                self.test_task_ids = [] # No test tasks

        print(f"Train Tasks: {len(self.train_task_ids)}")
        print(f"Test Tasks: {len(self.test_task_ids)}")

        # Get environment info from first task
        if self.train_task_ids:
            sample_id = self.train_task_ids[0]
        elif self.test_task_ids:
            sample_id = self.test_task_ids[0]
        else:
            sample_id = 0

        sample_env = self.task_dist.sample_task(sample_id)
        self.state_dim = sample_env.observation_space.shape[0]
        self.action_dim = sample_env.action_space.shape[0]
        sample_env.close()
        
        # Create model
        self.model = self._create_model()
        
        # Create meta-learner
        self.meta_learner = MetaMAML(
            model=self.model,
            inner_lr=config.get('inner_lr', 0.01),
            outer_lr=config.get('outer_lr', 0.001)
        )
        
        # Metrics storage
        self.metrics = defaultdict(list)
    
    def _create_model(self) -> nn.Module:
        """Create model based on method name."""
        hidden_dim = self.config.get('hidden_dim', 128)
        
        # For meta-learning, we predict next observation (state prediction task)
        output_dim = self.state_dim  # Predict next state, not action
        
        if self.method_name == 'ssm':
            from core.ssm import StateSpaceModel
            return StateSpaceModel(
                state_dim=hidden_dim,
                input_dim=self.state_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim
            ).to(self.device)
        
        elif self.method_name in ['mlp', 'lstm', 'gru', 'transformer']:
            return get_baseline_policy(
                self.method_name,
                self.state_dim,
                output_dim,  # Output next state prediction
                hidden_dim
            ).to(self.device)
        
        else:
            raise ValueError(f"Unknown method: {self.method_name}")
    
    def collect_episode_data(self, env, max_steps: int = 200) -> Dict[str, torch.Tensor]:
        """Collect data from one episode."""
        self.model.eval()
        
        obs, _ = env.reset()
        
        # Initialize hidden state if model is stateful
        if hasattr(self.model, 'init_hidden'):
            hidden_state = self.model.init_hidden(batch_size=1)
        else:
            hidden_state = None
        
        observations = []
        actions = []
        rewards = []
        next_observations = []
        
        for step in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if hidden_state is not None:
                    model_output, next_hidden_state = self.model(obs_tensor, hidden_state)
                else:
                    model_output = self.model(obs_tensor)
                    next_hidden_state = None
                
                # For state prediction models, we need to map output to action space
                # Use a simple projection: take first action_dim elements
                if model_output.shape[-1] >= self.action_dim:
                    action_logits = model_output[:, :self.action_dim]
                else:
                    # Fallback if dimensions mismatch (shouldn't happen with correct setup)
                    action_logits = torch.zeros(1, self.action_dim).to(self.device)

                # Sample action (for continuous control, use tanh squashing)
                action = torch.tanh(action_logits).cpu().numpy().flatten()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            
            obs = next_obs
            hidden_state = next_hidden_state
            
            if done:
                break
        
        # Convert to tensors (1, T, D) format
        if not observations:
             # Handle empty episode?
             return {}

        return {
            'observations': torch.tensor(np.array(observations), dtype=torch.float32).unsqueeze(0).to(self.device),
            'actions': torch.tensor(np.array(actions), dtype=torch.float32).unsqueeze(0).to(self.device),
            'rewards': torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device),
            'next_observations': torch.tensor(np.array(next_observations), dtype=torch.float32).unsqueeze(0).to(self.device)
        }
    
    def evaluate_policy(self, env, num_episodes: int = 10) -> float:
        """Evaluate policy on environment."""
        total_reward = 0.0

        for _ in range(num_episodes):
            data = self.collect_episode_data(env, max_steps=1000)
            if 'rewards' in data:
                total_reward += data['rewards'].sum().item()

        return total_reward / max(1, num_episodes)

    def meta_train_step(self, task_id: int) -> float:
        """Perform one meta-training step on a task."""
        env = self.task_dist.sample_task(task_id)
        
        # Collect support and query data
        support_data = self.collect_episode_data(env, max_steps=self.config.get('support_steps', 100))
        query_data = self.collect_episode_data(env, max_steps=self.config.get('query_steps', 100))
        
        env.close()
        
        if not support_data or not query_data:
            return 0.0

        # Prepare data for MetaMAML
        obs_support = support_data['observations']
        next_obs_support = support_data['next_observations']
        obs_query = query_data['observations']
        next_obs_query = query_data['next_observations']
        
        # Check if we have enough data
        if obs_support.shape[1] < 2 or obs_query.shape[1] < 2:
            return 0.0
        
        # Create tasks list
        tasks = [(obs_support, next_obs_support, obs_query, next_obs_query)]
        
        # Initialize hidden state if needed
        if hasattr(self.model, 'init_hidden'):
            initial_hidden = self.model.init_hidden(batch_size=1)
        else:
            initial_hidden = None
        
        # Meta-update
        meta_loss = self.meta_learner.meta_update(
            tasks=tasks,
            initial_hidden_state=initial_hidden,
            loss_fn=nn.MSELoss()
        )
        
        return meta_loss
    
    def meta_test(self, task_id: int, num_adapt_steps: int = 5) -> Dict[str, float]:
        """Test adaptation on a held-out task."""
        env = self.task_dist.sample_task(task_id)
        
        # Collect adaptation data (initial performance)
        adapt_data = self.collect_episode_data(env, max_steps=50)
        
        # Create adapter
        adapt_config = AdaptationConfig(
            learning_rate=self.config.get('adapt_lr', 0.01),
            num_steps=num_adapt_steps
        )
        adapter = Adapter(model=self.model, config=adapt_config, device=self.device)
        
        # Perform adaptation
        obs = adapt_data['observations']
        next_obs = adapt_data['next_observations']
        
        adaptation_losses = []
        
        if hasattr(self.model, 'init_hidden'):
            hidden_state = self.model.init_hidden(batch_size=1)
        else:
            hidden_state = None
        
        # Adapt on each timestep
        steps_taken = 0
        for t in range(min(obs.shape[1], num_adapt_steps)):
            x = obs[:, t, :].unsqueeze(1) # (1, 1, D) for compatibility
            y = next_obs[:, t, :].unsqueeze(1)
            
            x_flat = obs[:, t, :]
            y_flat = next_obs[:, t, :]

            if hidden_state is not None:
                # Adapter.update_step handles detach internally now
                loss, steps = adapter.update_step(x_flat, y_flat, hidden_state)
                steps_taken += steps

                # Update hidden state for next step
                with torch.no_grad():
                    _, hidden_state = self.model(x_flat, hidden_state)
                    # Ensure hidden state is detached for next iteration (double safety)
                    # hidden_state = hidden_state.detach()
                loss, _ = adapter.update_step(x, y, hidden_state)
                # Update hidden state - SSM returns single tensor, not tuple
                with torch.no_grad():
                    _, hidden_state = self.model(x, hidden_state)
                    hidden_state = hidden_state.detach()
            else:
                # For stateless models
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(x_flat)
                    loss = nn.MSELoss()(pred, y_flat).item()
            
            adaptation_losses.append(loss)
        
        # Evaluate final performance after adaptation
        final_reward = self.evaluate_policy(env, num_episodes=1)
        
        env.close()
        
        return {
            'initial_loss': adaptation_losses[0] if adaptation_losses else 0.0,
            'final_loss': adaptation_losses[-1] if adaptation_losses else 0.0,
            'adaptation_losses': adaptation_losses,
            'final_reward': final_reward,
            'num_steps': len(adaptation_losses)
        }

    def train(self, num_epochs: int):
        """Meta-training loop."""
        print(f"Meta-training on {len(self.train_task_ids)} tasks for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            epoch_losses = []

            # Sample tasks for this epoch
            # If we have fewer tasks than tasks_per_epoch, sample with replacement
            batch_size = self.config.get('tasks_per_epoch', 5)
            task_ids = np.random.choice(self.train_task_ids, size=batch_size, replace=True)

            for task_id in task_ids:
                loss = self.meta_train_step(int(task_id))
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            self.metrics['train_loss'].append(avg_loss)
            self.metrics['epoch'].append(epoch)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    def few_shot_test(self, task_id: int, num_shots: Union[int, List[int]] = [1, 5, 10]) -> Dict[int, float]:
        """
        Perform few-shot adaptation and evaluation.
        Args:
            task_id: Test task ID
            num_shots: List of shot counts (gradient steps) to evaluate at
        """
        if isinstance(num_shots, int):
            num_shots = [num_shots]

        sorted_shots = sorted(num_shots)
        max_shots = sorted_shots[-1]

        env = self.task_dist.sample_task(task_id)

        # We need to start from the meta-trained model
        # But since we modify the model during adaptation, we should probably clone it
        # or restore it afterwards. The Adapter modifies weights in-place?
        # Adapter implementation in `adaptation/test_time_adaptation.py` uses an optimizer
        # on the model parameters. So yes, it modifies in-place.
        # We should save state dict before adaptation.
        original_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}

        results = {}

        # Collect adaptation data (one long trajectory or multiple?)
        # "Shots" usually implies N examples/updates.
        # We will perform N updates using data collected on-policy.

        # Initial evaluation (0-shot)
        if 0 in num_shots:
            results[0] = self.evaluate_policy(env, num_episodes=3)

        # Adaptation loop
        # We adapt incrementally
        current_step = 0

        adapt_config = AdaptationConfig(
            learning_rate=self.config.get('adapt_lr', 0.01),
            num_steps=1 # We step manually
        )
        adapter = Adapter(model=self.model, config=adapt_config, device=self.device)

        # Reset env for adaptation
        obs, _ = env.reset()
        if hasattr(self.model, 'init_hidden'):
            hidden_state = self.model.init_hidden(batch_size=1)
        else:
            hidden_state = None

        for shot in range(1, max_shots + 1):
            # Collect one step of data
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if hidden_state is not None:
                    out, next_hidden = self.model(obs_tensor, hidden_state)
                else:
                    out = self.model(obs_tensor)
                    next_hidden = None

                if out.shape[-1] >= self.action_dim:
                    action_logits = out[:, :self.action_dim]
                else:
                    action_logits = torch.zeros(1, self.action_dim).to(self.device)
                action = torch.tanh(action_logits).cpu().numpy().flatten()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Adaptation update
            if hidden_state is not None:
                adapter.update_step(obs_tensor, next_obs_tensor, hidden_state)
            else:
                # Stateless update logic if needed, or skip
                pass

            # Move forward
            obs = next_obs
            if hidden_state is not None:
                hidden_state = next_hidden #.detach() # Adapter handles detach for gradient, but we need new state

            if done:
                obs, _ = env.reset()
                if hasattr(self.model, 'init_hidden'):
                    hidden_state = self.model.init_hidden(batch_size=1)

            # Evaluate if this shot count is requested
            if shot in sorted_shots:
                print(f"  Evaluating at {shot}-shot...")
                # Evaluate on separate episodes (don't mess up adaptation stream?
                # Actually, standard few-shot RL usually separates adaptation phase and eval phase)
                # But here we adapt online.
                # To evaluate "performance at K shots", we should pause, evaluate, resume.
                # But evaluation changes environment state.
                # Ideally we should checkpoint env, but Gym doesn't support that easily.
                # So we might just evaluate and then reset env for next adaptation steps?
                # Or maybe evaluation doesn't affect the 'adaptation' trajectory if we start a fresh episode for eval?
                # Yes, `evaluate_policy` runs fresh episodes.
                # But we need to persist the *model* state. The model weights are updated.
                # The hidden state for adaptation stream is lost if we use the same model for eval.
                # We should save hidden state of adaptation stream?

                # Save adaptation stream state
                saved_obs = obs
                saved_hidden = hidden_state

                score = self.evaluate_policy(env, num_episodes=3)
                results[shot] = score

                # Restore adaptation stream state?
                # Actually, since we reset env for eval, the old env state is lost unless we use a separate env instance.
                # `env` is the same instance. `evaluate_policy` resets it.
                # So we cannot resume adaptation seamlessly on the *same* trajectory.
                # BUT, in MAML RL, usually "K-shot" means "Train on K trajectories".
                # Here we are doing "Train on K steps".
                # If we break the trajectory, it's fine, we just start a new one for adaptation?
                # Let's assume we just continue adaptation from a reset env if needed.
                obs, _ = env.reset()
                if hasattr(self.model, 'init_hidden'):
                    hidden_state = self.model.init_hidden(batch_size=1)

        env.close()

        # Restore model
        self.model.load_state_dict(original_state_dict)

        return results

    def zero_shot_test(self, train_task_ids: List[int], test_task_ids: List[int]) -> Dict[int, float]:
        """
        Train on train tasks, evaluate on test tasks (no adaptation).
        """
        # 1. Train (if epochs > 0)
        epochs = self.config.get('epochs', 0)
        if epochs > 0:
            self.train_task_ids = train_task_ids # Override
            self.train(epochs)

        # 2. Evaluate on test tasks
        results = {}
        print(f"Evaluating Zero-Shot on {len(test_task_ids)} tasks...")
        for tid in test_task_ids:
            env = self.task_dist.sample_task(tid)
            score = self.evaluate_policy(env, num_episodes=5)
            results[tid] = score
            env.close()
            print(f"  Task {tid}: Reward={score:.2f}")

        avg_score = np.mean(list(results.values()))
        print(f"Zero-Shot Average Reward: {avg_score:.2f}")
        return results

    def run(self, num_epochs: int = 50, eval_interval: int = 10):
        """Run standard benchmark (train + periodic eval on held-out)."""
        print(f"\n{'='*70}")
        print(f"Running Benchmark: {self.task_dist_name} with {self.method_name.upper()}")
        print(f"{'='*70}")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"Num tasks: {self.task_dist.num_tasks}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Train loop
        for epoch in range(num_epochs):
            # Train step
            epoch_losses = []
            batch_size = self.config.get('tasks_per_epoch', 5)
            task_ids = np.random.choice(self.train_task_ids, size=batch_size, replace=True)
            
            for task_id in task_ids:
                loss = self.meta_train_step(int(task_id))
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            self.metrics['train_loss'].append(avg_loss)
            
            # Eval
            if (epoch + 1) % eval_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}")
                
                # Test on held-out tasks (if any)
                if self.test_task_ids:
                    test_id = np.random.choice(self.test_task_ids)
                    res = self.meta_test(test_id)
                    self.metrics['test_reward'].append(res['final_reward'])
                    print(f"  Test Task {test_id} - Reward: {res['final_reward']:.2f}")
                else:
                    # Test on a training task?
                    test_id = np.random.choice(self.train_task_ids)
                    res = self.meta_test(test_id)
                    print(f"  Train Task {test_id} (Eval) - Reward: {res['final_reward']:.2f}")

        elapsed_time = time.time() - start_time
        self.metrics['total_time'] = elapsed_time
        
        return self.metrics
    
    def save_results(self, output_dir: str = 'results'):
        """Save benchmark results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{self.task_dist_name}_{self.method_name}_results.json"
        filepath = output_path / filename
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {}
        for key, value in self.metrics.items():
            if isinstance(value, (list, np.ndarray)):
                metrics_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
            else:
                metrics_serializable[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
        
        with open(filepath, 'w') as f:
            json.dump({
                'task_distribution': self.task_dist_name,
                'method': self.method_name,
                'config': self.config,
                'metrics': metrics_serializable
            }, f, indent=2)
        
        print(f"Results saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Run serious benchmarks for SSM-MetaRL')
    parser.add_argument('--task', type=str, default='halfcheetah-vel',
                       choices=list_task_distributions(),
                       help='Task distribution to benchmark')
    parser.add_argument('--method', type=str, default='ssm',
                       choices=['all'] + ['ssm'] + list_baselines(),
                       help='Method to benchmark (or "all" for all methods)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of meta-training epochs')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension for models')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    # New args
    parser.add_argument('--few-shot', type=int, nargs='*',
                       help='Few-shot test with N shots (can specify multiple)')
    parser.add_argument('--zero-shot', action='store_true',
                       help='Run zero-shot evaluation')
    parser.add_argument('--train-tasks', type=int, nargs='+',
                       help='Task IDs for training')
    parser.add_argument('--test-tasks', type=int, nargs='+',
                       help='Task IDs for testing')

    args = parser.parse_args()
    
    # Configuration
    config = {
        'hidden_dim': args.hidden_dim,
        'inner_lr': 0.01,
        'outer_lr': 0.001,
        'adapt_lr': 0.01,
        'support_steps': 100,
        'query_steps': 100,
        'tasks_per_epoch': 5,
        'device': args.device,
        'epochs': args.epochs
    }
    
    if args.train_tasks:
        config['train_tasks'] = args.train_tasks
    if args.test_tasks:
        config['test_tasks'] = args.test_tasks

    # Determine which methods to run
    if args.method == 'all':
        methods = ['ssm'] + list_baselines()
    else:
        methods = [args.method]
    
    # Run benchmarks
    all_results = {}
    
    for method in methods:
        print(f"\n{'#'*70}")
        print(f"# Running: {method.upper()} on {args.task}")
        print(f"{'#'*70}")
        
        try:
            runner = BenchmarkRunner(args.task, method, config)
            
            if args.few_shot:
                # If args.few_shot is a list from nargs='*', use it.
                # If it's None, skip.
                # If it was defined as 'int' before, now 'nargs=*' makes it a list.
                # Example: --few-shot 1 5 10
                shots = args.few_shot if args.few_shot else [1, 5, 10]

                # Train first?
                runner.train(args.epochs)

                # Test on all test tasks? Or just one?
                # "results = runner.few_shot_test(task_id=args.test_task, ...)"
                # CLI didn't specify single test task. Let's run on all test tasks and average.
                print(f"Running Few-Shot Test ({shots} shots) on {len(runner.test_task_ids)} tasks...")

                avg_results = defaultdict(list)
                for test_id in runner.test_task_ids:
                    res = runner.few_shot_test(test_id, shots)
                    for k, v in res.items():
                        avg_results[k].append(v)

                # Aggregate
                final_res = {k: np.mean(v) for k, v in avg_results.items()}
                runner.metrics['few_shot_results'] = final_res
                print("Few-Shot Results:", final_res)

            elif args.zero_shot:
                runner.zero_shot_test(runner.train_task_ids, runner.test_task_ids)

            else:
                runner.run(num_epochs=args.epochs, eval_interval=10)

            runner.save_results(output_dir=args.output_dir)
            all_results[method] = runner.metrics
            
        except Exception as e:
            print(f"Error running {method}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETED")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
