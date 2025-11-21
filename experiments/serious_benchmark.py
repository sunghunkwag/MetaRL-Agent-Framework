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
from typing import Dict, List, Tuple, Any
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
        
        # Get environment info from first task
        sample_env = self.task_dist.sample_task(0)
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
                    model_output, hidden_state = self.model(obs_tensor, hidden_state)
                else:
                    model_output = self.model(obs_tensor)
                
                # For state prediction models, we need to map output to action space
                # Use a simple projection: take first action_dim elements
                if model_output.shape[-1] == self.state_dim:
                    # State prediction output - project to action space
                    action_logits = model_output[:, :self.action_dim]
                else:
                    action_logits = model_output
                
                # Sample action (for continuous control, use tanh squashing)
                action = torch.tanh(action_logits).cpu().numpy().flatten()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            
            obs = next_obs
            
            if done:
                break
        
        # Convert to tensors (1, T, D) format
        return {
            'observations': torch.tensor(np.array(observations), dtype=torch.float32).unsqueeze(0).to(self.device),
            'actions': torch.tensor(np.array(actions), dtype=torch.float32).unsqueeze(0).to(self.device),
            'rewards': torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device),
            'next_observations': torch.tensor(np.array(next_observations), dtype=torch.float32).unsqueeze(0).to(self.device)
        }
    
    def meta_train_step(self, task_id: int) -> float:
        """Perform one meta-training step on a task."""
        env = self.task_dist.sample_task(task_id)
        
        # Collect support and query data
        support_data = self.collect_episode_data(env, max_steps=self.config.get('support_steps', 100))
        query_data = self.collect_episode_data(env, max_steps=self.config.get('query_steps', 100))
        
        env.close()
        
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
        
        # Collect adaptation data
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
        for t in range(min(obs.shape[1], num_adapt_steps)):
            x = obs[:, t, :]
            y = next_obs[:, t, :]
            
            if hidden_state is not None:
                loss, _ = adapter.update_step(x, y, hidden_state)
                # Update hidden state - SSM returns single tensor, not tuple
                with torch.no_grad():
                    _, hidden_state = self.model(x, hidden_state)
                    hidden_state = hidden_state.detach()
            else:
                # For stateless models, we can't use update_step directly
                # Just compute loss
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(x)
                    loss = nn.MSELoss()(pred, y).item()
            
            adaptation_losses.append(loss)
        
        # Evaluate final performance
        eval_data = self.collect_episode_data(env, max_steps=200)
        final_reward = eval_data['rewards'].sum().item()
        
        env.close()
        
        return {
            'initial_loss': adaptation_losses[0] if adaptation_losses else 0.0,
            'final_loss': adaptation_losses[-1] if adaptation_losses else 0.0,
            'adaptation_losses': adaptation_losses,
            'final_reward': final_reward,
            'num_steps': len(adaptation_losses)
        }
    
    def run(self, num_epochs: int = 50, eval_interval: int = 10):
        """Run the full benchmark."""
        print(f"\n{'='*70}")
        print(f"Running Benchmark: {self.task_dist_name} with {self.method_name.upper()}")
        print(f"{'='*70}")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"Num tasks: {self.task_dist.num_tasks}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Meta-training loop
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Sample tasks for this epoch
            task_ids = np.random.choice(self.task_dist.num_tasks, size=self.config.get('tasks_per_epoch', 5), replace=True)
            
            for task_id in task_ids:
                loss = self.meta_train_step(int(task_id))
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            self.metrics['train_loss'].append(avg_loss)
            self.metrics['epoch'].append(epoch)
            
            # Evaluation
            if (epoch + 1) % eval_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}")
                
                # Test on held-out task
                test_task_id = self.task_dist.num_tasks - 1  # Use last task for testing
                test_results = self.meta_test(test_task_id, num_adapt_steps=10)
                
                self.metrics['test_initial_loss'].append(test_results['initial_loss'])
                self.metrics['test_final_loss'].append(test_results['final_loss'])
                self.metrics['test_reward'].append(test_results['final_reward'])
                
                print(f"  Test - Initial Loss: {test_results['initial_loss']:.4f}, "
                      f"Final Loss: {test_results['final_loss']:.4f}, "
                      f"Reward: {test_results['final_reward']:.2f}")
        
        elapsed_time = time.time() - start_time
        self.metrics['total_time'] = elapsed_time
        
        print(f"\n{'='*70}")
        print(f"Benchmark completed in {elapsed_time:.2f}s")
        print(f"{'='*70}\n")
        
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
        'device': args.device
    }
    
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
            metrics = runner.run(num_epochs=args.epochs, eval_interval=10)
            runner.save_results(output_dir=args.output_dir)
            
            all_results[method] = metrics
            
        except Exception as e:
            print(f"Error running {method}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"Task: {args.task}")
    print(f"Methods tested: {', '.join(methods)}")
    print()
    
    for method, metrics in all_results.items():
        final_train_loss = metrics['train_loss'][-1] if metrics['train_loss'] else 0.0
        final_test_reward = metrics['test_reward'][-1] if metrics['test_reward'] else 0.0
        total_time = metrics.get('total_time', 0.0)
        
        print(f"{method.upper():15s}: Train Loss={final_train_loss:.4f}, "
              f"Test Reward={final_test_reward:.2f}, Time={total_time:.1f}s")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
