#!/usr/bin/env python3
"""
Standalone evaluation script for DreamerV3 trained models.
Loads a checkpoint and runs evaluation episodes with visualization.
"""

import argparse
import functools
import pathlib
import sys
import torch
import numpy as np
from collections import defaultdict
import tools
from envs.highway import HighwayEnv
from envs.highway_base import ENV_NAME_MAPPING
from HanoiAgent import HanoiAgent
from ruamel.yaml import YAML

def load_config(config_names):
    """Load config from configs.yaml using the same method as dreamer.py."""
    yaml_loader = YAML(typ="safe")
    configs = yaml_loader.load((pathlib.Path(__file__).parent / "configs.yaml").read_text())
    
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value
    
    name_list = ["defaults", *config_names] if config_names else ["defaults"]
    defaults = {}
    for name in name_list:
        if name in configs:
            recursive_update(defaults, configs[name])
    
    # Convert to argparse namespace (same as dreamer.py)
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    
    config = parser.parse_args([])
    return config

def make_env(config, render=False):
    """Create environment based on task."""
    task = config.task

    # Highway-family tasks
    if task.startswith("highway_"):
        env_name = task.split("_", 1)[1]
    elif task in ENV_NAME_MAPPING:
        env_name = task
    else:
        raise ValueError(f"Unknown task: {task}")

    # Choose render_mode
    if render:
        render_mode = "human"        # visualize with window
    else:
        render_mode = "rgb_array"    # off-screen; no window

    env = HighwayEnv(
        name=env_name,
        size=tuple(config.size) if hasattr(config, "size") else (64, 64),
        obs_type=getattr(config, "highway_obs_type", "image"),
        action_type=getattr(config, "highway_action_type", "discrete"),
        action_repeat=config.action_repeat,
        vehicles_count=getattr(config, "highway_vehicles_count", 50),
        vehicles_density=getattr(config, "highway_vehicles_density", 1.5),
        use_reward_shaping=getattr(config, "highway_reward_shaping", True),
        render_mode=render_mode,     # <--- Added param
        offscreen_rendering=not getattr(config, "highway_visualize", False),
    )
    return env

def evaluate(config, agent, env, episodes=5, render=True, *args):
    """Run evaluation episodes."""
    results = defaultdict(list)
    
    for ep in range(episodes):
        obs, info = env.reset()
        agent.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {ep + 1}/{episodes}")
        
        while not done:
            # Get action from agent
            action = agent(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            # Print progress every 50 steps
            if steps % 50 == 0:
                print(f"  Step {steps}: reward={total_reward:.2f}")
        
        # Episode finished
        crashed = info.get("crashed", False)
        status = "CRASHED" if crashed else "SURVIVED"
        print(f"  Done: {steps} steps, reward={total_reward:.2f}, {status}")
        
        results["steps"].append(steps)
        results["reward"].append(total_reward)
        results["crashed"].append(crashed)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Episodes: {episodes}")
    print(f"Avg Steps: {np.mean(results['steps']):.1f} ± {np.std(results['steps']):.1f}")
    print(f"Avg Reward: {np.mean(results['reward']):.2f} ± {np.std(results['reward']):.2f}")
    print(f"Crash Rate: {np.mean(results['crashed'])*100:.1f}%")
    print("="*50)
    
    return results

class HanoiWrapper:
    """Wrapper for HANOI-WORLD agent for evaluation."""
   
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self._state = None
        
    def reset(self):
        self._state = None
        
    def __call__(self, obs):
        """Get action from observation."""
        with torch.no_grad():
            # Prepare observation
            # If env provides kinematics only, synthesize a blank image so the encoder path stays valid.
            if "image" not in obs and "kinematics" in obs:
                w, h = (64, 64)
                if hasattr(self.config, "size"):
                    w, h = self.config.size
                obs = dict(obs)
                obs["image"] = np.zeros((h, w, 3), dtype=np.uint8)
            obs_dict = {}
            for key, val in obs.items():
                if isinstance(val, np.ndarray):
                    dtype = torch.uint8 if val.dtype == np.uint8 else torch.float32
                    obs_dict[key] = torch.as_tensor(val, dtype=dtype).to(self.config.device)
                else:
                    obs_dict[key] = torch.tensor(val, dtype=torch.float32).to(self.config.device)
            
            # Add required fields
            if "is_first" not in obs_dict:
                is_first = 1.0 if self._state is None else 0.0
                obs_dict["is_first"] = torch.tensor([[is_first]], dtype=torch.float32).to(self.config.device)
            
            # Call agent
            action_dict, self._state = self.agent(obs_dict, self._state, training=False)
            
            # Extract action
            action = action_dict["action"].squeeze(0).cpu().numpy()
            
            return action

# Create dummy logger
class DummyLogger:
    def __init__(self):
        self.step = 0
    def scalar(self, *args, **kwargs): pass
    def image(self, *args, **kwargs): pass
    def video(self, *args, **kwargs): pass
    def write(self, *args, **kwargs): pass

def main():

    parser = argparse.ArgumentParser(description="Evaluate DreamerV3 model")
    parser.add_argument("--logdir", type=str, required=True, help="Path to training logdir")
    parser.add_argument("--config", type=str, default="highway", help="Config name from configs.yaml")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()
    
    logdir = pathlib.Path(args.logdir)
    print(f"Loading from: {logdir}")
    
    # Load config from yaml (same method as dreamer.py)
    config = load_config([args.config])
    config.device = args.device
    config.logdir = logdir
    
    print(f"Task: {config.task}")
    print(f"Device: {config.device}")
    
    # Create environment
    env = make_env(config)


    acts = env.action_space
    if hasattr(acts, "n"):
        config.num_actions = acts.n
    elif hasattr(acts, "shape"):
        config.num_actions = int(np.prod(acts.shape))
    else:
        raise ValueError(f"Unsupported action space: {acts}")

    # import the model
    config.embed = 128 # this is the choice of the embedding size

    logger = DummyLogger()

    agent = HanoiAgent(config=config,
                       logger=logger,
                       dataset=None,
                       encoder=None)
    # wrap the agent
    eval_agent = HanoiWrapper(agent=agent,
                              config=config)
    
    results = evaluate(config=config,
                       agent=eval_agent,
                       env=env,
                       episodes=args.episodes,
                       render=True)
    
    env.close()

if __name__ == "__main__":
    main()
