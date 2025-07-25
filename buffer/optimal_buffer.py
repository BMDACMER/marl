import torch as th
import numpy as np
import os
import json
from datetime import datetime


class OptimalBuffer:
    """Buffer specifically designed for storing optimal episode data"""
    
    def __init__(self, args):
        self.args = args
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        
        # Store complete trajectories of optimal episodes
        self.optimal_episodes = []
        self.current_episode = None
        
        # Statistics
        self.total_episodes = 0
        self.best_reward = float('-inf')
        self.best_episode_idx = -1
        
    def start_new_episode(self):
        """Start recording a new episode"""
        self.current_episode = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'avail_actions': [],
            'next_obs': [],
            'masks': [],
            'episode_return': 0,
            'step_count': 0,
            'episode_info': {}
        }
    
    def add_step(self, obs, actions, reward, avail_actions, next_obs, mask, info=None):
        """Add data for one time step"""
        if self.current_episode is None:
            self.start_new_episode()
        
        # Ensure correct data format
        if isinstance(obs, th.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(actions, th.Tensor):
            actions = actions.cpu().numpy()
        if isinstance(avail_actions, th.Tensor):
            avail_actions = avail_actions.cpu().numpy()
        if isinstance(next_obs, th.Tensor):
            next_obs = next_obs.cpu().numpy()
        
        self.current_episode['obs'].append(obs)
        self.current_episode['actions'].append(actions)
        self.current_episode['rewards'].append(reward)
        self.current_episode['avail_actions'].append(avail_actions)
        self.current_episode['next_obs'].append(next_obs)
        self.current_episode['masks'].append(mask)
        
        self.current_episode['episode_return'] += reward
        self.current_episode['step_count'] += 1
        
        if info:
            self.current_episode['episode_info'] = info
    
    def finish_episode(self):
        """Complete recording of current episode"""
        if self.current_episode is None:
            return
        
        # Convert to numpy arrays
        self.current_episode['obs'] = np.array(self.current_episode['obs'])
        self.current_episode['actions'] = np.array(self.current_episode['actions'])
        self.current_episode['rewards'] = np.array(self.current_episode['rewards'])
        self.current_episode['avail_actions'] = np.array(self.current_episode['avail_actions'])
        self.current_episode['next_obs'] = np.array(self.current_episode['next_obs'])
        self.current_episode['masks'] = np.array(self.current_episode['masks'])
        
        # Check if this is the best episode
        if self.current_episode['episode_return'] > self.best_reward:
            self.best_reward = self.current_episode['episode_return']
            self.best_episode_idx = len(self.optimal_episodes)
        
        # Add to episode list
        self.optimal_episodes.append(self.current_episode)
        self.total_episodes += 1
        
        self.current_episode = None
    
    def get_best_episode(self):
        """Get the best episode"""
        if self.best_episode_idx >= 0:
            return self.optimal_episodes[self.best_episode_idx]
        return None
    
    def get_all_episodes(self):
        """Get all episodes"""
        return self.optimal_episodes
    
    def convert_to_rl_format(self, episode_idx=None):
        """Convert episode data to reinforcement learning training format"""
        if episode_idx is None:
            episode_idx = self.best_episode_idx
        
        if episode_idx < 0 or episode_idx >= len(self.optimal_episodes):
            return None
        
        episode = self.optimal_episodes[episode_idx]
        
        # Convert to torch tensor format, compatible with EpisodeBuffer
        obs = th.from_numpy(episode['obs']).float()
        actions = th.from_numpy(episode['actions']).long()
        rewards = th.from_numpy(episode['rewards']).float()
        avail_actions = th.from_numpy(episode['avail_actions']).int()
        next_obs = th.from_numpy(episode['next_obs']).float()
        masks = th.from_numpy(episode['masks']).float()
        
        # Adjust dimensions to match EpisodeBuffer format [n_threads, episode_limit, n_agents, ...]
        batch_size = 1  # Single episode
        episode_len = obs.shape[0]
        n_agents = obs.shape[1] if len(obs.shape) > 1 else 1
        
        # Reshape data
        if len(obs.shape) == 2:  # [steps, obs_dim]
            obs = obs.unsqueeze(1)  # [steps, 1, obs_dim]
            next_obs = next_obs.unsqueeze(1)
        
        if len(actions.shape) == 1:  # [steps]
            actions = actions.unsqueeze(1)  # [steps, 1]
        
        if len(rewards.shape) == 1:  # [steps]
            rewards = rewards.unsqueeze(1)  # [steps, 1]
        
        if len(masks.shape) == 1:  # [steps]
            masks = masks.unsqueeze(1)  # [steps, 1]
        
        # Add batch dimension
        obs = obs.unsqueeze(0)  # [1, steps, agents, obs_dim]
        actions = actions.unsqueeze(0).unsqueeze(-1)  # [1, steps, agents, 1]
        rewards = rewards.unsqueeze(0).unsqueeze(-1)  # [1, steps, agents, 1]
        avail_actions = avail_actions.unsqueeze(0)  # [1, steps, agents, n_actions]
        next_obs = next_obs.unsqueeze(0)  # [1, steps, agents, obs_dim]
        masks = masks.unsqueeze(0).unsqueeze(-1)  # [1, steps, agents, 1]
        
        # Pad to episode_limit length
        if episode_len < self.args.episode_limit:
            pad_len = self.args.episode_limit - episode_len
            
            obs_pad = th.zeros(1, pad_len, obs.shape[2], obs.shape[3])
            obs = th.cat([obs, obs_pad], dim=1)
            
            actions_pad = th.zeros(1, pad_len, actions.shape[2], actions.shape[3], dtype=th.long)
            actions = th.cat([actions, actions_pad], dim=1)
            
            rewards_pad = th.zeros(1, pad_len, rewards.shape[2], rewards.shape[3])
            rewards = th.cat([rewards, rewards_pad], dim=1)
            
            avail_actions_pad = th.zeros(1, pad_len, avail_actions.shape[2], avail_actions.shape[3], dtype=th.int)
            avail_actions = th.cat([avail_actions, avail_actions_pad], dim=1)
            
            next_obs_pad = th.zeros(1, pad_len, next_obs.shape[2], next_obs.shape[3])
            next_obs = th.cat([next_obs, next_obs_pad], dim=1)
            
            masks_pad = th.zeros(1, pad_len, masks.shape[2], masks.shape[3])
            masks = th.cat([masks, masks_pad], dim=1)
        
        return {
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'avail_actions': avail_actions,
            'next_obs': next_obs,
            'masks': masks,
            'episode_info': episode['episode_info']
        }
    
    def save_to_file(self, save_path, episode_idx=None):
        """Save episode data to file"""
        if episode_idx is None:
            episode_idx = self.best_episode_idx
        
        if episode_idx < 0 or episode_idx >= len(self.optimal_episodes):
            print(f"Invalid episode index: {episode_idx}")
            return False
        
        # Create save directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save original data
        episode_data = self.optimal_episodes[episode_idx].copy()
        
        # Convert numpy arrays to lists for JSON serialization
        for key in ['obs', 'actions', 'rewards', 'avail_actions', 'next_obs', 'masks']:
            if isinstance(episode_data[key], np.ndarray):
                episode_data[key] = episode_data[key].tolist()
        
        # Add metadata
        episode_data['metadata'] = {
            'episode_idx': episode_idx,
            'total_episodes': self.total_episodes,
            'best_reward': self.best_reward,
            'save_time': datetime.now().isoformat(),
            'args': vars(self.args)
        }
        
        # Save JSON format
        json_path = save_path.replace('.pt', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False)
        
        # Save PyTorch format (for training)
        rl_format_data = self.convert_to_rl_format(episode_idx)
        if rl_format_data:
            th.save(rl_format_data, save_path)
            print(f"Optimal episode data saved:")
            print(f"  JSON format: {json_path}")
            print(f"  PyTorch format: {save_path}")
            return True
        
        return False
    
    def load_from_file(self, load_path):
        """Load episode data from file"""
        if load_path.endswith('.json'):
            with open(load_path, 'r', encoding='utf-8') as f:
                episode_data = json.load(f)
            
            # Convert back to numpy arrays
            for key in ['obs', 'actions', 'rewards', 'avail_actions', 'next_obs', 'masks']:
                if key in episode_data:
                    episode_data[key] = np.array(episode_data[key])
            
            # Remove metadata
            if 'metadata' in episode_data:
                metadata = episode_data.pop('metadata')
                self.total_episodes = metadata.get('total_episodes', 1)
                self.best_reward = metadata.get('best_reward', episode_data.get('episode_return', 0))
            
            self.optimal_episodes = [episode_data]
            self.best_episode_idx = 0
            
        elif load_path.endswith('.pt'):
            data = th.load(load_path)
            # Reconstruct episode data from PyTorch format
            # This needs to be implemented based on specific save format
            pass
        
        print(f"Optimal episode data loaded: {load_path}")
    
    def get_statistics(self):
        """Get statistics"""
        if not self.optimal_episodes:
            return {}
        
        rewards = [ep['episode_return'] for ep in self.optimal_episodes]
        step_counts = [ep['step_count'] for ep in self.optimal_episodes]
        
        return {
            'total_episodes': self.total_episodes,
            'best_reward': self.best_reward,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_steps': np.mean(step_counts),
            'std_steps': np.std(step_counts),
            'best_episode_idx': self.best_episode_idx
        }