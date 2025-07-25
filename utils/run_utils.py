import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import json


def load_models(args, agent):
    path = os.path.join("results", "models", args.folder, args.checkpoint_path)
    agent.load_models(path)


def train_models(args, agent, runner):
    time = datetime.today().strftime('%Y.%m.%d-%H-%M-%S')
    num_edge_node = str(args.edge_node_num)
    file_name = '-'.join([args.name, num_edge_node, args.log_tag, time])
    log_path = os.path.join(os.getcwd(), "results", "logs", args.folder, file_name)
    logger.add(log_path + ".log", format="{time} {level} {message}", filter="", level="INFO")
    """
    load offline model to guide online model
    """
    if args.offline_models_path != '':
        path = os.path.join(os.getcwd(), "results", "models", args.offline_models_path)
        logger.info("load offline model")
        agent.load_offline_models(path)
    """
    load offline buffer such as o2o_sac_cql
    """
    if args.buffers_path != '':
        path = os.path.join(os.getcwd(), "results", "buffers", args.buffers_path)
        logger.info('load heuristic buffer')
        agent.buffer.load(path)

        if args.normalize_rewards:
            logger.info('normalized reward')
            agent.buffer.normalize_rewards()

    path = os.path.join(os.getcwd(), "results", 'tensorboard', args.folder, file_name)
    writer = SummaryWriter(log_dir=path)
    test_steps = 0
    max_hop_dict_all = {}
    
    episode_rewards_list = []
    cumulative_reward_sum = 0
    training_episode_count = 0
    
    while runner.t_env <= args.t_max:
        episode_info = runner.run(test_mode=False)
        
        if 'episode_return' in episode_info:
            current_reward = episode_info['episode_return']
            episode_rewards_list.append(current_reward)
            cumulative_reward_sum += current_reward
            training_episode_count += 1
            
            writer.add_scalar('Training/Episode_Reward', current_reward, training_episode_count)
            
            # 计算累积平均奖励
            cumulative_avg_reward = cumulative_reward_sum / training_episode_count
            writer.add_scalar('Training/Cumulative_Avg_Reward', cumulative_avg_reward, training_episode_count)
            
            window_size = min(100, len(episode_rewards_list))
            if window_size > 0:
                recent_rewards = episode_rewards_list[-window_size:]
                moving_avg_reward = sum(recent_rewards) / len(recent_rewards)
                writer.add_scalar('Training/Moving_Avg_Reward_100', moving_avg_reward, training_episode_count)
            
            if training_episode_count % 50 == 0:
                logger.info(f"训练Episode {training_episode_count}: "
                           f"Current reward: {current_reward:.4f}, "
                           f"Cumulative average reward: {cumulative_avg_reward:.4f}, "
                           f"Moving average reward: {moving_avg_reward:.4f}")
        
        if runner.t_env // args.test_interval > test_steps:
            test_steps += 1
            res = test_models(args, runner)
            logger.info(res['success_rate_avg'])
            writer.add_scalar('success_rate_avg', res['success_rate_avg'], args.t_start + runner.t_env)
            writer.add_scalar('success_rate_std', res['success_rate_std'], args.t_start + runner.t_env)
            
            if 'episode_return_avg' in res:
                writer.add_scalar('Test/Episode_Return_Avg', res['episode_return_avg'], args.t_start + runner.t_env)
                writer.add_scalar('Test/Episode_Return_Std', res['episode_return_std'], args.t_start + runner.t_env)

            max_hop_dict = res['max_hop_dict']
            for max_hop, count in max_hop_dict.items():
                if max_hop in max_hop_dict_all:
                    max_hop_dict_all[max_hop] += count
                else:
                    max_hop_dict_all[max_hop] = count
    
    if episode_rewards_list:
        _plot_reward_cumulative_curve(writer, episode_rewards_list, file_name)
        
        _plot_reward_distribution(writer, episode_rewards_list)
        
        final_avg_reward = cumulative_reward_sum / len(episode_rewards_list)
        final_max_reward = max(episode_rewards_list)
        final_min_reward = min(episode_rewards_list)
        logger.info(f"Training completed - Total number of episodes: {len(episode_rewards_list)}, "
                   f"final_avg_reward: {final_avg_reward:.4f}, "
                   f"final_max_reward: {final_max_reward:.4f}, "
                   f"final_min_reward: {final_min_reward:.4f}")
    
    fig_hist, ax_hist = plt.subplots()
    max_hop_values = list(max_hop_dict_all.keys())
    task_count_values = list(max_hop_dict_all.values())
    ax_hist.bar(max_hop_values, task_count_values)
    ax_hist.set_xlabel('Max Hop')
    ax_hist.set_ylabel('Task Count')
    writer.add_figure('max_hop_vs_task_count_histogram', fig_hist)
    
    max_hop_values = sorted(max_hop_dict_all.keys())
    task_count_values = [max_hop_dict_all[key] for key in max_hop_values]
    fig, ax = plt.subplots()
    ax.plot(max_hop_values, task_count_values, marker='o')
    ax.set_xlabel('Max Hop')
    ax.set_ylabel('Task Count')
    writer.add_figure('max_hop_vs_task_count_line', fig)

    runner.close_env()
    writer.flush()
    writer.close()
    """
    save buffers
    """
    if args.save_buffers:
        buffers_file_name = '-'.join([file_name, str(agent.buffer.size)])
        buffers_path = os.path.join(os.getcwd(), "results", "buffers", args.folder, buffers_file_name)
        os.makedirs(buffers_path, exist_ok=True)
        agent.buffer.save(buffers_path)
    """
    save models
    """
    if args.save_models:
        models_path = os.path.join(os.getcwd(), "results", "models", args.folder, file_name)
        os.makedirs(models_path, exist_ok=True)
        agent.save_models(models_path)


def _plot_reward_cumulative_curve(writer, episode_rewards_list, file_name):
    episodes = list(range(1, len(episode_rewards_list) + 1))
    
    cumulative_avg_rewards = []
    cumulative_sum = 0
    for i, reward in enumerate(episode_rewards_list):
        cumulative_sum += reward
        cumulative_avg_rewards.append(cumulative_sum / (i + 1))
    
    window_size = 50
    moving_avg_rewards = []
    for i in range(len(episode_rewards_list)):
        start_idx = max(0, i - window_size + 1)
        window_rewards = episode_rewards_list[start_idx:i+1]
        moving_avg_rewards.append(sum(window_rewards) / len(window_rewards))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(episodes, episode_rewards_list, alpha=0.3, color='lightblue',
            linewidth=1, label='Original Episode reward')
    
    ax.plot(episodes, moving_avg_rewards, color='orange',
            linewidth=2, label=f'Moving average reward (window size={window_size})')
    
    ax.plot(episodes, cumulative_avg_rewards, color='red',
            linewidth=2.5, label='Cumulative average reward')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('reward')
    ax.set_title('Reward accumulation curve during the training process of PPO algorithm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    final_avg = cumulative_avg_rewards[-1]
    max_reward = max(episode_rewards_list)
    min_reward = min(episode_rewards_list)
    std_reward = np.std(episode_rewards_list)
    
    stats_text = f'Avg: {final_avg:.2f}\nStd: {std_reward:.2f}\nMax: {max_reward:.2f}\nMin: {min_reward:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    writer.add_figure('Training/Reward_Cumulative_Curve', fig)
    plt.close(fig)


def _plot_reward_distribution(writer, episode_rewards_list):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_bins = min(50, len(set(episode_rewards_list)))  # 自适应bin数量
    ax.hist(episode_rewards_list, bins=n_bins, alpha=0.7, 
            color='skyblue', edgecolor='black')
    
    mean_reward = np.mean(episode_rewards_list)
    # ax.axvline(mean_reward, color='red', linestyle='--', linewidth=2,
    #            label=f'avg: {mean_reward:.2f}')
    
    ax.set_xlabel('reward')
    ax.set_ylabel('frequency')
    ax.set_title('Episode reward distribution during training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    writer.add_figure('Training/Reward_Distribution', fig)
    
    plt.close(fig)


def train_expert_models(args, agent, runner):
    time = datetime.today().strftime('%Y.%m.%d-%H-%M-%S')
    file_name = '-'.join([args.name, args.log_tag, time])
    path = os.path.join(os.getcwd(), "results", "tensorboard", args.folder, file_name)
    writer = SummaryWriter(log_dir=path)
    test_steps = 0

    while runner.t_env <= args.t_max:
        t_expert = runner.t_env
        runner.run(test_mode=False, t_expert=t_expert)
        if runner.t_env // args.test_interval > test_steps:
            test_steps += 1

            res = test_models_llm(args, runner, t_expert)
            logger.info(res['success_rate_avg'])
            writer.add_scalar('success_rate_avg', res['success_rate_avg'], args.t_start + runner.t_env)
            writer.add_scalar('success_rate_std', res['success_rate_std'], args.t_start + runner.t_env)
    runner.close_env()
    writer.flush()
    writer.close()

    if args.save_models:
        models_path = os.path.join(os.getcwd(), "results", "models", args.folder, file_name)
        os.makedirs(models_path, exist_ok=True)
        agent.save_models(models_path)


def test_models_llm(args, runner, t_expert):
    success_rate_list = []
    # drop_rate_list = []
    # failure_rate_list = []
    # task_completion_time_list = []
    n_test_runs = max(1, args.test_nepisode // args.n_threads)
    for _ in range(n_test_runs):
        info = runner.run(test_mode=True, t_expert=t_expert)
        success_rate_list.append(info["success_rate"])
    res = {}
    res['success_rate_avg'] = np.mean(success_rate_list, axis=0)
    res['success_rate_std'] = np.std(success_rate_list, axis=0)
    return res


def train_offline_models(args, agent, runner):
    time = datetime.today().strftime('%Y.%m.%d-%H-%M-%S')
    file_name = '-'.join([args.name, args.log_tag, time])
    path = os.path.join(os.getcwd(), "results", "tensorboard", args.folder, file_name)
    writer = SummaryWriter(log_dir=path)
    current_train_times = 0

    """
    load buffers
    """
    path = os.path.join(os.getcwd(), "results", "buffers", args.buffers_path)
    agent.buffer.load(path)

    if args.normalize_rewards:
        agent.buffer.normalize_rewards()

    """
    load pre train offline model
    """
    if args.checkpoint_path != '':
        path = os.path.join(os.getcwd(), "results", "models", args.checkpoint_path)
        agent.load_models(path)

    """
    train offline model
    """
    while current_train_times < args.t_max:
        agent.train()
        current_train_times += 1
        if current_train_times % args.test_interval == 0:
            res = test_models(args, runner)
            writer.add_scalar('success_rate_avg', res['success_rate_avg'], current_train_times)
            writer.add_scalar('success_rate_std', res['success_rate_std'], current_train_times)

    """
    save models
    """
    save_path = os.path.join(os.getcwd(), "results", "models", args.folder, file_name)
    os.makedirs(save_path, exist_ok=True)
    agent.save_models(save_path)


def test_models(args, runner):
    success_rate_list = []
    max_hop_dict_all = {}
    episode_return_list = []
    
    n_test_runs = max(1, args.test_nepisode // args.n_threads)
    for _ in range(n_test_runs):
        info = runner.run(test_mode=True)
        success_rate_list.append(info["success_rate"])
        
        if "episode_return" in info:
            episode_return_list.append(info["episode_return"])
        
        max_hop_dict = info["max_hop_dict"]
        for max_hop, count in max_hop_dict.items():
            if max_hop in max_hop_dict_all:
                max_hop_dict_all[max_hop] += count
            else:
                max_hop_dict_all[max_hop] = count
    
    res = {}
    res['success_rate_avg'] = np.mean(success_rate_list, axis=0)
    res['success_rate_std'] = np.std(success_rate_list, axis=0)
    res['max_hop_dict'] = max_hop_dict_all
    
    if episode_return_list:
        res['episode_return_avg'] = np.mean(episode_return_list)
        res['episode_return_std'] = np.std(episode_return_list)
        res['episode_return_max'] = np.max(episode_return_list)
        res['episode_return_min'] = np.min(episode_return_list)
    
    return res


def only_test_models(args, runner):
    file_name = '-'.join([args.name, args.log_tag])
    """
    logs
    """
    log_path = os.path.join(os.getcwd(), "results", "logs", file_name)
    logger.add(log_path + ".log", format="{time} {level} {message}", filter="", level="INFO")
    success_rate = 0
    drop_rate = 0
    failure_rate = 0
    task_completion_time = 0
    success_rate_list = []
    n_test_runs = max(1, args.test_nepisode // args.n_threads)
    for _ in range(n_test_runs):
        info = runner.run(test_mode=True)
        success_rate += info["success_rate"]
        drop_rate += info["drop_rate"]
        failure_rate += info["failure_rate"]
        task_completion_time += info["task_completion_time"]
        success_rate_list.append(info["success_rate"])
    task_completion_time /= n_test_runs
    success_rate /= n_test_runs
    drop_rate /= n_test_runs
    failure_rate /= n_test_runs
    success_std = np.std(success_rate_list, axis=0)
    logger.info(
        f'success rate {success_rate} std {success_std} drop rate {drop_rate} failure rate {failure_rate} task completion time {task_completion_time}')
