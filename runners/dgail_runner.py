import torch as th
from envs.env_register import env_register


class DGAILRunner:

    def __init__(self, args, agent):
        """
        Initialize the DRAIL runner
        
        参数:
            args: Configuration parameters
            agent: DRAIL Agent instance
        """
        self.args = args
        self.agent = agent
        self.env = env_register[args.env_name](args)
        self.t_env = 0
        
        self.expert_usage_history = []
        self.reward_statistics = []
        self.episode_count = 0
        
        # ------------------------------------------------------------------
        self.bc_guidance_steps = getattr(args, 'bc_guidance_steps', 2_000)
        self.expert_guidance_steps = getattr(args, 'expert_guidance_steps', 6_000)
        # ------------------------------------------------------------------


    def _expert_ratio(self) -> float:
        if self.t_env < self.bc_guidance_steps:
            return 1.0
        if self.t_env < self.expert_guidance_steps:
            progress = (self.t_env - self.bc_guidance_steps) / (
                self.expert_guidance_steps - self.bc_guidance_steps
            )
            return max(0.2, 0.9 * (1 - progress))
        return 0.2

    def _take_env_step(self, episode_stats: dict, test_mode: bool) -> bool:
        obs = th.as_tensor(self.env.get_obs(), dtype=th.float)
        avail = th.as_tensor(self.env.get_avail_actions(), dtype=th.int)

        # ========== 决策阶段 ==========
        expert_actions = None
        if (not test_mode) and (th.rand(1).item() < self._expert_ratio()):
            try:
                expert_np = self.agent.get_expert_actions(self.env)
                if expert_np is not None:
                    expert_actions = th.tensor(expert_np, dtype=th.long, device=obs.device)
            except Exception as e:
                print(f"[WARN] Failed to acquire the expert action: {e}")

        # 智能体动作（在使用专家执行时仍计算一次供训练）
        agent_actions = self.agent.select_actions(obs.unsqueeze(0),
                                                  avail.unsqueeze(0),
                                                  test_mode)

        exec_actions = expert_actions if expert_actions is not None else agent_actions
        cpu_actions = exec_actions.to("cpu").numpy()
        if cpu_actions.ndim == 2 and cpu_actions.shape[0] == 1:
            cpu_actions = cpu_actions[0]

        reward, terminated, info = self.env.step(cpu_actions.tolist())

        episode_stats['episode_steps'] += 1
        episode_stats['episode_return'] += reward
        for k in ['task_completion_time', 'failure_task_number',
                  'drop_task_number', 'finish_task_number',
                  'success_finish_task_number']:
            episode_stats[k] += info.get(k, 0)

        if "max_hop" in info:
            mh = info["max_hop"]
            episode_stats['max_hop_dict'][mh] = episode_stats['max_hop_dict'].get(mh, 0) + 1

        if not test_mode:
            self.t_env += 1
            next_obs = th.as_tensor(self.env.get_obs(), dtype=th.float)

            store_actions = agent_actions if expert_actions is not None else exec_actions

            r_tensor = th.as_tensor(reward).unsqueeze(0).unsqueeze(0).repeat(self.args.n_agents, 1)
            m_tensor = th.as_tensor(1 - int(terminated)).unsqueeze(0).repeat(self.args.n_agents, 1)

            self.agent.buffer.insert(
                obs.unsqueeze(0), avail.unsqueeze(0),
                store_actions.unsqueeze(0).unsqueeze(-1),
                r_tensor.unsqueeze(0), m_tensor.unsqueeze(0),
                next_obs.unsqueeze(0)
            )

            if expert_actions is not None:
                self.agent.store_expert_data(obs, expert_actions)

        return bool(terminated)

    def run(self, test_mode: bool = False) -> dict:
        self.agent.buffer.reset()
        self.env.reset()

        stats = dict(task_completion_time=0, failure_task_number=0,
                     drop_task_number=0, finish_task_number=0,
                     success_finish_task_number=0, episode_return=0,
                     max_hop_dict={}, episode_steps=0)

        terminated = False
        while not terminated:
            terminated = self._take_env_step(stats, test_mode)

        # --------  Episode  --------
        finish_cnt = max(stats['finish_task_number'], 1)
        success_base = max(stats['success_finish_task_number'] + stats['drop_task_number'], 1)

        ep_info = {
            "episode_return": stats['episode_return'],
            "success_rate": stats['success_finish_task_number'] / finish_cnt,
            "drop_rate": stats['drop_task_number'] / finish_cnt,
            "failure_rate": stats['failure_task_number'] / finish_cnt,
            "task_completion_time": stats['task_completion_time'] / success_base,
            "max_hop_dict": stats['max_hop_dict'],
            "episode_steps": stats['episode_steps'],
            "expert_buffer_size": len(self.agent.expert_transitions),
        }

        # --------  train stage --------
        if not test_mode and len(self.agent.expert_transitions) > self.args.batch_size_run:
            train_stats = self.agent.train(self.t_env)
            if train_stats:
                ep_info.update(train_stats)
            self.agent.update_performance_monitoring(ep_info['success_rate'])

        self.episode_count += 1
        if not test_mode and self.episode_count % 10 == 0:
            self._print_detailed_stats(ep_info)

        return ep_info

    def _print_detailed_stats(self, episode_info):
        print(f"\n--- Episode {self.episode_count} | Env Steps {self.t_env} ---")
        print(f"  success rate: {episode_info.get('success_rate', 0):.4f} | "
              f"Episode return: {episode_info.get('episode_return', 0):.2f}")
        print(f"  expert buffer size: {episode_info.get('expert_buffer_size', 0)}")
        
        disc_loss = episode_info.get('discriminator_loss', -1)
        prob_expert = episode_info.get('prob_expert', -1)
        prob_agent = episode_info.get('prob_agent', -1)
        imitation_reward = episode_info.get('avg_imitation_reward', 0)
        combined_reward = episode_info.get('avg_combined_reward', 0)

        print(f"  loss of discriminator: {disc_loss:.4f} | discriminator output -> expert: {prob_expert:.4f}, agent: {prob_agent:.4f}")
        print(f"  imitation_reward: {imitation_reward:.4f} | combined_reward: {combined_reward:.4f}")
        print("-" * 50)

    def close_env(self):
        self.env.close()
