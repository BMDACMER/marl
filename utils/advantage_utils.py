import torch as th


def get_gae(args, rewards, obs_values, next_obs_value, masks):
    advantages = th.zeros(args.n_threads, args.episode_limit + 1, args.n_agents, 1, device=args.device)
    for t in range(args.episode_limit - 1, -1, -1):
        delta = rewards[:, t] + args.gamma * next_obs_value[:, t] * masks[:, t] - obs_values[:, t]
        advantages[:, t] = delta + args.gamma * args.lambda_ * advantages[:, t + 1] * masks[:, t]
    return advantages[:, :-1]


def get_returns(args, rewards, masks):
    returns = th.zeros(args.n_threads, args.episode_limit + 1, args.n_agents, 1, device=args.device)
    for t in range(args.episode_limit - 1, -1, -1):
        returns[:, t] = rewards[:, t] + args.gamma * returns[:, t + 1] * masks[:, t]
    return returns[:, :-1]


def n_step_returns(args, rewards, mask, values):
    n_step_values = th.zeros_like(values, device=args.device)
    for t_start in range(rewards.size(1)):
        t_start_return = th.zeros_like(values[:, 0], device=args.device)
        for step in range(args.q_nstep + 1):
            t = t_start + step
            if t >= rewards.size(1):
                break
            elif step == args.q_nstep or t == rewards.size(1) - 1:
                t_start_return += args.gamma ** step * values[:, t] * mask[:, t]
            else:
                t_start_return += args.gamma ** step * rewards[:, t] * mask[:, t]
        n_step_values[:, t_start] = t_start_return
    return n_step_values
