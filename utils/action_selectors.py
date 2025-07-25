import torch.nn.functional as F
from torch.distributions import Categorical
import torch as th


def random_selector(avail_actions):
    random_actions = Categorical(avail_actions).sample().long()
    return random_actions


def soft_selector(agent_outs):
    agent_outs = F.softmax(agent_outs, dim=-1)
    dist = Categorical(agent_outs)
    picked_actions = dist.sample().long()
    return picked_actions


def greedy_selector(agent_outs):
    picked_actions = agent_outs.argmax(dim=-1)
    return picked_actions


def train_epsilon_greedy_selector(args, agent_outs, avail_actions):
    random_numbers = th.rand(args.n_agents, 1)
    # ! epsilon
    pick_random = (random_numbers < args.epsilon).long().to(args.device)
    random_actions = Categorical(avail_actions).sample().unsqueeze(-1).long()
    agent_actions = agent_outs.argmax(dim=-1).unsqueeze(-1)
    picked_actions = pick_random * random_actions + (1 - pick_random) * agent_actions

    # delta = (args.epsilon - args.min_epsilon) / args.epsilon_anneal_time
    # args.epsilon = max(args.epsilon - delta * args.n_threads, args.min_epsilon)
    args.epsilon = max(args.min_epsilon, args.epsilon * (1 - args.epsilon_tau))
    return picked_actions


def test_epsilon_greedy_selector(args, agent_outs, avail_actions):
    random_numbers = th.rand(args.n_agents, 1)
    # ! min_epsilon
    pick_random = (random_numbers < args.min_epsilon).long().to(args.device)
    random_actions = Categorical(avail_actions).sample().unsqueeze(-1).long()
    agent_actions = agent_outs.argmax(dim=-1).unsqueeze(-1)
    picked_actions = pick_random * random_actions + (1 - pick_random) * agent_actions
    return picked_actions
