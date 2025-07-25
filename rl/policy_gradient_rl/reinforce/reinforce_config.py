def reinforce_config(args):
    args.name = "reinforce"
    args.algo_type = "rl"

    args.runner = "episode"
    args.n_threads = 1

    # args.runner = "parallel"
    # args.n_threads = 10

    args.hidden_dim = 64  # Size of hidden state for default rnn agent

    args.gamma = 0.99
    args.t_max = 300000
    args.lr = 0.0005  # Learning rate for agents
    args.normalize_rewards = True
    return args
