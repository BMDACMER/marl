def add_ac_config(parser):
    parser.add_argument('--name', type=str, default='ac')
    parser.add_argument('--algo_type', type=str, default='rl')
    parser.add_argument('--runner', type=str, default='episode')
    parser.add_argument('--n_threads', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--normalize_rewards', type=bool, default=True)
    parser.add_argument('--test_models', type=bool, default=False)

    return parser
