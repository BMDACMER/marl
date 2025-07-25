def add_d2sac_args(parser):
    parser.add_argument('--name', type=str, default='d2sac')
    parser.add_argument('--algo_type', type=str, default='rl')
    parser.add_argument('--runner', type=str, default='step')  # It shares the same Runner with SAC

    parser.add_argument('--n_threads', type=int, default=1)

    parser.add_argument('--soft_update', type=bool, default=True)
    parser.add_argument('--tau', type=float, default=0.0005)
    parser.add_argument('--hard_update', type=bool, default=False)
    parser.add_argument('--target_update_interval', type=int, default=8000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_size', type=int, default=1_000_000)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--normalize_rewards', type=bool, default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--start_training_size', type=int, default=10_000)
    parser.add_argument('--save_models', type=bool, default=True)

    parser.add_argument('--adaptive_alpha', action='store_true', default=True)
    parser.add_argument('--alpha', type=float, default=0.05)

    parser.add_argument('--offline', default=False, action='store_true')
    parser.add_argument('--use_cql', default=True, action='store_true')
    parser.add_argument('--cql_weight', type=float, default=0.1)
    parser.add_argument('--load_models', type=bool, default=False)
    parser.add_argument('--expert_buffers_path', type=str, default='')

    parser.add_argument('--diffusion_steps', type=int, default=4, help='The number of steps in the diffusion model T')
    parser.add_argument('--diffusion_beta', type=float, default=0.1, help='Diffuse noise intensity β')

    # critic
    parser.add_argument('--add_critic', default=True, action='store_true')

    parser.add_argument('--grad_clip', type=float, default=10.0, help='Gradient clipping threshold')
    return parser