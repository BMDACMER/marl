def add_gail_args(parser):

    parser.add_argument('--name', type=str, default='gail')
    parser.add_argument('--algo_type', type=str, default='rl')
    parser.add_argument('--runner', type=str, default='gail_runner')
    parser.add_argument('--expert_guidance_steps', type=int, default=40000)
    parser.add_argument('--bc_guidance_steps', type=int, default=2000)
    parser.add_argument('--max_expert_buffer_size', type=int, default=1000)
    parser.add_argument('--n_threads', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambda_', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--eps_clip', type=float, default=0.1)
    
    parser.add_argument('--bc_loss_weight', type=float, default=2)
    parser.add_argument('--bc_decay_rate', type=float, default=0.998)
    parser.add_argument('--min_bc_weight', type=float, default=0.3)

    parser.add_argument('--use_entropy', type=bool, default=True)
    parser.add_argument('--entropy_coef', type=float, default=0.005)
    parser.add_argument('--use_grad_clip', type=bool, default=True)
    parser.add_argument('--grad_norm_clip', type=float, default=10.0)
    
    parser.add_argument('--activate_fun', type=str, default='relu')
    parser.add_argument('--normalize_rewards', type=bool, default=True)
    parser.add_argument('--normalize_advantages', type=bool, default=True)
    
    parser.add_argument('--expert', type=bool, default=False)
    parser.add_argument('--load_models', type=bool, default=False)
    parser.add_argument('--save_models', type=bool, default=True)
    parser.add_argument('--save_buffers', type=bool, default=False)

    return parser 