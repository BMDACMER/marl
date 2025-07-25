def add_ilets_config(parser):
    parser.add_argument('--name', type=str, default='ilets')
    parser.add_argument('--algo_type', type=str, default='rl')
    parser.add_argument('--runner', type=str, default='ilets_runner')
    parser.add_argument('--n_threads', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--normalize_rewards', type=bool, default=True)
    
    parser.add_argument('--bc_loss_weight', type=float, default=2.0)
    parser.add_argument('--bc_decay_rate', type=float, default=0.9995)
    parser.add_argument('--min_bc_weight', type=float, default=0.8)
    parser.add_argument('--imitation_threshold', type=float, default=0.75)
    
    parser.add_argument('--max_expert_buffer_size', type=int, default=1000)
    parser.add_argument('--expert_collection_frequency', type=int, default=10)
    parser.add_argument('--expert_guidance_steps', type=int, default=30000)
    parser.add_argument('--bc_guidance_steps', type=int, default=8000)
    
    parser.add_argument('--value_loss_coeff', type=float, default=0.5)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--use_grad_clip', type=bool, default=True)
    parser.add_argument('--grad_norm_clip', type=float, default=5.0)
    
    parser.add_argument('--training_frequency', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1)
    
    parser.add_argument('--test_models', type=bool, default=False)
    parser.add_argument('--save_models', type=bool, default=True)
    parser.add_argument('--save_interval', type=int, default=5000)
    
    return parser
