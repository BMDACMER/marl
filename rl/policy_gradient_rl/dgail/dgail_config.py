def add_dgail_args(parser):
    
    # ===== Basic Configuration =====
    parser.add_argument('--name', type=str, default='dgail', help='Algorithm name')
    parser.add_argument('--algo_type', type=str, default='rl', help='Algorithm type')
    parser.add_argument('--runner', type=str, default='dgail_runner', help='Runner type, use DRAIL specific runner')
    parser.add_argument('--max_expert_buffer_size', type=int, default=20000, help='Maximum size of expert data buffer')
    
    # ===== Network Architecture Parameters =====
    parser.add_argument('--n_threads', type=int, default=1, help='Number of parallel threads')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer dimension')
    parser.add_argument('--activate_fun', type=str, default='relu', choices=['relu', 'tanh', 'leaky_relu'], help='Activation function type')
    
    # ===== PPO Training Parameters =====
    parser.add_argument('--batch_size_run', type=int, default=100, help='Number of episodes to run before each training')
    parser.add_argument('--ppo_batch_size', type=int, default=64, help='PPO update mini-batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lambda_', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--policy_lr', type=float, default=2e-5, help='Policy network learning rate')
    parser.add_argument('--value_lr', type=float, default=2e-5, help='Value network learning rate')
    parser.add_argument('--ppo_epochs', type=int, default=4, help='Number of PPO update epochs')
    parser.add_argument('--clip_param', type=float, default=0.2, help='PPO clipping parameter')
    parser.add_argument('--value_loss_coef', type=float, default=0.2, help='Value function loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=8e-4, help='Entropy regularization coefficient')
    
    # ===== Diffusion Discriminator Network Parameters =====
    parser.add_argument('--discriminator_lr', type=float, default=1e-4, help='Diffusion discriminator learning rate')
    parser.add_argument('--gail_discriminator_lr', type=float, default=1e-4, help='GAIL discriminator learning rate')
    parser.add_argument('--grad_norm_clip', type=float, default=1.0, help='Gradient clipping norm')
    
    # ===== Model Save and Load =====
    parser.add_argument('--load_models', type=bool, default=False, help='Whether to load pre-trained models')
    parser.add_argument('--save_models', type=bool, default=True, help='Whether to save trained models')
    
    # ===== Hybrid Algorithm Control Parameters =====
    parser.add_argument('--switch_to_gail_at_step', type=int, default=-1,
                        help='Switch to GAIL after step 8000. Set to -1 to always use simplified DGAIL')

    # ===== Dynamic Weight Annealing Parameters =====
    parser.add_argument('--imit_weight_start', type=float, default=1.5, help='Initial imitation reward weight')
    parser.add_argument('--imit_weight_end', type=float, default=0.8, help='Final imitation reward weight after annealing')
    parser.add_argument('--use_simplified_reward', type=bool, default=True, help='Use simplified reward calculation to avoid complex diffusion loss computation')

    # ===== Behavior Cloning (BC) Parameters =====
    parser.add_argument('--bc_loss_weight', type=float, default=0.3, help='Initial BC loss weight in PPO updates')
    parser.add_argument('--bc_decay_rate', type=float, default=0.995, help='BC weight decay coefficient after each training')
    parser.add_argument('--min_bc_weight', type=float, default=0.05, help='Minimum BC weight')
    parser.add_argument('--batch_expert_transitions', type=int, default=64, help='Batch size for expert transitions')

    return parser