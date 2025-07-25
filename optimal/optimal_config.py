def add_optimal_args(parser):
    """Add optimal algorithm related parameters"""
    parser.add_argument('--name', type=str, default='optimal')
    parser.add_argument('--algo_type', type=str, default='optimal')
    parser.add_argument('--runner', type=str, default='optimal')
    parser.add_argument('--n_threads', type=int, default=1)
    parser.add_argument('--test_models', type=bool, default=True)
    parser.add_argument('--test_nepisode', type=int, default=1)
    parser.add_argument('--t_max', type=int, default=1)   # Control training iterations
    # Compatibility parameters
    parser.add_argument('--load_models', type=bool, default=False)
    parser.add_argument('--save_models', type=bool, default=True)
    parser.add_argument('--save_buffers', type=bool, default=True)
    # Search related parameters
    parser.add_argument('--max_search_nodes', type=int, default=1000,
                       help='Maximum number of search nodes')
    parser.add_argument('--search_time_limit', type=int, default=360,
                       help='Search time limit (seconds)')
    parser.add_argument('--pruning_threshold', type=float, default=0.01, 
                       help='Pruning threshold')
    parser.add_argument('--max_action_combinations', type=int, default=1000, 
                       help='Maximum number of action combinations per node')
    # Save related parameters
    parser.add_argument('--save_optimal_episode', type=bool, default=True, 
                       help='Whether to save optimal episode')
    parser.add_argument('--save_search_log', type=bool, default=True,
                       help='Whether to save search log')

    return parser