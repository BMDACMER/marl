def add_edge_computing_env_args(parser):
    parser.add_argument('--env_name', type=str, default='edge_computing')
    parser.add_argument('--episode_limit', type=int, default=100)
    parser.add_argument('--time_slot', type=float, default=0.5)
    parser.add_argument('--mini_time_slot', type=float, default=0.05)
    parser.add_argument('--mini_time_slot_num', type=int, default=10)
    # Number of tasks [10, 20, 30, 40, 50]
    parser.add_argument('--edge_node_num', type=int, default=10)
    # Task arrival/generation rate [0.2, 0.4, 0.6, 0.8, 1.0]
    parser.add_argument('--task_probability_min', type=float, default=0)
    parser.add_argument('--task_probability_max', type=float, default=1.0)
    # Task size [2000, 4000, 6000, 8000, 10000]
    parser.add_argument('--task_size_min', type=int, default=1000)
    parser.add_argument('--task_size_max', type=int, default=2000)
    # Task complexity [1600, 2400, 3200, 4000, 4800]
    parser.add_argument('--task_complexity_min', type=int, default=800)
    parser.add_argument('--task_complexity_max', type=int, default=2400)
    # Task deadline
    parser.add_argument('--deadline', type=int, default=3.5)
    # CPU settings
    parser.add_argument('--single_core_cpu_capacity', type=int, default=3)
    parser.add_argument('--cpu_core_list', type=list, default=[4, 8, 12, 16, 20, 24, 28, 32])
    # Node transmission rate
    parser.add_argument('--transmission_rate_min', type=int, default=10)
    parser.add_argument('--transmission_rate_max', type=int, default=40)
    # Failure rate [0.1, 0.15, 0.2, 0.25, 0.3]
    parser.add_argument('--execution_failure_rate_min', type=float, default=0)
    parser.add_argument('--execution_failure_rate_max', type=float, default=0.3)
    parser.add_argument('--transmission_failure_rate_min', type=float, default=0)
    parser.add_argument('--transmission_failure_rate_max', type=float, default=0.1)

    parser.add_argument('--task_drop_penalty', type=float, default=-1)
    parser.add_argument('--task_failure_penalty', type=float, default=-1)
    parser.add_argument('--task_success_reward', type=float, default=1)
    parser.add_argument('--task_hop_penalty', type=float, default=-0.3)

    parser.add_argument('--edge_node_seed', type=int, default=500)
    parser.add_argument('--task_seed', type=int, default=100)
    parser.add_argument('--failure_seed', type=int, default=200)
    parser.add_argument('--seed', type=int, default=300)
    parser.add_argument('--rl_queue_coeff', type=float, default=1.0)

    parser.add_argument('--link_seed', type=int, default=601)

    # Configuring the k tasks to execute simultaneously
    parser.add_argument('--k', type=int, default=4)   # Number of tasks that can be executed in parallel, set range [3, 10]
    parser.add_argument('--beta', type=int, default=1.0)     # Computing node computing power

    return parser
