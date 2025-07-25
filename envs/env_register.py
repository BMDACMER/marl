from envs.edge_computing.multihop.edge_computing_env import EdgeComputingEnv
from envs.edge_computing.multihop.edge_computing_config import add_edge_computing_env_args

env_register = {
    'edge_computing': EdgeComputingEnv
}

env_config_register = {
    'edge_computing': add_edge_computing_env_args
}
