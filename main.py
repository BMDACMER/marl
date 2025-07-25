import argparse
import os
import sys
import threading
import time

import numpy as np
import torch as th

from agent_register import agent_register, agent_config_register
from default_config import add_default_args
from envs.env_register import env_register, env_config_register
from runners.runner_register import runner_register
from utils.run_utils import train_models, train_offline_models, only_test_models, load_models, train_expert_models
from loguru import logger

if __name__ == '__main__':
    env_name = 'edge_computing'
    algo = sys.argv[1].split("=")[1]
    add_env_args = env_config_register[env_name]
    add_algo_args = agent_config_register[algo]

    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser = add_default_args(parser)
    parser = add_env_args(parser)
    parser = add_algo_args(parser)
    args = parser.parse_args(sys.argv[1:])

    logger.info(args)

    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.set_num_threads(1)
    np.random.seed(args.seed)

    # 根据环境配置创建环境class
    env = env_register[args.env_name](args)
    env_info = env.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.obs_shape = env_info["obs_shape"]
    args.state_shape = env_info["state_shape"]
    env.close()

    # 根据agent配置创建agent class
    agent = agent_register[args.name](args)
    runner = runner_register[args.runner](args, agent)

    if args.load_models:
        logger.info(f'load models {args.name}')
        load_models(args, agent)

    if args.test_models:
        logger.info(f"run test models {args.name}")
        start_time = time.time()
        only_test_models(args, runner)
        end_time = time.time()
        logger.info(f"The average running time is ：{(end_time - start_time) / args.test_nepisode}")
    elif args.train_models:
        if args.offline:
            logger.info(f"run train offline models {args.name}")
            train_offline_models(args, agent, runner)
        elif args.expert:
            logger.info(f"run train expert-to-ppo, the expert is {args.name}")
            train_expert_models(args, agent, runner)
        else:
            logger.info(f"run train models {args.name}")
            train_models(args, agent, runner)

    # Clean up after finishing
    logger.info("Exiting Main")
    logger.info("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            logger.info("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            logger.info("Thread joined")
    logger.info("Exiting script")
    # Making sure framework really exits
    # os._exit(os.EX_OK)
    os._exit(0)
