from rl.policy_gradient_rl.ac.ac_agent import ActorCriticAgent
from rl.policy_gradient_rl.ac.ac_config import add_ac_config

# policy gradient ppo
from rl.policy_gradient_rl.ppo.ppo_agent import PPOAgent
from rl.policy_gradient_rl.ppo.ppo_config import add_ppo_args
# policy gradient sac
from rl.policy_gradient_rl.sac.sac_agent import SACAgent
from rl.policy_gradient_rl.sac.sac_config import add_sac_args

# optimal search algorithm
from optimal.optimal_agent import OptimalAgent
from optimal.optimal_config import add_optimal_args

# 在现有导入的基础上添加
from rl.policy_gradient_rl.gail.gail_agent import GAILAgent
from rl.policy_gradient_rl.gail.gail_config import add_gail_args

# 新增：DRAIL算法导入
from rl.policy_gradient_rl.dgail.dgail_agent import DGAILAgent
from rl.policy_gradient_rl.dgail.dgail_config import add_dgail_args

# 新增：ILETS算法导入
from rl.policy_gradient_rl.ilets.ilets_agent import ILETSAgent
from rl.policy_gradient_rl.ilets.ilets_config import add_ilets_config

# 新增：D2SAC算法导入
from rl.policy_gradient_rl.d2sac.d2sac_agent import D2SACAgent
from rl.policy_gradient_rl.d2sac.d2sac_config import add_d2sac_args

agent_register = {
    'ac': ActorCriticAgent,
    'sac': SACAgent,
    'ppo': PPOAgent,
    'optimal': OptimalAgent,
    'gail': GAILAgent,
    'dgail': DGAILAgent,
    'ilets': ILETSAgent,
    'd2sac': D2SACAgent,
}

agent_config_register = {
    'ac': add_ac_config,
    'sac': add_sac_args,
    'ppo': add_ppo_args,
    'optimal': add_optimal_args,
    'gail': add_gail_args,
    'dgail': add_dgail_args,
    'ilets': add_ilets_config,
    'd2sac': add_d2sac_args,
}
