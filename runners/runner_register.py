from runners.parallel_episode_runner import ParallelRunner
from runners.episode_runner import EpisodeRunner
from runners.step_runner import StepRunner
from runners.optimal_runner import OptimalRunner
from runners.gail_runner import GAILRunner
from runners.dgail_runner import DGAILRunner
from runners.ilets_runner import ILETSRunner

runner_register = {
    'parallel': ParallelRunner,
    'episode': EpisodeRunner,
    'step': StepRunner,
    'optimal': OptimalRunner,
    'gail_runner': GAILRunner,
    'dgail_runner': DGAILRunner,
    'ilets_runner': ILETSRunner,
}
