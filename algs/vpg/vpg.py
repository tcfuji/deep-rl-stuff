import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.vpg.core as core
from rl_utils.logx import EpochLogger
from rl_utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from rl_utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

class VPGBuffer:
    """
    A buffer for strong trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end a trajectory, or when one gets cut off by an epoch
        ending. This looks back in the buffer to where the trajectory started, and
        uses rewards and value estimates from the whole trajectory to compute
        advantage estimates with GAE-Lambda, as well as compute the rewards-to-go
        for each state, to use as the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended because
        the agent reached a terminal state (died), and otherwise should be V(s_T),
        the value function estimated for the last state. This allows us to
        bootstrap the reward-to-go calculation to account for timesteps beyond
        the arbitrary episode horizon (or epoch cutoff).
        """
