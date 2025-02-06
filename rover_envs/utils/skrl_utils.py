import copy
from typing import Any, Tuple

import torch
import tqdm
from isaaclab.envs import ManagerBasedRLEnv
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import IsaacLabWrapper, Wrapper, wrap_env
from skrl.trainers.torch import Trainer
from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG

# We use SKRL 1.1, consequently we cannot use the official
# `SkrlSequentialLogTrainer` class from the `omni.isaac.orbit_tasks.utils.wrappers.skrl` module.


def SkrlVecEnvWrapper(env: ManagerBasedRLEnv):
    """Wraps around Orbit environment for skrl.

    This function wraps around the Orbit environment. Since the :class:`ManagerBasedRLEnv` environment
    wrapping functionality is defined within the skrl library itself, this implementation
    is maintained for compatibility with the structure of the extension that contains it.
    Internally it calls the :func:`wrap_env` from the skrl library API.

    NOTE: Since we are using SKRL 1.1 - we have to restate the function.
    Please see "https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/
    isaac/orbit_tasks/utils/wrappers/skrl.py"


    Args:
        env: The environment to wrap around.

    Raises:
        ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`.

    Reference:
        https://skrl.readthedocs.io/en/latest/modules/skrl.envs.wrapping.html
    """
    # check that input is valid
    if not isinstance(env.unwrapped, ManagerBasedRLEnv):
        raise ValueError(
            f"The environment must be inherited from ManagerBasedRLEnv. Environment type: {type(env)}")
    # wrap and return the environment
    return wrap_env(env, wrapper="isaac-orbit")


class SkrlOrbitVecWrapper(IsaacLabWrapper):
    """ Wrapper for the Isaac Orbit environment.
    Note: The wrapper from SKRL breaks with nan values of in the observation space.
    This can sometimes happen within a ORBIT environment. This wrapper is used to handle the nan values.
    """

    def __init__(self, env: ManagerBasedRLEnv):
        self._env = env
        super().__init__(env)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        actions = actions.nan_to_num(
            nan=0.0000001, posinf=0.000001, neginf=0.00001)

        self._observations, reward, terminated, truncated, self._info = self._env.step(
            actions)
        self._obs_dict["policy"] = torch.nan_to_num(
            self._obs_dict["policy"], nan=0.0, posinf=0.0, neginf=0.0)
        return (
            self._observations["policy"],
            reward.view(-1, 1),
            terminated.view(-1, 1),
            truncated.view(-1, 1),
            self._info
        )

    def reset(self) -> Tuple[torch.Tensor, Any]:
        info = {}
        if self._reset_once:
            self._reset_once = False
            self._obs_dict, info = self._env.reset()

        self._obs_dict["policy"] = torch.nan_to_num(
            self._obs_dict["policy"], nan=0.0, posinf=0.0, neginf=0.0)
        return self._obs_dict["policy"].nan_to_num(nan=0.01), info


class SkrlSequentialLogTrainer(Trainer):
    """Sequential trainer with logging of episode information.

    This trainer inherits from the :class:`skrl.trainers.base_class.Trainer` class. It is used to
    train agents in a sequential manner (i.e., one after the other in each interaction with the
    environment). It is most suitable for on-policy RL agents such as PPO, A2C, etc.

    It modifies the :class:`skrl.trainers.torch.sequential.SequentialTrainer` class with the following
    differences:

    * It also log episode information to the agent's logger.
    * It does not close the environment at the end of the training.

    Reference:
        https://skrl.readthedocs.io/en/latest/modules/skrl.trainers.base_class.html

    NOTE: Since we are using SKRL 1.1 - we have to restate the function with some modifications.
    Please see "https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/
    isaac/orbit_tasks/utils/wrappers/skrl.py"

    """

    def __init__(
        self,
        env: Wrapper,
        agents: Agent | list[Agent],
        agents_scope: list[int] | None = None,
        cfg: dict | None = None,
    ):
        """Initializes the trainer.

        Args:
            env: Environment to train on.
            agents: Agents to train.
            agents_scope: Number of environments for each agent to
                train on. Defaults to None.
            cfg: Configuration dictionary. Defaults to None.
        """
        # update the config
        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        # store agents scope
        agents_scope = agents_scope if agents_scope is not None else []
        # initialize the base class
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)
        # init agents
        if self.env.num_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

    def train(self):
        """Train the agents sequentially.

        This method executes the training loop for the agents. It performs the following steps:

        * Pre-interaction: Perform any pre-interaction operations.
        * Compute actions: Compute the actions for the agents.
        * Step the environments: Step the environments with the computed actions.
        * Record the environments' transitions: Record the transitions from the environments.
        * Log custom environment data: Log custom environment data.
        * Post-interaction: Perform any post-interaction operations.
        * Reset the environments: Reset the environments if they are terminated or truncated.

        """
        # init agent
        self.agents.init(trainer_cfg=self.cfg)
        self.agents.set_running_mode("train")
        # reset env
        states, infos = self.env.reset()
        # training loop
        for timestep in tqdm.tqdm(range(self.timesteps), disable=self.disable_progressbar):
            # pre-interaction
            self.agents.pre_interaction(
                timestep=timestep, timesteps=self.timesteps)
            # compute actions
            with torch.no_grad():
                actions = self.agents.act(
                    states, timestep=timestep, timesteps=self.timesteps)[0]
            # step the environments
            next_states, rewards, terminated, truncated, infos = self.env.step(
                actions)
            # note: here we do not call render scene since it is done in the env.step() method
            # record the environments' transitions
            with torch.no_grad():
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )
            # log custom environment data
            if "episode" in infos:
                for k, v in infos["episode"].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        self.agents.track_data(f"EpisodeInfo / {k}", v.item())
            # post-interaction
            self.agents.post_interaction(
                timestep=timestep, timesteps=self.timesteps)
            # reset the environments
            # note: here we do not call reset scene since it is done in the env.step() method
            # update states
            states.copy_(next_states)

    def eval(self) -> None:
        """Evaluate the agents sequentially.

        This method executes the following steps in loop:

        * Compute actions: Compute the actions for the agents.
        * Step the environments: Step the environments with the computed actions.
        * Record the environments' transitions: Record the transitions from the environments.
        * Log custom environment data: Log custom environment data.
        """
        # set running mode
        if self.num_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("eval")
        else:
            self.agents.set_running_mode("eval")
        # single agent
        if self.num_agents == 1:
            self.single_agent_eval()
            return

        # reset env
        states, infos = self.env.reset()
        # evaluation loop
        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar):
            # compute actions
            with torch.no_grad():
                actions = torch.vstack([
                    agent.act(states[scope[0]: scope[1]],
                              timestep=timestep, timesteps=self.timesteps)[0]
                    for agent, scope in zip(self.agents, self.agents_scope)
                ])

            # step the environments
            next_states, rewards, terminated, truncated, infos = self.env.step(
                actions)

            with torch.no_grad():
                # write data to TensorBoard
                for agent, scope in zip(self.agents, self.agents_scope):
                    # track data
                    agent.record_transition(
                        states=states[scope[0]: scope[1]],
                        actions=actions[scope[0]: scope[1]],
                        rewards=rewards[scope[0]: scope[1]],
                        next_states=next_states[scope[0]: scope[1]],
                        terminated=terminated[scope[0]: scope[1]],
                        truncated=truncated[scope[0]: scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )
                    # log custom environment data
                    if "log" in infos:
                        for k, v in infos["log"].items():
                            if isinstance(v, torch.Tensor) and v.numel() == 1:
                                agent.track_data(k, v.item())
                    # perform post-interaction
                    super(type(agent), agent).post_interaction(
                        timestep=timestep, timesteps=self.timesteps)

                # reset environments
                # note: here we do not call reset scene since it is done in the env.step() method
                states.copy_(next_states)
