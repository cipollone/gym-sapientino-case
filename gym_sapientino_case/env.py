"""Gym interface."""

import importlib.resources
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Sequence

import gym
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)
from temprl.wrapper import TemporalGoalWrapper

from . import resources
from .gym_utils import SingleAgentWrapper
from .observations import ContinuousRobotFeatures
from .temporal_goal import SapientinoFluents, SapientinoGoal

# Default arguments
_sapientino_defaults = dict(
    reward_per_step=0.0,
    reward_outside_grid=0.0,
    reward_duplicate_beep=0.0,
    acceleration=0.2,
    angular_acceleration=10.0,
    max_velocity=0.4,
    min_velocity=0.0,
    max_angular_vel=40,
    initial_position=[2, 2],
    tg_reward=1.0,
)


class SapientinoCase(gym.Wrapper):
    """A specific instance of gym sapientino with non-markovian goal."""

    def __init__(
        self,
        colors: Sequence[str],
        params=None,
        map_file: Optional[Path] = None,
        logdir: Optional[str] = None,
    ):
        """Initialize.

        :param colors: the temporal goal is to visit these colors in the
            correct order.
        :params params: a dictionary of environment parameters.
            see this module source code: there are defaults as reference.
        :params map_file: path to a map file. See this package resource file
            for an example of a map.
        :param logdir: where to save logs.
        """
        # Defaults
        if params is None:
            params = _sapientino_defaults

        # Instantiate gym sapientino
        env = self._make_sapientino(params, map_file)

        # Define the fluent extractor
        fluents = SapientinoFluents(set(colors))

        # Define the temporal goal
        tg = SapientinoGoal(
            colors=colors,
            fluents=fluents,
            reward=params["tg_reward"],
            save_to=os.path.join(logdir, "reward-dfa.dot") if logdir else None,
        )

        # Add rewards of the temporal goal to the environment
        env = TemporalGoalWrapper(
            env=env,
            temp_goals=[tg],
        )

        # Choose a specific observation space
        env = ContinuousRobotFeatures(env)

        # Save
        super().__init__(env)

    @staticmethod
    def _make_sapientino(
        params=None,
        map_file: Optional[Path] = None,
    ) -> gym.Env:
        """Create the sapientino environment with some map."""
        # Define the robot
        agent_configuration = SapientinoAgentConfiguration(
            continuous=True,
            initial_position=params["initial_position"],
        )

        # Maybe I need to open the default map
        if map_file is not None:
            map_context = nullcontext(map_file)
        else:
            map_context = importlib.resources.path(resources, "map1.txt")

        # Define the environment
        with map_context as map_path:
            configuration = SapientinoConfiguration(
                [agent_configuration],
                path_to_map=map_path,
                reward_per_step=params["reward_per_step"],
                reward_outside_grid=params["reward_outside_grid"],
                reward_duplicate_beep=params["reward_duplicate_beep"],
                acceleration=params["acceleration"],
                angular_acceleration=params["angular_acceleration"],
                max_velocity=params["max_velocity"],
                min_velocity=params["min_velocity"],
                max_angular_vel=params["max_angular_vel"],
            )

        # Observation space
        env = SingleAgentWrapper(SapientinoDictSpace(configuration))

        return env
