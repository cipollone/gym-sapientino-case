"""Definitions on the sapientino environment, and fluents valuation."""

from typing import Dict, List, Optional, Sequence, Set

from gym import Env
from gym.spaces import Discrete
from gym_sapientino.core.types import color2int as enum_color2int
from pythomata.impl.simple import SimpleDFA
from temprl.wrapper import TemporalGoal, TemporalGoalWrapper

# These mappings are often useful (from colors to ID and vice versa)
color2int = {c.value: i for c, i in enum_color2int.items()}
int2color = {i: c for c, i in color2int.items()}


class SapientinoFluents:
    """Define the propositions in the sapientino environment.

    The propositions will evaluate to true if the agent is on a colored cell
    and last action was "bip" (visit).
    """

    def __init__(self, colors_set: Set[str]):
        """Initialize.

        :param colors_set: a set of colors among the ones used by sapientino;
            this will be the set of fluents to evaluate.
        """
        self.fluents = colors_set
        if not self.fluents.issubset(color2int):
            raise ValueError(str(colors_set) + " contains invalid colors")

    def evaluate(self, obs: Dict[str, float], action: int) -> Set[str]:
        """Respects AbstractFluents.evaluate."""
        beeps = obs["beep"] > 0
        if not beeps:
            true_fluents = set()  # type: Set[str]
        else:
            color_id = obs["color"]
            color_name = int2color[color_id]
            if color_name == "blank":
                true_fluents = set()
            else:
                if color_name not in self.fluents:
                    raise RuntimeError("Unexpected color: " + color_name)
                true_fluents = {color_name}

        return frozenset({f for f in self.fluents if f in true_fluents})


class SapientinoGoal(TemporalGoal):
    """Temporal goals for sapientino environments.

    This class defines temporal goals for all the sapientino environments.
    The goal is to visit all the given colors (these are positions in
    the environments) in some fixed order.

    Right now, just for efficiency, the automaton is defined directly and not
    built from a temporal formula.
    """

    def __init__(
        self,
        colors: Sequence[str],
        fluents: SapientinoFluents,
        reward: Optional[float] = 1.0,
        save_to: Optional[str] = None,
    ):
        """Initialize.

        :param colors: a sequence of colors, these are the positions that
            the agent must reach with the correct order.
        :param fluents: a fluents evaluator. All colors must be fluents, so
            that we know when the agent is in each position.
        :param reward: reward suplied when reward is reached.
        :param save_to: path where the automaton should be exported.
        """
        # Check
        if not all((color in fluents.fluents for color in colors)):
            raise ValueError("Some color has no associated fluent to evaluate it")

        # Make automaton for this sequence
        automaton = self._make_sapientino_automaton(colors)

        automaton = automaton.renumbering()

        # Super
        TemporalGoal.__init__(
            self,
            formula=None,  # Provinding automaton directly
            reward=reward,
            automaton=automaton,
            labels=set(colors),
            extract_fluents=fluents.evaluate,
            reward_shaping=False,
            zero_terminal_state=False,
        )

        # Maybe save
        if save_to:
            self.automaton.to_graphviz().render(save_to)

    @staticmethod
    def _make_sapientino_automaton(colors: Sequence[str]) -> SimpleDFA:
        """Make the automaton from a sequence of colors."""
        # All possible interpretations
        #  I don't consider those that would never happen to simplify the dfa
        alphabet = {frozenset({c}) for c in colors}
        alphabet.add(frozenset())

        nb_states = len(colors) + 2
        initial_state = 0
        current_state = initial_state
        sink = nb_states - 1
        accepting = nb_states - 2
        states = {initial_state, sink}
        transitions = {}
        for c in colors:
            next_state = current_state + 1
            for symbol in alphabet:
                if len(symbol) == 0:
                    transitions.setdefault(current_state, {})[symbol] = current_state
                elif c in symbol:
                    transitions.setdefault(current_state, {})[symbol] = next_state
                else:
                    transitions.setdefault(current_state, {})[symbol] = sink
            current_state = next_state
            states.add(current_state)

        for symbol in alphabet:
            transitions.setdefault(current_state, {})[symbol] = sink
            transitions.setdefault(sink, {})[symbol] = sink

        dfa = SimpleDFA(states, alphabet, initial_state, {accepting}, transitions)
        return dfa.trim().complete()

    @property
    def observation_space(self) -> Discrete:
        """Return the observation space.

        NOTE: Temprl returns automata states+1, we don't want that
        if we already have a complete automaton.
        """
        return Discrete(len(self._automaton.states))


class MyTemporalGoalWrapper(TemporalGoalWrapper):
    """Custom version of TemporalGoalWrapper."""

    def __init__(
        self,
        env: Env,
        temp_goals: List[TemporalGoal],
        end_on_success: bool = True,
        end_on_failure: bool = False,
    ):
        """Initialize.

        :param env: gym environment to wrap.
        :param temp_goals: list of temporal goals.
        :param end_on_success: if true, episode terminates when the agent
            reaches the reward.
        :param end_on_failure: if true, episode terminates when the agent
            reaches a failure state.
        """
        # Super
        TemporalGoalWrapper.__init__(self, env=env, temp_goals=temp_goals)

        # Store
        self.__end_on_success = end_on_success
        self.__end_on_failure = end_on_failure

    def step(self, action):
        """Do the step."""
        # Step
        state, reward, done, info = super().step(action)

        # Reward
        for tg in self.temp_goals:
            if tg.is_true():
                reward += tg.reward

        # Termination
        failure_done = self.__end_on_failure and all(
            tg.is_failed() for tg in self.temp_goals
        )
        success_done = self.__end_on_success and all(
            tg.is_true() for tg in self.temp_goals
        )
        done = done or failure_done or success_done

        return state, reward, done, info
