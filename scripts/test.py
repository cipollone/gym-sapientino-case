"""Try the environment to see how the agent moves at random."""

import random
import time

from gym.wrappers import TimeLimit

from gym_sapientino_case.env import SapientinoCase


def main():
    """Just visualizes a random exploration of the environment."""
    # Define the environment
    #   NOTE: this uses the defaults, but you can experiments with all options
    env = SapientinoCase(
        colors=["red", "green", "blue", "yellow"],
        logdir=".",
    )

    # Limit the length of the episode
    env = TimeLimit(env, 100)

    # Episodes
    while True:

        # Init episode
        obs = env.reset()
        done = False
        cum_reward = 0.0

        # Print
        print(f"\n> Env reset.\nInitial observation {obs}")

        while not done:
            # Render
            env.render()

            # Compute action (random here)
            action = random.randint(0, env.action_space.n - 1)

            # Move env
            obs, reward, done, _ = env.step(action)
            cum_reward += reward

            # Print
            print(
                "Step.",
                f"Action {action}",
                f"Observation {obs}",
                f"Reward {reward}",
                f"Done {done}",
                sep="\n  ",
            )

            # Let us see the screen
            time.sleep(0.1)


if __name__ == "__main__":
    main()
