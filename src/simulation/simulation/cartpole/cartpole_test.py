import gymnasium as gym
from cartpole import CartPoleAgent
import pandas as pd

objects = pd.read_pickle('src/simulation/cartpole.pkl')
trained_q_values = dict(objects)

print("Q Table read from cartpole.pkl")

# * Now lets try the trained thing
try:
    env = gym.make("CartPole-v1", render_mode="human")

    inference_agent = CartPoleAgent(
        env=env,
        learning_rate=0,
        initial_epsilon=0,  # The Q values should not change and the model should not act randomly.
        epsilon_decay=0,
        final_epsilon=0,
    )
    # inference_agent.q_values = cartpole_agent.q_values
    inference_agent.q_values = trained_q_values
    obs, _ = env.reset()
    total_reward = 0
    while True:
        # Choose an action: 0 = push cart left, 1 = push cart right
        action = inference_agent.get_action(obs=obs)

        # Take the action and see what happens
        observation, reward, terminated, truncated, _ = env.step(action)

        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)

        total_reward += reward
        episode_over = terminated or truncated

        if terminated or truncated:
            print(f"Episode finished! Total Reward: {total_reward}")
            break

        obs = observation
    env.close()
except KeyboardInterrupt as e:

    env.close()