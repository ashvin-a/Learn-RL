import gymnasium as gym
from collections import defaultdict
import numpy as np
from tqdm import tqdm  # Progress bar
import pickle


class CartPoleAgent:

    def __init__(
        self,
        env: gym.Env,  # The environment
        learning_rate: float,  # How fast the agent updates the Q-value
        initial_epsilon: float,  # Starting exploration rate
        epsilon_decay: float,  # The rate at which the exploration rate reduces after each episode
        final_epsilon: float,  # Final exploration rate
        discount_factor: float = 0.95,  # Value of future rewards
    ):

        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    # ? See https://gymnasium.farama.org/environments/classic_control/cart_pole/ for more details.
    # Define global bins (or put them inside __init__)
    # We slice the ranges into chunks.
    # 1. Cart Position: -4.8 to 4.8
    # 2. Cart Velocity: -Inf to Inf (we clip it to -3 to 3)
    # 3. Pole Angle: -0.418 to 0.418
    # 4. Pole Velocity: -Inf to Inf (we clip it to -3 to 3)

    def _discretize(
        self, obs: list[float, float, float, float]
    ) -> tuple[int, int, int, int]:
        x, x_dot, theta, theta_dot = obs
        # Define bins if not defined globally
        pos_bins = np.linspace(-4.8, 4.8, 10)
        vel_bins = np.linspace(-3.0, 3.0, 10)
        angle_bins = np.linspace(-0.209, 0.209, 10)
        ang_vel_bins = np.linspace(-3.0, 3.0, 10)

        return (
            np.digitize(x, pos_bins),
            np.digitize(x_dot, vel_bins),
            np.digitize(theta, angle_bins),
            np.digitize(theta_dot, ang_vel_bins),
        )

    def get_action(self, obs: list[float, float, float, float]) -> int:
        """
        Returns 0 = push cart left, 1 = push cart right
        """
        discretised_obs = self._discretize(obs)
        # with some probability epsilon, the agent should explore
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[discretised_obs]))

    def update(
        self,
        obs: list[float, float, float, float],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: list[float, float, float, float],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        discretised_obs = self._discretize(obs)
        discretised_next_obs = self._discretize(next_obs)
        future_q_value = (not terminated) * np.max(self.q_values[discretised_next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[discretised_obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[discretised_obs][action] = (
            self.q_values[discretised_obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

if __name__ == "__main__":
    # Create our training environment - a cart with a pole that needs balancing
    number_of_episodes = 100000
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=number_of_episodes)

    # Reset environment to start a new episode
    observation, info = env.reset()
    # observation: what the agent can "see" - cart position, velocity, pole angle, etc.
    # info: extra debugging information (usually not needed for basic learning)

    print(f"Starting observation: {observation}")
    # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
    # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

    
    # * Lets train this son of a gun
    cartpole_agent = CartPoleAgent(
        env=env,
        learning_rate=0.09,
        initial_epsilon=1.0,
        epsilon_decay=(
            1.0 / (number_of_episodes / 2)
        ),  # (initial_epsilon/(number_of_episodes/2))
        final_epsilon=0.1,
        discount_factor=0.95,
    )

    for episode in tqdm(range(number_of_episodes)):

        observation, info = env.reset()
        done = False

        while not done:
            action = cartpole_agent.get_action(obs=observation)

            next_observation, reward, terminated, truncated, info = env.step(action=action)

            cartpole_agent.update(
                obs=observation,
                action=action,
                reward=reward,
                terminated=terminated,
                next_obs=next_observation,
            )

            done = terminated or truncated
            observation = next_observation

        cartpole_agent.decay_epsilon()

    # #! Lets save the trained thing

    with open("src/simulation/cartpole.pkl", "wb") as f:
        trained_q_values = dict(cartpole_agent.q_values)
        pickle.dump(trained_q_values, f)
    print("Q Table Saved to cartpole.pkl")


