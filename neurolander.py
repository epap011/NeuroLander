import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent

def train_agent(agent, env, num_episodes):
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, agent.state_shape[0]])
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_shape[0]])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()

            if done:
                agent.rewards_history.append(total_reward)
                print(f"Episode: {episode + 1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

        agent.update_epsilon()

if __name__ == "__main__":
    env = gym.make("LunarLander-v3", render_mode="human")
    
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n

    agent = DQNAgent(state_shape, num_actions)
    
    train_agent(agent, env, num_episodes=500)

    agent.model.save("dqn_model.h5")
    np.save("rewards_history.npy", agent.rewards_history)
    np.save("loss_history.npy", agent.loss_history)
    
    env.close()