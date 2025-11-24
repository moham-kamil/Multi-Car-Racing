import gym
import gym_multi_car_racing
from collections import deque
from dqn_multicar import CarRacingDQNAgent
from dqn_multicar import process_state_image, generate_state_frame_stack_from_queue

if __name__ == '__main__':
    play_episodes = 10

    env = gym.make('MultiCarRacing-v0')

    agent0 = CarRacingDQNAgent(epsilon=0)
    agent1 = CarRacingDQNAgent(epsilon=0)

    agent0.load('save/agent0_episode1000.h5')
    agent1.load('save/agent1_episode1000.h5')

    for e in range(play_episodes):
        init_states = env.reset()
        init_states = [process_state_image(s) for s in init_states]

        state_stack_0 = deque([init_states[0]] * agent0.frame_stack_num, maxlen=agent0.frame_stack_num)
        state_stack_1 = deque([init_states[1]] * agent1.frame_stack_num, maxlen=agent1.frame_stack_num)

        total_reward0 = 0
        total_reward1 = 0
        time_frame_counter = 1

        while True:
            env.render()

            current_stack_0 = generate_state_frame_stack_from_queue(state_stack_0)
            current_stack_1 = generate_state_frame_stack_from_queue(state_stack_1)

            action0 = agent0.act(current_stack_0)
            action1 = agent1.act(current_stack_1)
            actions = [action0, action1]

            next_states, rewards, done, info = env.step(actions)

            total_reward0 += rewards[0]
            total_reward1 += rewards[1]

            next_state0 = process_state_image(next_states[0])
            next_state1 = process_state_image(next_states[1])
            state_stack_0.append(next_state0)
            state_stack_1.append(next_state1)

            if done:
                print(
                    "Episode: {}/{}, Time Frames: {}, Agent 0 Total Reward: {:.2f}, Agent 1 Total Reward: {:.2f}".format(
                        e + 1, play_episodes, time_frame_counter, total_reward0, total_reward1))
                break

            time_frame_counter += 1

    env.close()
