import argparse
import gym
import cv2
import numpy as np
from collections import deque
import gym_multi_car_racing
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


RENDER                        = True
STARTING_EPISODE              = 1
ENDING_EPISODE                = 1000
SKIP_FRAMES                   = 2
TRAINING_BATCH_SIZE           = 64
SAVE_TRAINING_FREQUENCY       = 25
UPDATE_TARGET_MODEL_FREQUENCY = 5
NUM_AGENTS                    = 2

class CarRacingDQNAgent:
    def __init__(
        self,
        action_space    = [
            (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
            (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
            (-1, 0,   0), (0, 0,   0), (1, 0,   0)
        ],
        frame_stack_num = 3,
        memory_size     = 5000,
        gamma           = 0.95,  # discount rate
        epsilon         = 1.0,   # exploration rate
        epsilon_min     = 0.1,
        epsilon_decay   = 0.9999,
        learning_rate   = 0.001
    ):
        self.action_space    = action_space
        self.frame_stack_num = frame_stack_num
        self.memory          = deque(maxlen=memory_size)
        self.gamma           = gamma
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.learning_rate   = learning_rate
        self.model           = self.build_model()
        self.target_model    = self.build_model()
        self.update_target_model()

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, self.frame_stack_num)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate, epsilon=1e-7))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))

    def act(self, state):
        if np.random.rand() > self.epsilon:
            act_values = self.model.predict(np.expand_dims(state, axis=0))
            action_index = np.argmax(act_values[0])
        else:
            action_index = random.randrange(len(self.action_space))
        return self.action_space[action_index]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []
        for state, action_index, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0))[0]
            if done:
                target[action_index] = reward
            else:
                t = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                target[action_index] = reward + self.gamma * np.amax(t)
            train_state.append(state)
            train_target.append(target)
        self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.target_model.save_weights(name)

def process_state_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state

def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)
    return np.transpose(frame_stack, (1, 2, 0))

def main():
    parser = argparse.ArgumentParser(description='Training multi-agent DQN on MultiCarRacing-v0.')
    parser.add_argument('-m', '--model', help='Path to a saved model to continue training from.')
    parser.add_argument('-s', '--start', type=int, help='Starting episode, default=1.')
    parser.add_argument('-e', '--end', type=int, help='Ending episode, default=1000.')
    parser.add_argument('-p', '--epsilon', type=float, default=1.0, help='Starting epsilon, default=1.0.')
    args = parser.parse_args()

    if args.start:
        global STARTING_EPISODE
        STARTING_EPISODE = args.start
    if args.end:
        global ENDING_EPISODE
        ENDING_EPISODE = args.end

    env = gym.make('MultiCarRacing-v0')


    agents = [CarRacingDQNAgent(epsilon=args.epsilon) for _ in range(NUM_AGENTS)]

    if args.model:
        for i in range(NUM_AGENTS):
            agents[i].load(args.model)

    for e in range(STARTING_EPISODE, ENDING_EPISODE + 1):
        init_states = env.reset()
        init_states = [process_state_image(s) for s in init_states]

        frame_stacks = [
            deque([init_states[i]] * agents[i].frame_stack_num, maxlen=agents[i].frame_stack_num)
            for i in range(NUM_AGENTS)
        ]

        total_rewards = [0.0 for _ in range(NUM_AGENTS)]
        negative_reward_counters = [0 for _ in range(NUM_AGENTS)]
        time_frame_counter = 1
        done = False

        while True:
            if RENDER:
                env.render()

            actions = []
            stacked_states = []
            for i in range(NUM_AGENTS):
                current_stack = generate_state_frame_stack_from_queue(frame_stacks[i])
                stacked_states.append(current_stack)
                action = agents[i].act(current_stack)
                actions.append(action)

            accumulated_rewards = [0.0 for _ in range(NUM_AGENTS)]
            for _ in range(SKIP_FRAMES + 1):
                next_states, rewards, done_flag, info = env.step(actions)

                for i in range(NUM_AGENTS):
                    accumulated_rewards[i] += rewards[i]
                if done_flag:
                    done = True
                    break

            for i in range(NUM_AGENTS):

                if actions[i][1] == 1 and actions[i][2] == 0:
                    accumulated_rewards[i] *= 1.5

                total_rewards[i] += accumulated_rewards[i]

                if time_frame_counter > 100 and accumulated_rewards[i] < 0:
                    negative_reward_counters[i] += 1
                else:
                    negative_reward_counters[i] = 0

                if negative_reward_counters[i] >= 25:
                    done = True

            if done or any(r < 0 for r in total_rewards):

                print(
                    f"Episode: {e}/{ENDING_EPISODE}, "
                    f"Scores(Time Frames): {time_frame_counter}, "
                    f"Total Rewards: {total_rewards}, "
                    f"Epsilons: {[f'{ag.epsilon:.2f}' for ag in agents]}"
                )
                break


            processed_next_states = [process_state_image(ns) for ns in next_states]
            for i in range(NUM_AGENTS):
                frame_stacks[i].append(processed_next_states[i])
                next_stack = generate_state_frame_stack_from_queue(frame_stacks[i])
                agents[i].memorize(
                    stacked_states[i],
                    actions[i],
                    accumulated_rewards[i],
                    next_stack,
                    done
                )

            for i in range(NUM_AGENTS):
                if len(agents[i].memory) > TRAINING_BATCH_SIZE:
                    agents[i].replay(TRAINING_BATCH_SIZE)

            time_frame_counter += 1

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            for i in range(NUM_AGENTS):
                agents[i].update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            for i in range(NUM_AGENTS):
                agents[i].save(f'./save/agent{i}_episode{e}.h5')

    env.close()

if __name__ == "__main__":
    main()
