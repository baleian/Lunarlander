import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# 정책 신경망과 가치 신경망 생성
class A2C(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(A2C, self).__init__()
        self.actor_fc = Dense(64, activation='relu')
        self.actor_out = Dense(action_size, activation='softmax')
        self.critic_fc1 = Dense(64, activation='relu')
        self.critic_fc2 = Dense(64, activation='relu')
        self.critic_out = Dense(1)
        self.build(input_shape=(None, state_size))

    def call(self, x):
        actor_x = self.actor_fc(x)
        policy = self.actor_out(actor_x)

        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)
        return policy, value


class ActorModel(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(__class__, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.out = Dense(action_size, activation='softmax')
        self.build(input_shape=(None, state_size))

    def call(self, x):
        x = self.dense1(x)
        policy = self.out(x)
        return policy


class CriticModel(tf.keras.Model):
    def __init__(self, state_size):
        super(__class__, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.out = Dense(1)
        self.build(input_shape=(None, state_size))

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        value = self.out(x)
        return value


# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False

        # 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        # 정책신경망과 가치신경망 생성
        self.model = A2C(self.state_size, self.action_size)
        # self.actor_model = ActorModel(self.state_size, self.action_size)
        # self.critic_model = CriticModel(self.state_size)

        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정
        self.optimizer = Adam(lr=self.learning_rate)

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        policy, _ = self.model(state)
        policy = np.array(policy[0])
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # def train_model(self, state, action, reward, next_state, done):
    #     actor_model_params = self.actor_model.trainable_variables
    #     critic_model_params = self.critic_model.trainable_variables
    #
    #     act = np.zeros(action_size)
    #     act[action] = 1
    #
    #     with tf.GradientTape(persistent=True) as tape:
    #         policy = self.actor_model(state)
    #         value = self.critic_model(state)
    #         next_value = self.critic_model(next_state)
    #         target = reward + (1 - done) * self.discount_factor * next_value[0]
    #
    #         # one_hot_action = tf.one_hot([action], self.action_size)
    #         action_prob = tf.reduce_sum([act] * policy, axis=1)
    #         cross_entropy = -tf.math.log(action_prob)
    #         advantage = target - value[0]
    #         # actor_loss = 0.1 * tf.reduce_mean(cross_entropy * advantage)
    #         actor_loss = 0.1 * cross_entropy * advantage
    #
    #         critic_loss = tf.square(target - value[0])
    #         # critic_loss = tf.reduce_mean(critic_loss)
    #
    #     actor_grads = tape.gradient(actor_loss, actor_model_params)
    #     critic_grads = tape.gradient(critic_loss, critic_model_params)
    #     del tape
    #
    #     self.optimizer.apply_gradients(zip(actor_grads, actor_model_params))
    #     self.optimizer.apply_gradients(zip(critic_grads, critic_model_params))
    #
    #     return np.array(cross_entropy)[0]

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            policy, value = self.model(state)
            _, next_value = self.model(next_state)
            target = reward + (1 - done) * self.discount_factor * next_value[0]

            # 정책 신경망 오류 함수 구하기
            one_hot_action = tf.one_hot([action], self.action_size)
            action_prob = tf.reduce_sum(one_hot_action * policy, axis=1)
            cross_entropy = -tf.math.log(action_prob) # + 1e-5)
            advantage = tf.stop_gradient(target - value[0])
            actor_loss = tf.reduce_mean(cross_entropy * advantage)

            # 가치 신경망 오류 함수 구하기
            critic_loss = tf.square(tf.stop_gradient(target) - value[0])
            critic_loss = tf.reduce_mean(critic_loss)

            # 하나의 오류 함수로 만들기
            loss = 0.1 * actor_loss + critic_loss

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return np.array(loss)


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('LunarLander-v2')
    SEED = 123
    env.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = 8
    action_size = 4

    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(state_size, action_size)

    scores = []
    steps = []
    cross_entropies = []

    for episode in range(1, 3001):
        done = False
        score = 0
        step = 0
        cross_entropy = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while not done:
            if agent.render:
                env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            cross_entropy += agent.train_model(state, action, reward, next_state, done)
            score += reward
            state = next_state
            step += 1
        scores.append(score)
        steps.append(step)
        cross_entropies.append(cross_entropy / step)

        print("episode: {:4d} | score: {:4.2f} | step: {:4d} | cross_entropy: {:.3f} | score: {:4.2f} | step: {:4.1f} | cross_entropy: {:.3f} | total score: {}".format(
            episode,
            scores[-1],
            steps[-1],
            cross_entropies[-1],
            np.mean(scores[-100:]),
            np.mean(steps[-100:]),
            np.mean(cross_entropies[-100:]),
            np.sum(scores),
        ))
