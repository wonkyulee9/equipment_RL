import numpy as np
from dqn_model import DQNAgent, state_input
from simulator import gen

#  Parameters                                       # 파라미터 설정
NEPOCH = 0                                          # Number of epochs to train
load_epoch = 40                                     # Epoch # to load model
XY = 'Y'                                            # X or Y
batch_size = 64

# Load data and initialize simulator
# if NEPOCH is 0 no training, so skip
if NEPOCH != 0:
    env = gen('DQN_realdata_{}_temp'.format(XY))
    env.data_shuff()
    batch_size = 64

# Initialize Agent
agent = DQNAgent(10, 5)

# load_epoch = 0 starts training from scratch, otherwise loads the model at a certain point
if load_epoch != 0:
    agent_th.epsilon = 0.0
    agent_th.model.load_weights('weights_{}_count/{}_model_epoch{}.h5'.format(XY,XY,load_epoch))

# Training
for e in range(load_epoch, NEPOCH + load_epoch):
    state = env.reset()
    state, _, _ = state_input(state)
    state = np.reshape(state, [1, 10])
    mean = 0
    total = 0
    r = 1000
    num_actions = [0, 0, 0, 0, 0]
    for st in range(r):
        action = agent_th.act(state)
        next_state, reward, done = env.next_state(2, action)

        next_state, _, _ = state_input(next_state)
        next_state = np.reshape(next_state, [1, 10])
        agent_th.remember(state, action, reward, next_state, False)

        total += reward
        mean += np.mean(np.abs(next_state))
        num_actions[action] += 1
        state = next_state

        if done:
            r = st
            break
        if len(agent_th.memory) > batch_size:
            agent_th.replay(batch_size)

    print("episode: {}/{}, total: {}/{}, mean: {:.4}, actions: {}, e: {:.2}, "
          "sess : {}, actions : {}".format(e + 1, load_epoch + NEPOCH, total, r + 1,
            mean / (r + 1), r + 1, agent_th.epsilon, env.sess,
            num_actions))

    if (e + 1) % 10 == 0 :
        agent_th.model.save('weights_{}_count/{}_model_epoch{}.h5'.format(XY,XY,e + 1))                     #Save model every 10 episodes


# Testing
env = gen('DQN_testdata_{}_temp'.format(XY))
state = env.reset()

for s in range(31):
    sess_states = np.empty([0, 3])
    for st in range(1000):
        sess_states = np.insert(sess_states, len(sess_states), state, axis=0)
        state2, a, b = state_input(state)
        state1 = (a - b)/2
        state1 = np.reshape(state1, [1, 10])
        state2 = np.reshape(state2, [1, 10])

        action1 = agent.act(state1)
        action2 = agent.act(state2)

        next_state, reward, done = env.next_state(action1, action2)

        state = next_state

        if done:
            sess_states = np.insert(sess_states, len(sess_states), state, axis=0)
            state = env.reset()
            break
    # Save resulting data to a file
    np.savetxt('result_Y_count' + '/' + env.flist[s].split('.')[0] + '.csv', sess_states, delimiter=',')
