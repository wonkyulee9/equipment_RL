# This code uses the trained dqn models to give real-time decisions

from dqn_model import DQNAgent, state_input
from preprocess import findzero, interpolate, del_nondata, check_minmax
import csv
import numpy as np


x_path = ''                   # path to best X model
y_path = ''                   # path to best Y model
measure_path = 'measure.csv'  # path to log file (needs to be preprocessed)

min = [0, 0, 0, 0, 0, 0, 0, 0]    # min/max boundaries
max = [0, 0, 0, 0, 0, 0, 0, 0]
# Target values
target = [0, 0, 0, 0, 0, 0]

action_space = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])


def get_actions(measure_path):
    agent_x, agent_y = DQNAgent(10, 5), DQNAgent(10, 5)
    agent_x.epsilon, agent_y.epsilon = 0.0, 0.0
    agent_x.model.load_weights(x_path)
    agent_y.model.load_weights(y_path)

    with open(measure_path, 'r') as measure_file:
        temp = list(csv.reader(measure_file))
    measure_file.close()

    if len(temp) < 101:
        print("Error: file has insufficient data")              # Prompt error when insufficient data
        return

    temp = del_nondata(temp)                                    

    temp = np.asarray(temp)[1:, 1:].astype(np.float32)        
    data = np.empty([100, 8], dtype=np.float32)
    data[:, :6] = temp[-100:, 1:7]
    data[:, 6:8] = temp[-100:, 9:11]

    data = check_minmax(data, 0)                            

    # Data is redundant, 2 data per time frame, so only keep half by averaging IF both data is not 0
    for x in range(len(data)):
        if x % 2 == 1:
            for y in range(8):
                if (data[x][y] != 0) and (data[x-1][y] != 0):
                    data[x][y] = (data[x][y] + data[x-1][y]) / 2
                elif data[x][y] == 0:
                    data[x][y] = data[x-1][y]
    data = np.delete(data, 2*np.arange(50), 0)

    # Find 0 / error and interpoalte
    zlist = findzero(data, 0)
    data = interpolate(zlist, data)

    #####
    data[:, 3] = data[:, 2:4].mean(1)
    data[:, 2] = data[:, :2].mean(1)
    data = np.delete(data, [0, 1], 1)

    # Normalize using target data
    data = data - target

    # Average/reshape
    ts, a12, a34 = state_input(data[:, [0, 1, 5]])
    th, a5, a6 = state_input(data[:, [2, 3, 4]])

    # DQN의 판단을 받아와서 해당 action의 값을 return
    x_a = action_space[agent_x.act(np.reshape((a12-a34)/2, [1, 10]))]
    x_b = action_space[agent_x.act(np.reshape(ts, [1, 10]))]
    y_a = action_space[agent_y.act(np.reshape((a5-a6)/2, [1, 10]))]
    y_b = action_space[agent_y.act(np.reshape(th, [1, 10]))]
    return [x_a, x_b, y_a, y_b]

# Test
# control = get_actions(measure_path)
# print(control)
