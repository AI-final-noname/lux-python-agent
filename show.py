import csv

agent_place = ['PPO_1', 'DQN_1', 'DQN_scratch', 'DQN_scratch', 'DQN_scratch']
algo = ['PPO', 'DQN', 'DQN_scratch', 'DDQN', 'PRDQN']

file_path = './Result/whole.txt'

with open(file_path, newline='') as csvfile:
    rows = list(csv.reader(csvfile))
    agents = rows[0]
    win = rows[1:]
    num_agents = len(rows[0])

    agent_list = []
    for k in range(num_agents):
        win_rate = 0.0
        for l in range(num_agents):
            if k != l:
                win_rate += float(win[k][l])
        win_rate /= (num_agents - 1)
        agent_list.append((agents[k], win_rate))
    table = [[0.0 for i in range(5)] for j in range(5)]
    for i in range(25):
        table[i // 5][i % 5] = round(agent_list[i][1], 2)
    for i in range(5):
        print(*table[i])
