import csv
import os.path
import copy

agent_place = ['PPO_1', 'DQN_1', 'DQN_scratch', 'DQN_scratch', 'DQN_scratch']
algo = ['PPO', 'DQN', 'DQN_scratch', 'DDQN', 'PRDQN']


def is_subseq(x, y):
    it = iter(y)
    return all(c in it for c in x)


for i in range(5):
    for j in range(5):
        file_path = f'./Result/{i+1}_{j+1}.txt'
        csvfile = open(file_path, newline='')

        rows = list(csv.reader(csvfile))
        agents = rows[0]
        win = rows[1:]

        for k in range(6):
            for l in range(6):
                win[k][l] = float(win[k][l])

        order = {}
        for k in range(6):
            file_name = os.path.basename(agents[k])
            if is_subseq('0100000', file_name):
                file_name = f'{i+1}_{j+1}_0_100000'
                order[k] = 0
            elif is_subseq('0200000', file_name):
                file_name = f'{i+1}_{j+1}_0_200000'
                order[k] = 1
            elif is_subseq('0300000', file_name):
                file_name = f'{i+1}_{j+1}_0_300000'
                order[k] = 2
            elif is_subseq('1100000', file_name):
                file_name = f'{i+1}_{j+1}_1_100000'
                order[k] = 3
            elif is_subseq('1200000', file_name):
                file_name = f'{i+1}_{j+1}_1_200000'
                order[k] = 4
            elif is_subseq('1300000', file_name):
                file_name = f'{i+1}_{j+1}_1_300000'
                order[k] = 5
            else:
                raise ValueError(f'Value error')
            agents[k] = file_name

        new_agents = copy.deepcopy(agents)
        new_win = copy.deepcopy(win)
        for k in range(6):
            new_agents[order[k]] = agents[k]
        for k in range(6):
            for l in range(6):
                new_win[order[k]][order[l]] = win[k][l]
        print(new_agents + new_win)

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows([new_agents] + new_win)
