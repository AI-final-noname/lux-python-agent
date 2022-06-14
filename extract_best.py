import csv

agent_place = ['PPO_1', 'DQN_1', 'DQN_scratch', 'DQN_scratch', 'DQN_scratch']
algo = ['PPO', 'DQN', 'DQN_scratch', 'DDQN', 'PRDQN']

for i in range(5):
    for j in range(5):
        file_path = f'./Result/{i+1}_{j+1}.txt'
        with open(file_path, newline='') as csvfile:
            rows = list(csv.reader(csvfile))
            agents = rows[0]
            win = rows[1:]

            agent_list = []
            best_win_rate, best_model = 0.0, ''
            for k in range(6):
                win_rate = 0.0
                for l in range(6):
                    if k != l:
                        win_rate += float(win[k][l])
                win_rate /= 5
                if win_rate >= best_win_rate:
                    best_win_rate = win_rate
                    best_model = agents[k]
            print(agent_place[j], best_model, algo[j])
