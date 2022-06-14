# Playing LUX AI using Reinforcement Learning

![](https://raw.githubusercontent.com/Lux-AI-Challenge/Lux-Design-2021/master/assets/daynightshift.gif)

[Lux AI Challenge](https://www.lux-ai.org/) is an on-going Kaggle competition. In this repo, we tried several popular Reinforcement Learning algorithms and compare their results.

The agents are implemented based on the [Lux AI 2021 python game engine and gym](https://github.com/glmcdona/LuxPythonEnvGym/)

## Requirements

All codes in this project are tested with Ubuntu 20.04 and Python 3.7.0. Higher version of Python may produce errors.

Other requirements are in `requirements.txt`. Just simply run: `pip install -r requirements.txt`.

## Usage

### Training models

In this project, we implemented 5 different reward metrics and 5 different algorithms (PPO, DQN, DQN(self implemented), DDQN, Prioritized Replay DDQN). The reward metrics are in the `Agents` directory. PPO and DQN are from the `stable_baselines3` package. The other three are in the `DQN_scratch` directory.

To train a model, just run `train_PPO.py`, `train_DQN.py`, `train_DQN_scratch.py`. Some parameters can be modified in the training process:

* If running `train_PPO.py`, `train_DQN.py` on command line, some arguments like learning rate, training iterations can be modified. Please refer to the code to get the list of parameters.
* To change other reward metrics, just change the `from Agents.agent_1 import AgentPolicy` line.

### Testing results

Now we have all the trained models. We have two phases of testing: The first phase involves models with the same reward metric and training algorithm, and the second phase involves different phases.

To run the first phase, create `agent.txt`, and put all the trained models in the order: Agent directory, trained model directory, training algorithm and path for saving the result. For example:

```
Agents.agent_1 models/directory_to_your_model PPO Results/1_1.txt
Agents.agent_2 models/directory_to_your_another_model DQN Results/1_2.txt
```

Then run `ranking.py`.

The second phase is a similar one, but put the model in the order: Agent directory, trained model path, training algorithm. Then run `rank_whole.py`.

## Results

Files with filename `a_b.txt` is the result of using reward metric a and training algorithm b. The reward metric order is the order in `Agents` directory. The training algorithm order is PPO, DQN, DQN(self implemented), DDQN, Prioritized Replay DDQN. Each file is a csv file showing the winning rate of each pair of agents.

The file `whole.txt` is the final ranking results.
