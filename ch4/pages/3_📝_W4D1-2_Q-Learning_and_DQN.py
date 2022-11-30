import os
if not os.path.exists("./images"):
    os.chdir("./ch4")
from st_dependencies import *
styling()
import plotly.io as pio
import re
import json

def img_to_html(img_path, width):
    with open("images/" + img_path, "rb") as file:
        img_bytes = file.read()
    encoded = base64.b64encode(img_bytes).decode()
    return f"<img style='width:{width}px;max-width:100%;st-bottom:25px' src='data:image/png;base64,{encoded}' class='img-fluid'>"
def st_image(name, width):
    st.markdown(img_to_html(name, width=width), unsafe_allow_html=True)

def read_from_html(filename):
    filename = f"images/{filename}.html"
    with open(filename) as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html)[0]
    call_args = json.loads(f'[{call_arg_str}]')
    try:
        plotly_json = {'data': call_args[1], 'layout': call_args[2]}
        fig = pio.from_json(json.dumps(plotly_json))
    except:
        del call_args[2]["template"]["data"]["scatter"][0]["fillpattern"]
        plotly_json = {'data': call_args[1], 'layout': call_args[2]}
        fig = pio.from_json(json.dumps(plotly_json))
    return fig

# def get_fig_dict():
#     names = [f"rosenbrock_{i}" for i in range(1, 5)]
#     return {name: read_from_html(name) for name in names}

# if "fig_dict" not in st.session_state:
#     fig_dict = get_fig_dict()
#     st.session_state["fig_dict"] = fig_dict
# else:
#     fig_dict = st.session_state["fig_dict"]

def section_home():
    st.markdown(r"""
## 1️⃣ Q-learning

Now, we deal with situations where the environment is a black-box, and the agent must learn the rules of the world via interaction with it. This is different from everything else we've done so far, e.g. in the previous section we could calculate optimal policies by using the tensors $R$ and $T$, which we will now assume the agent doesn't have direct knowledge of.

We call algorithms which have access to the transition probability distribution and reward function **model-based algorithms**. **Q-learning** is a **model-free algorithm**. From the original paper introducing Q-learning:""")

    cols = st.columns([1, 10, 1])
    with cols[1]:
        st.markdown(r"""
*[Q-learning] provides agents with the capability of learning to act optimally in Markovian domains by experiencing the consequences of actions, without requiring them to build maps of the domains.*
""")
    st.markdown("")
    st.markdown(r"""
## 2️⃣ DQN

In this section, you'll implement Deep Q-Learning, often referred to as DQN for "Deep Q-Network". This was used in a landmark paper Playing Atari with [Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).

You'll then apply the technique of DQN to master the famous CartPole environment (below).
""")
    st.markdown("<img src='https://miro.medium.com/max/1200/1*v8KcdjfVGf39yvTpXDTCGQ.gif' style='max-width:100%;margin-top:0px'>", unsafe_allow_html=True)
    st.markdown("""
## 3️⃣ Bonus

If you have time, then you can move onto bonus exercises! These include harder games like Acrobot and MountainCar, as well as more difficult implementations like Dueling DQN.
""")

def section_1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#learning-objectives">Learning Objectives</a></li>
   <li><a class="contents-el" href="#readings">Readings</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#optional-readings">Optional Readings</a></li>
   </ul></li>
   <li><a class="contents-el" href="#recap-of-gymenv">Recap of <code>gym.Env</code></a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#the-step-method">The <code>step</code> method</a></li>
       <li><a class="contents-el" href="#the-render-method">The <code>render()</code> method</a></li>
       <li><a class="contents-el" href="#observation-and-action-types">Observation and Action Types</a></li>
       <li><a class="contents-el" href="#registering-an-environment">Registering an Environment</a></li>
       <li><a class="contents-el" href="#timelimit-wrapper">TimeLimit Wrapper</a></li>
   </ul></li>
   <li><a class="contents-el" href="#cheater-agent">Cheater Agent</a></li>
   <li><a class="contents-el" href="#sarsa:-on-policy-td-control">SARSA: On-Policy TD Control</a></li>
   <li><a class="contents-el" href="#q-learning:-off-policy-td-control">Q-Learning: Off-policy TD Control</a></li>
   <li><a class="contents-el" href="#explore-vs-exploit">Explore vs. Exploit</a></li>
   <li><a class="contents-el" href="#tips">Tips</a></li>
   <li><a class="contents-el" href="#other-environments">Other Environments</a></li>
   <li><a class="contents-el" href="#tabular-methods">Tabular Methods</a></li>
   <li><a class="contents-el" href="#bonus">Bonus</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#build-your-own-cliffwalking-environment">Build your own CliffWalking environment</a></li>
       <li><a class="contents-el" href="#monte-carlo-q-learning">Monte-Carlo Q-learning</a></li>
       <li><a class="contents-el" href="#lr-scheduler">LR scheduler</a></li>
       <li><a class="contents-el" href="#other-environments">Other environments</a></li>
       <li><a class="contents-el" href="#double-q-learning">Double-Q learning</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
# Q-learning

Now, we deal with situations where the environment is a black-box, and the agent must learn the rules of the world via interaction with it. This is different from everything else we've done so far, e.g. in the previous section we could calculate optimal policies by using the tensors $R$ and $T$, which we will now assume the agent doesn't have direct knowledge of.

We call algorithms which have access to the transition probability distribution and reward function **model-based algorithms**. **Q-learning** is a **model-free algorithm**. From the original paper introducing Q-learning:""")

    cols = st.columns([1, 10, 1])
    with cols[1]:
        st.markdown("""
*[Q-learning] provides agents with the capability of learning to act optimally in Markovian domains by experiencing the consequences of actions, without requiring them to build maps of the domains.*
""")
    st.markdown(r"""
The "Q" part of Q-learning refers to the function $Q$ which we encountered last week - the expected rewards for an action $a$ taken in a particular state $s$, based on some policy $\pi$.""")

    st.info(r"""
## Learning Objectives

* Understand the `Agent` class provided as well as the `DiscreteEnviroGym` wrapper around `gym.Env` that makes the `.T` and `.R` dynamics avaliable to peek at.
* Understand and implement both **SARSA** and **Q-Learning**.
* Compare SARSA vs. Q-learning on an environment, spend some time hyperparameter tuning.
* Understand that SARSA acts more cautiously as it learns on-policy, while Q-Learning learns off-policy and uses the best action to update, not the action that was taken.""")

    st.markdown(r"""

## Readings

Don't worry about absorbing every detail, we will repeat a lot of the details here. Don't worry too much about the maths, we will also cover that here.

- [Sutton and Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
    - Chapter 6, Section 6.1, 6.3 (Especially Example 6.4)
    - Note that Section 6.1 talks about temporal difference (TD) updates for the value function $V$. We will instead be using TD updates for the Q-value $Q$.
    - Don't worry about the references to Monte Carlo in Chapter 5.

### Optional Readings

- [Q-Learning](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf) The original paper where Q-learning is first described.

```python
from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np
import gym
import gym.spaces
import gym.envs.registration
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from PIL import Image, ImageDraw

import utils
from w3d5_chapter4_tabular.solutions import *

MAIN = __name__ == "__main__"

```

Today and tomorrow, we'll be using OpenAI Gym, which provides a uniform interface to many different RL environments including Atari games. Gym was released in 2016 and details of the API have changed significantly over the years. We are using version 0.23.1, so ensure that any documentation you use refers to the same version.""")

    with st.expander("What's the difference between observation and state?"):
        st.markdown("""
We use the word *observation* here as some environments are *partially observable*, the agent receives not an exact description of the state they are in, but merely an observation giving a partial description (for our gridworld, it could be a description of which cells directly adjacent to the agent are free to move into, rather than precisely which state they are in. This means that the agent
would be unable to distinguish the cell north of the wall from the cell south of the wall. Returning the state as the observation is a special case, and we will often refer to one or the other as required.
""")
    st.markdown("""
Again, we'll be using NumPy for this section, and we'll start off with our gridworld environment from last week:""")
    st_image("gridworld.png", 300)
    st.markdown("")
    st.markdown(r"""
but this time we'll use it within the `gym` framework. 

## Recap of `gym.Env`

Let's have a speed recap of the key features the `gym.Env` class provides, and see how we can use it to wrap our gridworld environment from last week.

### The `step` method

The environment's `step` method takes the action selected by the agent and returns four values: `obs`, `reward`, `done`, and the `info` dictionary.

`obs` and `reward` is the next observation and reward that the agent receives based on the action chosen.

`done` indicates if the environment has entered a terminal state and ended. Here, both the goal-states (+1 and -1) are terminal. Early termination is equivalent to an infinite trajectory where the agent remains trapped for all future states, and always receives reward zero.

`info` can contain anything extra that doesn't fit into the uniform interface - it's up to the environment what to put into it. A good use of this is for debugging information that the agent isn't "supposed" to see, like the dynamics of the environment. Agents that cheat and peek at `info` are helpful because we know that they should obtain the maximum possible rewards; if they aren't, then there's a bug. We will throw the entire underlying environment into `info`, from which an agent could cheat by peeking at the values for `T` and `R`.

### The `render()` method

Render is only used for debugging or entertainment, and what it does is up to the environment. It might render a little popup window showing the Atari game, or it might give you a RGB image tensor, or just some ASCII text describing what's happening.

### Observation and Action Types

A `gym.Env` is a generic type: both the type of the observations and the type of the actions depends on the specifics of the environment.

We're only dealing with the simplest case: a discrete set of actions which are the same in every state. In general, the actions could be continuous, or depend on the state.

---

Below, we define a class that allows us to use our old environment definition from last week, and wrap it in a `gym.Env` instance so we can learn from experience instead.

Read the code below carefully and make sure you understand how the Gym environment API works.

```python
ObsType = int
ActType = int

class DiscreteEnviroGym(gym.Env):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete

    def __init__(self, env: Environment):
        super().__init__()
        self.env = env
        self.observation_space = gym.spaces.Discrete(env.num_states)
        self.action_space = gym.spaces.Discrete(env.num_actions)
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        '''
        Samples from the underlying dynamics of the environment
        '''
        (states, rewards, probs) = self.env.dynamics(self.pos, action)
        idx = self.np_random.choice(len(states), p=probs)
        (new_state, reward) = (states[idx], rewards[idx])
        self.pos = new_state
        done = self.pos in self.env.terminal
        return (new_state, reward, done, {"env": self.env})

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        super().reset(seed=seed)
        self.pos = self.env.start
        return (self.pos, {"env": self.env}) if return_info else self.pos

    def render(self, mode="human"):
        assert mode == "human", f"Mode {mode} not supported!"

```

### Registering an Environment

User code normally won't use the constructor of an `Env` directly for two reasons:

- Usually, we want to wrap our `Env` in one or more wrapper classes.
- If we want to test our agent on a variety of environments, it's annoying to have to import all the `Env` classes directly.

The `register` function stores information about our `Env` in a registry so that a later call to `gym.make` can look it up using the `id` string that is passed in.

By convention, the `id` strings have a suffix with a version number. There can be multiple versions of the "same" environment with different parameters, and benchmarks should always report the version number for a fair comparison. For instance, `id="NorvigGrid-v0"` below.

### TimeLimit Wrapper

As defined, our environment might take a very long time to terminate: A policy that actively avoids
the terminal states and hides in the bottom-left corner would almost surely terminate through
a long sequence of slippery moves, but this could take a long time.
By setting `max_episode_steps` here, we cause our `env` to be wrapped in a `TimeLimit` wrapper class which terminates the episode after that number of steps.

Note that the time limit is also an essential part of the problem definition: if it were larger or shorter, there would be more or less time to explore, which means that different algorithms (or at least different hyperparameters) would then have improved performance. We would obviously want to choose a rather
conservative value such that any reasonably strong policy would be able to reach a terminal state in time.

For our toy gridworld environment, we choose the (rather pessimistic) bound of 100 moves.


```python
gym.envs.registration.register(
    id="NorvigGrid-v0",
    entry_point=DiscreteEnviroGym,
    max_episode_steps=100,
    nondeterministic=True,
    kwargs={"env": Norvig(penalty=-0.04)},
)

gym.envs.registration.register(
    id="ToyGym-v0", 
    entry_point=DiscreteEnviroGym, 
    max_episode_steps=2, 
    nondeterministic=False, 
    kwargs={"env": Toy()}
)
```

Provided is the `RandomAgent` subclass which should pick an action at random, using the random number generator provided by gym. This is useful as a baseline to ensure the environment has no bugs. If your later agents are doing worse than random, you have a bug!


```python
@dataclass
class Experience:
    '''A class for storing one piece of experience during an episode run'''
    obs: ObsType
    act: ActType
    reward: float
    new_obs: ObsType
    new_act: Optional[ActType] = None

@dataclass
class AgentConfig:
    '''Hyperparameters for agents'''
    epsilon: float = 0.1
    lr: float = 0.05
    optimism: float = 0

defaultConfig = AgentConfig()

class Agent:
    '''Base class for agents interacting with an environment (you do not need to add any implementation here)'''
    rng: np.random.Generator

    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma: float = 0.99, seed: int = 0):
        self.env = env
        self.reset(seed)
        self.config = config
        self.gamma = gamma
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n
        self.name = type(self).__name__

    def get_action(self, obs: ObsType) -> ActType:
        raise NotImplementedError()

    def observe(self, exp: Experience) -> None:
        '''
        Agent observes experience, and updates model as appropriate.
        Implementation depends on type of agent.
        '''
        pass

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def run_episode(self, seed) -> List[int]:
        '''
        Simulates one episode of interaction, agent learns as appropriate
        Inputs:
            seed : Seed for the random number generator
        Outputs:
            The rewards obtained during the episode
        '''
        rewards = []
        obs = self.env.reset(seed=seed)
        self.reset(seed=seed)
        done = False
        while not done:
            act = self.get_action(obs)
            (new_obs, reward, done, info) = self.env.step(act)
            exp = Experience(obs, act, reward, new_obs)
            self.observe(exp)
            rewards.append(reward)
            obs = new_obs
        return rewards

    def train(self, n_runs=500):
        '''
        Run a batch of episodes, and return the total reward obtained per episode
        Inputs:
            n_runs : The number of episodes to simulate
        Outputs:
            The discounted sum of rewards obtained for each episode
        '''
        all_rewards = []
        for seed in trange(n_runs):
            rewards = self.run_episode(seed)
            all_rewards.append(utils.sum_rewards(rewards, self.gamma))
        return all_rewards

class Random(Agent):
    def get_action(self, obs: ObsType) -> ActType:
        return self.rng.integers(0, self.num_actions)
```

## Cheater Agent

Implement the cheating agent that peeks at the info and finds the optimal policy directly using your previous code. If your agent gets more than this in the long run, you have a bug! You should solve for the optimal policy once when `Cheater` is initalised, and then use that to define `get_action`.

Check that your cheating agent outperforms the random agent. The cheating agent represents the best possible behavior, as it omnisciently always knows to play optimally.

On the environment `ToyGym-v0`, (assuming $\gamma = 0.99$) the cheating agent should always get reward $2 \gamma = 1.98$,
and the random agent should get a fluctuating reward, with average $\frac{2 \gamma + 1}{2} = 1.49$. 

Hint: Use `env.unwrapped.env` to extract the `Environment` wrapped inside `gym.Env`, to get access to the underlying dynamics.""")

    with st.expander("Help - I get 'AttributeError: 'DiscreteEnviroGym' object has no attribute 'num_states''."):
        st.markdown(r"""
This is probably because you're passing the `DiscreteEnviroGym` object to your `find_optimal_policy` function. In the following line of code:

```python
env_toy = gym.make("ToyGym-v0")
```

the object `env_toy` wraps around the `Toy` environment you used last week. As mentioned, you'll need to use `env.unwrapped.env` to access this environment, and its dynamics.
""")

    st.markdown(r"""
```python
class Cheater(Agent):
    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma=0.99, seed=0):
        pass

    def get_action(self, obs):
        pass


if MAIN:
    env_toy = gym.make("ToyGym-v0")
    agents_toy = [Cheater(env_toy), Random(env_toy)]
    for agent in agents_toy:
        returns = agent.train(n_runs=100)
        plt.plot(utils.cummean(returns), label=agent.name)
    plt.legend()
    plt.title(f"Avg. reward on {env_toy.spec.name}")
    plt.show()

```

## SARSA: On-Policy TD Control

Now we wish to train an agent on the same gridworld environment as before, but this time it doesn't have access to the underlying dynamics (`T` and `R`). The rough idea here is to try and estimate the Q-value function directly from samples received from the environment. Recall that the optimal Q-value function satisfies 
$$
Q^*(s,a) = \mathbb{E}_{\pi^*} \left[ \sum_{i=t+1}^\infty \gamma^{i-t}r_i  \mid s_t = s, a_t = a\right]
= \mathbb{E}_{\pi^*} \left[r + \gamma \max_{a'} Q^*(s', a') \right]
= \sum_{s'} T(s' \mid s,a) \left( R(s,a,s') + \gamma \max_{a'} Q^*(s',a') \right)
$$
where
* $s'$ represents the next state after $s$,
* $a'$ the next action after $a$
* $r$ is the reward obtained from taking action $a$ in state $s$
* the expectation $\mathbb{E}_{\pi^*}$ is with respect to both the optimal policy $\pi^*$, as well as the stochasticity in the environment itself.

So, for any particular episode $s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2, r_3,\ldots$ we have that
*on average* the value of $Q^*(s_t, a_t)$ should be equal to the *actual reward*
$r_t$ recieved when choosing action $a_t$ in state $s_t$, plus $\gamma$ times the
Q-value of the next state $s_{t+1}$ and next action $a_{t+1}$.
$$
Q^*(s_t,a_t) =
\mathbb{E}_{\pi^*} \left[r + \gamma \max_{a'} Q^*(s', a') \right]
\approx r_{t+1} + \gamma  Q^*(s_{t+1}, a_{t+1})
$$
where $a_{t+1} = \pi^*(s_{t+1}) = \argmax_{a} Q^*(s_{t+1}, a)$.


Letting $Q$ denote our best current estimate of $Q^*$, the error $\delta_t := r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t)$  in this "guess" is called the **TD error**, and tells us in which direction we should bump our estimate of $Q^*$.
Of course, this estimate might be wildly inaccurate (even for the same state-action pair!), due to the stochasticity of the environment, or poor estimates of $Q$. So, we update our estimate slightly in the direction of $\delta_t$, much like stochastic gradient descent does. The update rule for Q-learning (with learning rate $\eta > 0$) is
$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \eta \left( r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t) \right)
$$
This update depends on the information $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$, and so is called **SARSA** learning. Note that SARSA learns *on-policy*, in that it only learns from data that was actually generated by the current policy $\pi$, derived from the current estimate of $Q$, $\pi(s) = \argmax_a Q(s,a)$.""")

    st_image("sarsa.png", 700)
    st.markdown("")
    st.markdown(r"""
## Q-Learning: Off-policy TD Control

At the end of the day, what SARSA is essentially doing is estimating $Q^\pi$ by using the rewards gathered by following policy $\pi$. But we don't actually care about $Q^\pi$, what we care about is $Q^*$. Q-Learning provides a slight modification to SARSA, by modifying the TD-error $\delta_t$ to use the action that $\pi$ *should* have taken in state $s_t$ (namely $\argmax_a Q(s_t,a)$) rather than the action $a_t = \pi(s_t)$ that was actually taken.
$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \eta \left( r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t,a_t) \right)
$$
Note that each Q-learning update depends on the information $(s_t, a_t, r_{t+1}, s_{t+1})$.
This means that Q-learning tries to estimate $Q^*$ directly, regardless of what policy $\pi$ generated the episode, and so Q-Learning learns *off-policy*.""")

    st_image("qlearn.png", 700)
    st.markdown("")

    st.markdown(r"""
Note - if you're still confused by the difference between Q-learning and SARSA, you can read [this StackOverflow answer](https://stackoverflow.com/questions/6848828/what-is-the-difference-between-q-learning-and-sarsa).

## Explore vs. Exploit

Lastly, methods to learn the Q-value often have trouble exploring. If a state-action pair $(s,a)$ with low Q-value $Q^*(s,a)$ just so happens to return a high reward by chance, the greedy policy with respect to $Q$ will often choose action $a$ in state $s$ instead of exploring potentially other good actions first. To remedy this, we use instead an $\epsilon$-greedy policy with respect to the current Q-value estimates: With probability $\epsilon$, a random action is chosen, and with probability $1-\epsilon$ the greedy action $\argmax_a Q(s,a)$ is chosen. The exploration probability $\epsilon$ is a hyperparameter that for now we will set to a constant $\epsilon = 0.1$, but more sophisticated techniques include the use of a schedule to start exploring often early, and then decay exploration as times goes on.

We also have the choice of how the estimate $Q(s,a)$ is initialized. By choosing "optimistic" values (initial values that are much higher than what we expect $Q^*(s,a)$ to actually be), this will encourage the greedy policy to hop between different actions in each state when they discover they weren't as valuable as first thought.

We will implement an `EpsilonGreedy` agent that keeps track of the current Q-value estimates, and selects an action based on the epsilon greedy policy.

Both `SARSA` and `QLearning` will inherit from `EpsilonGreedy`, and differ in how they update the Q-value estimates.

- Keep track of an estimate of the Q-value of each state-action pair.
- Epsilon greedy exploration: with probability `epsilon`, take a random action; otherwise take the action with the highest average observed reward (according to your current Q-value estimates).
    - Remember that your `AgentConfig` object contains epsilon, as well as the optimism value and learning rate.
- Optimistic initial values: initialize each arm's reward estimate with the `optimism` value.
- Compare the performance of your Q-learning and SARSA agent again the random and cheating agents.
- Try and tweak the hyperparameters from the default values of `epsilon = 0.1`, `optimism = 1`, `lr = 0.1` to see what effect this has. How fast can you get your
agents to perform?

## Tips

- Use `self.rng.random()` to generate random numbers in the range $[0,1)$, and `self.rng.integers(0, n)` for random integers in the range $0, 1, \ldots, n-1$.
- The random agent results in very long episodes, which slows evaluation. You can remove them from the experiment once you've convinced yourself that your agents are doing something intelligent and outperforming a random agent.
- Leave $\gamma =0.99$ for now.""")

    with st.expander("Help - I'm not sure what methods I should be rewriting in QLearning and SARSA."):
        st.markdown(r"""
Your `EpsilonGreedy` agent already has a method for getting actions, which should use the `self.Q` object. This will be the same for `QLearning` and `SARSA`.

The code you need to add is the `observe` method. Recall from the code earlier that `observe` takes an `Experience` object, which stores data `obs`, `act`, `reward`, `new_obs`, and `new_act`. In mathematical notation, these correspond to $s_t$, $a_t$, $r_{t+1}$, $s_{t+1}$ and $a_{t+1}$.

For `SARSA`, there's an added complication. We want SARSA to directly update the Q-value estimates after action $a_{t+1}$ is taken, as we need all of $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$ to perform a SARSA update. This means you will need to override the `run_episode` function to adjust when the Q-values are updated.
""")
    with st.expander("Help - I'm still confused about the 'run_episode' function."):
        st.markdown(r"""
The main loop of the original `run_episode` function looked like this:

```python
while not done:
    act = self.get_action(obs)
    (new_obs, reward, done, info) = self.env.step(act)
    exp = Experience(obs, act, reward, new_obs)
    self.observe(exp)
    rewards.append(reward)
    obs = new_obs
```

The problem here is that we don't have `new_act` in our `Experience` dataclass, because we only keep track of one action at a time. We can fix this by defining `new_act = self.get_action(new_obs)` **after** `new_obs` is defined. In this way, we can pass all of `(obs, act, reward, new_obs, new_act)` into our `Experience` dataclass.
""")

    with st.expander("What output should I expect to see?"):
        st.markdown(r"""

""")

    st.markdown(r"""
```python
class EpsilonGreedy(Agent):
    '''
    A class for SARSA and Q-Learning to inherit from.
    '''

    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma: float = 0.99, seed: int = 0):
        '''TODO: YOUR CODE HERE'''

    def get_action(self, obs: ObsType) -> ActType:
        '''
        Selects an action using epsilon-greedy with respect to Q-value estimates
        '''
        "TODO: YOUR CODE HERE"

class QLearning(EpsilonGreedy):
    '''TODO: YOUR CODE HERE'''

class SARSA(EpsilonGreedy):
    '''TODO: YOUR CODE HERE'''

```

Compare the performance of SARSA and Q-Learning on the gridworld environment v.s. the cheating agent and the random agent. Try to tune the hyperparameters to get the best performance you can.
- Which seems to work better? SARSA or Q-Learning?
- Does the optimism parameter seems to help?
- What's the best choice of exploration parameter $\epsilon$?
- The feedback from the environment is very noisy. At the moment, the code provided plots the cumulative average reward per episode. You might want to try plotting a sliding average instead, or an exponential weighted moving average (see `utils.py`).


```python
if MAIN:
    env_norvig = gym.make("NorvigGrid-v0")
    "TODO: Init config_norvig using the AgentConfig class"
    n_runs = 1000
    gamma = 0.99
    seed = 1
    args_nor = (env_norvig, config_norvig, gamma, seed)
    agents_norvig = [Cheater(*args_nor), QLearning(*args_nor), SARSA(*args_nor), Random(*args_nor)]
    returns_norvig = {}
    for agent in agents_norvig:
        returns_norvig[agent.name] = agent.train(n_runs)
if MAIN:
    for agent in agents_norvig:
        name = agent.name
        plt.plot(utils.cummean(returns_norvig[name]), label=name)
    plt.legend()
    plt.title(f"Avg. reward on {env_toy.spec.name}")
    plt.show()

```

## Other Environments

`gym` provides a large set of environments with which to test agents against. We can see all available environments by running `gym.envs.registry.all()`

Have a look at [the gym library](https://www.gymlibrary.dev/environments/toy_text/) for descriptions of these environments. As written, our SARSA and Q-Learning agents will only work with environments that have both discrete observation and discrete action spaces.

Modify the above code to use environment `gym.make("CliffWalking-v0")` instead (see [this link](https://www.gymlibrary.dev/environments/toy_text/cliff_walking/)). We have the following graph from Sutton & Barto, Example 6.6, that displays the sum of reward obtained for each episode, as well as the policies obtained (SARSA takes the safer path, Q-Learning takes the optimal path). You may want to check out [this post](https://towardsdatascience.com/walking-off-the-cliff-with-off-policy-reinforcement-learning-7fdbcdfe31ff).""")


    st_image("cliff_pi.png", 400)
    st_image("cliff.png", 400)
    st.markdown("")
    st.markdown(r"""

Can you replicate a similar result?

Some notes for this task:
* The cheating agent will not function on this environment (as standard gym environments aren't required to include the probability distributions in the `info` field as we do), though based on the description it shouldn't be too hard to work out the optimal policy and corresponding reward per episode. You should plot this as a horizontal line to compare against the reward obtained by your agents.
    * One of the bonus exercises we've suggested is to write your own version of `CliffWalking-v0` by writing a class similar to the `Norvig` class you have been working with. If you do this correctly, then you'll also be able to make a cheating agent.
* The random agent will take a *very* long time to accidentally stumble into the goal state, and will slow down learning. You should probably neglect it.
* Use $\gamma = 1$ as described in Sutton & Barto, Example 6.6.
* You can plot a horizontal line with plt.axhline(y=optimal_return, color='r', linestyle='-')
* Try tweaking the learning rate and epsilon (start with $\epsilon = 0.1$) to try and cause SARSA to take the cautious path, while Q-Learning takes the risky path.
* We've included some helper functions to display the value of each state, as well as the policy an agent would take, given the Q-value function.""")

    with st.expander("Question - why is it okay to use gamma=1 here?"):
        st.markdown(r"""
The penalty term `-1` makes sure that the agent continually penalised until it hits the terminal state. Unlike our `Norvig` environment, there is no wall to get stuck in perpetually, rather hitting the cliff will send you back to the start, so the agent must eventually reach the terminal state.""")
    st.markdown(r"""

```python
def show_cliff_value(Q: Arr, title: Optional[str] = None):
    '''
    Displays the value of each state in CliffWalking-v0 given a Q-value table.
    '''
    V = Q.max(axis=-1).reshape(4, 12)
    fig = px.imshow(V, text_auto=".2f", title=title)
    fig.show()

def show_cliff_policy(Q: Arr):
    '''
    Displays the greedy policy for CliffWalking-v0 given a Q-value table.
    '''
    pi = Q.argmax(axis=-1).reshape((4, 12))
    objects = {(3, 0): "green", (3, 11): "red"} | {(3, i): "black" for i in range(1, 11)}
    img = Image.new(mode="RGB", size=(1200, 400), color="white")
    draw = ImageDraw.Draw(img)
    for x in range(0, img.width+1, 100):
        draw.line([(x, 0), (x, img.height)], fill="black", width=4)
    for y in range(0, img.height+1, 100):
        draw.line([(0, y), (img.width, y)], fill="black", width=4)
    for x in range(12):
        for y in range(4):
            draw.regular_polygon((50+x*100, 50+y*100, 20), 3, rotation=-int(90*pi[y][x]), fill="black")
            if (y, x) in objects:
                draw.regular_polygon((50+x*100, 50+y*100, 40), 4, fill=objects[(y, x)])
    display(img.resize((600, 200)))

"TODO: YOUR CODE HERE"
```""")
    with st.expander("Help - my value function registers zero in the cliff and terminal squares, and I don't know why."):
        st.markdown("""This is expected. Your value function $Q$ is intialised to zero, and it never gets updated on these squares because your agent never actually acts from those squares. When it moves into the cliff it immediately reverts back to the start state, and when it moves into the terminal state the game stops.""")
    st.markdown(r"""

## Tabular Methods

The methods used here are called tabular methods, because they create a lookup table from `(state, action)` to the Q value. This is pure memorization, and if our reward function was sampled from the space of all functions, this is usually the best you can do because there's no structure that you can exploit to do better.

We can hope to do better on most "natural" reward functions that do have structure. For example in a game of poker, there is structure in both of the actions (betting $100$ will have a similar reward to betting $99$ or $101$), and between states (having a pair of threes in your hand is similar to having a pair of twos or fours). We need to take advantage of this, otherwise there are just too many states and actions to have any hope of training an agent.

One idea is to use domain knowledge to hand-code a function that "buckets" states or actions into a smaller number of equivalence classes and use those as the states and actions in a smaller version of the problem (see Sutton and Barto, Section 9.5). This was one component in the RL agent [Libratus: The Superhuman AI for No-Limit Poker](https://www.cs.cmu.edu/~noamb/papers/17-IJCAI-Libratus.pdf). The details are beyond the scope of today's material, but I found them fascinating.

If you don't have domain knowledge to leverage, or you care specifically about making your algorithm "general", you can follow the approach that we'll be using in Part 2️⃣: make a neural network that takes in a state (technically, an observation) and outputs a value for each action. We then train the neural network using environmental interaction as training data.

## Bonus

You can progress immediately to part 2️⃣, or if you like you can do the following exercises.

### Build your own CliffWalking environment

You can modify the code used to define the `Norvig` class to define your own version of `CliffWalking-v0`. 

You can do this without guidance, or you can get some more guidance from the dropdowns below. **Hint 1** offers vague guidance, **Hint 2** offers more specific direction.""")

    with st.expander("Hint 1"):
        st.markdown(r"""
You'll need to modify the `__init__`, `render` and `dynamics` functions we used for `Norvig`.

The main way in which the `CliffWalking` environment differs from the `Norvig` gridworld is that the former has cliffs while the latter has walls. Cliffs and walls have different behaviour; you can see how the cliffs affect the agent by visiting the documentation page for `CliffWalking-v0`.
""")

    with st.expander("Hint 2"):
        st.markdown(r"""

There are three methods we used when writing the `Norvig` class: `__init__`, `render` and `dynamics`. Here is a rough outline of the changes you'll need to make for each of these:

#### `__init__`

This mainly just involves changing the dimensions of the space, position of the start and terminal states, and parameters like `penalty`. Also, rather than walls, you'll need to define the position of the **cliffs** (which behave differently).

#### `render`

We've already given you functions to use here: `print_cliff_policy` and `show_cliff_value`. You can call one or both of these in your environment's `render` method. 

Remember that these functions take $Q$ as input rather than $\pi$.

#### `dynamics`

You'll need to modify `dynamics` in the following two ways:

* Remove the slippage probability (although it would be interesting to experiment with this and see what effect it has!)
* Remove the "when you hit a wall, you get trapped forever" behaviour, and replace it with "when you hit a cliff, you get a reward of -100 and go back to the start state".
""")
        st.markdown("")

    st.markdown(r"""
### Monte-Carlo Q-learning

Implement Monte-Carlo Q-learning (Chapter 5 Sutton and Barto) and $\text{TD}(\lambda)$ with eligibility traces (Chapter 7 Sutton and Barto).

### LR scheduler

Try using a schedule for the exploration rate $\epsilon$ (Large values early to encourage exploration, low values later once the agent has sufficient statistics to play optimally).

Would Q-Learning or SARSA be better off with a scheduled exploration rate?

The Sutton book mentions that if $\epsilon$ is gradually reduced, both methods asymptotically converge to the optimal policy. Is this what you find?

### Other environments

Try other environments like [Frozen Lake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) and [BlackJack](https://www.gymlibrary.dev/environments/toy_text/blackjack/). Note that BlackJack uses `Tuple(Discrete(32), Discrete(11), Discrete(2))` as it's observation space, so you will have to write some glue code to convert this back and forth between an observation space of `Discrete(32 * 11 * 2)` to work with our agents as written.

### Double-Q learning

Read Sutton and Barto Section 6.7 Maximisation Bias and Double Learning. Implement Double-Q learning, and compare it's performance against SARSA and Q-Learning.
""")

def section_2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#learning-objectives">Learning Objectives</a></li>
    <li><a class="contents-el" href="#readings">Readings</a></li>
    <li><a class="contents-el" href="#fast-feedback-loops">Fast Feedback Loops</a></li>
    <li><a class="contents-el" href="#cartpole">CartPole</a></li>
    <li><a class="contents-el" href="#outline-of-the-exercises">Outline of the Exercises</a></li>
    <li><a class="contents-el" href="#the-q-network">The Q-Network</a></li>
    <li><a class="contents-el" href="#replay-buffer">Replay Buffer</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#correlated-states">Correlated States</a></li>
        <li><a class="contents-el" href="#uniform-sampling">Uniform Sampling</a></li>
    </ul></li>
    <li><a class="contents-el" href="#environment-resets">Environment Resets</a></li>
    <li><a class="contents-el" href="#exploration">Exploration</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#reward-shaping">Reward Shaping</a></li>
        <li><a class="contents-el" href="#reward-hacking">Reward Hacking</a></li>
        <li><a class="contents-el" href="#advanced-exploration">Advanced Exploration</a></li>
    </ul></li>
    <li><a class="contents-el" href="#epsilon-greedy-policy">Epsilon Greedy Policy</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#hints:">Hints:</a></li>
    </ul></li>
    <li><a class="contents-el" href="#probe-environments">Probe Environments</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#additional-probe-environments">Additional Probe Environments</a></li>
    </ul></li>
    <li><a class="contents-el" href="#main-dqn-algorithm">Main DQN Algorithm</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#logging-metrics">Logging Metrics</a></li>
        <li><a class="contents-el" href="#weights-and-biases">Weights and Biases</a></li>
        <li><a class="contents-el" href="#expected-behavior-of-the-loss">Expected Behavior of the Loss</a></li>
    </ul></li>
    <li><a class="contents-el" href="#hints">Hints</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""

# Deep Q-Learning

In this section, you'll implement Deep Q-Learning, often referred to as DQN for "Deep Q-Network". This was used in a landmark paper [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).

At the time, the paper was very exicitng: The agent would play the game by only looking at the same screen pixel data that a human player would be looking at, rather than a description of where the enemies in the game world are. The idea that convolutional neural networks could look at Atari game pixels and "see" gameplay-relevant features like a Space Invader was new and noteworthy. In 2022, we take for granted that convnets work, so we're going to focus on the RL aspect solely, and not the vision component.""")

    st.info(r"""
## Learning Objectives

* Play `CartPole` to appreciate how difficult the task is for humans!
* Implement the Q-network that will be used to estimate the Q-value for an (observation, action) pair.
* Implement the replay buffer that stores experience from interaction with the environment, from which batches of training data will be sampled.
* Implement the linear schedule for the exploration constant.
* Implement an epsilon greedy policy based on the Q-Network and linearly decaying epsilon.
* Implement some trivial debugging environments to sanity-check the DQN implementation.
* Put all the pieces together, and implement the DQN algorithm, which should learn to balance the pole in `CartPole` until termination in 500 timesteps.
""")

    st.markdown(r"""

## Readings

* [Deep Q Networks Explained](https://www.lesswrong.com/posts/kyvCNgx9oAwJCuevo/deep-q-networks-explained)
    * A high-level distillation as to how DQN works.
* [Andy Jones - Debugging RL, Without the Agonizing Pain](https://andyljones.com/posts/rl-debugging.html)
    * Useful tips for debugging your code when it's not working.
    * The "probe environments" (a collection of simple environments of increasing complexity) section will be our first line of defense against bugs.""")

    with st.expander("Interesting Resources (not required reading)"):
        st.markdown(r"""
- [An Outsider's Tour of Reinforcement Learning](http://www.argmin.net/2018/06/25/outsider-rl/) - comparison of RL techniques with the engineering discipline of control theory.
- [Towards Characterizing Divergence in Deep Q-Learning](https://arxiv.org/pdf/1903.08894.pdf) - analysis of what causes learning to diverge
- [Divergence in Deep Q-Learning: Tips and Tricks](https://amanhussain.com/post/divergence-deep-q-learning/) - includes some plots of average returns for comparison
- [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures) - 2017 bootcamp with video and slides. Good if you like videos.
- [DQN debugging using OpenAI gym Cartpole](https://adgefficiency.com/dqn-debugging/) - random dude's adventures in trying to get it to work.
- [CleanRL DQN](https://github.com/vwxyzjn/cleanrl) - single file implementations of RL algorithms. Your starter code today is based on this; try not to spoiler yourself by looking at the solutions too early!
- [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html) - 2018 article describing difficulties preventing industrial adoption of RL.
- [Deep Reinforcement Learning Works - Now What?](https://tesslerc.github.io/posts/drl_works_now_what/) - 2020 response to the previous article highlighting recent progress.
- [Seed RL](https://github.com/google-research/seed_rl) - example of distributed RL using Docker and GCP.""")
        st.markdown("")

    st.markdown(r"""
## Fast Feedback Loops

We want to have faster feedback loops, and learning from Atari pixels doesn't achieve that. It might take 15 minutes per training run to get an agent to do well on Breakout, and that's if your implementation is relatively optimized. Even waiting 5 minutes to learn Pong from pixels is going to limit your ability to iterate, compared to using environments that are as simple as possible.

## CartPole

The classic environment "CartPole-v1" is simple to understand, yet hard enough for a RL agent to be interesting, by the end of the day your agent will be able to do this and more! (Click to watch!)


[![CartPole](https://img.youtube.com/vi/46wjA6dqxOM/0.jpg)](https://www.youtube.com/watch?v=46wjA6dqxOM "CartPole")

If you like, run `python play_cartpole.py` (locally, not on the remote machine) to try having a go at the task yourself! Use Left/Right to move the cart, R to reset, and Q to quit. By default, the cart will alternate Left/Right actions (there's no no-op action) if you're not pressing a button.

The description of the task is [here](https://www.gymlibrary.dev/environments/classic_control/cart_pole/). Note that unlike the previous environments, the observation here is now continuous. You can see the source for CartPole [here](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py); don't worry about the implementation but do read the documentation to understand the format of the actions and observations. In particular, take note of the following descriptions of the action space and observation space:

```
### Action Space
The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction of the fixed force the cart is pushed with.
| Num | Action                 |
|-----|------------------------|
| 0   | Push cart to the left  |
| 1   | Push cart to the right |

### Observation Space
The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

| Num | Observation           | Min                 | Max               |
|-----|-----------------------|---------------------|-------------------|
| 0   | Cart Position         | -4.8                | 4.8               |
| 1   | Cart Velocity         | -Inf                | Inf               |
| 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
| 3   | Pole Angular Velocity | -Inf                | Inf               |
```

The simple physics involved would be very easy for a model-based algorithm to fit, (this is a common assignment in control theory using [proportional-integral-derivative](https://en.wikipedia.org/wiki/PID_controller) (PID) controllers) but today we're doing it model-free: your agent has no idea that these observations represent positions or velocities, and it has no idea what the laws of physics are. The network has to learn in which direction to bump the cart in response to the current state of the world.

Each environment can have different versions registered to it. By consulting [the Gym source](https://github.com/openai/gym/blob/master/gym/envs/__init__.py) you can see that CartPole-v0 and CartPole-v1 are the same environment, except that v1 has longer episodes. Again, a minor change like this can affect what algorithms score well; an agent might consistently survive for 200 steps in an unstable fashion that means it would fall over if ran for 500 steps.

## Outline of the Exercises

- Implement the Q-network that maps a state to an estimated value for each action.
- Implement the policy which chooses actions based on the Q-network, plus epsilon greedy randomness
to encourage exploration.
- Implement a replay buffer to store experiences $e_t = (s_t, a_t, r_{t+1}, s_{t+1})$.
- Piece everything together into a training loop and train your agent.

```python
import argparse
import os
import sys
import random
import time
import re
from dataclasses import dataclass
from distutils.util import strtobool
from typing import Any, List, Optional, Union, Tuple, Iterable
import gym
import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from gym.spaces import Discrete, Box
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from numpy.random import Generator
import gym.envs.registration
import pandas as pd
from w3d5_chapter4_tabular.utils import make_env
from w4d2_chapter4_dqn import utils

MAIN = __name__ == "__main__"
os.environ["SDL_VIDEODRIVER"] = "dummy"

```

## The Q-Network

The Q-Network takes in an observation and outputs a number for each available action predicting how good it is, mimicking he behaviour of our Q-value table from last week.
For best results, the architecture of the Q-network can be customized to each particular problem. For example, [the architecture of OpenAI Five](https://cdn.openai.com/research-covers/openai-five/network-architecture.pdf) used to play DOTA 2 is pretty complex and involves LSTMs.

For learning from pixels, a simple convolutional network and some fully connected layers does quite well. Where we have already processed features here, it's even easier: an MLP of this size should be plenty large for any environment today. Your code should support running the network on either GPU or CPU, but for CartPole it was actually faster to use CPU on my hardware.

Implement the Q-network using a standard MLP, constructed of alternating Linear and ReLU layers.
The size of the input will match the dimensionality of the observation space, and the
size of the output will match the number of actions to choose from (associating a reward to each.)
The dimensions of the hidden_sizes are provided.

Here is a diagram of what our particular Q-Network will look like for CartPole.""")

    st.write("""<figure style="max-width:300px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNplkEtrwzAMgP-K0WkDB5ash5LDTtmhUDa616UeRY3VxSx-4AdjlP73Oc3WpkwHIaTvA0l7aK0kqOHDo-vYSyMMyxHSdmw0RI6tigeKX9Z_jsMhFmYtYGFciuzKbsMmdOiIXwt4Z0Vxx5bKEPoyM2M1YVhZ3Zy4J1q-lushT73q7GWYs_nsQqj-CbdnYT7jzCS9wTYqa8JJXL1hn6nHFI87T5Dj1uNlZCRw0OQ1Kpmfsh_aAmJHmgTUuZS0w9RHAcIcMpqcxEj3UkXrod5hH4gDpmifv00LdfSJ_qBGYX6p_qUOP7mqd2g" /></figure>""", unsafe_allow_html=True)
    # graph TD
    #     subgraph Deep Q-Network
    #         In["Input (obs_shape,)"] --> Linear1["Linear(obs_shape, 120)"] --> ReLU1[ReLU] --> Linear2["Linear(120, 84)"] --> ReLU2[ReLU] --> Linear3["Linear(84, num_actions)"] --> QVal["Output (num_actions,)"]
    #     end

    with st.expander("Why do we not include a ReLU at the end?"):
        st.markdown("If you end with a ReLU, then your network can only predict 0 or positive Q-values. This will cause problems as soon as you encounter an environment with negative rewards, or you try to do some scaling of the rewards.")
    with st.expander("CartPole-v1 gives +1 reward on every timestep. Why would the network not just learn the constant +1 function regardless of observation?"):
        st.markdown("""The network is learning Q-values (the sum of all future expected discounted rewards from this state/action pair), not rewards. Correspondingly, once the agent has learned a good policy, the Q-value associated with state action pair (pole is slightly left of vertical, move cart left) should be large, as we would expect a long episode (and correspondingly lots of reward) by taking actions to help to balance the pole. Pairs like (cart near right boundary, move cart right) cause the episode to terminate, and as such the network will learn low Q-values.""")

    st.markdown(r"""
```python
class QNetwork(nn.Module):
    def __init__(self, dim_observation: int, num_actions: int, hidden_sizes: list[int] = [120, 84]):
        super().__init__()
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        pass

if MAIN:
    net = QNetwork(dim_observation=4, num_actions=2)
    n_params = sum((p.nelement() for p in net.parameters()))
    print(net)
    print(f"Total number of parameters: {n_params}")
    print("You should manually verify network is Linear-ReLU-Linear-ReLU-Linear")
    assert n_params == 10934
```

## Replay Buffer

The goal of DQN is to reduce the reinforcement learning problem to a supervised learning problem.
In supervised learning, training examples should be drawn **i.i.d**. from some distribution, and we hope to generalize to future examples from that distribution.

In RL, the distribution of experiences $e_t = (s_t, a_t, r_{t+1}, s_{t+1})$ to train from depend on the policy $\pi$ followed, which depends on the current state of the Q-value network, so DQN is always chasing a moving target. This is why the training loss curve isn't going to have a nice steady decrease like in supervised learning. We will extend experiences to $e_t = (o_t, a_t, r_{t+1}, o_{t+1}, d_{t+1})$. Here, $d_{t+1}$ is a boolean indicating that $o_{t+1}$ is a terminal observation, and that no further interaction happened beyond $s_{t+1}$ in the episode from which it was generated.

### Correlated States

Due to DQN using a neural network to learn the Q-values, the value of many state-action pairs are aggregated together (unlike tabular Q-learning which learns independently the value of each state-action pair). For example, consider a game of chess. The board will have some encoding as a vector, but visually similar board states might have wildly different consequences for the best move. Another problem is that states within an episode are highly correlated and not i.i.d. at all. A few bad moves from the start of the game might doom the rest of the game regardless how well the agent tries to recover, whereas a few bad moves near the end of the game might not matter if the agent has a very strong lead, or is so far behind the game is already lost. Training mostly on an episode where the agent opened the game poorly might disincentive good moves to recover, as these too will have poor Q-value estimates.

### Uniform Sampling

To recover from this problem and make the environment look "more i.i.d", a simple strategy that works decently well is to pick a buffer size, store experiences and uniformly sample out of that buffer. Intuitively, if we want the policy to play well in all sorts of states, the sampled batch should be a representative sample of all the diverse scenarios that can happen in the environment.

For complex environments, this implies a very large batch size (or doing something better than uniform sampling). [OpenAI Five](https://cdn.openai.com/dota-2.pdf) used batch sizes of over 2 million experiences for Dota 2.

The capacity of the replay buffer is yet another hyperparameter; if it's too small then it's just going to be full of recent and correlated examples. But if it's too large, we pay an increasing cost in memory usage and the information may be too old to be relevant.

Implement `ReplayBuffer`. It only needs to handle a discrete action space, and you can assume observations are some shape of dtype `np.float32`, and actions are of dtype `np.int64`. The replay buffer will store experiences $e_t = (o_t, a_t, r_{t+1}, o_{t+1}, d_{t+1})$ in a circular queue. If the buffer is already full, the oldest experience is overwritten.

```python
@dataclass
class ReplayBufferSamples:
    '''
    Samples from the replay buffer, converted to PyTorch for use in neural network training.
    obs: shape (sample_size, *observation_shape), dtype t.float
    actions: shape (sample_size, ) dtype t.int
    rewards: shape (sample_size, ), dtype t.float
    dones: shape (sample_size, ), dtype t.bool
    next_observations: shape (sample_size, *observation_shape), dtype t.float
    '''
    rng: Generator
    observations: t.Tensor
    actions: t.Tensor
    rewards: t.Tensor
    dones: t.Tensor
    next_observations: t.Tensor


class ReplayBuffer:
    rng: Generator
    observations: t.Tensor
    actions: t.Tensor
    rewards: t.Tensor
    dones: t.Tensor
    next_observations: t.Tensor

    def __init__(self, buffer_size: int, num_actions: int, observation_shape: tuple, num_environments: int, seed: int):
        assert num_environments == 1, "This buffer only supports SyncVectorEnv with 1 environment inside."
        pass

    def add(
        self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, next_obs: np.ndarray
    ) -> None:
        '''
        obs: shape (num_environments, *observation_shape) 
            Observation before the action
        actions: shape (num_environments, ) 
            Action chosen by the agent
        rewards: shape (num_environments, ) 
            Reward after the action
        dones: shape (num_environments, ) 
            If True, the episode ended and was reset automatically
        next_obs: shape (num_environments, *observation_shape) 
            Observation after the action
            If done is True, this should be the terminal observation, NOT the first observation of the next episode.
        '''
        pass

    def sample(self, sample_size: int, device: t.device) -> ReplayBufferSamples:
        '''Uniformly sample sample_size entries from the buffer and convert them to PyTorch tensors on device.
        Sampling is with replacement, and sample_size may be larger than the buffer size.
        '''
        pass

if MAIN:
    utils.test_replay_buffer_single(ReplayBuffer)
    utils.test_replay_buffer_deterministic(ReplayBuffer)
    utils.test_replay_buffer_wraparound(ReplayBuffer)
```

## Environment Resets

There's a subtlety to the Gym API around what happens when the agent fails and the episode is terminated. Our environment is set up to automatically reset at the end of an episode, but when this happens the `next_obs` returned from `step` is actually the initial observation of the new episode.

What we want to store in the replay buffer is the final observation of the old episode. The code to do this is shown below.

- Run the code and inspect the replay buffer contents. Referring back to the [CartPole source](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py), do the numbers make sense?
- Look at the sample, and see if it looks properly shuffled.


```python
if MAIN:
    rb = ReplayBuffer(buffer_size=256, num_actions=2, observation_shape=(4,), num_environments=1, seed=0)
    envs = gym.vector.SyncVectorEnv([utils.make_env("CartPole-v1", 0, 0, False, "test")])
    obs = envs.reset()
    for i in range(512):
        actions = np.array([0])
        (next_obs, rewards, dones, infos) = envs.step(actions)
        rb.add(obs, actions, rewards, dones, next_obs)
        obs = next_obs
    sample = rb.sample(128, t.device("cpu"))
    columns = ["cart_pos", "cart_v", "pole_angle", "pole_v"]
    df = pd.DataFrame(rb.observations, columns=columns)
    df.plot(subplots=True, title="Replay Buffer")
    df2 = pd.DataFrame(sample.observations, columns=columns)
    df2.plot(subplots=True, title="Shuffled Replay Buffer")
```""")

    with st.expander("Click to reveal the kind of graphs you should be getting."):
        st_image("replaybuffer.png", 420)
        st_image("shuffledbuffer.png", 420)

    st.markdown(r"""
## Exploration

DQN makes no attempt to explore intelligently. The exploration strategy is the same as
for Q-Learning: agents take a random action with probability epsilon, but now we gradually
decrease epsilon. The Q-network is also randomly initialized (rather than initialized with zeros),
so its predictions of what is the best action to take are also pretty random to start.

Some games like [Montezuma's Revenge](https://paperswithcode.com/task/montezumas-revenge) have sparse rewards that require more advanced exploration methods to obtain. The player is required to collect specific keys to unlock specific doors, but unlike humans, DQN has no prior knowledge about what a key or a door is, and it turns out that bumbling around randomly has too low of a probability of correctly matching a key to its door. Even if the agent does manage to do this, the long separation between finding the key and going to the door makes it hard to learn that picking the key up was important.

As a result, DQN scored an embarrassing 0% of average human performance on this game.

### Reward Shaping

One solution to sparse rewards is to use human knowledge to define auxillary reward functions that are more dense and made the problem easier (in exchange for leaking in side knowledge and making
the algorithm more specific to the problem at hand). What could possibly go wrong?

The canonical example is for a game called [CoastRunners](https://openai.com/blog/faulty-reward-functions/), where the goal was given to maximize the score (hoping that the agent would learn to race around the map). Instead, it found it could gain more score by driving in a loop picking up power-ups just as they respawn, crashing and setting the boat alight in the process.

### Reward Hacking

For Montezuma's Revenge, the reward was shaped by giving a small reward for picking up the key. One time this was tried, the reward was given slightly too early and the agent learned it could go close to the key without quite picking it up, obtain the auxillary reward, and then back up and repeat.

[![Montezuma Reward Hacking](https://img.youtube.com/vi/_sFp1ffKIc8/0.jpg)](https://www.youtube.com/watch?v=_sFp1ffKIc8 "Montezuma Reward Hacking")

A collected list of examples of Reward Hacking can be found [here](https://docs.google.com/spreadsheets/d/e/2PACX-1vRPiprOaC3HsCf5Tuum8bRfzYUiKLRqJmbOoC-32JorNdfyTiRRsR7Ea5eWtvsWzuxo8bjOxCG84dAg/pubhtml).


### Advanced Exploration

It would be better if the agent didn't require these auxillary rewards to be hardcoded by humans, but instead reply on other signals from the environment that a state might be worth exploring. One idea is that a state which is "surprising" or "novel" (according to the agent's current belief of how the environment works) in some sense might be valuable. Designing an agent to be innately curious presents a potential solution to exploration, as the agent will focus exploration in areas it is unfamiliar with. In 2018, OpenAI released [Random Network Distillation](https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/) which made progress in formalizing this notion, by measuring the agent's ability to predict the output of a neural network on visited states. States that are hard to predict are poorly explored, and thus highly rewarded. In 2019, an excellent paper [First return, then explore](https://arxiv.org/pdf/2004.12919v6.pdf) found an even better approach. Such reward shaping can also be gamed, leading to the noisy TV problem, where agents that seek novelty become entranced by a source of randomness in the environment (like a analog TV out of tune displaying white noise), and ignore everything else in the environment.

For now, implement the basic linearly decreasing exploration schedule.""")

    with st.expander("Solution - Plot of the Intended Schedule"):
        st_image("newplot.png", 560)

    st.markdown(r"""
```python
def linear_schedule(
    current_step: int, start_e: float, end_e: float, exploration_fraction: float, total_timesteps: int
) -> float:
    '''Return the appropriate epsilon for the current step.

    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).

    It should stay at end_e for the rest of the episode.
    '''
    pass


if MAIN:
    epsilons = [
        linear_schedule(step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500)
        for step in range(500)
    ]
    utils.test_linear_schedule(linear_schedule)

```

## Epsilon Greedy Policy

In DQN, the policy is implicitly defined by the Q-network: we take the action with the maximum predicted reward. This gives a bias towards optimism. By estimating the maximum of a set of values $v_1, \ldots, v_n$ using the maximum of some noisy estimates $\hat{v}_1, \ldots, \hat{v}_n$ with $\hat{v}_i \approx v$, we get unlucky and get very large positive noise on some samples, which the maximum then chooses. Hence, the agent will choose actions that the Q-network is overly optimistic about.

See Sutton and Barto, Section 6.7 if you'd like a more detailed explanation, or the original [Double Q-Learning](https://proceedings.neurips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf) paper which notes this maximisation bias, and introduces a method to correct for it using two separate Q-value estimators, each used to update the other.

### Hints:

- Don't forget to convert the result back to a `np.darray`.
- Use `rng.random()` to generate random numbers in the range $[0,1)$, and `rng.integers(0, n, size)` for an array of shape `size` random integers in the range $0, 1, \ldots, n-1$.
- Use `envs.single_action_space.n` to retrieve the number of possible actions.

```python
def epsilon_greedy_policy(
    envs: gym.vector.SyncVectorEnv, q_network: QNetwork, rng: Generator, obs: t.Tensor, epsilon: float
) -> np.ndarray:
    '''With probability epsilon, take a random action. Otherwise, take a greedy action according to the q_network.
    Inputs:
        envs : gym.vector.SyncVectorEnv, the family of environments to run against
        q_network : QNetwork, the network used to approximate the Q-value function
        obs : The current observation
        epsilon : exploration percentage
    Outputs:
        actions: (n_environments, ) the sampled action for each environment.
    '''
    pass

if MAIN:
    utils.test_epsilon_greedy_policy(epsilon_greedy_policy)
```

## Probe Environments

Extremely simple probe environments are a great way to debug your algorithm. The first one is given to you.

Let's try and break down how this environment works. We see that the function `step` always returns the same thing. The observation and reward are always the same, and `done` is always true (i.e. the episode always terminates after one action). We expect the agent to rapidly learn that the value of the constant observation `[0.0]` is `+1`.

### A note on action spaces

The action space we're using here is `gym.spaces.Box`. This means we're dealing with real-valued quantities, i.e. continuous not discrete. The first two arguments of `Box` are `low` and `high`, and these define a box in $\mathbb{R}^n$. For instance, if these arrays are `(0, 0)` and `(1, 1)` respectively, this defines the box $0 \leq x, y \leq 1$ in 2D space.

```python
ObsType = np.ndarray
ActType = int

class Probe1(gym.Env):
    '''One action, observation of [0.0], one timestep long, +1 reward.

    We expect the agent to rapidly learn that the value of the constant [0.0] observation is +1.0. Note we're using a continuous observation space for consistency with CartPole.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([0]), np.array([0]))
        self.action_space = Discrete(1)
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        return (np.array([0]), 1.0, True, {})

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        super().reset(seed=seed)
        if return_info:
            return (np.array([0.0]), {})
        return np.array([0.0])

gym.envs.registration.register(id="Probe1-v0", entry_point=Probe1)
if MAIN:
    env = gym.make("Probe1-v0")
    assert env.observation_space.shape == (1,)
    assert env.action_space.shape == ()
```

### Additional Probe Environments

Feel free to skip ahead for now, and implement these as needed to debug your model. 

Each implementation should be very similar to `Probe1` above. If you aren't sure whether you've implemented them correctly, you can check them against the solutions.

Note - you can access a uniform random number generator using `self.np_random.random()`.

```python
class Probe2(gym.Env):
    '''One action, observation of [-1.0] or [+1.0], one timestep long, reward equals observation.

    We expect the agent to rapidly learn the value of each observation is equal to the observation.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        pass

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        pass

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        pass

gym.envs.registration.register(id="Probe2-v0", entry_point=Probe2)

class Probe3(gym.Env):
    '''One action, [0.0] then [1.0] observation, two timesteps, +1 reward at the end.

    We expect the agent to rapidly learn the discounted value of the initial observation.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        pass

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        pass

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        pass

gym.envs.registration.register(id="Probe3-v0", entry_point=Probe3)

class Probe4(gym.Env):
    '''Two actions, [0.0] observation, one timestep, reward is -1.0 or +1.0 dependent on the action.

    We expect the agent to learn to choose the +1.0 action.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        pass

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        pass

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        pass

gym.envs.registration.register(id="Probe4-v0", entry_point=Probe4)

class Probe5(gym.Env):
    '''Two actions, random 0/1 observation, one timestep, reward is 1 if action equals observation otherwise -1.

    We expect the agent to learn to match its action to the observation.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        pass

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        pass

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        pass

gym.envs.registration.register(id="Probe5-v0", entry_point=Probe5)
```

## Main DQN Algorithm

We now combine all the elements we have designed thus far into the final DQN algorithm. Here, we assume the environment returns three parameters $(s_{new}, r, d)$, a new state $s_{new}$, a reward $r$ and a boolean $d$ indicating whether interaction has terminated yet.

Our Q-value function $Q(s,a)$ is now a network $Q(s,a ; \theta)$ parameterised by weights $\theta$. The key idea, as in Q-learning, is to ensure the Q-value function satisfies the optimal Bellman equation
$$
Q(s,a ; \theta)
= \mathbb{E}_{s',r \sim p(\cdot \mid s,a)} \left[r + \gamma \max_{a'} Q(s', a' ;\theta) \right]
$$

$$
\delta_t = \mathbb{E} \left[ r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]
$$
Letting $y_t = r_t + \gamma \max_a Q(s_{t+1}, a)$, we can use the expected squared TD-Error $\delta_t^2 = (y_t - Q(s_t, a_t))^2$ as the loss function to optimize against. Since we want the model to learn from a variety of experiences (recall that supervised learning is assuming i.i.d) we approximate the expectation by sampling a batch $B = \{s^i, a^i, r^i, s^i_\text{new}\}$ of experiences from the replay buffer, and try to adjust $\theta$ to make the loss
$$
L(\theta) = \frac{1}{|B|} \sum_{i=1}^B \left( r^i +
\gamma \max_a Q(s^i_\text{new}, a ; \theta_\text{target}) - Q(s^i, a^i ; \theta) \right)^2
$$
smaller via gradient descent. Here, $\theta_\text{target}$ is a previous copy of the parameters $\theta$. Every so often, we then update the target $\theta_\text{target} \leftarrow \theta$ as the agent improves it's Q-values from experience. We don't do this every step, because this improves training stability (your network isn't continually chasing a moving target).

Note that $\llbracket S \rrbracket$ is 1 if $S$ is True, and 0 if $S$ is False.""")

    st_image("dqn_algo.png", 550)
    st.markdown("")
    st.markdown("")
    st.markdown(r"""
On line 13, we need to check if the environment is still running before adding the $\gamma \max_{a'}Q(s^i_\text{new}, a' ; \theta_\text{target})$ term, as terminal states don't have future rewards, so we zero the second term if $d^i = \text{True}$, indicating that the episode has terminated.

All the boilerplate code is provided for you. You just need to fill in the three placeholders as indicated, referring to the above algorithm, modified from the original [Algorithm 1, DQN paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). The original algorithm assumes a preprocessing step to turn pixels into useful features to then train on. The observation provided by `CartPole-v1` is already a set of useful features that full encapsulates the entire state of the world (the position/velocity of the cart, and the angle/angular velocity of the pole), so we will store and train from these directly.


```python
@dataclass
class DQNArgs:
    exp_name: str = os.path.basename(globals().get("__file__", "DQN_implementation").rstrip(".py"))
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "CartPoleDQN"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 0.00025
    buffer_size: int = 10000
    gamma: float = 0.99
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 10

arg_help_strings = dict(
    exp_name = "the name of this experiment",
    seed = "seed of the experiment",
    torch_deterministic = "if toggled, `torch.backends.cudnn.deterministic=False`",
    cuda = "if toggled, cuda will be enabled by default",
    track = "if toggled, this experiment will be tracked with Weights and Biases",
    wandb_project_name = "the wandb's project name",
    wandb_entity = "the entity (team) of wandb's project",
    capture_video = "whether to capture videos of the agent performances (check out `videos` folder)",
    env_id = "the id of the environment",
    total_timesteps = "total timesteps of the experiments",
    learning_rate = "the learning rate of the optimizer",
    buffer_size = "the replay memory buffer size",
    gamma = "the discount factor gamma",
    target_network_frequency = "the timesteps it takes to update the target network",
    batch_size = "the batch size of samples from the replay memory",
    start_e = "the starting epsilon for exploration",
    end_e = "the ending epsilon for exploration",
    exploration_fraction = "the fraction of `total-timesteps` it takes from start-e to go end-e",
    learning_starts = "timestep to start learning",
    train_frequency = "the frequency of training",
)
toggles = ["torch_deterministic", "cuda", "track", "capture_video"]

def parse_args(arg_help_strings=arg_help_strings, toggles=toggles) -> DQNArgs:
    parser = argparse.ArgumentParser()
    for (name, field) in DQNArgs.__dataclass_fields__.items():
        flag = "--" + name.replace("_", "-")
        type_function = field.type if field.type != bool else lambda x: bool(strtobool(x))
        toggle_kwargs = {"nargs": "?", "const": True} if name in toggles else {}
        parser.add_argument(
            flag, type=type_function, default=field.default, help=arg_help_strings[name], **toggle_kwargs
        )
    return DQNArgs(**vars(parser.parse_args()))

def setup(args: DQNArgs) -> Tuple[str, SummaryWriter, np.random.Generator, t.device, gym.vector.SyncVectorEnv]:
    '''Helper function to set up useful variables for the DQN implementation'''
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{key}|{value}|" for (key, value) in vars(args).items()]),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv([utils.make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"
    return (run_name, writer, rng, device, envs)

def log(
    writer: SummaryWriter,
    start_time: float,
    step: int,
    predicted_q_vals: t.Tensor,
    loss: Union[float, t.Tensor],
    infos: Iterable[dict],
    epsilon: float,
):
    '''Helper function to write relevant info to TensorBoard logs, and print some things to stdout'''
    if step % 100 == 0:
        writer.add_scalar("losses/td_loss", loss, step)
        writer.add_scalar("losses/q_values", predicted_q_vals.mean().item(), step)
        writer.add_scalar("charts/SPS", int(step / (time.time() - start_time)), step)
        if step % 10000 == 0:
            print("SPS:", int(step / (time.time() - start_time)))
    for info in infos:
        if "episode" in info.keys():
            print(f"global_step={step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], step)
            writer.add_scalar("charts/epsilon", epsilon, step)
            break
    # OPTIONAL: ADD CODE HERE TO LOG TO WANDB

if MAIN:
    if "ipykernel_launcher" in os.path.basename(sys.argv[0]):
        filename = globals().get("__file__", "<filename of this script>")
        print(f"Try running this file from the command line instead: python {os.path.basename(filename)} --help")
        args = DQNArgs()
    else:
        args = parse_args()
    # train_dqn(args)
```

All of the above is boilerplate to interface with Weights and Biases, write information to logs, and read arguments from the commandline. You should only worry about `DQNArgs`, which is a datastructure preinitialized with sensible hyperparameters for CartPole (though you can adjust them using command line arguments.)

After you've written your `train_dqn` function (see below), you should move the `if MAIN` block to the bottom of your Python file, and comment out the `train_dqn` line.

This file will read arguments from the command line. Run

```
python <your-script>.py -h
```

to see the arguments it takes. The hyperparameters are already set to reasonable values by default, have a look inside `w4d2_chapter4_dqn.utils` if you're interested.

A sample invocation that will let you see your agent's progress in Weights and Biases, including videos of it playing episodes would look like:

```
python <your-script>.py --track --capture-video
```

You can do this from **VSCode's terminal**. Note - it can be annoying to have all your `utils` tests happening every time you run your file from the terminal, and there's no easy way of letting Python know whether you're running it from the terminal or from inside the file. One way around this is to add a boolean global variable `TESTING` at the top of your file, and changing the `if MAIN` loops so that they only run when `TESTING = True`. That way, you can set `TESTING = False` when you want to run the file from the terminal. Another solution is just to run your Python file the way you normally would (including the `parse_args` function), but it's preferable to use the terminal.

Don't be discouraged if it's not working - it's normal for debugging RL to take longer than you would expect. Add asserts or your own tests, implement an appropriate probe environment, try anything in the Andy Jones post that sounds promising, and try to notice confusion. Reinforcement Learning is often so tricky as even if the algorithm has bugs, the agent might still learn something useful regardless (albeit maybe not as well), or even if everything is correct, the agent might just fail to learn anything useful (like how DQN failed to do anything on Montezuma's Revenge.)

Since the environment is already know to be one DQN can solve, and we've already provided hyperparameters that work for this environment, hopefully that's isolated a lot of the problems one would usually have with solving real world problems with RL.

### Logging Metrics

You can view your metrics either in the IDE using Tensorboard (VS Code has built-in Tensorboard support) or remotely on Weights and Biases. Some of the logs won't work properly until you define a variable with the expected name.

### Weights and Biases

In previous parts, we've just trained the agent, and then plotted the reward per episode after training. For small toy examples that train in a few seconds this is fine, but for longer runs we'd like to watch the run live and make sure the agent is doing something interesting (especially if we were planning to run the model overnight.)

Luckily, **Weights and Biases** has got us covered! When you run your experiments, you'll be able to view not only *live plots* of the loss and average reward per episode while the agent is training - you can also log and view animations, which visualise your agent's progress in real time! This is all handled via the argument `monitor_gym=True` in `wandb.init`. The way this works is very simple - Weights and Biases just looks for the videos logged automatically by `gym` (which will also be saved in your `gym` folder).

Note that forms of `wandb` logging other than video logging won't be done automatically for you; you'll have to add those like you've done previously. You can do this by editing the `log` function.

### Expected Behavior of the Loss

In supervised learning, we want our loss to always be decreasing and it's a bad sign if it's actually increasing, as that's the only metric we care about. In RL, it's the total reward per epsiode that should be (noisily) increasing over time.

Our agent's loss function just reflects how close together the Q-network's estimates are to the experiences currently sampled from the replay buffer, which might not adequately represent what the world actually looks like.

This means that once the agent starts to learn something and do better at the problem, it's expected for the loss to increase. The loss here is just the TD-error, the difference between how valuable the agent thinks the (state-action) is, v.s. the best  current bootstrapped estimate of the actual Q-value.

For example, the Q-network initially learned some state was bad, because an agent that reached them was just flapping around randomly and died shortly after. But now it's getting evidence that the same state is good, now that the agent that reached the state has a better idea what to do next. A higher loss is thus actually a good sign that something is happening (the agent hasn't stagnated), but it's not clear if it's learning anything useful without also checking how the total reward per episode has changed.

## Hints

* When gathering experiences, make sure you have a line `obs = next_obs` at an appropriate location, or you'll just keep passing in the same observation on every iteration.
* Use `net2.load_state_dict(net1.state_dict())` to copy the parameters from `net1` into an identically shaped network `net2`. (Useful for updating target network.)
* Use `torch.inference_mode()` when computing thins from the target network.
* `args` contains a lot of the parameters you'll need to use, e.g. `learning_rate` for your optimizer's learning rate.
    * Make sure not to confuse this learning rate with your `epsilon`. The former is used in your optimizer and is kept fixed; the latter is used in your agent's policy and follows the `linear_schedule`.
* Pay attention to the difference between observation shape and number of observations.
    * Observation shape is returned from `envs.single_observation_space.shape`. In the case of our CartPole environment, this is simply `(4,)` (see earlier description of CartPole), although in other environments this might be more than one-dimensional. For instance, the observation shape might be `(height, width)` for playing Atari games (because each pixel represents a different observation).
    * The number of observations is simply equal to the product of the elements in `obs_shape`. In the case of CartPole, this is just 4.

Below are a few other hints to guide you through the exercises, in the form of dropdowns. You should use these liberally, because otherwise you might find yourself stuck for a while - it's hard to code up difficult functions like this when you're not already very familiar with the libraries involved.
""")

    with st.expander("Help - I'm not sure how to use the outputs of my Q network in the formulas."):
        st.markdown(r"""
Your network `q_network` will take in an observation (or multiple observations), and return the value for taking each action following that observation.

In order to get $Q(s, a; \theta)$, you just evaluate your output at action `a`. In order to get $\max_{a'}Q(s, a'; \theta)$, you need to take the maximum over your outputs.
""")
    with st.expander("Help - I don't understand when to use my base network and target network."):
        st.markdown(r"""
Your base network is the one that actually gets updated via gradient descent, via the loss function given in the section above. It corresponds to the function $Q(s, a ; \theta)$ in that section.

Your target network is $Q(s, a; \theta_\text{target})$ in the section above. It is also used in the formula for calculating loss. It is only updated once every `args.target_network_frequency` steps. 

The idea beyhind this is that using the target network's Q values to train the main Q-network will improve the stability of the training (because your network isn't continually chasing a moving target).
""")

    st.markdown("""
If you need more detailed instructions for what to do in some of the code sections, then you can look at the dropdowns.
""")
    with st.expander("Guidance for (1)"):
        st.markdown(r"""
You should do the following things, in order:

* Define `num_actions` and `obs_shape` from your `envs` object.
* Define both your networks, and your optimizer (which network's parameters should you pass in?).
    * You can get your optimizer's learning rate from `args`. The other parameters should be left as default.
* Create your `ReplayBuffer` object.
    * Note that you can get the number of environments in `envs` using `len(envs.envs)`.
""")
        st.markdown("")
    with st.expander("Guidance for (2)"):
        st.markdown(r"""
You should do the following things, in order:

* Set `epsilon` using your `linear_schedule` function.
* Choose your actions using your `epsilon_greedy_policy` function
* Return a tuple of `(next_obs, rewards, dones, infos)` from your environment's `step` function.
""")
        st.markdown("")
    with st.expander("Guidance for (3)"):
        st.markdown(r"""
You should do the following things, in order:

* Use the `sample` method from your replay buffer to return $\{s^i, a^i, r^i, d^i, s_\text{new}^i\}$ (here I'm following the notation at the start of the **Main DQN Algorithm** section of this page).
* Calculate your loss function $L(\theta)$ using both your base and target networks.
    * Remember to use `t.inference_mode()` as appropriate.
* Do the standard optimizer things: zero the gradient, perform backprop, step your optimizer.
""")
        st.markdown("")
    with st.expander("Guidance for (4)"):
        st.markdown(r"""
(4) should just be one line. You'll need the function `load_state_dict`.
""")
        st.markdown("")

    st.markdown(r"""
Note, if you get dependency errors then you might have to upgrade to a more recent version of `gym` (I found 0.25.2 worked for me). You might also need to work your way through some error messages by e.g. doing some `pip install`s.

```python
def train_dqn(args: DQNArgs):
    (run_name, writer, rng, device, envs) = setup(args)
    "YOUR CODE: Create your Q-network, Adam optimizer, and replay buffer here."
    start_time = time.time()
    obs = envs.reset()
    for step in range(args.total_timesteps):
        "YOUR CODE: Sample actions according to the epsilon greedy policy using the linear schedule for epsilon, and then step the environment"
        rb.add(obs, actions, rewards, dones, next_obs)
        obs = next_obs
        if step > args.learning_starts and step % args.train_frequency == 0:
            "YOUR CODE: Sample from the replay buffer, compute the TD target, compute TD loss, and perform an optimizer step."
            log(writer, start_time, step, predicted_q_vals, loss, infos, epsilon)

    "If running one of the Probe environments, will test if the learned q-values are\n    sensible after training. Useful for debugging."
    probe_batches = [t.tensor([[0.0]]), t.tensor([[-1.0], [+1.0]]), t.tensor([[0.0], [1.0]]), t.tensor([[0.0]]), t.tensor([[0.0], [1.0]])]
    probe_expected = [t.tensor([[1.0]]), t.tensor([[-1.0], [+1.0]]), t.tensor([[args.gamma], [1.0]]), t.tensor([[-1.0, 1.0]]), t.tensor([[1.0, -1.0], [-1.0, 1.0]])]
    probe_tolerances = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]
    if re.match(r"Probe(\d)-v0", args.env_id):
        probe_no = int(re.match(r"Probe(\d)-v0", args.env_id).group(1)) - 1
        value = q_network(probe_batches[probe_no])
        print("Value: ", value)
        t.testing.assert_close(value, probe_expected[probe_no], atol=probe_tolerances[probe_no], rtol=0)


    if args.env_id == "Probe1-v0":
        batch = t.tensor([[0.0]]).to(device)
        value = q_network(batch)
        print("Value: ", value)
        expected = t.tensor([[1.0]]).to(device)
        t.testing.assert_close(value, expected, atol=5e-4, rtol=0)
    elif args.env_id == "Probe2-v0":
        batch = t.tensor([[-1.0], [+1.0]]).to(device)
        value = q_network(batch)
        print("Value:", value)
        expected = batch
        t.testing.assert_close(value, expected, atol=5e-4, rtol=0)
    elif args.env_id == "Probe3-v0":
        batch = t.tensor([[0.0], [1.0]]).to(device)
        value = q_network(batch)
        print("Value: ", value)
        expected = t.tensor([[args.gamma], [1.0]]).to(device)
        t.testing.assert_close(value, expected, atol=5e-4, rtol=0)
    elif args.env_id == "Probe4-v0":
        batch = t.tensor([[0.0]]).to(device)
        value = q_network(batch)
        expected = t.tensor([[-1.0, 1.0]]).to(device)
        print("Value: ", value)
        t.testing.assert_close(value, expected, atol=5e-4, rtol=0)
    elif args.env_id == "Probe5-v0":
        batch = t.tensor([[0.0], [1.0]]).to(device)
        value = q_network(batch)
        expected = t.tensor([[1.0, -1.0], [-1.0, 1.0]]).to(device)
        print("Value: ", value)
        t.testing.assert_close(value, expected, atol=1e-3, rtol=0)

    envs.close()
    writer.close()
```

If your training is successful, then you should see your pole balancing for the full `n_runs=500` timesteps. At this point, you can move on to the bonus exercises!
""")

def section_3():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#beyond-cartpole">Beyond CartPole</a></li>
    <li><a class="contents-el" href="#target-network">Target Network</a></li>
    <li><a class="contents-el" href="#shrink-the-brain">Shrink the Brain</a></li>
    <li><a class="contents-el" href="#dueling-dqn">Dueling DQN</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Bonus

## Beyond CartPole

If things go well and your agent masters CartPole, the next harder challenges are [Acrobot-v1](https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py), and [MountainCar-v0](https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py). These also have discrete action spaces, which are the only type we're dealing with today. Feel free to Google for appropriate hyperparameters for these other problems - in a real RL problem you would have to do hyperparameter search using the techniques we learned on a previous day because bad hyperparameters in RL often completely fail to learn, even if the algorithm is perfectly correct.

There are many more exciting environments to play in, but generally they're going to require more compute and more optimization than we have time for today. If you finish the main material, some ones I like are:

- [Minimalistic Gridworld Environments](https://github.com/Farama-Foundation/gym-minigrid) - a fast gridworld environment for experiments with sparse rewards and natural language instruction.
- [microRTS](https://github.com/santiontanon/microrts) - a small real-time strategy game suitable for experimentation.
- [Megastep](https://andyljones.com/megastep/) - RL environment that runs fully on the GPU (fast!)
- [Procgen](https://github.com/openai/procgen) - A family of 16 procedurally generated gym environments to measure the ability for an agent to generalize. Optimized to run quickly on the CPU.

In particular, we will likely be working with Procgen when we get to the PPO section, so if you can get your algorithm working on Procgen that will put you in a great position for the rest of the week. Even if you don't have time to get your algorithm working, playing around with Procgen environments will still be a very useful exercise.

## Target Network

Why have the target network? Modify the DQN code from the last section, but this time use the same network for both the target and the Q-value network, rather than updating the target every so often. 

Compare the performance of this against using the target network.

## Shrink the Brain

Can DQN still learn to solve CartPole with a Q-network with fewer parameters? Could we get away with three-quarters or even half as many parameters? Try comparing the resulting training curves with a shrunken version of the Q-network. What about the same number of parameters, but with more/less layers, and less/more parameters per layer?

## Dueling DQN

Implement dueling DQN according to [the paper](https://arxiv.org/pdf/1511.06581.pdf) and compare its performance.
""")

func_list = [section_home, section_1, section_2, section_3]

page_list = ["🏠 Home", "1️⃣ Q-learning", "2️⃣ DQN", "3️⃣ Bonus"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()

