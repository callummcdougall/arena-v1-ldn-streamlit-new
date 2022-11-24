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
    st.markdown("""
## 1️⃣ Multi-Armed Bandit

In Part 1 we'll study the multi-armed bandit problem, which is simple yet captures introduces many of the difficulties in RL. Many practical problems can be posed in terms of generalizations of the multi-armed bandit. For example, the Hyperband algorithm for hyperparameter optimization is based on the multi-armed bandit with an infinite number of arms.
""")

def section1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#readings">Readings</a></li>
   <li><a class="contents-el" href="#learning-objectives">Learning Objectives</a></li>
   <li><a class="contents-el" href="#intro-to-openai-gym">Intro to OpenAI Gym</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#the-info-dictionary">The <code>info</code> dictionary</a></li>
       <li><a class="contents-el" href="#the-render()-method">The <code>render()</code> method</a></li>
       <li><a class="contents-el" href="#observation-and-action-types">Observation and Action Types</a></li>
       <li><a class="contents-el" href="#registering-an-environment">Registering an Environment</a></li>
       <li><a class="contents-el" href="#timelimit-wrapper">TimeLimit Wrapper</a></li>
       <li><a class="contents-el" href="#a-note-on-(pseudo)-rngs">A Note on (pseudo) RNGs</a></li>
       <li><a class="contents-el" href="#reward-averaging">Reward Averaging</a></li>
   </ul></li>
   <li><a class="contents-el" href="#cheater-agent">Cheater Agent</a></li>
   <li><a class="contents-el" href="#the-authentic-rl-experience">The Authentic RL Experience</a></li>
   <li><a class="contents-el" href="#ucbactionselection">UCBActionSelection</a></li>
   <li><a class="contents-el" href="#bonus">Bonus</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""

## Readings

* [Sutton and Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf), Chapter 2
    * Section 2.1: A k-armed Bandit Problem through to Section 2.7 Upper-Confidence-Bound Action Section
    * We won't cover Gradient Bandits. Don't worry about these.
    * Don't worry too much about all the math for the moment if you can't follow it.

## Learning Objectives

* Understand the anatomy of a `gym.Env`, so that you feel comfortable using them and writing your own
* Practice RL in the tabular (lookup table) setting before adding the complications of neural networks
* Understand the difficulty of optimal exploration
* Understand that performance is extremely variable in RL, and how to measure performance

```python
import os
from typing import Optional, Union
import gym
import gym.envs.registration
import gym.spaces
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from tqdm.auto import tqdm

MAIN = __name__ == "__main__"
max_episode_steps = 1000
N_RUNS = 200
```

## Intro to OpenAI Gym

Today and tomorrow, we'll be using OpenAI Gym, which provides a uniform interface to many different RL environments including Atari games. Gym was released in 2016 and details of the API have changed significantly over the years. We are using version 0.23.1, so ensure that any documentation you use refers to the same version.

Below, we've provided a simple environment for the multi-armed bandit described in the Sutton and Barto reading. Here, an action is an integer indicating the choice of arm to pull, and an observation is the constant integer 0, since there's nothing to observe here. Even though the agent does in some sense observe the reward, the reward is always a separate variable from the observation.

We're using NumPy for this section. PyTorch provides GPU support and autograd, neither of which is of any use to environment code or the code we're writing today.

Read the `MultiArmedBandit` code carefully and make sure you understand how the Gym environment API works.

### The `info` dictionary

The environment's `step` method returns four values: `obs`, `reward`, `done`, and the `info` dictionary.

`info` can contain anything extra that doesn't fit into the uniform interface - it's up to the environment what to put into it. A good use of this is for debugging information that the agent isn't "supposed" to see. In this case, we'll return the index of the actual best arm. This would allow us to measure how often the agent chooses the best arm, but it would also allow us to build a "cheating" agent that peeks at this information to make its decision.

Cheating agents are helpful because we know that they should obtain the maximum possible rewards; if they aren't, then there's a bug.

### The `render()` method

Render is only used for debugging or entertainment, and what it does is up to the environment. It might render a little popup window showing the Atari game, or it might give you a RGB image tensor, or just some ASCII text describing what's happening. In this case, we'll just make a little plot showing the distribution of rewards for each arm of the bandit.

### Observation and Action Types

A `gym.Env` is a generic type: both the type of the observations and the type of the actions depends on the specifics of the environment.

We're only dealing with the simplest case: a discrete set of actions which are the same in every state. In general, the actions could be continuous, or depend on the state.

```python
ObsType = int
ActType = int

class MultiArmedBandit(gym.Env):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete
    num_arms: int
    stationary: bool
    arm_reward_means: np.ndarray
    arm_star: int

    def __init__(self, num_arms=10, stationary=True):
        super().__init__()
        self.num_arms = num_arms
        self.stationary = stationary
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(num_arms)
        self.reset()

    def step(self, arm: ActType) -> tuple[ObsType, float, bool, dict]:
        '''
        Note: some documentation references a new style which has (termination, truncation) bools in place of the done bool.
        '''
        assert self.action_space.contains(arm)
        if not self.stationary:
            q_drift = self.np_random.normal(loc=0.0, scale=0.01, size=self.num_arms)
            self.arm_reward_means += q_drift
            self.best_arm = int(np.argmax(self.arm_reward_means))
        reward = self.np_random.normal(loc=self.arm_reward_means[arm], scale=1.0)
        obs = 0
        done = False
        info = dict(best_arm=self.best_arm)
        return (obs, reward, done, info)

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        super().reset(seed=seed)
        if self.stationary:
            self.arm_reward_means = self.np_random.normal(loc=0.0, scale=1.0, size=self.num_arms)
        else:
            self.arm_reward_means = np.zeros(shape=[self.num_arms])
        self.best_arm = int(np.argmax(self.arm_reward_means))
        if return_info:
            return (0, dict())
        else:
            return 0

    def render(self, mode="human"):
        assert mode == "human", f"Mode {mode} not supported!"
        bandit_samples = []
        for arm in range(self.action_space.n):
            bandit_samples += [np.random.normal(loc=self.arm_reward_means[arm], scale=1.0, size=1000)]
        plt.violinplot(bandit_samples, showmeans=True)
        plt.xlabel("Bandit Arm")
        plt.ylabel("Reward Distribution")
        plt.show()
```

### Registering an Environment

User code normally won't use the constructor of an `Env` directly for two reasons:

- Usually, we want to wrap our `Env` in one or more wrapper classes.
- If we want to test our agent on a variety of environments, it's annoying to have to import all the `Env` classes directly.

The `register` function stores information about our `Env` in a registry so that a later call to `gym.make` can look it up using the `id` string that is passed in.

By convention, the `id` strings have a suffix with a version number. There can be multiple versions of the "same" environment with different parameters, and benchmarks should always report the version number for a fair comparison. For instance, `id="ArmedBanditTestbed-v0"` below.

### TimeLimit Wrapper

As defined, our environment never terminates; the `done` flag is always False so the agent would keep playing forever. By setting `max_episode_steps` here, we cause our env to be wrapped in a `TimeLimit` wrapper class which terminates the episode after that number of steps.

Note that the time limit is also an essential part of the problem definition: if it were larger or shorter, there would be more or less time to explore, which means that different algorithms (or at least different hyperparameters) would then have improved performance.

```python
gym.envs.registration.register(
    id="ArmedBanditTestbed-v0",
    entry_point=MultiArmedBandit,
    max_episode_steps=max_episode_steps,
    nondeterministic=True,
    reward_threshold=1.0,
    kwargs={"num_arms": 10, "stationary": True},
)
if MAIN:
    env = gym.make("ArmedBanditTestbed-v0")
    print("Our env inside its wrappers looks like: ", env)
```

### A Note on (pseudo) RNGs

The PRNG that `gym.Env` provides as `self.np_random` is from the [PCG family](https://www.pcg-random.org/index.html). In RL code, you often need massive quantities of pseudorandomly generated numbers, so it's important to have a generator that is both very fast and has good quality output.

When you call `np.random.randint` or similar, you're using the old-school Mersenne Twister algorithm which is both slower and has inferior quality output to PCG. Since Numpy 1.17, you can use `np.random.default_rng()` to get a PCG generator and then use its `integers` method to get random integers.

- Implement the `RandomAgent` subclass which should pick an arm at random.
    - This is useful as a baseline to ensure the environment has no bugs. If your later agents are doing worse than random, you have a bug!
- Verify that `RandomAgent` pulls the optimal arm with frequency roughly `1/num_arms`.
- Verify that the average reward is very roughly zero. This is the case since the mean reward for each arm is centered on zero.

```python
class Agent:
    '''Base class for agents in a multi-armed bandit environment (you do not need to add any implementation here)'''

    rng: np.random.Generator

    def __init__(self, num_arms: int, seed: int):
        self.num_arms = num_arms
        self.reset(seed)

    def get_action(self) -> ActType:
        raise NotImplementedError()

    def observe(self, action: ActType, reward: float, info: dict) -> None:
        pass

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

def run_episode(env: gym.Env, agent: Agent, seed: int):

    (rewards, was_best) = ([], [])

    env.reset(seed=seed)
    agent.reset(seed=seed)

    done = False
    while not done:
        arm = agent.get_action()
        (obs, reward, done, info) = env.step(arm)
        agent.observe(arm, reward, info)
        rewards.append(reward)
        was_best.append(1 if arm == info["best_arm"] else 0)
    
    rewards = np.array(rewards, dtype=float)
    was_best = np.array(was_best, dtype=int)
    return (rewards, was_best)

def test_agent(env: gym.Env, agent: Agent, n_runs=200):
    all_rewards = []
    all_was_bests = []
    for seed in tqdm(range(n_runs)):
        (rewards, corrects) = run_episode(env, agent, seed)
        all_rewards.append(rewards)
        all_was_bests.append(corrects)
    return (np.array(all_rewards), np.array(all_was_bests))

class RandomAgent(Agent):

    def get_action(self) -> ActType:
        pass

    def __repr__(self):
        # Useful when plotting multiple agents with `plot_rewards`
        return "RandomAgent"

if MAIN:
    "TODO: YOUR CODE HERE"
```

### Reward Averaging

Now implement these features as detailed in Sutton and Barto section 2.3, titled "Incremental Implementation":

- Track the moving average of observed reward for each arm.
- Epsilon greedy exploration: with probability `epsilon`, take a random action; otherwise take the action with the highest average observed reward.
- Optimistic initial values: initialize each arm's reward estimate with the `optimism` value.

Remember to call `super().__init__(num_arms, seed)` in your `__init__()` method.""")

    with st.expander("Hint - average reward formula"):
        st.markdown(r"""
$$Q_k = Q_{k-1} + \frac{1}{k}[R_k - Q_{k-1}]$$

Where $k$ is the number of times the action has been taken, $R_k$ is the reward from the kth time the action was taken, and $Q_{k-1}$ is the average reward from the previous times this action was taken (this notation departs slightly from the S&B notation, but may be more helpful for our implementation).

**Important - $k$ is not the total number of timesteps, it's the total number of times you've taken this particular action.**
""")

    st.markdown(r"""
We've given you a function for plotting multiple agents' reward trajectories on the same graph, with an optional moving average parameter to make the graph smoother.

```python
def moving_avg(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_rewards(
    all_rewards: List[np.ndarray], 
    names: List[str],
    moving_avg_window: Optional[int] = 15,
):
    fig = go.Figure(layout=dict(template="simple_white", title_text="Mean reward over all runs"))
    for rewards, name in zip(all_rewards, names):
        rewards_avg = rewards.mean(axis=0)
        if moving_avg_window is not None:
            rewards_avg = moving_avg(rewards_avg, moving_avg_window)
        fig.add_trace(go.Scatter(y=rewards_avg, mode="lines", name=name))
    fig.show()

class RewardAveraging(Agent):
    def __init__(self, num_arms: int, seed: int, epsilon: float, optimism: float):
        pass

    def get_action(self):
        pass

    def observe(self, action, reward, info):
        pass

    def reset(self, seed: int):
        pass

    def __repr__(self):
        pass

if MAIN:
    num_arms = 10
    stationary = True
    names, all_rewards = [], []
    env = gym.make("ArmedBanditTestbed-v0", num_arms=num_arms, stationary=stationary)

    for optimism in [0, 5]:
        agent = RewardAveraging(num_arms, 0, epsilon=0.1, optimism=optimism)
        (rewards, num_correct) = test_agent(env, agent, n_runs=N_RUNS, base_seed=1)
        names.append(str(agent))
        all_rewards.append(rewards)
        print(agent)
        print(f" -> Frequency of correct arm: {num_correct.mean():.4f}")
        print(f" -> Average reward: {rewards.mean():.4f}")

    plot_rewards(all_rewards, names, moving_avg_window=15)
```

## Cheater Agent

Implement the cheating agent and see how much reward it can get. If your agent gets more than this in the long run, you have a bug!""")

    with st.expander("Help - I'm not sure how to implement the cheater."):
        st.markdown("""
Recall that your `env.step` method returns a tuple of `(obs, reward, done, info)`, and the `info` dict contains `best_arm`.

In `observe`, you should read the `best_arm` object from the `info` dictionary, and store it. Then, in `get_action`, you should always return this best arm.

Note, you'll have to make a special case for when best arm hasn't been observed yet (in this case, your action doesn't matter).
""")

    st.markdown("""
```python
class CheatyMcCheater(Agent):
    def __init__(self, num_arms: int, seed: int):
        pass

    def get_action(self):
        pass

    def observe(self, action, reward, info):
        pass

    def repr(self):
        pass

if MAIN:
    cheater = CheatyMcCheater(num_arms, 0)
    reward_averaging = RewardAveraging(num_arms, 0, epsilon=0.1, optimism=0)
    random = RandomAgent(num_arms, 0)

    names = []
    all_rewards = []

    for agent in [cheater, reward_averaging, random]:
        (rewards, num_correct) = test_agent(env, agent, n_runs=N_RUNS, base_seed=1)
        names.append(str(agent))
        all_rewards.append(rewards)

    plot_rewards(all_rewards, names, moving_avg_window=15)
```

## The Authentic RL Experience

It would be nice if we could say something like "optimistic reward averaging is a good/bad feature that improves/decreases performance in bandit problems." Unfortunately, we can't justifiably claim either at this point.

Usually, RL code fails silently, which makes it difficult to be confident that you don't have any bugs. I had a bug in my first attempt that made both versions appear to perform equally, and there were only 13 lines of code, and I had written similar code before.

The number of hyperparameters also grows rapidly, and hyperparameters have interactions with each other. Even in this simple problem, we already have two different ways to encourage exploration (`epsilon` and `optimism`), and it's not clear whether it's better to use one or the other, or both in some combination. It's actually worse than that, because `epsilon` should probably be annealed down at some rate.

Even in this single comparison, we trained 200 agents for each version. Is that a big number or a small number to estimate the effect size? Probably we should like, compute some statistics? And test with a different number of arms - maybe we need more exploration for more arms? The time needed for a rigorous evaluation is going to increase quickly.

We're using 0.23.1 of `gym`, which is not the latest version. 0.24.0 and 0.24.1 according to the [release notes](https://github.com/openai/gym/releases) have "large bugs" and the maintainers "highly discourage using these releases". How confident are we in the quality of the library code we're relying on?

As we continue onward to more complicated algorithms, keep an eye out for small discrepancies or minor confusions. Look for opportunities to check and cross-check everything, and be humble with claims.

## UCBActionSelection

Once you feel good about your `RewardAveraging` implementation, you should implement `UCBActionSelection`.

This should store the same moving average rewards for each action as `RewardAveraging` did, but instead of taking actions using the epsilon-greedy strategy it should use Equation 2.8 in Section 2.6 to select actions using the upper confidence bound. It's also useful to add a small epsilon term to the $N_t(a)$ term, to handle instances where some of the actions haven't been taken yet.

You should expect to see a small improvement over `RewardAveraging` using this strategy.

```python
class UCBActionSelection(Agent):
    def __init__(self, num_arms: int, seed: int, c: float):
        pass

    def get_action(self):
        pass

    def observe(self, action, reward, info):
        pass

    def reset(self, seed: int):
        pass

if MAIN:
    env = gym.make("ArmedBanditTestbed-v0", num_arms=num_arms, stationary=stationary)
    ucb = UCBActionSelection(num_arms, 0, c=2.0)
    (all_rewards, all_corrects) = test_agent(env, ucb, n_runs=N_RUNS)
    print(f"Frequency of correct arm: {all_corrects.mean()}")
    print(f"Average reward: {all_rewards.mean()}")
    plot_rewards(all_rewards)
```

## Bonus
* Implement the gradient bandit algorithm.
* Implement an environment and an agent for the contextual bandit problem.
* Complete the exercises at the end of Chapter 2 of Sutton and Barto.
""")

func_list = [section_home, section1]

page_list = ["🏠 Home", "1️⃣ Multi-Armed Bandit"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()

