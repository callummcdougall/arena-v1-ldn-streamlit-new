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
## 1️⃣ Tabular RL and Policy Improvement

Today, we'll start to get into the weeds of some of the mathematical formulations of RL. Some important concepts to cover are Markov processes and the Bellman equation. We'll then put this into practice on some basic gridworld environments.
""")

def section_1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#reinforcement-learning,-known-environments">Reinforcement Learning, Known Environments</a></li>
   <li><a class="contents-el" href="#readings">Readings</a></li>
   <li><a class="contents-el" href="#what-is-reinforcement-learning?">What is Reinforcement Learning?</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#theory-exercises">Theory Exercises</a></li>
   </ul></li>
   <li><a class="contents-el" href="#tabular-rl,-known-environments">Tabular RL, Known Environments</a></li>
   <li><a class="contents-el" href="#policy-evaluation">Policy Evaluation</a></li>
   <li><a class="contents-el" href="#exact-policy-evaluation">Exact Policy Evaluation</a></li>
   <li><a class="contents-el" href="#policy-improvement">Policy Improvement</a></li>
   <li><a class="contents-el" href="#bonus">Bonus</a></li>
</ul>
""", unsafe_allow_html=True)
    
    st.markdown(r"""
## Reinforcement Learning, Known Environments

We are presented with a environment, and the
transition function that indicates how the environment will transition from state to state based
on the action chosen. We will see how we can turn the problem of finding the best agent into
an optimization problem, which we can then solve iteratively.

Here, we assume environments small enough where
visiting all pairs of states and actions is tractable. These types of models don't learn
any relationships between states that can be treated similarly, but keep track of an estimate
of how valuable each state is in a large lookup table.

## Readings

Don't worry about absorbing every detail, we will repeat a lot of the details here.
Don't worry too much about the maths, we will also cover that here.

- [Sutton and Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- Chapter 3, Sections 3.1, 3.2, 3.3, 3.5, 3.6
- Chapter 4, Sections 4.1, 4.2, 4.3, 4.4


```python
from gettext import find
from typing import Optional, Union
import numpy as np
import gym
import gym.spaces
import gym.envs.registration
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import utils

MAIN = __name__ == "__main__"
Arr = np.ndarray

```

## What is Reinforcement Learning?

In reinforcement learning, the agent interacts with an environment in a loop: The agent
in state $s$ issues an **action** $a$ to the environment, and the environment replies
with **state, reward** pairs $(s',r)$. We will assume the environment is **Markovian**, in that
the next state $s'$ and reward $r$ depend solely on the current state $s$
and the action $a$ chosen by the agent
(as opposed to environments which may depend on actions taken far in the past.)""")

    st_image("agent-diagram.png", 500)
    st.markdown("")
    st.markdown("")
    st.markdown(r"""

Between the agent and environment, an **interaction history** is generated:
$$
s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2, r_3, \ldots
$$""")

    st.markdown("")
    st.info(r"""
#### Pedantic note #1

Some authors disagree on precisely where the next timestep occours.
We (following Sutton and Barto) define the timestep transition to be when the environment
acts, so given state $s_t$ the agent returns action $a_t$ (in the same time step), but
given $(s_t, a_t)$ the environment generates $(s_{t+1}, r_{t+1})$.

Other authors notate the history as
$$
s_0, a_0, r_0, s_1, a_1, r_1, s_2, a_2, r_2, \ldots
$$
often when the reward $r_t$ is a deterministic function of the current state $s_t$ and
action chosen $a_t$, and the environment's only job is to select the next state
$s_{t+1}$ given $(s_t, a_t)$.

There's nothing much we can do about this, so be aware of the difference when reading
other sources.
""")

    st.markdown("")
    st.markdown(r"""
The agent chooses actions using a policy $\pi$, which we can think of as either
a deterministic function $a = \pi(s)$ from states to actions, or more generally a stochastic
function from which actions are sampled, $a \sim \pi(\cdot | s)$.

Implicitly, we have assumed that the agent need only be Markovian as well. Is this
a reasonable assumption?""")

    with st.expander("Answer"):
        st.markdown("""
The environment dynamics depend only on the current state, so the agent needs only the current
state to decide how to act optimally (for example, in a game of tic-tac-toe, knowledge of the current state
of the board is sufficient to determine the optimal move, how the board got to that state is irrelevant.)
""")
    st.markdown(r"""
The environment samples (state, reward) pairs from a probability distribution conditioned on
the current state $s$ and the action $a$ the policy chose in that state,
$(s', r) \sim p(\cdot | s, a)$. In the case where the environment is also deterministic,
we write $(s', r) = p(s, a)$.

The goal of the agent is to choose a policy that maximizes the **expected discounted return**, the sum of rewards it would expect to obtain by following it's currently
chosen **policy** $\pi$. We call the expected discounted return from a state $s$
following policy $\pi$ the **state value function** $V_{\pi}(s)$, or simply **value function**, as 
$$
V_{\pi}(s) = \mathbb{E}_{\pi} \left[ \sum_{i=t}^\infty \gamma^{i-t} r_{i+1} \Bigg| s_t = s \right]
$$
where the expectation is with respect to sampling actions from $\pi$, and
(implicitly) sampling states and rewards from $p$.""")

    st.info(r"""
#### Pedantic note #2

Technically $V_\pi$ also depends on the choice of environment $p$ and discount factor $\gamma$, but usually during training the only thing we are optimizing for
is the choice of policy $\pi$ (the environment $p$ is fixed and the discount $\gamma$
is either provided or fixed as a hyperparameter before training), so we ignore these dependencies instead of writing $V_{\pi, p, \gamma}$.
""")
    with st.expander("Why discount?"):
        st.markdown(r"""
We would like a way to signal to the agent that reward now is better than
reward later.
If we didn't discount, the sum $\sum_{i=t}^\infty r_i$ may diverge.
This leads to strange behavior, as we can't meaningfully compare the returns
for sequences of rewards that diverge. Trying to sum the sequence $1,1,1,\ldots$ or
$2,2,2,2\ldots$ or even $2, -1, 2 , -1, \ldots$ all diverge to positive infinity,
and so are all "equally good"
even though it's clear that $2,2,2,2\ldots$ would be the most desirable.

An agent with a policy that leads to infinite expected return might become lazy,
as waiting around doing nothing for a thousand years, and then actually optimally
leads to the same return (infinite) as playing optimal from the first time step.

Worse still, we can have reward sequences for which the sum may never approach
any finite value, nor diverge to $\pm \infty$ (like summing $-1, 1, -1 ,1 , \ldots$).
We could otherwise patch this by requiring that the agent has a finite number of interactions
with the environment (this is often true, and is called an **episodic** environment that we will
see later) or restrict ourselves to environments for which the expected return for the optimal
policy is finite, but these can often be undesirable constraints to place.
""")
    with st.expander("Why geometric discount?"):
        st.markdown(r"""
In general, we could consider a more general discount function $\Gamma : \mathbb{N} \to [0,1)$ and define the discounted return as $\sum_{i=t}^\infty \Gamma(i) r_i$. The geometric discount $\Gamma(i) = \gamma^i$ is commonly used, but other discounts include the hyperbolic discount $\Gamma(i) = \frac{1}{1 + iD}$, where $D>0$ is a hyperparameter. (Humans are [often said](https://chris-said.io/2018/02/04/hyperbolic-discounting/) to act as if they use a hyperbolic discount.) Other than  being mathematically convenient to work with (as the sum of geometric discounts has an elegant closed form expression $\sum_{i=0}^\infty \gamma^i  = \frac{1}{1-\gamma}$), geometric is preferred as it is *time consistant*, that is, the discount from one time step to the next remains constant. 
Rewards at timestep $t+1$ are always worth a factor of $\gamma$ less than rewards on timestep $t$
$$
\frac{\Gamma(t+1)}{\Gamma(t)} = \frac{\gamma^{t+1}}{\gamma^t} = \gamma
$$
whereas for the hyperbolic reward, the amount discounted from one step to the next
is a function of the timestep itself, and decays to 1 (no discount)
$$
\frac{\Gamma(t+1)}{\Gamma(t)} =  \frac{1+tD}{1+(t+1)D} \to 1
$$
so very little discounting is done once rewards are far away enough in the future.
This would make our value function a function of not only state but also time:
If the agent finds itself in the same state it was in before, it stands to reason
the value should be the same (assuming the policy hasn't changed.)""")

    st.markdown(r"""
Note we can write the value function in the following recursive matter:
$$
V_\pi(s) = \sum_a \pi(a | s) \sum_{s', r} p(s',r \mid s, a) \left( r + \gamma V_\pi(s') \right)
$$

(Optional) Try to prove this for yourself!
""")
    with st.expander("Hint"):
        st.markdown(r"""
The expectation can be fully expanded as
$$
\mathbb{E}_{\pi} \left[ \sum_{i=t}^\infty \gamma^{i-t} r_i \Bigg| s_t = s\right]
$$
$$
= \lim_{m \to \infty} \sum_{a_t} \pi(a_t | s_t)
\sum_{s_{t+1}, r_{t+1}} p(s_{t+1}, r_{t+1} | s_t, a_t)
\ldots
\sum_{a_m} \pi(a_m | s_m)
\sum_{s_{m+1}, r_{m+1}} p(s_{m+1}, r_{m+1} | s_m, a_m) \sum_{i=t}^m \gamma^{i-t} r_i
$$
You can remove the first term from the reward sum
$$
\sum_{i=t}^\infty \gamma^{i-t} r_i
= r_t + \gamma \sum_{i=t+1}^\infty \gamma^{i-(t+1)} r_i
$$
and unroll the first two sums of the expectation. Rearrange the rest of the
sum so it looks like the value $V_\pi(s')$ of the next state $s'$.
""")

    with st.expander("Solution"):
        st.markdown(r"""
We roll out the sum:
$$
\mathbb{E}_{\pi} \left[ \sum_{i=t}^\infty \gamma^{i-t} r_i \Bigg| s_t = s\right]
= \mathbb{E}_{\pi} \left[r_t + \gamma \sum_{i=t+1}^\infty \gamma^{i-(t+1)} r_i \Bigg| s_t = s\right]
= \mathbb{E}_{\pi} [r_t | s_t = s] + \gamma \mathbb{E}_{\pi} \left[ \sum_{i=t+1}^\infty \gamma^{i-(t+1)} r_i \Bigg| s_t = s\right]
$$

Focusing on the first part, we can compute the expected immediate reward $\mathbb{E}_\pi[r]$
by summing over all actions $a$ that $\pi$ might take (weighted by $\pi(a|s)$, how likely $\pi$ would
choose $a$ in state $s$), and then sum over all future state/reward pairs $s',r$ (weighted
by how likely $p$ would choose them)
$$
\mathbb{E}_{\pi} [r_t]
= \sum_a \pi(a | s) \sum_{s',r} p(s',r | s,a) r
$$

Similarly, we can expand out the expectation over the sum of future rewards by considering all
possible ways of transitioning to the next state $s_{t+1}$, weighted by their probability
$$
\gamma \mathbb{E}_{\pi} \left[ \sum_{i=t+1}^\infty \gamma^{i-(t+1)} r_i \Bigg| s_t = s\right]
=
\sum_a \pi(a | s) \sum_{s',r} p(s',r | s,a) \gamma \mathbb{E}_{\pi} \left[ \sum_{i=t+1}^\infty \gamma^{i-(t+1)} r_i \Bigg| s_{t+1} = s'\right]
$$
$$
= \sum_a \pi(a | s) \sum_{s',r} p(s',r | s,a) \gamma V_\pi(s')
$$
Combining the results, we get
$$
V_\pi(s) = \sum_a \pi(a | s) \sum_{s', r} p(s',r \mid s, a) \left( r + \gamma V_\pi(s') \right)
$$
as required.
""")

    st.markdown(r"""
This recursive formulation of the value function is called the **Bellman equation**, and can be thought of as "The value of the current state is the reward the agent gets now, plus the value for the next state."

We can also define the **action-value function**, or **Q-value** of state $s$ and action $a$ following policy $\pi$:
$$
Q_\pi(s,a) = \mathbb{E}\left[ \sum_{i=t}^\infty \gamma^{i-t}r_{t+1} \Bigg| s_t=s, a_t=a   \right]
$$
which can be written recursively much like the value function can:
$$
Q_\pi(s,a) = \sum_{s',r} p(s',r \mid s,a) \left( r + \gamma \sum_{a'} \pi(a' \mid s') Q_\pi(s', a') \right)
$$

### Theory Exercises

(These are mostly to check your understanding of the readings on RL.
Feel free to skip ahead if you're already happy with the definition.)

Consider the following environment: There are two actions $A = \{a_L, a_R\}$, three states $S = \{s_0, s_L, s_R\}$ and three rewards $R = \{0,1,2\}$. The environment is deterministic, and can be represented by the following transition diagram:""")

    st_image("markov-diagram.png", 400)
    st.markdown("")
    st.markdown(r"""

The edges represent the state transitions given an action, as well as the reward received. For example, in state $s_0$, taking action $a_L$ means the new state is $s_L$, and the reward received is $+1$. (The transitions for $s_L$ and $s_R$ are independent of action taken.)

##### 1. How many choices of deterministic policy are there for this environment?""")

    with st.expander("Answer"):
        st.markdown("""There are 3 states, and 2 actions for each state, so there are $2^3 = 8$ choices of deterministic policy $\pi$.""")

    st.markdown(r"""
We say that two policies $\pi_1$ and $\pi_2$ are **equivalent** if $\forall s \in S. V_{\pi_1}(s) = V_{\pi_2}(s)$.

This gives us effectively two choices of deterministic policies, $\pi_L(s_0) = s_L$ and $\pi_R(s_0) = s_R$. (It is irrelevant what those policies do in the other states.)

A policy $\pi_1$ is **better** than $\pi_2$ (denoted $\pi_1 \geq \pi_2$) if
$\forall s \in S. V_{\pi_1}(s) \geq V_{\pi_2}(s)$.

##### 2. Compute the value $V_{\pi}(s_0)$ for $\pi = \pi_L$ and $\pi = \pi_R$.
Which policy is better? Does the answer depend on the choice of
discount factor $\gamma$? If so, how?""")

    with st.expander("Answer"):
        st.markdown(r"""
Following the first policy, this gives
$$
V_{\pi_L}(s_0) = 1 + \gamma V_{\pi_L}(s_L) = 1 + \gamma(0 + V_{\pi_L}(s_0)) = 1 +\gamma^2 V_{\pi_L}(s_0)
$$
Rearranging, this gives
$$
V_{\pi_L}(s_0) = \frac{1}{1-\gamma^2}
$$
Following the second policy, this gives
$$
V_{\pi_R}(s_0) = 0 + \gamma V_{\pi_R}(s_R) = \gamma(2 + \gamma V_{\pi_R}(s_0)) = 2 \gamma + \gamma^2 V_{\pi_R}(s_0)
$$
Rearranging, this gives
$$
V_{\pi_R}(s_0) = \frac{2\gamma}{1-\gamma^2}
$$
Therefore,
$$
\pi^* = \begin{cases}
\pi_L & \gamma < 1/2 \\
\pi_L \text{ or } \pi_R & \gamma = 1/2 \\
\pi_R & \gamma > 1/2
\end{cases}
$$
which makes sense, an agent that discounts heavily ($\gamma < 1/2$) is shortsighted,
and will choose the reward 1 now, over the reward 2 later.""")

    st.markdown(r"""
An **optimal** policy (denoted $\pi^*$) is a policy that is better than all other policies.
There may be more than one optimal policy, so we refer to any of them as $\pi^*$, with the understanding that since all optimal policies have the same value $V_{\pi^*}$ for all states, it doesn't actually matter which is chosen.

It is possible to prove that, for any environment, an optimal policy exists. I can go through this proof later in the day, using the whiteboard.

## Tabular RL, Known Environments

For the moment, we focus on environments for which the agent has access to $p$, the function describing the underlying dynamics of the environment, which will allow us to solve the Bellman equation explicitly. While unrealistic, it means we can explicitly solve the Bellman equation. Later on we will remove this assumption and treat the environment as a black box from which the agent can sample from.

We will simplify things a bit further, and assume that the environment samples states from a probability distribution $T(\cdot | s, a)$ conditioned on the current state $s$ and the action $a$ the policy $\pi$ chose in that state. Normally, the reward is also considered to be a stochastic function of both state and action $r \sim  R(\cdot \mid s,a)$, but we will assume the reward is a deterministic
function $R(s,a,s')$ of the current state, action and next state, and offload the randomness in the rewards to the next state $s'$ sampled from $T$.

This (together with assuming $\pi$ is deterministic) gives a simpler recursive form of the value function
$$
V_\pi(s) = \sum_{s'} T(s' \mid s, a) \Big( R(s,a,s') + \gamma V_\pi (s') \Big)
\text{ where } a = \pi(s)
$$

Below, we've provided a simple environment for a gridworld taken from [Russell and Norvig](http://aima.cs.berkeley.edu/). The agent can choose one of four actions: `up` (0), `right` (1), `down` (2) and `left` (3), encoded as numbers. The observation is just the state that the agent is in (encoded as a number from 0 to 11).""")

    st_image("gridworld.png", 300)
    st.markdown("")
    st.markdown(r"""
The result of choosing an action from an empty cell will move the agent in that direction, unless they would bump into a wall or walk outside the grid, in which case the next state is unchanged. Both the terminal states "trap" the agent, and any movement from one of the terminal states leaves the agent in the same state, and no reward is received. The agent receives a small penalty for each move $r = -0.04$ to encourage them to move to a terminal state quickly, unless the agent is moving into a terminal state, in which case they recieve reward either $+1$ or $-1$ as appropriate.

Lastly, the environment is slippery, and the agent only has a 70\% chance of moving in the direction chosen, with a 10% chance each of attempting to move in the other three cardinal directions.

Provided is a class that allows us to define environments with known dynamics. The only parts you should be concerned
with are
* `.num_states`, which gives the number of states,
* `.num_actions`, which gives the number of actions
* `.T`, a 3-tensor of shape  `(num_states,num_actions,num_states)` representing the probability $T(s_{next} \mid s, a)$ = `T[s,a,s_next]`
* `.R`, the reward function, encoded as a vector of shape `(num_states,num_actions,num_states)` that returns the reward $R(s,a,s')$ associated with entering state $s'$ from state $s$ by taking action $a$.

This environment also provides two additional parameters which we will not use now, but need for part 3 where the environment is treated as a black box, and agents learn from interaction.
* `.start`, the state with which interaction with the environment begins. By default, assumed to be state zero.
* `.terminal`, a list of all the terminal states (with which interaction with the environment ends). By default, terminal states are empty.

```python
class Environment:
    def __init__(self, num_states: int, num_actions: int, start=0, terminal=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.start = start
        if terminal is None:
            self.terminal = np.array([], dtype=int)
        else:
            self.terminal = terminal
        (self.T, self.R) = self.build()

    def build(self):
        '''
        Constructs the T and R tensors from the dynamics of the environment.
        Outputs:
            T : (num_states, num_actions, num_states) State transition probabilities
            R : (num_states, num_actions, num_states) Reward function
        '''
        num_states = self.num_states
        num_actions = self.num_actions
        T = np.zeros((num_states, num_actions, num_states))
        R = np.zeros((num_states, num_actions, num_states))
        for s in range(num_states):
            for a in range(num_actions):
                (states, rewards, probs) = self.dynamics(s, a)
                (all_s, all_r, all_p) = self.out_pad(states, rewards, probs)
                T[s, a, all_s] = all_p
                R[s, a, all_s] = all_r
        return (T, R)

    def dynamics(self, state: int, action: int) -> tuple[Arr, Arr, Arr]:
        '''
        Computes the distribution over possible outcomes for a given state
        and action.
        Inputs:
            state : int (index of state)
            action : int (index of action)
        Outputs:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair
        '''
        raise NotImplementedError

    def render(pi: Arr):
        '''
        Takes a policy pi, and draws an image of the behavior of that policy,
        if applicable.
        Inputs:
            pi : (num_actions,) a policy
        Outputs:
            None
        '''
        raise NotImplementedError

    def out_pad(self, states: Arr, rewards: Arr, probs: Arr):
        '''
        Inputs:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair
        Outputs:
            states  : (num_states,) all the next states
            rewards : (num_states,) rewards for each next state transition
            probs   : (num_states,) likelihood of each state-reward pair (including
                           probability zero outcomes.)
        '''
        out_s = np.arange(self.num_states)
        out_r = np.zeros(self.num_states)
        out_p = np.zeros(self.num_states)
        for i in range(len(states)):
            idx = states[i]
            out_r[idx] += rewards[i]
            out_p[idx] += probs[i]
        return (out_s, out_r, out_p)
```

For example, here is the toy environment from above implemented in this format.

```python
class Toy(Environment):
    def dynamics(self, state: int, action: int):
        (S0, SL, SR) = (0, 1, 2)
        LEFT = 0
        num_states = 3
        num_actions = 2
        assert 0 <= state < self.num_states and 0 <= action < self.num_actions
        if state == S0:
            if action == LEFT:
                (next_state, reward) = (SL, 1)
            else:
                (next_state, reward) = (SR, 0)
        elif state == SL:
            (next_state, reward) = (0, 0)
        elif state == SR:
            (next_state, reward) = (0, 2)
        return (np.array([next_state]), np.array([reward]), np.array([1]))

    def __init__(self):
        super().__init__(3, 2)

```

Given a definition for the `dynamics` function, the `Environment` class
automatically generates `T` and `R` for us.

```python
if MAIN:
    toy = Toy()
    print(toy.T)
    print(toy.R)

```
We also provide an implementation of the gridworld environment above. We include a definition of `render`, which given a policy, prints out a grid showing the direction the policy will try to move in from each cell.

```python
class Norvig(Environment):
    def dynamics(self, state: int, action: int) -> tuple[Arr, Arr, Arr]:
        def state_index(state):
            assert 0 <= state[0] < self.width and 0 <= state[1] < self.height, print(state)
            pos = state[0] + state[1] * self.width
            assert 0 <= pos < self.num_states, print(state, pos)
            return pos

        pos = self.states[state]
        move = self.actions[action]
        if state in self.terminal or state in self.walls:
            return (np.array([state]), np.array([0]), np.array([1]))
        out_probs = np.zeros(self.num_actions) + 0.1
        out_probs[action] = 0.7
        out_states = np.zeros(self.num_actions, dtype=int) + self.num_actions
        out_rewards = np.zeros(self.num_actions) + self.penalty
        new_states = [pos + x for x in self.actions]
        for (i, s_new) in enumerate(new_states):
            if not (0 <= s_new[0] < self.width and 0 <= s_new[1] < self.height):
                out_states[i] = state
                continue
            new_state = state_index(s_new)
            if new_state in self.walls:
                out_states[i] = state
            else:
                out_states[i] = new_state
            for idx in range(len(self.terminal)):
                if new_state == self.terminal[idx]:
                    out_rewards[i] = self.goal_rewards[idx]
        return (out_states, out_rewards, out_probs)

    def render(self, pi: Arr):
        assert len(pi) == self.num_states
        emoji = ["⬆️", "➡️", "⬇️", "⬅️"]
        grid = [emoji[act] for act in pi]
        grid[3] = "🟩"
        grid[7] = "🟥"
        grid[5] = "⬛"
        print(str(grid[0:4]) + "\n" + str(grid[4:8]) + "\n" + str(grid[8:]))

    def __init__(self, penalty=-0.04):
        self.height = 3
        self.width = 4
        self.penalty = penalty
        num_states = self.height * self.width
        num_actions = 4
        self.states = np.array([[x, y] for y in range(self.height) for x in range(self.width)])
        self.actions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        self.dim = (self.height, self.width)
        terminal = np.array([3, 7], dtype=int)
        self.walls = np.array([5], dtype=int)
        self.goal_rewards = np.array([1.0, -1])
        super().__init__(num_states, num_actions, start=8, terminal=terminal)
```

## Policy Evaluation

At the moment, we would like to determine the value function $V_\pi$ of some policy $\pi$. We will **assume policies are deterministic**, and encode policies as a lookup table from states to actions (so $\pi$ will be a vector of shape `(num_states,)`, where each element is an integer `a` in the range `0 <= a < num_actions`, representing one of the possible actions to choose for that state.)

Firstly, we will use the Bellman equation as an update rule: Given a current estimate $\hat{V}_\pi$ of the value function $V_{\pi}$, we can obtain a better estimate by using the Bellman equation, sweeping over all states.
$$
\forall s. \hat{V}_\pi(s) \leftarrow \sum_{s'} T(s, \pi(s), s') \left( R(s,a,s') + \gamma \hat{V}_\pi(s) \right)
$$
We continue looping this update rule until the result stabilizes: $\max_s |\hat{V}^{new}(s) - \hat{V}^{old}(s)| < \epsilon$ for some small $\epsilon > 0$. Use $\hat{V}_\pi(s) = 0$ as your initial guess.

```python
def policy_eval_numerical(env: Environment, pi: Arr, gamma=0.99, eps=1e-08) -> Arr:
    '''
    Numerically evaluates the value of a given policy by iterating the Bellman equation
    Inputs:
        env: Environment
        pi : shape (num_states,) - The policy to evaluate
        gamma: float - Discount factor
        eps  : float - Tolerance
    Outputs:
        value : float (num_states,) - The value function for policy pi
    '''
    pass

if MAIN:
    utils.test_policy_eval(policy_eval_numerical, exact=False)
```

## Exact Policy Evaluation

Essentially what we are doing in the previous step is numerically solving the Bellman equation. Since the Bellman equation essentially gives us a set of simultaneous equations, we can solve it explicitly rather than iterating the Bellman update rule.

Given a policy $\pi$, consider $\mathbf{v} \in \mathbb{R}^{|S|}$ to be a vector representing the value function $V_\pi$ for each state.
$$
\mathbf{v} = [V_\pi(s_1), \ldots, V_\pi(s_{N})]
$$
and recall the Bellman equation:
$$
\mathbf{v}_i = \sum_{s'} T(s' \mid s_i,\pi(s)) \left( R(s,\pi(s),s') + \gamma V_\pi(s') \right)
$$
We can define two matrices $P^\pi$ and $R^\pi$, both of shape `(num_states, num_states)`
as
$$
P^\pi_{i,j} = T(i, \pi(i), j) \quad R^\pi_{i,j} = R(i, \pi(i), j)
$$
$P^\pi$ can be thought of as a probability transition matrix from current state to next state,
and $R^\pi$ is the reward function given current
state and next state, assuming actions are chosen by $\pi$.
$$
\mathbf{v}_i = \sum_{j} P^\pi_{i,j} \left( R^\pi_{i,j} + \gamma \mathbf{v}_j \right)
$$
$$
\mathbf{v}_i = \sum_{j} P^\pi_{i,j}  R^\pi_{i,j} +  \gamma \sum_{j} P^\pi_{i,j} \mathbf{v}_j
$$
We can define $\mathbf{r}^\pi_i = \sum_{j} P^\pi_{i,j}  (R^\pi)_{i,j} = (P^\pi  (R^\pi)^T)_{ii}$
$$
\mathbf{v}_i = \mathbf{r}_i^\pi + \gamma (P^\pi \mathbf{v})_i
$$
A little matrix algebra, and we obtain:
$$
\mathbf{v} = (I - \gamma P^\pi)^{-1} \mathbf{r}^\pi
$$
wich gives us a closed form solution for the value function $\mathbf{v}$.

Is the inverse $(I - \gamma P^\pi)^{-1}$ guaranteed to exist?
Discuss with your partner, and ask Callum if you're not sure.""")

    with st.expander("Help - I don't know how to compute the intermediate terms!"):
        st.markdown(r"""
Try writing $P^\pi$ and $\mathbf{r}^\pi$ using a for-loop over both $i$ and $j$.
Once you've got it working, go back and write it efficiently
using numpy operations.""")

    st.markdown(r"""
```python
def policy_eval_exact(env: Environment, pi: Arr, gamma=0.99) -> Arr:
    pass

if MAIN:
    utils.test_policy_eval(policy_eval_exact, exact=True)
```

## Policy Improvement

So, we now have a way to compute the value of a policy. What we are really interested in is finding better policies. One way we can do this is to compare how well $\pi$ performs on a state versus the value obtained by choosing another action instead, that is, the Q-value. If there is an action $a'$ for which $Q_{\pi}(s,a') > V_\pi(s) \equiv Q_\pi(s, \pi(s))$, then we would prefer that $\pi$ take action $a'$ in state $s$ rather than whatever action $\pi(s)$ currently is. In general, we can select the action that maximizes the Bellman equation:
$$
\argmax_a \sum_{s'} T(s' \mid s, a) (R(s,a,s') + \gamma V_{\pi}(s)) \geq \sum_{s',r}
T(s' \mid s, \pi(a)) \left( R(s,\pi(a),s') + \gamma V_{\pi}(s) \right) = V_{\pi}(s)
$$

This gives us an update rule for the policy. Given the value function $V_{\pi}(s)$
for policy $\pi$, we define an improved policy $\pi^\text{better}$ as follows:
$$
\pi^\text{better}(s) = \argmax_a \sum_{s'} T(s' \mid s, a) (R(s,a,s') + \gamma V_{\pi}(s))
$$

```python
def policy_improvement(env: Environment, V: Arr, gamma=0.99) -> Arr:
    '''
    Inputs:
        env: Environment
        V  : (num_states,) value of each state following some policy pi
    Outputs:
        pi_better : vector (num_states,) of actions representing a new policy obtained via policy iteration
    '''
    pass

if MAIN:
    utils.test_policy_improvement(policy_improvement)
```

Putting these together, we now have an algorithm to find the optimal policy for an environment.
$$
\pi_0 \overset{E}{\to} V_{\pi_0}
\overset{I}{\to} \pi_1 \overset{E}{\to} V_{\pi_1}
\overset{I}{\to} \pi_2 \overset{E}{\to} V_{\pi_2} \overset{I}{\to}  \ldots
$$
We alternate policy evaluation ($\overset{E}{\to}$) and policy improvement ($\overset{I}{\to}$), each step a monotonic improvement, until the policy no longer changes ($\pi_n = \pi_{n+1}$), at which point we have an optimal policy, as our current policy will satisfy the optimal Bellman equations:
$$
V_{\pi^*} = \argmax_a \sum_{s',r} T(s' \mid s,a) (R(s,a,s') + \gamma V_{\pi^*}(s'))
$$

Don't forget that policies should be of `dtype=int`, rather than floats!

You should play around with the `penalty` value for the gridworld environment
and see how this affects the optimal policy found.

Note that since the optimal policy is not unique, the automated tests will merely check that your optimal policy has the same value function as the optimal policy found by the solution.

```python
def find_optimal_policy(env: Environment, gamma=0.99):
    '''
    Inputs:
        env: environment
    Outputs:
        pi : (num_states,) int, of actions represeting an optimal policy
    '''
    pass

if MAIN:
    utils.test_find_optimal_policy(find_optimal_policy)
    penalty = -0.04
    norvig = Norvig(penalty)
    pi_opt = find_optimal_policy(norvig, gamma=0.99)
    norvig.render(pi_opt)
```

## Bonus

- Implement and test your policy evaluation method on other environments.
- Complete some exercises in Chapters 3 and 4 of Sutton and Barto.
- Modify the tabular RL solvers to allow stochastic policies or to allow $\gamma=1$ on episodic environments (may need to change how environments are defined.)
""")

func_list = [section_home, section_1]

page_list = ["🏠 Home", "1️⃣ Tabular RL"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()

