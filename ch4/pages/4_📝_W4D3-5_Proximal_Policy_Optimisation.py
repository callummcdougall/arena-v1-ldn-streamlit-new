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
## 1️⃣ PPO: Mathematical Background

In this section, we'll be discussing some of the mathematical derivations behind PPO. This section isn't absolutely essential to understand in its entirity, so feel free to skim over this section. 

## 2️⃣ PPO: Implementation

These exercises make up the bulk of the section. We'll walk through an implementation of the PPO algorithm.

## 3️⃣ Bonus

A fwe exercises are suggested here, that go beyond basic implementations. You might find yourselves working on one of these for the rest of the RL week.
""")

def section_1():
    st.info("""
This section will be filled in soon. In the meantime, here's my rough notes, based on the [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html) documentation. Feel free to skip past this section and start reading the material in section 2.
""")
    st.markdown(r"""
## Why is PPO different to DQN?
- DQN (or Q-learning more broadly) is about learning the function $Q$, and then we can derive the policy by e.g. argmaxing
	- Relevant implementation questions are "what model do we have for $Q$ (e.g. tabular or NN)" and "how do we perform updates"
- PPO (or policy gradient algs more broadly) is about learning the policy function $\pi$
	- Relevant implementational questions are "how do we estimate the policy gradient" and "how do we perform gradient steps"
## Core Definitions
==**Finite-horizon discounted return**==
- $R(\tau)=\sum_{t=0}^{\infty} \gamma^t r_t$ 
- Or we can use the ==**undiscounted return**== and assume the time-horizon is finite, which we often will since it makes derivations easier
==**Policy gradient**==
- $\nabla_\theta J(\pi_\theta)$, where we have $J\left(\pi_\theta\right)=\underset{\tau \sim \pi_\theta}{\mathrm{E}}[R(\tau)]$
- We can perform ==**policy gradient ascent**== via $\theta_{k+1}=\theta_k+\left.\alpha \nabla_\theta J\left(\pi_\theta\right)\right|_{\theta_k}$
==**Log-derivative trick**==
- $\nabla_\theta P(\tau \mid \theta)=P(\tau \mid \theta) \nabla_\theta \log P(\tau \mid \theta)$ where tau is the tuple of states and actions $(s_0, a_0, ..., s_{T+1}$)
- This is useful because $P$ can be factored as the product of a bunch of probabilities, i.e. it's very small
==**On-Policy Action-Value Function**==
- $Q_\pi(s, a)=\underset{\tau \sim \pi}{\mathrm{E}}\big[R(\tau) \mid s_0=s, a_0=a\big]$
- Bellman equation is: $Q_\pi(s, a)=\underset{s^{\prime} \sim P}{\mathrm{E}}\left[r(s, a)+\gamma \underset{a^{\prime} \sim \pi}{\mathrm{E}}\left[Q_\pi\left(s^{\prime}, a^{\prime}\right)\right]\right]$
==**On-Policy Value Function**==
- $V_\pi(s)=\underset{a \sim \pi}{\mathrm{E}}\left[Q^\pi(s, a)\right]$
- Bellman equation is: $V_\pi(s)=\underset{\substack{a \sim \pi \\ s^{\prime} \sim P}}{\mathrm{E}}\left[r(s, a)+\gamma V_\pi\left(s^{\prime}\right)\right]$
==**Advantage Function**==
- $A_\pi(s, a)=Q_\pi(s, a)-V_\pi(s)$
- i.e. *how much better action $a$ is than others*

## Vanilla Policy Gradient
- Introduce the definitions above
- We can express the policy gradient in terms of the transition probabilities:$$\nabla_\theta J\left(\pi_\theta\right)=\underset{\tau \sim \pi_\theta}{\mathrm{E}}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) R(\tau)\right]$$ - in other words, we learn a policy parameterized by $\theta$, and perform gradient updates on $\theta$.
- Why does this work? 
	- Derivation: write out $J$ as an intergral over $\tau$, swap integral and derivative, do the log-derivative trick, then drop out state transition probabilities because those aren't a function of our policy
- Intuition?
	- If there's a trajectory with positive reward, then we increase the probabilities of taking all actions in that trajectory
	- The change in each $\pi_\theta (a_t | s_t)$ is a weighted sum of their impact on the reward across all transitions in which this particular state-action pair appears
- How does this help us? 
	- We can use the ==**Vanilla Policy Gradient Algorithm**== - estimating the gradient by:$$\hat{g}=\frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^T \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) R(\tau)$$ where $\mathcal{D}$ is some set of trajectories that we choose to measure.

### Expected Grad-Log-Prob Lemma
- Lemma
	- If we swap out $R(\tau)$ in the expression for policy gradient above,
	- replacing it with a function that doesn't depend on the trajectory,
	- then the expected value of that expression is zero.
- Intuition
	- It's just like the expression above, but the reward is just constant, so there's no policy direction!
- Corollary
	- We can swap out $R(\tau)$ in the expresson above with $\sum_{t^{\prime}=t}^T R\left(s_{t^{\prime}}, a_{t^{\prime}}, s_{t^{\prime}+1}\right)$, because terms before this will drop out
	- We call this the ==**reward-to-go policy gradient**==
	- We can also subtract a function $b(s_t)$, which (when used in this way) we refer to as a ==**baseline**==:$$\nabla_\theta J\left(\pi_\theta\right)=\underset{\tau \sim \pi_\theta}{\mathrm{E}}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right)\left(\sum_{t^{\prime}=t}^T R\left(s_{t^{\prime}}, a_{t^{\prime}}, s_{t^{\prime}+1}\right)-b\left(s_t\right)\right)\right]$$

### Some rewriting
- Let's first rewrite our policy gradient in a different way, by separating past and future:

$$
\begin{align*}\nabla_\theta J\left(\pi_\theta\right)&=\sum_{t=0}^T\underset{\tau_{:t} \sim \pi_\theta}{\mathrm{E}}\left[ \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) \underset{\tau_{t:}\sim \pi_\theta}{\mathrm{E}} \bigg[\;\sum_{t^{\prime}=t}^T R\left(s_{t^{\prime}}, a_{t^{\prime}}, s_{t^{\prime}+1}\right)-b\left(s_t\right) \;\bigg|\; \tau_{:t}\;\bigg]\right]\\&=\sum_{t=0}^T\underset{\tau_{:t} \sim \pi_\theta}{\mathrm{E}}\bigg[ \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) \;\Phi_t\bigg]\end{align*}
$$
where $\tau_{:t}$ is the trajectory up to $t$, and $\tau_{t:}$ is the trajectory beyond.
- Up to a constant, $\Phi_t$ is equal to the ==**On-Policy Action-Value Function**== $Q_{\pi_\theta}(s_t, a_t)$
- We can actually use this for $\Phi_t$, but a more common choice is the ==**Advantage Function**== $\Phi_t=A_{\pi_\theta}\left(s_t, a_t\right)=Q_{\pi_\theta}\left(s_t, a_t\right)-V_{\pi_\theta}(s_t)$
- Why? These both have the same expected value, but we're trying to **reduce variance**
	- It's reasonable to guess that $\Phi_t$ has the lowest variance
- So we've now basically reduced the problem to trying to find not-too-biased, low-variance estimators for the advantage function
	- Then we can construct a policy gradient estimator of the same form as $\hat{g}$ above, but swapping out $R(\tau)$ with our estimate
- We do this via ==**Generalized Average Estimation**==, or GAE


## GAE
- Let's now move back to temporal discounting (TD)
- Define the ==**temporally discounted residual**== of the value function $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$
	- Lemma: $A_{\pi, \gamma}(s_t, a_t) = \mathbb{E}_{s_{t+1}}\big[\delta_t^{V_{\pi, \gamma}}\big]$
	- Proof: follows from $Q(s_t, a_t) = \gamma V(s_{t+1}) + r_t$ (when conditioned on next state $s_{t+1}$)
	- Intuition?
		- Note that this looks a lot like the TD error term we used to update  $Q$ in Q-learning; it told us in what direction we should bump our estimate of $Q^*$
		- $\delta_t^V$ will be zero when no value is lost from going to state $s_t$ to $s_{t+1}$, which is only true in expectation when the advantage of our policy is zero (i.e. there's no more advantage to take!)
- $\delta_t^V$ expresses the notion *what will my (time-discounted) advantage be if I take one greedy step before re-evaluating at my new state?* 
	- But we want to look further ahead than this!
	- How can we express the notion *what will my (time-discounted) advantage be if I take one greedy step before re-evaluating at my new state?*
	- Answer - with ==**GAE**==
- Consider the following terms:
$$
\begin{array}{ll}
\hat{A}_t^{(1)}:=\delta_t^V & =-V\left(s_t\right)+r_t+\gamma V\left(s_{t+1}\right) \\
\hat{A}_t^{(2)}:=\delta_t^V+\gamma \delta_{t+1}^V & =-V\left(s_t\right)+r_t+\gamma r_{t+1}+\gamma^2 V\left(s_{t+2}\right) \\
\hat{A}_t^{(3)}:=\delta_t^V+\gamma \delta_{t+1}^V+\gamma^2 \delta_{t+2}^V & =-V\left(s_t\right)+r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+\gamma^3 V\left(s_{t+3}\right)
\end{array}
$$
and
$$
\hat{A}_t^{(k)}:=\sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V=-V\left(s_t\right)+r_t+\gamma r_{t+1}+\cdots+\gamma^{k-1} r_{t+k-1}+\gamma^k V\left(s_{t+k}\right)
$$
- We call these the ==**k-step advantages**==, because they answer the question *"how much better off I am if I sequentially take the $n$ best actions available (by my current estimate) and then reevaluate things from this new state?"*
- The limit is $\hat{A}_t^{(\infty)}=\sum_{l=0}^{\infty} \gamma^l \delta_{t+l}^V=-V\left(s_t\right)+\sum_{l=0}^{\infty} \gamma^l r_{t+l}$, which is just the empirical returns minus the function value baseline
- It also seems impractical to use this as our estimate, because we have to compute an infinity of steps ahead for this to work
- We take the ==**Generalised Advantage Estimator**== $\text{GAE}(\gamma, \lambda)$, which is the exponentially-weighted average of these $k$-step advantages
$$
\begin{aligned}
\hat{A}_t^{\mathrm{GAE}(\gamma, \lambda)}:&=(1-\lambda)\left(\hat{A}_t^{(1)}+\lambda \hat{A}_t^{(2)}+\lambda^2 \hat{A}_t^{(3)}+\ldots\right) \\
&=\sum_{l=0}^{\infty}(\gamma \lambda)^l \delta_{t+l}^V
\end{aligned}
$$
- $\lambda$ is the ==**discount factor**==
	- If it is higher, then you put more weight on the later advantages, i.e. "what happens when I take a large number of greedy steps before computing my value function"

# PPO
- We've basically covered everything that goes into PPO now, the rest is just a few more [details](https://arxiv.org/pdf/1707.06347.pdf)
	- I'll discuss a couple of them below
### On vs Off-Policy
- ==**Off-policy**== = learning the value of the optimal policy independently of the agent's actions
- ==**On-policy**== = learning the value of the policy being carried out by the agent
- Examples:
	- Q-learning is off-policy, since it updates its Q-values using the Q-values of the next state and the *greedy action* (i.e. the action that would be taken assuming a greedy policy is followed)
	- SARSA is on-policy, since it updates its Q-values using the Q-values of the next state and the *current policy's action*
		- The distinction disappears if the current policy is a greedy policy. However, such an agent would not be good since it never explores.
	- PPO is on-policy, since it only learns from experiences that were generated from the current policy
### Actor and Critic
- We have two different networks: an actor and a critic
- The ==**actor**== learns the policy, i.e. its parameters are the $\theta$ in all the equations we've discussed so far
	- It takes in observations $s_t$ and outputs logits, which we turn into a probability distribution $\pi(\;\cdot\;|s_t)$
	- Updates are via gradient ascent on the function described below
- The ==**critic**== learns the value
	- It takes observations $s_t$ and returns estimates for $V(s_t)$, from which we can compute our GAEs which are used to update the actor
	- Updates are via gradient descent on the MSE loss between its estimates and the actual observed returns
### Loss function
- Given discussions so far, we might expect to perform gradient ascent on a function approximation that looks like this:
$$
\begin{align*}
\theta_{k+1}&=\arg \max _\theta \underset{s, a \sim \pi_{\theta_k}}{\mathbb{E}_t}\Big[L(s_t, a_t, \theta)\Big]\\ L(s_t, a_t, \theta)&=\mathbb{E}_t\left[\log \pi_\theta\left(a_t \mid s_t\right) \hat{A}_t\right] \end{align*}
$$
- Problem - this often leads to *"destructively large policy updates"*
- Instead, we perform ==**clipping**==. We do the following:
$$
\begin{align*}\theta_{k+1}&=\arg \max _\theta \underset{s, a \sim \pi_{\theta_k}}{\mathrm{E}}\left[L\left(s, a, \theta_k, \theta\right)\right]\\L\left(s, a, \theta_k, \theta\right)&=\min \left(\frac{\pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)} \hat{A}_t, \; \operatorname{clip}\left(\frac{\pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t\right)\end{align*}
$$
- Intuition?
	- We are positively weighting things by their probability, but we're also making sure that we don't stray too far from our previous policy with each step
	- If the estimated advantage is positive, this reduces to:$$L\left(s, a, \theta_k, \theta\right)=\min \left(\frac{\pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)}, 1+\epsilon\right)\hat{A}_t$$which means we stop this from being too positive an update. If the estimated average is negative, the converse applies.
- It's similar to ==**Trust Region Policy Optimisation**==
	- That's when we perform constrained optimisation; maximising $\frac{\pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)} \hat{A}_t$ within some ==**trust region**== for the likelihood ratio
	- Yet another option is to perform unconstrained optimisation, but with a KL divergence penalty term for $\pi_\theta$ and $\pi_{\theta_k}$
### Entropy bonus
- We don't want to converge to deterministic policies quickly, this would be bad!
- We incentivise exploration by providing a bonus term for having higher entropy
	- e.g. in CartPole environment, entropy is just the entropy of a Bernoulli distribution (since there are two possible actions, left and right)
""")

def section_2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#readings">Readings</a></li>
   <li><a class="contents-el" href="#optional-reading">Optional Reading</a></li>
   <li><a class="contents-el" href="#learning-objectives">Learning Objectives</a></li>
   <li><a class="contents-el" href="#on-policy-vs-off-policy">On-Policy vs Off-Policy</a></li>
   <li><a class="contents-el" href="#actor-critic-methods">Actor-Critic Methods</a></li>
   <li><a class="contents-el" href="#notes-on-todays-workflow">Notes on today's workflow</a></li>
   <li><a class="contents-el" href="#references-not-required-reading">References (not required reading)</a></li>
   <li><a class="contents-el" href="#generalized-advantage-estimation-detail-#5">Generalized Advantage Estimation (detail #5)</a></li>
   <li><a class="contents-el" href="#minibatch-update-detail-#6">Minibatch Update (detail #6)</a></li>
   <li><a class="contents-el" href="#loss-function">Loss Function</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#gradient-ascent">Gradient Ascent</a></li>
       <li><a class="contents-el" href="#clipped-surrogate-loss">Clipped Surrogate Loss</a></li>
       <li><a class="contents-el" href="#minibatch-advantage-normalization-detail-7">Minibatch Advantage Normalization (detail #7)</a></li>
       <li><a class="contents-el" href="#value-function-loss-detail-10">Value Function Loss (detail #9)</a></li>
       <li><a class="contents-el" href="#entropy-bonus-detail-10">Entropy Bonus (detail #10)</a></li>
       <li><a class="contents-el" href="#entropy-diagnostic">Entropy Diagnostic</a></li>
   </ul></li>
   <li><a class="contents-el" href="#putting-it-all-together">Putting It All Together</a></li>
   <li><a class="contents-el" href="#debug-variables-detail-#12">Debug Variables (detail #12)</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#update-frequency">Update Frequency</a></li>
   </ul></li>
   <li><a class="contents-el" href="#reward-shaping">Reward Shaping</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
import os
import random
import time
import sys
import re
from dataclasses import dataclass
import numpy as np
import torch
import torch as t
import gym
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gym.spaces import Discrete
from einops import rearrange
from w4d3_chapter4_ppo.utils import make_env, ppo_parse_args
from w4d3_chapter4_ppo import solutions, utils

# PPO: Implementation

In this section, you'll be implementing the Proximal Policy Gradient algorithm!""")

    st.info(r"""
## Learning Objectives

* Understand some of the core implementational details of PPO.
* Understand Actor-Critic networks and why they're used. Implement them.
* Compute Generalised Advantage Estimation (GAE).
* Compute the loss function from the PPO paper (clipped surrogate loss with entropy bonus).
* Put all the parts together, into a working implementation of PPO.
* Shape the rewards to make the task easier to learn
* Design a reward function to incentivise novel behaviour
""")
    st.markdown(r"""

## Readings

* [Spinning Up in Deep RL - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
    * You don't need to follow all the derivations, but try to have a qualitative understanding of what all the symbols represent.
    * You might also prefer reading the section **1️⃣ PPO: Mathematical Background** instead.
* [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#solving-pong-in-5-minutes-with-ppo--envpool)
    * he good news is that you won't need all 37 of these today, so no need to read to the end.
    * We will be tackling the 13 "core" details, not in the same order as presented here. Some of the sections below are labelled with the number they correspond to in this page (e.g. **Minibatch Update (detail #6)**).

You might find it helpful to make a physical checklist of the 13 items and marking them as you go with how confident you are in your implementation. If things aren't working, this will help you notice if you missed one, or focus on the sections most likely to be bugged.

## Optional Reading

* [Spinning Up in Deep RL - Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html#background)
    * PPO is a fancier version of vanilla policy gradient, so if you're struggling to understand PPO it may help to look at the simpler setting first.
* [Andy Jones - Debugging RL, Without the Agonizing Pain](https://andyljones.com/posts/rl-debugging.html)
    * You've already read this previously but it will come in handy again.
    * You'll want to reuse your probe environments from yesterday, or you can import them from the solution if you didn't implement them all.

## On-Policy vs Off-Policy

Broadly, RL algorithms can be categorized as off-policy or on-policy. DQN learns from a replay buffer of old experiences that could have been generated by an old policy quite different than the current one. This means it is off-policy.

PPO will only learn from experiences that were generated by the current policy, which is why it's called on-policy. We will generate batch of experiences, train on them once, and then discard them.

## Actor-Critic Methods

In DQN, there was no neural network directly representing the policy; the policy was "sometimes act randomly, otherwise take the action with max q-value". In PPO, we're going to have two neural networks:

- The actor network learns the policy: it takes observations $o$ and outputs logits, which we can normalize into a probability distribution $\pi(\cdot | o)$  which we can sample from to determine our action $a \sim \pi(\cdot | o)$.
- The critic network learns the value: it takes observations $o$ and outputs an estimate of the optimal value function $\hat{V}^*(o)$. Again, we're going to equivocate between states and observations as is tradition. The critic acts like a movie critic, it just watches what's happening without taking any actions and forms an opinion on whether states are good or bad.

Unlike DQN, PPO can also be used for environments with continuous action spaces.

## Notes on today's workflow

Your implementation might get huge benchmark scores by the end of the day, but don't worry if it struggles to learn the simplest of tasks. RL can be frustrating because the feedback you get is extremely noisy: the agent can fail even with correct code, and succeed with buggy code. Forming a systematic process for coping with the confusion and uncertainty is the point of today, more so than producing a working PPO implementation.

Some parts of your process could include:

- Forming hypotheses about why it isn't working, and thinking about what tests you could write, or where you could set a breakpoint to confirm the hypothesis.
- Implementing some of the even more basic Gym environments and testing your agent on those.
- Getting a sense for the meaning of various logged metrics, and what this implies about the training process
- Noticing confusion and sections that don't make sense, and investigating this instead of hand-waving over it.

## References (not required reading)

- [The Policy of Truth](http://www.argmin.net/2018/02/20/reinforce/) - a contrarian take on why Policy Gradients are actually a "terrible algorithm" that is "legitimately bad" and "never a good idea".
- [Tricks from Deep RL Bootcamp at UC Berkeley](https://github.com/williamFalcon/DeepRLHacks/blob/master/README.md) - more debugging tips that may be of use.
- [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/pdf/2006.05990.pdf) - Google Brain researchers trained over 250K agents to figure out what really affects performance. The answers may surprise you.
- [Lilian Weng Blog](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#ppo)
- [A Closer Look At Deep Policy Gradients](https://arxiv.org/pdf/1811.02553.pdf)
- [Where Did My Optimum Go?: An Empirical Analysis of Gradient Descent Optimization in Policy Gradient Methods](https://arxiv.org/pdf/1810.02525.pdf)
- [Independent Policy Gradient Methods for Competitive Reinforcement Learning](https://papers.nips.cc/paper/2020/file/3b2acfe2e38102074656ed938abf4ac3-Supplemental.pdf) - requirements for multi-agent Policy Gradient to converge to Nash equilibrium.


```python
import argparse
import os
import random
import time
import sys
from distutils.util import strtobool
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch as t
import gym
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gym.spaces import Discrete
from typing import Any, List, Optional, Union, Tuple, Iterable
from einops import rearrange
from rl_utils import ppo_parse_args, make_env
import part4_dqn_solution

MAIN = __name__ == "__main__"
RUNNING_FROM_FILE = "ipykernel_launcher" in os.path.basename(sys.argv[0])
```

### Actor-Critic Agent Implementation (detail #2)

Implement the `Agent` class according to the diagram, inspecting `envs` to determine the observation shape and number of actions. We are doing separate Actor and Critic networks because detail #13 notes that is performs better than a single shared network in simple environments. Note that today `envs` will actually have multiple instances of the environment inside, unlike yesterday's DQN which had only one instance inside.

Use `layer_init` to initialize each `Linear`, overriding the standard deviation according to the diagram. What is the benefit of using a small standard deviation for the last actor layer?
""")

    st.write("""<figure style="max-width:510px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNqNkU9LAzEQxb_KMicLW-3W0sOihWC9CR7sbVPKbDJ1A012yZ-DlH53s8buqhV0IMnk8QZ-LzmCaCVBCa8WuybbrLnJYrlQJ-HBKq9EEvt6UobQFhWH1F21tdu5BjvKs-ViwmGbTaerbIOmKap-T_dkno9jy8WFf37hv_3uLyZ3tb1ZOS_vi_Pgc_DcJDwy8gc8E761Izv7Jzz7Qv9x_4ueJXwO_TmIv2YwQe9QeNUaN6aZXc-GQCwmquLaDqEgB01Wo5Lxm469zME3pIlDGVtJewwHz4GbU7SGTqKnR6lidCj3eHCUAwbfvrwZAaW3gc6mtcL4TvrTdXoHrTShmw" /></figure>""", unsafe_allow_html=True)
    # graph TD
    #     subgraph Critic
    #         Linear1["Linear(obs_shape, 64)"] --> Tanh1[Tanh] --> Linear2["Linear(64, 64)"] --> Tanh2[Tanh] --> Linear3["Linear(64, 1)<br/>std=1"] --> Out

    #     end
    #     subgraph Actor
    #         ALinear1["Linear(obs_shape, 64)"] --> ATanh1[Tanh]--> ALinear2["Linear(64, 64)"] --> ATanh2["Tanh"] --> ALinear3["Linear(64, num_actions)<br/>std=0.01"] --> AOut[Out]
    #     end
    st.markdown(r"""
```python
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        pass
```

## Generalized Advantage Estimation (detail #5)

The advantage function $A_\pi(s,a)$ indicates how much better choosing action $a$ would be in state $s$ as compared to the value obtained by letting $\pi$ choose the action (or if $\pi$ is stochastic, compared to the on expectation value by letting $\pi$ decide).
$$
A_\pi(s,a) = Q_\pi(s,a) - V_\pi(s)
$$

There are various ways to compute advantages - follow detail #5 closely for today.


Given a batch of experiences, we want to compute each `advantage[t][env]`. This is equation $(11)$ of the [PPO paper](https://arxiv.org/pdf/1707.06347.pdf).

Implement `compute_advantages`. I recommend using a reversed for loop over `t` to get it working, and not worrying about trying to completely vectorize it.

Remember that the sum in $(11)$ should be truncated at the first instance when the episode is terminated (i.e. `done=True`). This is another reason why using a for loop is easier than vectorization!

```python
@t.inference_mode()
def compute_advantages(
    next_value: t.Tensor,
    next_done: t.Tensor,
    rewards: t.Tensor,
    values: t.Tensor,
    dones: t.Tensor,
    device: t.device,
    gamma: float,
    gae_lambda: float,
) -> t.Tensor:
    '''Compute advantages using Generalized Advantage Estimation.

    next_value: shape (1, env) - represents V(s_{t+1}) which is needed for the last advantage term
    next_done: shape (env,)
    rewards: shape (t, env)
    values: shape (t, env)
    dones: shape (t, env)

    Return: shape (t, env)
    '''
    pass

if MAIN and RUNNING_FROM_FILE:
    utils.test_compute_advantages(compute_advantages)
```""")

    with st.expander("Help - I'm confused about how to calculate advantages."):
        st.markdown(r"""
You can calculate all the deltas explicitly, using:

```python
deltas = rewards + gamma * next_values * (1.0 - next_dones) - values
```

where `next_values` and `next_dones` are created by concatenating `(values, next_value)` and `(dones, next_done)` respectively and dropping the first element (i.e. the one at timestep $t=0$).

When calculating the advantages from the deltas, it might help to work backwards, i.e. start with $\hat{A}_{T-1}$ and calculate them recursively. You can go from $\hat{A}_{t}$ to $\hat{A}_{t-1}$ by multiplying by a scale factor (which might be zero depending on the value of `dones[t]`) and adding $\delta_{t-1}$.
""")
    st.markdown(r"""

## Minibatch Update (detail #6)

After generating our experiences that have `(t, env)` dimensions, we need to:

- Flatten the `(t, env)` dimensions into one batch dimension
- Split the batch into minibatches, so we can take an optimizer step for each minibatch.

If we just randomly sampled the minibatch each time, some of our experiences might not appear in any minibatch due to random chance. This would be wasteful - we're going to discard all these experiences immediately after training, so there's no second chance for the experience to be used, unlike if it was in a replay buffer.

Implement the following functions so that each experience appears exactly once.

A few notes, either to help with this exercise or provide some more context for the things we're tracking:
* `Minibatch` stores the returns, which are just advantages + values.
* `logprobs` will be the same shape as `actions`. It stores the log probabilities of those corresponding actions (i.e. `logprobs` will be the max of the `actor` network's output, and `actions` will be the argmax of that same output).""")
    
    st.markdown(r"""

```python
@dataclass
class Minibatch:
    obs: t.Tensor
    logprobs: t.Tensor
    actions: t.Tensor
    advantages: t.Tensor
    returns: t.Tensor
    values: t.Tensor

def minibatch_indexes(batch_size: int, minibatch_size: int) -> list[np.ndarray]:
    '''Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

    Each index should appear exactly once.
    '''
    assert batch_size % minibatch_size == 0
    pass

if MAIN and RUNNING_FROM_FILE:
    test_minibatch_indexes(minibatch_indexes)

def make_minibatches(
    obs: t.Tensor,
    logprobs: t.Tensor,
    actions: t.Tensor,
    advantages: t.Tensor,
    values: t.Tensor,
    obs_shape: tuple,
    action_shape: tuple,
    batch_size: int,
    minibatch_size: int,
) -> list[Minibatch]:
    '''Flatten the environment and steps dimension into one batch dimension, then shuffle and split into minibatches.'''
    pass
```

## Loss Function

The overall loss function is given by Eq 9 in the paper and is the sum of three terms - we'll implement each term individually.

### Gradient Ascent

The convention we've used in these exercises for signs is that **your function outputs should be the expressions in equation $(9)$**, in other words you will compute $L_t^{CLIP}(\theta)$, $c_1 L_t^{VF}(\theta)$ and $c_2 S[\pi_\theta](s_t)$. You can then either perform gradient descent on the **negative** of the expression in $(9)$, or perform gradient ascent on the expression by passing `maximize=True` into your Adam optimizer when you initialise it.

### Clipped Surrogate Loss

For each minibatch, calculate $L^{CLIP}$ from Eq 7 in the paper. This will allow us to improve the parameters of our actor.

Note - in the paper, don't confuse $r_{t}$ which is reward at time $t$ with $r_{t}(\theta)$, which is the probability ratio between the current policy (output of the actor) and the old policy (stored in `mb_logprobs`).

### Minibatch Advantage Normalization (detail #7)

Pay attention to the normalization instructions in detail #7 when implementing this loss function.

You can use the `probs.log_prob` method to get the log probabilities that correspond to the actions in `mb_action`.

Note - if you're wondering why we're using a `Categorical` type rather than just using `log_prob` directly, it's because we'll be using them to sample actions later on in our `train_ppo` function. Also, categoricals have a useful method for returning the entropy of a distribution (which will be useful for the entropy term in the loss function).

```python
def calc_policy_loss(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, mb_logprobs: t.Tensor, clip_coef: float
) -> t.Tensor:
    '''Return the policy loss, suitable for maximisation with gradient ascent.

    probs: a distribution containing the actor's unnormalized logits of shape (minibatch, num_actions)

    clip_coef: amount of clipping, denoted by epsilon in Eq 7.

    normalize: if true, normalize mb_advantages to have mean 0, variance 1
    '''
    pass

if MAIN and RUNNING_FROM_FILE:
    test_calc_policy_loss(calc_policy_loss)
```

### Value Function Loss (detail #9)

The value function loss lets us improve the parameters of our critic. Today we're going to implement the simple form: this is just 1/2 the mean squared difference between the **critic's prediction** and the **observed returns**. We're defining returns as `returns = advantages + values`.

The PPO paper did a more complicated thing with clipping, but we're going to deviate from the paper and NOT clip, since detail #9 gives evidence that it isn't beneficial.

Implement `calc_value_function_loss` which returns the term denoted $c_1 L_t^{VF}$ in Eq 9.

```python
def calc_value_function_loss(critic: nn.Sequential, mb_obs: t.Tensor, mb_returns: t.Tensor, v_coef: float) -> t.Tensor:
    '''Compute the value function portion of the loss function.

    v_coef: the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    pass

if MAIN and RUNNING_FROM_FILE:
    utils.test_calc_value_function_loss(calc_value_function_loss)
```

### Entropy Bonus (detail #10)

The entropy bonus term is intended to incentivize exploration by increasing the entropy of the actions distribution. For a discrete probability distribution $p$, the entropy $H$ is defined as
$$
H(p) = \sum_x p(x) \ln \frac{1}{p(x)}
$$
If $p(x) = 0$, then we define $0 \ln \frac{1}{0} := 0$ (by taking the limit as $p(x) \to 0$).
You should understand what entropy of a discrete distribution means, but you don't have to implement it yourself: `probs.entropy` computes it using the above formula but in a numerically stable way, and in
a way that handles the case where $p(x) = 0$.""")

    with st.expander("Exercise: in CartPole, what are the minimum and maximum values that entropy can take? What behaviors correspond to each of these cases?"):
        st.markdown(r"""
The minimum entropy is zero, under the policy "always move left" or "always move right".

The minimum entropy is $\ln(2) \approx 0.693$ under the uniform random policy over the 2 actions.
""")
    st.markdown(r"""
### Entropy Diagnostic

Separately from its role in the loss function, the entropy of our action distribution is a useful diagnostic to have: if the entropy of agent's actions is near the maximum, it's playing nearly randomly which means it isn't learning anything (assuming the optimal policy isn't random). If it is near the minimum especially early in training, then the agent might not be exploring enough.

Implement `calc_entropy_loss`.

Tip: make sure the sign is correct; for gradient descent, to actually increase entropy this term needs to be negative.

```python
def calc_entropy_loss(probs: Categorical, ent_coef: float):
    '''Return the entropy loss term.

    ent_coef: the coefficient for the entropy loss, which weights its contribution to the overall loss. Denoted by c_2 in the paper.
    '''
    pass

if MAIN and RUNNING_FROM_FILE:
    utils.test_calc_entropy_loss(calc_entropy_loss)
```

## Adam Optimizer and Scheduler (details #3 and #4)

Even though Adam is already an adaptive learning rate optimizer, empirically it's still beneficial to decay the learning rate.

Implement a linear decay from `initial_lr` to `end_lr` over num_updates steps. Also, make sure you read details #3 and #4 so you don't miss any of the Adam implementational details.

Note, the training terminates after `num_updates`, so you don't need to worry about what the learning rate will be after this point.

```python
class PPOScheduler:
    def __init__(self, optimizer, initial_lr: float, end_lr: float, num_updates: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_updates = num_updates
        self.n_step_calls = 0

    def step(self):
        '''Implement linear learning rate decay so that after num_updates calls to step, the learning rate is end_lr.'''
        pass

def make_optimizer(agent: Agent, num_updates: int, initial_lr: float, end_lr: float) -> tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.'''
    pass
```

## Putting It All Together

Again, we've provided the boilerplate for you. It looks worse than it is - a lot of it is just tracking metrics for debugging. Implement the sections marked with placeholders.""")

    with st.expander("Help - I get the error 'AssertionError: tensor(1, device='cuda:0') (<class 'torch.Tensor'>) invalid'."):
        st.markdown(r"""
The actions passed into `envs.step` should probably be numpy arrays, not tensors. Convert them using `.cpu().numpy()`.
""")

    with st.expander("Help - I get 'RuntimeError: Trying to backward through the graph a second time...'."):
        st.markdown(r"""
You should be doing part 1 of coding (the **rollout phase**) in inference mode. This is just designed to sample actions, not for actual network updates.""")

    st.markdown(r"""

If you need more detailed instructions for what to do in some of the code sections, then you can look at the dropdowns. You should attempt each section of code for at least 20-30 mins before you look at these dropdowns.""")

    with st.expander("Guidance for (1)"):
        st.markdown(r"""
For each value of `i in range(0, args.num_envs)`, you need to fill in the `i`th row of your tensors `obs`, `dones`, `values`, `actions`, `logprobs` and `rewards`. You get each of these items in the following ways (and in the following order):
* Values are found from inputting `next_obs` to your critic.
* You can get the logits for your policy $\pi_\theta$ by inputting `next_obs` to your actor, and from these you can get:
    * A distribution object, via using `Categorical`.
    * Actions, using the `sample` method of `Categorical` objects.
    * Logprobs, which are the log-probabilities of your distribution object corresponding to the sampled actions.
    * Rewards, by passing your actions into the `envs.step` function.
""")
    with st.expander("Guidance for (2)"):
        st.markdown(r"""

""")

    st.markdown(r"""

```python
@dataclass
class PPOArgs:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "PPOCart"
    wandb_entity: str = None
    capture_video: bool = False
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 0.00025
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 512
    minibatch_size: int = 128

def train_ppo(args):
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
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    action_shape = envs.single_action_space.shape
    assert action_shape is not None
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"
    agent = Agent(envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    (optimizer, scheduler) = make_optimizer(agent, num_updates, args.learning_rate, 0.0)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    global_step = 0
    old_approx_kl = 0.0
    approx_kl = 0.0
    value_loss = t.tensor(0.0)
    policy_loss = t.tensor(0.0)
    entropy_loss = t.tensor(0.0)
    clipfracs = []
    info = []
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    for _ in range(num_updates):
        for i in range(0, args.num_steps):
            "YOUR CODE: Rollout phase (see detail #1)"
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break
        next_value = rearrange(agent.critic(next_obs), "env 1 -> 1 env")
        advantages = compute_advantages(
            next_value, next_done, rewards, values, dones, device, args.gamma, args.gae_lambda
        )
        clipfracs.clear()
        for _ in range(args.update_epochs):
            minibatches = make_minibatches(
                obs,
                logprobs,
                actions,
                advantages,
                values,
                envs.single_observation_space.shape,
                action_shape,
                args.batch_size,
                args.minibatch_size,
            )
            for mb in minibatches:
                "YOUR CODE: compute loss on the minibatch and step the optimizer (not the scheduler). Do detail #11 (global gradient clipping) here using nn.utils.clip_grad_norm_."
        scheduler.step()
        (y_pred, y_true) = (mb.values.cpu().numpy(), mb.returns.cpu().numpy())
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        with torch.no_grad():
            newlogprob: t.Tensor = probs.log_prob(mb.actions)
            logratio = newlogprob - mb.logprobs
            ratio = logratio.exp()
            old_approx_kl = (-logratio).mean().item()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if global_step % 10 == 0:
            print("steps per second (SPS):", int(global_step / (time.time() - start_time)))
    envs.close()
    writer.close()

if MAIN:
    if "ipykernel_launcher" in os.path.basename(sys.argv[0]):
        filename = globals().get("__file__", "<filename of this script>")
        print(f"Try running this file from the command line instead: python {os.path.basename(filename)} --help")
        args = PPOArgs()
    else:
        args = ppo_parse_args()
    train_ppo(args)
```

## Debug Variables (detail #12)

Go through and check each of the debug variables that are logged. Make sure your implementation computes or calculates the values and that you have an understanding of what they mean and what they should look like.

### Update Frequency

Note that the debug values are currently only logged once per update, meaning some are computed from the last minibatch of the last epoch in the update. This isn't necessarily the best thing to do, but if you log too often it can slow down training. You can experiment with logging more often, or tracking the average over the update or even an exponentially moving average.

## Reward Shaping

Recall the [docs](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) and [source code](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) for the `CartPole` environment.

The current rewards for `CartPole` encourage the agent to keep the episode running for as long as possible, which it then needs to associate with balancing the pole.

Here, we inherit from `CartPoleEnv` so that we can modify the dynamics of the environment.

Try to modify the reward to make the task as easy to learn as possible. Compare this against your performance on the original environment, and see if the agent learns faster with your shaped reward. If you can bound the reward on each timestep between 0 and 1, this will make comparing the results to `CartPole-v1` easier.

```python
from gym.envs.classic_control.cartpole import CartPoleEnv
import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled
import math

class EasyCart(CartPoleEnv):
    def step(self, action):
        (obs, rew, done, info) = super().step(action)
        "TODO: YOUR CODE HERE"

gym.envs.registration.register(id="EasyCart-v0", entry_point=EasyCart, max_episode_steps=500)
```

Now, change the environment such that the reward incentivises the agent to "dance".

It's up to you to define what qualifies as "dancing". Work out a sensible definition, and the reward function to incentive it. You may change the termination conditions of the environment if you think it helps teaching the cart to dance.
""")

def section_3():
    st.markdown(r"""
## Bonus

### Continuous Action Spaces

The `MountainCar-v0` environment has discrete actions, but there's also a version `MountainCarContinuous-v0` with continuous action spaces. Unlike DQN, PPO can handle continuous actions with minor modifications. Try to adapt your agent; you'll need to handle `gym.spaces.Box` instead of `gym.spaces.Discrete` and make note of the "9 details for continuous action domains" section of the reading.

### Vectorized Advantage Calculation

Try optimizing away the for-loop in your advantage calculation. It's tricky, so an easier version of this is: find a vectorized calculation and try to explain what it does.

### RLHF

Details coming soon!
""")

func_list = [section_home, section_1, section_2, section_3]

page_list = ["🏠 Home", "1️⃣ PPO: Mathematical Background", "2️⃣ PPO: Implementation", "3️⃣ Bonus"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()

