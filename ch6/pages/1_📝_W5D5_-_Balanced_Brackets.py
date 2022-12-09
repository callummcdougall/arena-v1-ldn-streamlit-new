import os
if not os.path.exists("./images"):
    os.chdir("./ch6")
import re, json
import plotly.io as pio

from st_dependencies import *
styling()

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

NAMES = ["attribution_fig"]
def get_fig_dict():
    return {name: read_from_html(name) for name in NAMES}
if "fig_dict" not in st.session_state:
    st.session_state["fig_dict"] = {}
if NAMES[0] not in st.session_state["fig_dict"]:
    st.session_state["fig_dict"] |= get_fig_dict()
fig_dict = st.session_state["fig_dict"]

def section_home():
    st.markdown(r"""
## 1️⃣ Bracket classifier

We'll start by looking at our bracket classifier and dataset, and see how it works. We'll also write our own hand-coded solution to the balanced bracket problem (understanding what a closed-form solution looks like will be helpful as we discover how our transformer is solving the problem).

## 2️⃣ Going backwards

If we want to investigate which heads cause the model to classify a bracket string as balanced or unbalanced, we need to work our way backwards from the input. Eventually, we can find things like the **resudiaul stream unbalanced directions**, which are the directions of vectors in the residual stream which contribute most to the model's decision to classify a string as unbalanced.

This will require learning how to use **hooks**, to capture inputs and outputs of intermediate layers in the model.

## 3️⃣ Total elevation circuit

In this section (which is the meat of the exercises), we'll hone in on a particular circuit and try to figure out what kinds of composition it is using.

## 4️⃣ Finding adversarial examples

Now that we have a basic understanding of how our transformer is classifying states
""")

def section_1():
    st.markdown(r"""

# W2D5 - Interpretability on an algorithmic model

One of the many behaviors that a large language model learns is the ability to tell if a sequence of nested parentheses is balanced. For example, `(())()`, `()()`, and `(()())` are balanced sequences, while `)()`, `())()`, and `((()((())))` are not.

In training, text containing balanced parentheses is much more common than text with imbalanced parentheses - particularly, source code scraped from GitHub is mostly valid syntactically. A pretraining objective like "predict the next token" thus incentivizes the model to learn that a close parenthesis is more likely when the sequence is unbalanced, and very unlikely if the sequence is currently balanced.

Some questions we'd like to be able to answer are:

- How robust is this behavior? On what inputs does it fail and why?
- How does this behavior generalize out of distribution? For example, can it handle nesting depths or sequence lengths not seen in training?

If we treat the model as a black box function and only consider the input/output pairs that it produces, then we're very limited in what we can guarantee about the behavior, even if we use a lot of compute to check many inputs. This motivates interpretibility: by digging into the internals, can we obtain insight into these questions? If the model is not robust, can we directly find adversarial examples that cause it to confidently predict the wrong thing? Let's find out!

## Life On The Frontier

Unlike many of the days in the curriculum which cover classic papers and well-trodden topics, today you're at the research frontier, covering current research at Redwood. This is pretty cool, but also means you should expect that things will be more confusing and complicated than other days. TAs might not know answers because in fact nobody knows the answer yet, or might be hard to explain because nobody knows how to explain it properly yet.

Feel free to ask Nix Goldowsky-Dill questions, and feel free to go "off-road" and follow your curiosity - you might discover uncharted lands :)

## Today's Toy Model

Today we'll study a small transformer that is trained to only classify whether a sequence of parentheses is balanced or not. It's small so we can run experiments quickly, but big enough to perform well on the task. The weights and architecture are provided for you.

### Model architecture:

The model resembles BERT and GPT, but isn't identical to either. Read through `w2d5_transformer.py` for the definitive model reference. To summarize:

* Positional embeddings are sinusoidal (non-learned).
* It has `hidden_size` (aka `d_model`, aka `embed_width`) of 56.
* It has bidirectional attention, like BERT.
* It has 3 attention layers and 3 MLPs.
* Each attention layer has two heads, and each head has head_size (aka query/key/value_width) of hidden_size/num_heads == 28.
* The MLP hidden layer has 56 neurons (i.e. its linear layers are square matrices).
* The input of each attention layer and each MLP is first layernormed, like in GPT.
* There's a LayerNorm on the residual stream after all the attention layers and MLPs have been added into it (this is also like GPT).
* We affix a linear layer to the position-0 residual stream (i.e. that of the [BEGIN] token) which goes down to a length-two vector, which we then softmax to get our classification probabilities.

To refer to attention heads, we'll use a shorthand "layer.head" where both layer and head are zero-indexed. So `2.1` is the second attention head (index 1) in the third layer (index 2). With this shorthand, the network graph looks like:""")

    st_excalidraw("brackets-transformer", 1100)
    # st.write("""<figure style="max-width:400px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNp9k19vgjAUxb8K6bMulEey-DI1MWHORN9gWaq9ajPaklISjfjdV-j4YyHw1PvrObcnl_aBTpICCtFFkezqHZaJ8MyXF0cLPiTPpAChc7tRfQf5C8KbzxdeyYSGC6iyRit-BBrXy_ejWtQlZeLyXWsjcgflb0RMKO2Tz2g3gHhIxmTBkDiydbTtl-XtR0jFS2_NBElrx9bUcV3aDl4FWjWFajyqjJgAohqay7Pm5FZ6e7uo-Vehs0J3U9rJnGkmnUEZasfUbJN0YlZdt4bY7a0ft-Gtw3_z3Zn2zGN6PKHvYHOeKdwWBvkvv8xpgFs3dq24nxYP0o7o8YS-g81542nxy81xGgStO3CtQT9tMEg7oscT-g42542nDboLbN0gKJohDooTRs2LfVQ4QfoKHBIUmiWFMylSnaBEPI20yCjRsKJMS4XCM0lzmCFSaLm_ixMKtSqgES0ZMe-d_6uefzWyPj4" /></figure>""", unsafe_allow_html=True)

# ```mermaid
# graph TD
#     subgraph Components
#         Token --> |integer|TokenEmbed[Token<br>Embedding] --> Layer0In[add] --> Layer0MLPIn[add] --> Layer1In[add] --> Layer1MLPIn[add] --> Layer2In[add] --> Layer2MLPIn[add] --> FLNIn[add] --> |x_norm| FinalLayerNorm[Final Layer Norm] --> |x_decoder|Linear --> |x_softmax| Softmax --> Output
#         Position --> |integer|PosEmbed[Positional<br>Embedding] --> Layer0In
#         Layer0In --> LN0[LayerNorm] --> 0.0 --> Layer0MLPIn
#         LN0[LayerNorm] --> 0.1 --> Layer0MLPIn
#         Layer0MLPIn --> LN0MLP[LayerNorm] --> MLP0 --> Layer1In
#         Layer1In --> LN1[LayerNorm] --> 1.0 --> Layer1MLPIn
#         LN1[LayerNorm] --> 1.1 --> Layer1MLPIn
#         Layer1MLPIn --> LN1MLP[LayerNorm] --> MLP1 --> Layer2In
#         Layer2In --> LN2[LayerNorm] --> 2.0 --> Layer2MLPIn
#         LN2[LayerNorm] --> 2.1 --> Layer2MLPIn
#         Layer2MLPIn --> LN2MLP[LayerNorm] --> MLP2 --> FLNIn
#     end
# ```
    st.markdown(r"""
```python
import functools
import json
import os
from typing import Any, List, Tuple, Union
import matplotlib.pyplot as plt
import torch
import torch as t
import torch.nn.functional as F
from fancy_einsum import einsum
from sklearn.linear_model import LinearRegression
from torch import nn
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from einops import rearrange, repeat
import pandas as pd
import numpy as np

import w5d5_testss
from w5d5_transformer import ParenTransformer, SimpleTokenizer

MAIN = __name__ == "__main__"
DEVICE = t.device("cpu")
```

## Tokenizer

There are only five tokens in our vocabulary: `[start]`, `[pad]`, `[end]`, `(`, and `)` in that order.

## Dataset

Each training example consists of `[start]`, up to 40 parens, `[end]`, and then as many `[pad]` as necessary.

In the dataset we're using, half the sequences are balanced, and half are unbalanced. Having an equal distribution is on purpose to make it easier for the model.

As is good practice, examine the dataset and plot the distribution of sequence lengths.


```python
if MAIN:
    model = ParenTransformer(ntoken=5, nclasses=2, d_model=56, nhead=2, d_hid=56, nlayers=3).to(DEVICE)
    state_dict = t.load("w2d5_state_dict.pt")
    model.to(DEVICE)
    model.load_simple_transformer_state_dict(state_dict)
    model.eval()
    tokenizer = SimpleTokenizer("()")
    with open("w2d5_data.json") as f:
        data_tuples: List[Tuple[str, bool]] = json.load(f)
        print(f"loaded {len(data_tuples)} examples")
    assert isinstance(data_tuples, list)


class DataSet:
    '''A dataset containing sequences, is_balanced labels, and tokenized sequences'''

    def __init__(self, data_tuples: list):
        '''
        data_tuples is List[Tuple[str, bool]] signifying sequence and label
        '''
        self.strs = [x[0] for x in data_tuples]
        self.isbal = t.tensor([x[1] for x in data_tuples]).to(device=DEVICE, dtype=t.bool)
        self.toks = tokenizer.tokenize(self.strs).to(DEVICE)
        self.open_proportion = t.tensor([s.count("(") / len(s) for s in self.strs])
        self.starts_open = t.tensor([s[0] == "(" for s in self.strs]).bool()

    def __len__(self) -> int:
        return len(self.strs)

    def __getitem__(self, idx) -> Union["DataSet", tuple[str, t.Tensor, t.Tensor]]:
        if type(idx) == slice:
            return self.__class__(list(zip(self.strs[idx], self.isbal[idx])))
        return (self.strs[idx], self.isbal[idx], self.toks[idx])

    @property
    def seq_length(self) -> int:
        return self.toks.size(-1)

    @classmethod
    def with_length(cls, data_tuples: list[tuple[str, bool]], selected_len: int) -> "DataSet":
        return cls([(s, b) for (s, b) in data_tuples if len(s) == selected_len])

    @classmethod
    def with_start_char(cls, data_tuples: list[tuple[str, bool]], start_char: str) -> "DataSet":
        return cls([(s, b) for (s, b) in data_tuples if s[0] == start_char])


if MAIN:
    N_SAMPLES = 5000 if not IS_CI else 100
    data_tuples = data_tuples[:N_SAMPLES]
    data = DataSet(data_tuples)
    "TODO: YOUR CODE HERE"

```

## Hand-Written Solution

A nice property of using such a simple problem is we can write a correct solution by hand. Take a minute to implement this using a for loop and if statements.

```python
def is_balanced_forloop(parens: str) -> bool:
    '''Return True if the parens are balanced.

    Parens is just the ( and ) characters, no begin or end tokens.
    '''
    pass

if MAIN:
    examples = ["()", "))()()()()())()(())(()))(()(()(()(", "((()()()()))", "(()()()(()(())()", "()(()(((())())()))"]
    labels = [True, False, True, False, True]
    for (parens, expected) in zip(examples, labels):
        actual = is_balanced_forloop(parens)
        assert expected == actual, f"{parens}: expected {expected} got {actual}"
    print("is_balanced_forloop ok!")
```

## Hand-Written Solution - Vectorized

A transformer has an inductive bias towards vectorized operations, because at each sequence position the same weights "execute", just on different data. So if we want to "think like a transformer", we want to get away from procedural for/if statements and think about what sorts of solutions can be represented in a small number of transformer weights.

Being able to represent a solutions in matrix weights is necessary, but not sufficient to show that a transformer could learn that solution through running SGD on some input data. It could be the case that some simple solution exists, but a different solution is an attractor when you start from random initialization and use current optimizer algorithms.
""")
    with st.expander(r"""Hint - Vectorized"""):
        st.markdown(r"""You can do this by indexing into a lookup table of size `vocab_size` and a `t.cumsum`. The lookup table represents the change in "nesting depth""")

    with st.expander("Solution - Vectorized"):
        st.markdown(r"""
One solution is to map begin, pad, and end tokens to zero, map open paren to 1 and close paren to -1. Then calculate the cumulative sum of the sequence. Your sequence is unbalanced if and only if:

- The last element of the cumulative sum is nonzero
- Any element of the cumulative sum is negative""")
        st.markdown("")

    st.markdown(r"""
```python
def plot_all_neurons(model, data, layer):
    neurons_in_d = out_by_neuron_in_20_dir(model, data, layer)[data.starts_open, 1, :].detach().flatten()
    neuron_numbers = repeat(t.arange(model.d_model), "n -> (s n)", s=data.starts_open.sum())
    data_open_proportion = repeat(data.open_proportion[data.starts_open], "s -> (s n)", n=model.d_model)
    df = pd.DataFrame({
        "Output in 2.0 direction": neurons_in_d,
        "Neuron number": neuron_numbers,
        "Open-proportion": data_open_proportion
    })
    fig = px.scatter(df, x="Open-proportion", y="Output in 2.0 direction", animation_frame="Neuron number", title=f"Neuron contributions from layer {layer}", template="simple_white", height=500, width=800).update_traces(marker_size=3, opacity=0.8)
    fig.update_layout(xaxis_range=[0, 1], yaxis_range=[-5, 5])
    fig.show()

if MAIN:
    plot_all_neurons(model, data, 0)
    plot_all_neurons(model, data, 1)
```

## The Model's Solution

It turns out that the model solves the problem like this:

At each position `i`, the model looks at the slice starting at the current position and going to the end: `seq[i:]`. It then computes (count of closed parens minus count of open parens) for that slice to generate the output at that position.

We'll refer to this output as the "elevation" at `i`, or equivalently the elevation for each suffix `seq[i:]`. Elevation means the same thing as "nesting depth", but we'll plot elevation increasing on the positive y-axis so this term feels a little more natural.

Rephrasing in terms of elevations, the sequence is imbalanced if one or both of the following is true:
- `elevation[0]` is non-zero
- `any(elevation < 0)`

For English readers, it's natural to process the sequence from left to right and think about prefix slices `seq[:i]` instead of suffixes, but the model is bidirectional and has no idea what English is. This model happened to learn the equally valid solution of going right-to-left.

We'll spend today inspecting different parts of the network to try to get a first-pass understanding of how various layers implement this algorithm. However, we'll also see that neural networks are complicated, even those trained for simple tasks, and we'll only be able to explore a minority of the pieces of the puzzle.

## Running the Model

Our model perfectly learned all 5000 training examples, and has confidence exceeding 99.99% in its correct predictions. Is this a robust model?

Note that we would normally check performance on a held-out test set here, but let's continue for now.

```python
if MAIN:
    toks = tokenizer.tokenize(examples).to(DEVICE)
    out = model(toks)
    prob_balanced = out.exp()[:, 1]
    print("Model confidence:\n" + "\n".join([f"{ex:34} : {prob:.4%}" for ex, prob in zip(examples, prob_balanced)]))

def run_model_on_data(model: ParenTransformer, data: DataSet, batch_size: int = 200) -> t.Tensor:
    '''Return probability that each example is balanced'''
    ln_probs = []
    for i in range(0, len(data.strs), batch_size):
        toks = data.toks[i : i + batch_size]
        with t.no_grad():
            out = model(toks)
        ln_probs.append(out)
    out = t.cat(ln_probs).exp()
    assert out.shape == (len(data), 2)
    return out

if MAIN:
    test_set = data
    n_correct = t.sum((run_model_on_data(model, test_set).argmax(-1) == test_set.isbal).int())
    print(f"\nModel got {n_correct} out of {len(data)} training examples correct!")
```""")

def section_2():
    st.markdown(r"""
## Moving backward

Suppose we run the model on some sequence and it outputs the classification probabilities `[0.99, 0.01]`, i.e. highly confident classification as "unbalanced".

We'd like to know _why_ the model had this output, and we'll do so by moving backwards through the network, and figuring out the correspondence between facts about earlier activations and facts about the final output. We want to build a chain of connections through different places in the computational graph of the model, repeatedly reducing our questions about later values to questions about earlier values.

Let's start with an easy one. Notice that the final classification probabilities only depend on the difference between the class logits, as softmax is invariant to constant additions. So rather than asking, "What led to this probability on balanced?", we can equivalently ask, "What led to this difference in logits?". Let's move another step backward. Since the logits each a linear function of the output of the final LayerNorm, their difference will be some linear function as well. In other words, we can find a vector in the space of LayerNorm outputs such that the logit difference will be the dot product of the LayerNorm's output with that vector.

We now want some way to tell which parts of the model are doing something meaningful. We will do this by identifying a single direction in the embedding space of the start token that we claim to be the "unbalanced direction": the direction that most indicates that the input string is unbalanced. It is important to note that it might be that other directions are important as well (in particular because of layer norm), but for a first approximation this works well.

We'll do this by starting from the model outputs and working backwards, finding the unbalanced direction at each stage.

The final part of the model is the classification head, which has three stages:

$$
\begin{align*}
\hat{y} &= \text{softmax}(\text{logits}_{..., 0, ...})\\
\text{logits} &= W \cdot x_\text{linear}\\
x_{\text{linear}} &= \text{layer\_norm}(x_\text{norm})
\end{align*}
$$

Where the input to the layer norm, $x_\text{norm}$, is the output of the final attention block.

Some shape notes:

- $\hat{y}$ has shape `(batch, 2)`, where $\hat{y}_0$ is the probability that the input was unbalanced, and $\hat{y}_1$ is the probability that it was balanced.
- $\text{logits}$ has shape `(batch, seq_len, 2)`
- We only use the logit values for the start token: $\text{logits}_{..., 0, ...}$ which has shape `(batch, 2)`

Writing these equations in code (and chronologically):

```python
x_linear = self.norm(x_norm)
logits = self.decoder(x_linear)
y_hat = t.softmax(logits[:, 0, :], -1)
```

In some of the notation below, **we'll omit the batch dimension for simplicity** (hopefully it'll be clear when we're doing this - please ask if there's any ambiguity).

### Stage 1: Translating through softmax

Let's get $\hat y$ as a function of the logits. Luckily, this is easy. Since we're doing the softmax over two elements, it simplifies to the sigmoid of the difference of the two logits:

$$
\text{softmax}(\begin{bmatrix} \text{logit}_0 \\ \text{logit}_1 \end{bmatrix})_0 = \frac{e^{\text{logit}_0}}{e^{\text{logit}_0} + e^{\text{logit}_1}} = \frac{1}{1 + e^{\text{logit}_1 - \text{logit}_0}} = \text{sigmoid}(\text{logit}_0 - \text{logit}_1)
$$

```python
x_linear = self.norm(x_norm)
logits = self.decoder(x_linear)
y_hat = t.sigmoid(logits[0, :] - logits[1, :])
```

Since sigmoid is monotonic, a large value of $\hat{y}_0$ follows from logits with a large $\text{logit}_0 - \text{logit}_1$. From now on, we'll only ask "What leads to a large difference in logits?"

### Stage 2: Translating through linear

The next step we encounter is the final linear layer (called `decoder`), $\text{logits} = W \cdot x_{\text{linear}}$, where:

* $W$ are the weights of the linear layer with shape `(2, 56)` (remember that weight matrices are stored as `(out_features, in_features)`), and
* $x_{\text{linear}}$ has shape `(seq_len, 56)`.

We can now put the difference in logits as a function of $W$ and $x_{\text{linear}}$ like this:

$\text{logit}_0 - \text{logit}_1 = W_{[0, :]}x_{\text{linear}} - W_{[1, :]}x_{\text{linear}} = (W_{[0, :]} - W_{[1, :]})x_{\text{linear}}$

```python
logit_diff = (self.decoder.weight[0, :] - self.decoder.weight[1, :]) @ x_linear
y_hat = t.sigmoid(logit_diff)
```

So a high difference in the logits follows from a high dot product of the output of the LayerNorm with the vector $(W_{[0, :]} - W_{[1, :]})$. We can now ask, "What leads to LayerNorm's output having high dot product with this vector?".

Use the weights of the final linear layer (`model.decoder`) to identify the direction in the space that goes into the linear layer (and out of the LN) corresponding to an 'unbalanced' classification. Hint: this is a one line function.

```python
def get_post_final_ln_dir(model: ParenTransformer) -> t.Tensor:
    pass
```

If you're feeling stuck, scoll down to the diagram at the end of section **Step 3: Translating through LayerNorm**.

### Step 3: Translating through LayerNorm

We want to find the unbalanced direction before the final layer norm, since this is where we can write the residual stream as a sum of terms. LayerNorm messes with this sort of direction analysis, since it is nonlinear. For today, however, we will approximate it with a linear fit. This is good enough to allow for interesting analysis (see for yourself that the $R^2$ values are very high for the fit)!

With a linear approximation to LayerNorm, which I'll use the matrix $L$ for, we can translate "What is the dot product of the output of the LayerNorm with the unbalanced-vector?" to a question about the input to the LN. We simply write:

$$
\begin{align*}
x_{\text{linear}} &= \text{layer\_norm}(x_\text{norm}) \approx L \cdot x_\text{norm}\\ 
\\
(W_{[0, :]} - W_{[1, :]})x_{\text{linear}} &\approx ((W_{[0, :]} - W_{[1, :]})L)x_{\text{norm}}
\end{align*}
$$

""")
    with st.expander("A note on what this linear approximation is actually representing:"):
        st.info(r"""

To clarify - we are approximating the LayerNorm operation as a linear operation from `hidden_size -> hidden_size`, i.e.:
$$
x_\text{linear}[i,j,k] = \sum_{l=1}^{hidden\_size} L[k,l] \cdot x_\text{norm}[i,j,l]
$$
In reality this isn't the case, because LayerNorm transforms each vector in the embedding dimension by subtracting that vector's mean and dividing by its standard deviation:
$$
\begin{align*}
x_\text{linear}[i,j,k] &= \frac{x_\text{norm}[i,j,k] - \mu(x_\text{norm}[i,j,:])}{\sigma(x_\text{norm}[i,j,:])}\\
&= \frac{x_\text{norm}[i,j,k] - \frac{1}{hidden\_size} \displaystyle\sum_{l=1}^{hidden\_size}x_\text{norm}[i,j,l]}{\sigma(x_\text{norm}[i,j,:])}
\end{align*}
$$
If the standard deviation of the vector in $j$th sequence position is approximately the same across all **sequences in a batch**, then we can see that the LayerNorm operation is approximately linear (if we use a different regression fit $L$ for each sequence position, which we will for most of today). Furthermore, if we assume that the standard deviation is also approximately the same across all **sequence positions**, then using the same $L$ for all sequence positions is a good approximation. By printing the $r^2$ of the linear regression fit (which is the proportion of variance explained by the model), we can see that this is indeed the case. 

Note that plotting the coefficients of this regression probably won't show a pattern quite as precise as this one, because of the classic problem of **correlated features**. However, this doesn't really matter for the purposes of our analysis, because we want to answer the question "What leads to the input to the LayerNorm having a high dot-product with this new vector?". As an example, suppose two features in the LayerNorm inputs are always identical and their coefficients are $c_1$, $c_2$ respectively. Our model could just as easily learn the coefficients $c_1 + \delta$, $c_2 - \delta$, but this wouldn't change the dot product of LayerNorm's input with the unbalanced vector (assuming we are using inputs that also have the property wherein the two features are identical).
""")

    st.markdown(r"""

We can then ask "What leads to the _input_ to the LayerNorm having a high dot-product with this new vector?"

### Introduction to hooks

In order to get the inputs and outputs of the LN so we can fit them (and to do plenty of other investigations today), we need to be able to extract activations from running the network. In order to do this we will use [forward hooks](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html). Forward hooks can also be used to modify the output of a module, but we will not be doing this today.

These hooks are functions that are attached to a pytorch module and run everytime the forward function is called. The syntax you'll need to use for calling a hook is as follows:
* Define a hook function `fn`, which takes three inputs: `module`, `input`, and `output`. `module` is the module that the hook is attached to, `input` is a tuple of the inputs to the module, and `output` is the output of the module. The hook function should return `None`.
* Call `module.register_forward_hook(fn)` to attach the hook to the module. This will return a handle that you can use to remove the hook later.
* Call the `run_model_on_data` function to run the model on some data. The hook will be called on every forward pass of the model.
* Call `handle.remove()` to remove the hook.

You should implement the `get_inputs` and `get_outputs` functions below. The easiest way to do this is by appending to a list in the hook function, and then concatenating the list into a tensor at the end.
""")

    st.info("""
A note on the ~EasyTransformer~ **TransformerLens** library

We'll end up replicating many parts of Neel's **TransformerLens** library that retrieves input activations by the end of the day, and you'll grow to appreciate how helpful it is. (We'll wait until next Monday to officially introduce the library though.)""")

    st.markdown(r"""
```python
def get_inputs(model: ParenTransformer, data: DataSet, module: nn.Module) -> t.Tensor:
    '''
    Get the inputs to a particular submodule of the model when run on the data.
    Returns a tensor of size (data_pts, seq_pos, emb_size).
    '''
    pass

def get_outputs(model: ParenTransformer, data: DataSet, module: nn.Module) -> t.Tensor:
    '''
    Get the outputs from a particular submodule of the model when run on the data.
    Returns a tensor of size (data_pts, seq_pos, emb_size).
    '''
    pass

if MAIN:
    w5d5_tests.test_get_inputs(get_inputs, model, data)
    w5d5_tests.test_get_outputs(get_outputs, model, data)
```

Now, use these functions and the [sklearn LinearRegression class](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) to find a linear fit to the inputs and outputs of `model.norm`.

The argument below takes `seq_pos` as an input. If this is an integer, then we are fitting only for that sequence position. If `seq_pos = None`, then we are fitting for all sequence positions (we aggregate the sequence and batch dimensions before performing our regression). You should include a fit coefficient in your linear regression (this is the default for `LinearRegression`).

```python
def get_ln_fit(
    model: ParenTransformer, data: DataSet, ln_module: nn.LayerNorm, seq_pos: Union[None, int]
) -> Tuple[LinearRegression, t.Tensor]:
    '''
    if seq_pos is None, find best fit for all sequence positions. Otherwise, fit only for given seq_pos.

    Returns: A tuple of a (fitted) sklearn LinearRegression object and a dimensionless tensor containing the r^2 of the fit (hint: wrap a value in torch.tensor() to make a dimensionless tensor)
    '''
    pass


if MAIN:
    (final_ln_fit, r2) = get_ln_fit(model, data, model.norm, seq_pos=0)
    print("r^2: ", r2)
    w5d5_tests.test_final_ln_fit(model, data, get_ln_fit)

```

Armed with our linear fit, we can now identify the direction in the residual stream before the final layer norm that most points in the direction of unbalanced evidence.


```python
def get_pre_final_ln_dir(model: ParenTransformer, data: DataSet) -> t.Tensor:
    pass


if MAIN:
    w5d5_tests.test_pre_final_ln_dir(model, data, get_pre_final_ln_dir)

```

If you're still confused by any of this, the following diagram might help:""")

    st_excalidraw("brackets-transformer-tracing-back-output", 1500)
    st.markdown(r"""

## Writing the residual stream as a sum of terms

Recall from yesterday that at any point during the forward pass of a model, a residual stream can be thought of as a sum of terms: one term for the contribution of each attention head or MLP so far, plus one for the input embeddings. In our journey backward, we've arrived at the last stop of the residual stream, when it is a sum of all the MLP contributions and all the attention contributions in the entire model.

If we want to figure out when the value in the residual stream at this location has high dot-product with the unbalanced direction, we can ask ***"Which components' outputs tend to cause the residual stream to have high dot product with this direction, and why?"***. Note that we're narrowing in on the components who are making direct contributions to the classification, i.e. they're causing a classification of "unbalanced" by directly writing a value to the residual stream with high dot product in the unbalanced direction, rather than by writing a value to the residual stream that influences some future head. We'll get into indirect contributions later.

In order to answer this question, we need the following tools:
 - A way to break down the input to the LN by component.
 - A tool to identify a direction in the embedding space that causes the network to output 'unbalanced' (we already have this)

### Output-by-head hooks

These functions can be used to capture the output of an MLP or an attention layer. However, we also want to be able to get the output of an individual attention head.

Write a function that returns the output by head instead, for a given layer. You'll need the hook functions you wrote earlier.

Each of the linear layers in the attention layers have bias terms. For getting the output by head, we will ignore the bias that comes from `model.W_O`, since this is not cleanly attributable to any individual head.

Reminder: PyTorch stores weights for linear layers in the shape `(out_features, in_features)`.
""")

    with st.expander("Help - I'm confused about the linear algebra here."):
        st.markdown(r"""
We can write the output weight matrix $W_O$ as a stack of matrices, and do the same for the result vectors $r$ which form the input to this layer. If $H$ is the number of heads, then we have:
$$
W_O\left[\begin{array}{l}
r^{1} \\
r^{2} \\
\ldots \\
r^{H}
\end{array}\right]=\left[W_O^{1}, W_O^{2}, \ldots, W_O^{H}\right] \cdot\left[\begin{array}{c}
r^{1} \\
r^{2} \\
\ldots \\
r^{H}
\end{array}\right]=\sum_{h=1}^{H} W_O^{h} r^{h}
$$

(for more on this, see [this section](https://transformer-circuits.pub/2021/framework/index.html#architecture-attn-independent) of the Mathematical Frameworks paper.)
""")
    with st.expander("Help - I'm confused about the implementation here."):
        st.markdown(r"""
Your function should do the following:

* Capture the inputs to this layer using the input hook function you wrote earlier - call the inputs `r`.
* Reshape the inputs so that heads go along their own dimension (see expander above).
* Extract the weights from the model directly, and reshape them so that heads go along their own dimension.
* Perform the matrix multiplication shown in the expander above (but keeping the terms in the sum separate). You should definitely use `einsum` for this!
""")
        st.markdown("")
    st.markdown(r"""
```python
def get_out_by_head(model: ParenTransformer, data: DataSet, layer: int) -> t.Tensor:
    '''
    Get the output of the heads in a particular layer when the model is run on the data.
    Returns a tensor of shape (batch, num_heads, seq, embed_width)
    '''
    pass

if MAIN:
    w5d5_tests.test_get_out_by_head(get_out_by_head, model, data)
```

### Breaking down the residual stream by component

Use your hook tools to create a tensor of shape `[num_components, dataset_size, seq_pos]`, where the number of components = 10.

This is a termwise representation of the input to the final layer norm from each component (recall that we can see each head as writing something to the residual stream, which is eventually fed into the final layer norm). The order of the components in your function's output should be:

* `embeddings`, i.e. the sum of token and positional embeddings (corresponding to the direct path through the model, which avoids all attention layers and MLPs)
* For each of the layers `layer = 0, 1, 2`:
    * Head `layer.0`, i.e. the output of the first attention head in layer `layer`
    * Head `layer.1`, i.e. the output of the second attention head in layer `layer`
    * MLP `layer`, i.e. the output of the MLP in layer `layer`

(The only term missing the `W_O`-bias from each of the attention layers).

```python
def get_out_by_components(model: ParenTransformer, data: DataSet) -> t.Tensor:
    '''
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2]
    '''
    pass

if MAIN:
    w5d5_tests.test_get_out_by_component(get_out_by_components, model, data)
```

Now, confirm that input to the final layer norm is the sum of the output of each component and the output projection biases.

```python
if MAIN:
    biases = sum([model.layers[l].self_attn.W_O.bias for l in (0, 1, 2)]).clone()
    out_by_components = "YOUR CODE"
    summed_terms = "YOUR CODE"
    pre_final_ln = "YOUR CODE"
    t.testing.assert_close(summed_terms, pre_final_ln)
```

### Which heads write in this direction? On what sorts of inputs?

To figure out which components are directly important for the the model's output being "unbalanced", we can see which components tend to output a vector to the position-0 residual stream with higher dot product in the unbalanced direction for actually unbalanced inputs.

The idea is that, if a component is important for correctly classifying unbalanced inputs, then its vector output when fed unbalanced bracket strings will have a higher dot product in the unbalanced direction than when it is fed balanced bracket strings.

In this section, we'll plot histograms of the dot product for each component. This will allow us to observe which components are significant.

For example, suppose that one of our components produced bimodal output like this:""")

    st_image("exampleplot.png", 750)

    st.markdown(r"""
This would be **strong evidence that this component is important for the model's output being unbalanced**, since it's pushing the unbalanced bracket inputs further in the unbalanced direction (i.e. the direction which ends up contributing to the inputs being classified as unbalanced) relative to the balanced inputs.

In the `MAIN` block below, you should compute a `(10, N_SAMPLES)` tensor containing, for each sample, for each component, the dot product of the component's output with the unbalanced direction on this sample. You should normalize it by subtracting the mean of the dot product of this component's output with the unbalanced direction on balanced samples - this will make sure the histogram corresponding to the balanced samples is centered at 0 (like in the figure above), which will make it easier to interpret. Remember, it's only the **difference between the dot product on unbalanced and balanced samples** that we care about (since adding a constant to both logits doesn't change the model's probabilistic output).

The `hists_per_comp` function to plot these histograms has been written for you - all you need to do is calculate the `magnitudes` object and supply it to that function.

```python
def hists_per_comp(magnitudes, data):
    num_comps = magnitudes.shape[0]
    titles = {
        (1, 1): "embeddings",
        (2, 1): "head 0.0",
        (2, 2): "head 0.1",
        (2, 3): "mlp 0",
        (3, 1): "head 1.0",
        (3, 2): "head 1.1",
        (3, 3): "mlp 1",
        (4, 1): "head 2.0",
        (4, 2): "head 2.1",
        (4, 3): "mlp 2"
    }
    assert num_comps == len(titles)

    fig = make_subplots(rows=4, cols=3)
    for ((row, col), title), mag in zip(titles.items(), magnitudes):
        fig.add_trace(go.Histogram(x=mag[data.isbal].numpy(), name="Balanced", marker_color="blue", opacity=0.5, legendgroup = '1', showlegend=title=="embeddings"), row=row, col=col)
        fig.add_trace(go.Histogram(x=mag[~data.isbal].numpy(), name="Unbalanced", marker_color="red", opacity=0.5, legendgroup = '2', showlegend=title=="embeddings"), row=row, col=col)
        fig.update_xaxes(title_text=title, range=[-10, 20], row=row, col=col)
    fig.update_layout(width=1200, height=1200, barmode="overlay", legend=dict(yanchor="top", y=0.92, xanchor="left", x=0.4), title="Histograms of component significance")
    fig.show()
    return fig

if MAIN:
    "TODO: YOUR CODE HERE"
```""")

    with st.expander("Click here to see the output you should be getting."):
        st.plotly_chart(fig_dict["attribution_fig"])

    with st.expander("Which heads do you think are the most important, and can you guess why that might be?"):
        st.markdown(r"""
The heads in layer 2 (i.e. `2.0` and `2.1`) seem to be the most important, because the unbalanced brackets are being pushed much further to the right than the balanced brackets. 

We might guess that some kind of composition is going on here. The outputs of layer 0 heads can't be involved in composition because they in effect work like a one-layer transformer. But the later layers can participate in composition, because their inputs come from not just the embeddings, but also the outputs of the previous layer. This means they can perform more complex computations.""")

    st.markdown(r"""
### Head influence by type of failures

Those histograms showed us which heads were important, but it doesn't tell us what these heads are doing, however. In order to get some indication of that, let's focus in on the two heads in layer 2 and see how much they write in our chosen direction on different types of inputs. In particular, we can classify inputs by if they pass the 'overall elevation' and 'nowhere negative' tests.

We'll also ignore sentences that start with a close paren, as the behaviour is somewhat different on them (they can be classified as unbalanced immediately, so they don't require more complicated logic).

Define, so that the graphing works:
* **`negative_failure`**
    * This is an `(N_SAMPLES,)` boolean vector that is true for sequences whose elevation (when reading from right to left) ever dips negative, i.e. there's an open paren that is never closed.
* **`total_elevation_failure`**
    * This is an `(N_SAMPLES,)` boolean vector that is true for sequences whose total elevation is not exactly 0. In other words, for sentences with uneven numbers of open and close parens.
* **`h20_in_d`**
    * This is an `(N_SAMPLES,)` float vector equal to head 2.0's contribution to the position-0 residual stream in the unbalanced direction, normalized by subtracting its average unbalancedness contribution to this stream over _balanced sequences_.
* **`h21_in_d`**
    * Same as above but head 2.1

For the first two of these, you will find it helpful to refer back to your `is_balanced_vectorized` code (although remember you're reading **right to left** here). You can get the last two from your `magnitudes` tensor.

```python
if MAIN:
    negative_failure = None
    total_elevation_failure = None
    h20_in_d = None
    h21_in_d = None

    failure_types = np.full(len(h20_in_d), "", dtype=np.dtype("U32"))
    failure_types_dict = {
        "both failures": negative_failure & total_elevation_failure,
        "just neg failure": negative_failure & ~total_elevation_failure,
        "just total elevation failure": ~negative_failure & total_elevation_failure,
        "balanced": ~negative_failure & ~total_elevation_failure
    }
    for name, mask in failure_types_dict.items():
        failure_types = np.where(mask, name, failure_types)
    failures_df = pd.DataFrame({
        "Head 2.0 contribution": h20_in_d,
        "Head 2.1 contribution": h21_in_d,
        "Failure type": failure_types
    })[data.starts_open.tolist()]
    fig = px.scatter(
        failures_df, 
        x="Head 2.0 contribution", y="Head 2.1 contribution", color="Failure type", 
        title="h20 vs h21 for different failure types", template="simple_white", height=600, width=800,
        category_orders={"color": failure_types_dict.keys()}
    ).update_traces(marker_size=4)
    fig.show()
```

Look at the above graph and think about what the roles of the different heads are!""")

    with st.expander("Nix's thoughts -- Read after thinking for yourself"):
        st.markdown(r"""

The primary thing to take away is that 2.0 is responsible for checking the overall counts of open and close parentheses, and that 2.1 is responsible for making sure that the elevation never goes negative.

Aside: the actual story is a bit more complicated than that. Both heads will often pick up on failures that are not their responsibility, and output in the 'unbalanced' direction. This is in fact incentived by log-loss: the loss is slightly lower if both heads unanimously output 'unbalanced' on unbalanced sequences rather than if only the head 'responsible' for it does so. The heads in layer one do some logic that helps with this, although we'll not cover it today.

One way to think of it is that the heads specialized on being very reliable on their class of failures, and then sometimes will sucessfully pick up on the other type.

""")

    st.markdown(r"""
In most of the rest of these exercises, we'll focus on the overall elevation circuit as implemented by head 2.0. As an additional way to get intuition about what head 2.0 is doing, let's graph its output against the overall proportion of the sequence that is an open-paren.

**Before you do this, think about what you expect to see in this plot.**

```python
if MAIN:
    fig = px.scatter(
        x=data.open_proportion, y=h20_in_d, color=failure_types, 
        title="Head 2.0 contribution vs proportion of open brackets '('", template="simple_white", height=500, width=800,
        labels={"x": "Open-proportion", "y": "Head 2.0 contribution"}, category_orders={"color": failure_types_dict.keys()}
    ).update_traces(marker_size=4, opacity=0.5).update_layout(legend_title_text='Failure type')
    fig.show()
```

Think about how this fits in with your understanding of what 2.0 is doing.""")

def section_3():
    st.markdown(r"""
# Understanding the total elevation circuit

## Attention pattern of the responsible head

Which tokens is 2.0 paying attention to when the query is an open paren at token 0? Recall that we focus on sequences that start with an open paren because sequences that don't can be ruled out immediately, so more sophisticated behavior is unnecessary.

Write a function that extracts the attention patterns for a given head when run on a batch of inputs. Our code will show you the average attention pattern paid by the query for residual stream 0 when that position is an open paren.

Specifically:
* Use `get_inputs` from earlier, on the self-attention module in the layer in question.
* You can use the `attention_pattern_pre_softmax` function to get the pattern, then mask the padding (elements of the batch might be different lengths, and thus be suffixed with padding).""")

    with st.expander("How do I find the padding?"):
        st.markdown(r"""
`data.toks == tokenizer.PAD_TOKEN` will give you a boolean matrix of which positions in which batch elements are padding and which aren't.""")

    st.markdown(r"""
```python
def get_attn_probs(model: ParenTransformer, tokenizer: SimpleTokenizer, data: DataSet, layer: int, head: int) -> t.Tensor:
    '''
    Returns: (N_SAMPLES, max_seq_len, max_seq_len) tensor that sums to 1 over the last dimension.
    '''
    pass

if MAIN:
    attn_probs = get_attn_probs(model, tokenizer, data, 2, 0)
    attn_probs_open = attn_probs[data.starts_open].mean(0)[[0]]
    px.bar(
        y=attn_probs_open.squeeze().numpy(), labels={"y": "Probability", "x": "Key Position"},
        template="simple_white", height=500, width=600, title="Avg Attention Probabilities for '(' query from query 0"
    ).update_layout(showlegend=False, hovermode='x unified')
```

You should see an average attention of around 0.5 on position 1, and an average of about 0 for all other tokens. So `2.0` is just copying information from residual stream 1 to residual stream 0. In other words, `2.0` passes residual stream 1 through its `W_OV` circuit (after `LayerNorm`ing, of course), weighted by some amount which we'll pretend is constant. The plot thickens. Now we can ask, "What is the direction in residual stream 1 that, when passed through `2.0`'s `W_OV`, creates a vector in the unbalanced direction in residual stream 0?"

### Identifying meaningful direction before this head

We again need to propagate the direction back, this time through the OV matrix of `2.0` and a linear fit to the layernorm. This will require functions that can find these matrices for us.

Remember, we're interested in the 1st sequence position in the residual stream (see the bar chart above). So when fitting our regression to the layernorm before attention head `2.0`, we'll use `seq_pos=1`.

If you're confused by this, then use the dropdown below for a version of the diagram from earlier which includes the contribution from `2.0`.""")

    with st.expander("Diagram"):
        st_excalidraw("brackets-transformer-tracing-back-output-2", 1500)

    st.markdown(r"""
```python
def get_WV(model: ParenTransformer, layer: int, head: int) -> t.Tensor:
    '''
    Returns the W_V matrix of a head. Should be a CPU tensor of size (d_model / num_heads, d_model)
    '''
    pass

def get_WO(model: ParenTransformer, layer: int, head: int) -> t.Tensor:
    '''
    Returns the W_O matrix of a head. Should be a CPU tensor of size (d_model, d_model / num_heads)
    '''
    pass

def get_WOV(model: ParenTransformer, layer: int, head: int) -> t.Tensor:
    return get_WO(model, layer, head) @ get_WV(model, layer, head)

def get_pre_20_dir(model, data):
    '''
    Returns the direction propagated back through the OV matrix of 2.0 and then through the layernorm before the layer 2 attention heads.
    '''
    pass

if MAIN:
    w5d5_tests.test_get_WV(model, get_WV)
    w5d5_tests.test_get_WO(model, get_WO)
    w5d5_tests.test_get_pre_20_dir(model, data, get_pre_20_dir)
if MAIN:
    pre_20_d = get_pre_20_dir(model, data)
    in_d_20 = t.inner(out_by_components[:7, :, 1, :], pre_20_d).detach()
    titles = [
        "embeddings",
        "head 0.0",
        "head 0.1",
        "mlp 0",
        "head 1.0",
        "head 1.1",
        "mlp 1",
        "head 2.0",
        "head 2.1",
        "mlp 2",
    ]
    (fig, axs) = plt.subplots(7, 1, figsize=(6, 11), sharex=True)
    for i in range(7):
        ax: plt.Axes = axs[i]
        normed = in_d_20[i] - in_d_20[i, data.isbal].mean(0)
        ax.hist(normed[data.starts_open & data.isbal.clone()].numpy(), alpha=0.5, bins=75, label="bal")
        ax.hist(normed[data.starts_open & ~data.isbal.clone()].numpy(), alpha=0.5, bins=75, label="unbal")
        ax.title.set_text(titles[i])
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
```

What do you observe?""")

    with st.expander("Some things to notice"):
        st.markdown(r"""
We can see that mlp0 and especially mlp1 are very important. This makes sense -- one thing that mlps are especially capable of doing is turning more continious features ('what proportion of characters in this input are open parens?') into sharp discontinious features ('is that proportion exactly 0.5?').

Head 1.1 also has some importance, although we will not be able to dig into this today. It turns out that one of the main things it does is incorporate information about when there is a negative elevation failure into this overall elevation branch. This allows the heads to agree the prompt is unbalanced when it is obviously so, even if the overall count of opens and closes would allow it to be balanced.""")

    st.markdown(r"""
In order to get a better look at what MLPs 0 and 1 are doing more thoughly, we can look at their output as a function of the overall open-proportion.

```python
if MAIN:
    plt.scatter(data.open_proportion[data.starts_open], in_d_20[3, data.starts_open], s=2)
    plt.ylabel("amount mlp 0 writes in the unbalanced direction for head 2.0")
    plt.xlabel("open-proprtion of sequence")
    plt.show()

if MAIN:
    plt.scatter(data.open_proportion[data.starts_open], in_d_20[6, data.starts_open], s=2)
    plt.ylabel("amount mlp 1 writes in the unbalanced direction for head 2.0")
    plt.xlabel("open-proprtion of sequence")
    plt.show()
```

### Breaking down an MLP's contribution by neuron

Yesterday you learned that an attention layer can be broken down as a sum of separate contributions from each head. It turns out that we can do something similar with MLPs, breaking them down as a sum of per-neuron contributions. I've hidden this decomposition in a dropdown in case you feel motivated to try to discover it for youself; simply opening up the box is fine though.

Ignoring biases, let $MLP(\vec x) = A f(B\vec x)$ for matrices $A, B$. Note that $f(B\vec x)$ is what we refer to as the neuron activations, let $n$ be its length (the intermediate size of the MLP). Write $MLP$ as a sum of $n$ functions of $\vec x$.

""")

    with st.markdown("Answer and discussion"):
        st.expander(r"""
One way to conceptualize the matrix-vector multiplication $V \vec y$ is as a weighted sum of the columns of $V$, thus making $y$ the list of coefficients on the columns in this weighted sum. In other words, $MLP(\vec x) = \sum_{i=0}^{n-1}A_{[;,i]}f(B\vec x)_i$. But we can actually simplify further, as $f(B\vec x)_i = f(B_{[i,:]} \vec x)$; i.e. the dot product of $\vec x$ and the $i$-th row of $B$ (and not the rest of $B$!).

Thus $MLP(\vec x) = \sum_{i=0}^{n-1}A_{[;,i]}f(B_{[i,:]}\vec x)$, or $d + \sum_{i=0}^{n-1}A_{[;,i]}f(B_{[i,:]}\vec x + c_i)$ if we include biases on the Linear layers.

We can view the $i$-th row of $B$ as being the "in-direction" of neuron $i$, as the activation of neuron $i$ depends on how high the dot product between $x$ and that row is. And then we can think of the $i$-th column of A as signifying neuron $i$'s special output vector, which it scales by its activation and then adds into the residual stream. So an MLP consists of $n$ neurons, each of whose activations are calculated by comparing $x$ to their individual in-directions, and then each contributes some scaled version of its out-direction to the residual stream. This is a neat-enough equation that Buck jokes he's going to get it tattooed on his arm.
""")

    st.markdown(r"""
```python
def out_by_neuron(model, data, layer):
    '''
    Return shape: [len(data), seq_len, neurons, out]
    '''
    pass

@functools.cache
def out_by_neuron_in_20_dir(model, data, layer):
    pass
```

Now, try to identify several individual neurons that are especially important to `2.0`.

For instance, you can do this by seeing which neurons have the largest difference between how much they write in our chosen direction on balanced and unbalanced sequences (especially unbalanced sequences beginning with an open paren).

Use the `plot_neuron` command to get a sense of what an individual neuron does on differen open-proportions.

One note: now that we are deep in the internals of the network, our assumption that a single direction captures most of the meaningful things going on in this overall-elevation circuit is highly questionable. This is especially true for using our `2.0` direction to analyize the output of `mlp0`, as one of the main ways this mlp has influence is through more indirect paths (such as `mlp0 -> mlp1 -> 2.0`) which are not the ones we chose our direction to capture. Thus, it is good to be aware that the intuitions you get about what different layers or neurons are doing are likely to be incomplete.

```python
def plot_neuron(model, data, layer, neuron_number):
    neurons_in_d = out_by_neuron_in_20_dir(model, data, layer)
    plt.scatter(data.open_proportion[data.starts_open], neurons_in_d[data.starts_open, 1, neuron_number].detach(), s=2)
    plt.xlabel("open-proportion")
    plt.ylabel("output in 2.0 direction")
    plt.show()

if MAIN:
    "TODO: YOUR CODE HERE"
```
""")

    with st.expander("Some observations:"):
        st.markdown(r"""
The important neurons in layer 1 can be put into three broad categories:
 - Some neurons detect when the open-proprtion is greater than 1/2. As a few examples, look at neurons 1.53, 1.39, 1.8 in layer 1. There are some in layer 0 as well, such as 0.33 or 0.43. Overall these seem more common in Layer 1.
 - Some neurons detect when the open-proprtion is less than 1/2. For instance, neurons 0.21, and 0.7. These are much more rare in layer 1, but you can see some such as 1.50 and 1.6

The network could just use these two types of neurons, and compose them to measure if the open-proportion exactly equals 1/2 by adding them together. But we also see in layer 1 that there are many neurons that output this composed property. As a few examples, look at 1.10 and 1.3. It's much harder for a single neuron in layer 0 to do this by themselves, given that ReLU is monotonic and it requires the output to be a non-monotonic function of the open-paren proportion. It is possible, however, to take advantage of the layernorm before MLP0 to approximate this -- 0.19 and 0.34 are good examples of this.""")

    st.markdown(r"""
## Understanding how the open-proportion is calculated - Head 0.0

Up to this point we've been working backwards from the logits and through the internals of the network. We'll now change tactics somewhat, and start working from the input embeddings forwards. In particular, we want to understand how the network calcuates the open-proportion of the sequence in the first place!

The key will end up being head 0.0. Let's start by examining it's attention pattern.

### 0.0 Attention Pattern

We're going to calculate the attention pattern directly from weights. First, write a function to get the Q and K weights, then another to find the attention pattern for a given sequence of embeddings. We're going to want to look at the attention pattern for different queries, so the function will take separate query and key embeddings. For a given query, we can then average accross all potential keys.


```python
def get_Q_and_K(model: ParenTransformer, layer: int, head: int) -> Tuple[t.Tensor, t.Tensor]:
    '''
    Get the Q and K weight matrices for the attention head at the given indices.

    Return: Tuple of two tensors, both with shape (embedding_size, head_size)
    '''
    pass


def qk_calc_termwise(
    model: ParenTransformer, layer: int, head: int, q_embedding: t.Tensor, k_embedding: t.Tensor
) -> t.Tensor:
    '''
    Get the pre-softmax attention scores that would be calculated by the given attention head from the given embeddings.

    q_embedding: tensor of shape (seq_len, embedding_size)
    k_embedding: tensor of shape (seq_len, embedding_size)

    Returns: tensor of shape (seq_len, seq_len)
    '''
    pass


if MAIN:
    w5d5_tests.qk_test(model, get_Q_and_K)
    w5d5_tests.test_qk_calc_termwise(model, tokenizer, qk_calc_termwise)

```

Get the token embeddings for the '(' or ')' tokens.


```python
def embedding(model: ParenTransformer, tokenizer: SimpleTokenizer, char: str) -> torch.Tensor:
    assert char in ("(", ")")
    pass


if MAIN:
    open_emb = embedding(model, tokenizer, "(")
    closed_emb = embedding(model, tokenizer, ")")
    w5d5_tests.embedding_test(model, tokenizer, embedding)

```

We'll plot the attention pattern in the case when the query is an '('.


```python
if MAIN:
    open_emb = embedding(model, tokenizer, "(")
    closed_emb = embedding(model, tokenizer, ")")
    pos_embeds = model.pos_encoder.pe
    open_emb_ln_per_seqpos = model.layers[0].norm1(open_emb.to(DEVICE) + pos_embeds[1:41])
    close_emb_ln_per_seqpos = model.layers[0].norm1(closed_emb.to(DEVICE) + pos_embeds[1:41])
    attn_score_open_open = qk_calc_termwise(model, 0, 0, open_emb_ln_per_seqpos, open_emb_ln_per_seqpos)
    attn_score_open_close = qk_calc_termwise(model, 0, 0, open_emb_ln_per_seqpos, close_emb_ln_per_seqpos)
    attn_score_open_avg = (attn_score_open_open + attn_score_open_close) / 2
    attn_prob_open = (attn_score_open_avg / 28**0.5).softmax(-1).detach().clone().numpy()
    plt.matshow(attn_prob_open, cmap="magma")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title("Predicted Attention Probabilities for ( query")
    plt.gcf().set_size_inches(8, 6)
    plt.colorbar()
    plt.tight_layout()

```

Now lets compare this to the empirical attention pattern on the dataset. We'll look at the attention pattern for both '(' and ')' as queries.


```python
def avg_attn_probs_0_0(
    model: ParenTransformer, data: DataSet, tokenizer: SimpleTokenizer, query_token: int
) -> t.Tensor:
    '''
    Calculate the average attention probs for the 0.0 attention head for the provided data when the query is the given query token.
    Returns a tensor of shape (seq, seq)
    '''
    pass


if MAIN:
    data_len_40 = DataSet.with_length(data_tuples, 40)
    for paren in ("(", ")"):
        tok = tokenizer.t_to_i[paren]
        attn_probs_mean = avg_attn_probs_0_0(model, data_len_40, tokenizer, tok).detach().clone()
        plt.matshow(attn_probs_mean, cmap="magma")
        plt.ylabel("query position")
        plt.xlabel("key position")
        plt.title(f"with query = {paren}")
        plt.show()

```

What do you think might be going on in each attention pattern?
""")

    with st.expander("Hint"):
        st.markdown(r"""
Consider only the attention pattern a query position one. This is what the attention head uses to calculate the open-proportion for the overall elevation circuit.
""")

    with st.expander("Why are the two attention patterns different?"):
        st.markdown(r"""
In the case where the sequence starts with a closed paren, the sequence is obviously unbalanced and the head does not need to know the open proportion.

Now, we'll plot just the attention pattern for the first query position.


```python
if MAIN:
    tok = tokenizer.t_to_i["("]
    attn_probs_mean = avg_attn_probs_0_0(model, data_len_40, tokenizer, tok).detach().clone()
    plt.plot(range(42), attn_probs_mean[1])
    plt.ylim(0, None)
    plt.xlim(0, 42)
    plt.xlabel("Sequence position")
    plt.ylabel("Average attention")
    plt.show()
```
""")

    with st.expander("What does this attention pattern mean?"):
        st.markdown(r"""
This adds the values from all key positions with roughly equal magnitude.""")

    st.markdown(r"""
### The 0.0 OV circuit
Next, lets look at what is being added, in other words, the OV for each embedding. We'll use the weights directly.

To calculate the OV for each embedding, we'll need to grab the V and O weights from the model, and combine them for the OV.

If the layernorm in layer 0 is approximately linear, and the attention pattern of 0.0 is uniform over successive tokens, then the output of 0.0 onto residual stream $i$ can be broken down as $\sum_{j=i}^n \frac 1 {n - i + 1} W_{OV}(LN(pos_j + tok_j)) = \sum_{j=i}^n \frac 1 {n - i + 1} W_{OV}(LN(pos_j)) + \frac a {n - i + 1} W_{OV}(LN("("))) + \frac b {n - i + 1} W_{OV}(LN(")"))$. The sum here is a constant, so it doesn't play a role in distinguishing balanced and unbalanced sequences.

Show that $W_{OV}(LN("(")))$ and $W_{OV}(LN(")")))$ are pointed away from each other and are of similarly magnitudes, demonstrating that this head is "tallying" the open and close parens that come after it.


```python
def embedding_V_0_0(model, emb_in: t.Tensor) -> t.Tensor:
    pass


def embedding_OV_0_0(model, emb_in: t.Tensor) -> t.Tensor:
    pass


if MAIN:
    "TODO: YOUR CODE HERE"

```


Great! Let's stop and take stock of what we've learned about this circuit. Head 0.0 pays attention uniformly to the suffix following each token, tallying up the amount of open and close parens that it sees and writing that value to the residual stream. This means that it writes a vector representing the total elevation to residual stream 1. The MLPs in residual stream 1 then operate nonlinearly on this tally, writing vectors to the residual stream that distinguish between the cases of zero and non-zero total elevation. Head 2.0 copies this signal to residual stream 0, where it then goes through the classifier and leads to a classification as unbalanced. Our first-pass understanding of this behavior is complete.""")

def section_4():
    st.markdown(r"""
# Finding adversarial examples

Our model gets around 1 in a ten thousand examples wrong on the dataset we've been using. Armed with our understanding of the model, can we find a misclassified input by hand? I recommend stopping reading now and trying your hand at applying what you've learned so far to find a misclassified sequence. If this doesn't work, look at a few hints.

## A series of hints...
""")
    with st.expander(r"""What kind of misclassification? """):

        st.markdown(r"""
 
To make our lives easier, we'd like to find a sequence that is, in some sense, "just barely" unbalanced. In other words, we want to give the model many signs of a balanced sequence and as few as possible of unbalancedness. Knowing what we've learned about how the model assesses balancedness, how might we do so?
""")
    with st.expander(r"""Hint: How to get to "just barely" """):

        st.markdown(r"""
 
Of the two types of unbalancedness, it's much easier to find advexes (adversarial examples) that have negative elevation at some point, rather than non-zero total elevation. I think this is because having the elevation being negative in just one place can be swamped by the elevation being fine everywhere else. Recall that the model decides whether, at each token position, the suffix beginning with that token has equal numbers of open and close parens. To make our sequence just barely unbalanced, we want to make almost all of its suffixes have equal numbers of open and close parens. We do this by using a sequence of the form "A)(B", where A and B are balanced substrings. The positions of the open paren next to the B will thus be the only position in the whole sequence on which the elevation drops below zero, and it will drop just to -1.
""")
    with st.expander(r"""Hint: How long should A and B be? """):

        st.markdown(r"""
 
Head 0.0 will be counting the parens in the suffix coming on or after the open paren in our )(. We want to pick A and B such that, when this calculation is being performed, 0.0 will be paying as little attention as possible to the open paren itself, and more to the rest of the suffix, which will be balanced. Looking at 0.0's attention pattern when the query is an open paren, how can we do this?
""")
    with st.expander(r"""Hint: The right lengths of A and B """):

        st.markdown(r"""
 
We want the )( to fall in the high-twenties of a length-40 sequence, as the attention pattern for head 0.0 shows that when an open paren is in this range, it attends bizarrely strongly to the tokens at position 38-40 ish (averaged across token type). Since attentions must sum to 1, strong attention somewhere else means less attention paid to this token.
""")
    with st.expander(r"""Exercise: make an adversarial example that is unbalanced but gets balanced confidence >10%"""):

        st.markdown(r"""


This gets confidence 24%. Formatted for readability:

    "()()()()() ()()()()() ()()()())( ()()()()()".replace(" ", "")

""")
    with st.expander(r"""Hint: tips for getting even higher (erroneous) confidence"""):
        st.markdown(r"""""")

    st.markdown(r"""This is good, but really what we'd like is to somehow use the extra attention paid to the final tokens to fully balance out the extra open paren. This means having that attention be applied to close parens in particular. This way, although there will in fact be fewer close parens than opens in the sequence starting with our troublesome open paren, they will have extra weight in the sum.
""")
    with st.expander(r"""Exercise: get an adversarial attack with 98% misclassification confidence"""):
        st.markdown(r"""We'll make the final tokens (with all the extra attention) are close parens. This gets 98% confidence.

    "()()()()() ()()()()() ()()()())( ()()((()))".replace(" ", "")""")

 
    with st.expander(r"""Exercise: Play around a bit and find an attack with >99.9% confidence"""):
        st.markdown(r"""This is the best one I've found (w/ 99.9265% of being balanced):

```python
"()()()()() ()()()()() ()())(()() (())((()))".replace(" ", "")
```""")

    st.markdown(r"""
```python
if MAIN:
    print("Update the examples list below below find adversarial examples")
    examples = ["()", "(()"]
    toks = tokenizer.tokenize(examples).to(DEVICE)
    out = model(toks)
    print("\n".join([f"{ex}: {p:.4%} balanced confidence" for (ex, p) in zip(examples, out.exp()[:, 1])]))
```
""")

func_list = [section_home, section_1, section_2, section_3, section_4]

page_list = ["🏠 Home", "1️⃣ Bracket classifier", "2️⃣ Going backwards", "3️⃣ Total elevation circuit", "4️⃣ Finding adversarial examples"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()
