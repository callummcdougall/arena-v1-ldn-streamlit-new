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

NAMES = ["attribution_fig", "attribution_fig_2", "failure_types_fig", "failure_types_fig_2", "logit_diff_from_patching", "line", "attn_induction_score","distil_plot", "ov_copying", "scatter_evals"]
def complete_fig_dict(fig_dict):
    for name in NAMES:
        if name not in fig_dict:
            fig_dict[name] = read_from_html(name)
    return fig_dict
if "fig_dict" not in st.session_state:
    st.session_state["fig_dict"] = {}
fig_dict_old = st.session_state["fig_dict"]
fig_dict = complete_fig_dict(fig_dict_old)
if len(fig_dict) > len(fig_dict_old):
    st.session_state["fig_dict"] = fig_dict

def section_home():
    st.markdown(r"""
# TransformerLens

Today is designed to get you introduced to Neel Nanda's **TransformerLens** library, which we'll be using for the rest of the interpretability chapter. The hope is that, having previously had to write our own code to do things like visualize attention heads, we'll have a better understanding of the features of TransformerLens that make this more convenient.""")

    st.info(r"""
Today's material is transcribed directly from Neel Nanda's TransformerLens intro. If you prefer to go through the original Colab, you can find it [here](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/new-demo/New_Demo.ipynb#scrollTo=cfdePPmEkR8y) (or alternatively you can pull it from Colab into a VSCode notebook).

Whenever the first person is used here, it's referring to Neel.""")

    st.markdown(r"""

## Setup

This section is just for doing basic setup and installations.

First, you'll need to install TransformerLens and the visualisation library circuitsvis:

```python
pip install git+https://github.com/neelnanda-io/TransformerLens.git@new-demo
pip install circuitsvis
```

Testing that circuitsvis works:

```python
import circuitsvis as cv
cv.examples.hello("Bob")
```

Additionally, it's useful to run the following at the top of your notebook / python file:

```python
from IPython import get_ipython
ipython = get_ipython()
# Code to automatically update the HookedTransformer code as its edited without restarting the kernel
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")
```

Lastly, here are the remaining imports and useful functions:

```python
import plotly.io as pio
pio.renderers.default = "notebook_connected" # or use "browser" if you want plots to open with browser

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from einops import repeat, rearrange, reduce
from fancy_einsum import einsum
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from torchtyping import TensorType as TT
from torchtyping import patch_typeguard
from typeguard import typechecked
from typing import List, Union, Optional, Tuple
from functools import partial
from tqdm import tqdm
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML, display

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

def imshow(tensor, renderer=None, xaxis="", yaxis="", caxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)
```
""")

def section_1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#loading-and-running-models">Loading and Running Models</a></li>
   <li><a class="contents-el" href="#caching-all-activations">Caching all Activations</a></li>
   <li><a class="contents-el" href="#hooks-intervening-on-activations">Hooks: Intervening on Activations</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#activation-patching-on-the-indirect-object-identification-task">Activation Patching on the Indirect Object Identification Task</a></li>
   </ul></li>
   <li><a class="contents-el" href="#hooks-accessing-activations">Hooks: Accessing Activations</a></li>
   <li><a class="contents-el" href="#available-models">Available Models</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#an-overview-of-the-important-open-source-models-in-the-library">An overview of the important open source models in the library</a></li>
       <li><a class="contents-el" href="#an-overview-of-some-interpretability-friendly-models-i-ve-trained-and-included">An overview of some interpretability-friendly models I've trained and included</a></li>
   </ul></li>
   <li><a class="contents-el" href="#other-resources:">Other Resources:</a></li>
   <li><a class="contents-el" href="#transformer-architecture">Transformer architecture</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#parameter-names">Parameter Names</a></li>
       <li><a class="contents-el" href="#activation-hook-names">Activation + Hook Names</a></li>
       <li><a class="contents-el" href="#folding-layernorm-for-the-curious">Folding LayerNorm (For the Curious)</a></li>
    </ul></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
# Introduction

This is a demo notebook for [TransformerLens](https://github.com/neelnanda-io/TransformerLens), **a library I ([Neel Nanda](neelnanda.io)) wrote for doing [mechanistic interpretability](https://distill.pub/2020/circuits/zoom-in/) of GPT-2 Style language models.** The goal of mechanistic interpretability is to take a trained model and reverse engineer the algorithms the model learned during training from its weights. It is a fact about the world today that we have computer programs that can essentially speak English at a human level (GPT-3, PaLM, etc), yet we have no idea how they work nor how to write one ourselves. This offends me greatly, and I would like to solve this! Mechanistic interpretability is a very young and small field, and there are a *lot* of open problems - if you would like to help, please try working on one! **Check out my [list of concrete open problems](https://docs.google.com/document/d/1WONBzNqfKIxERejrrPlQMyKqg7jSFW92x5UMXNrMdPo/edit#) to figure out where to start.**

I wrote this library because after I left the Anthropic interpretability team and started doing independent research, I got extremely frustrated by the state of open source tooling. There's a lot of excellent infrastructure like HuggingFace and DeepSpeed to *use* or *train* models, but very little to dig into their internals and reverse engineer how they work. **This library tries to solve that**, and to make it easy to get into the field even if you don't work at an industry org with real infrastructure! The core features were heavily inspired by [Anthropic's excellent Garcon tool](https://transformer-circuits.pub/2021/garcon/index.html). Credit to Nelson Elhage and Chris Olah for building Garcon and showing me the value of good infrastructure for accelerating exploratory research!

The core design principle I've followed is to enable exploratory analysis - one of the most fun parts of mechanistic interpretability compared to normal ML is the extremely short feedback loops! The point of this library is to keep the gap between having an experiment idea and seeing the results as small as possible, to make it easy for **research to feel like play** and to enter a flow state. This notebook demonstrates how the library works and how to use it, but if you want to see how well it works for exploratory research, check out [my notebook analysing Indirect Objection Identification](TODO: link) or [my recording of myself doing research](https://www.youtube.com/watch?v=yo4QvDn-vsU)!

## Loading and Running Models

TransformerLens comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. For this demo notebook we'll look at GPT-2 Small, an 80M parameter model, see the Available Models section for info on the rest.

```python
device = "cuda" if t.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained("gpt2-small", device=device)
```

To try the model the model out, let's find the loss on this text! Models can be run on a single string or a tensor of tokens (shape: `[batch, position]`, all integers), and the possible return types are: 
* `"logits"` (shape [batch, position, d_vocab], floats), 
* `"loss"` (the cross-entropy loss when predicting the next token), 
* `"both"` (a tuple of (logits, loss)) 
* `None` (run the model, but don't calculate the logits - this is faster when we only want to use intermediate activations)

```python
model_description_text = '''## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. See [model_details.md](TODO: link) for a description of all supported models. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!'''
loss = model(model_description_text, return_type="loss")
print("Model loss:", loss)
```

## Caching all Activations

The first basic operation when doing mechanistic interpretability is to break open the black box of the model and look at all of the internal activations of a model. This can be done with `logits, cache = model.run_with_cache(tokens)`. Let's try this out on the first line of the abstract of the GPT-2 paper.""")

    with st.expander("On `remove_batch_dim`"):
        st.markdown(r"""
Every activation inside the model begins with a batch dimension. Here, because we only entered a single batch dimension, that dimension is always length 1 and kinda annoying, so passing in the `remove_batch_dim=True` keyword removes it. `gpt2_cache_no_batch_dim = gpt2_cache.remove_batch_dim()` would have achieved the same effect.
""")

    st.markdown(r"""
```python
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = model.to_tokens(gpt2_text)
print(gpt2_tokens.device)
gpt2_logits, gpt2_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True)
```
Let's visualize the attention pattern of all the heads in layer 0, using [Alan Cooney's CircuitsVis library](https://github.com/alan-cooney/CircuitsVis) (based on [Anthropic's PySvelte library](TODO)). 

We look this the attention pattern in `gpt2_cache`, an `ActivationCache` object, by entering in the name of the activation, followed by the layer index (here, the activation is called "attn" and the layer index is 0). This has shape [head_index, destination_position, source_position], and we use the `model.to_str_tokens` method to convert the text to a list of tokens as strings, since there is an attention weight between each pair of tokens.

This visualization is interactive! Try hovering over a token or head, and click to lock. The grid on the top left and for each head is the attention pattern as a destination position by source position grid. It's lower triangular because GPT-2 has **causal attention**, attention can only look backwards, so information can only move forwards in the network.

See the ActivationCache section for more on what `gpt2_cache` can do (TODO link)

```python
print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0, "attn"]
print(attention_pattern.shape)
gpt2_str_tokens = model.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
cv.attention.attention_heads(tokens=gpt2_str_tokens, attention=attention_pattern)
```""")
    st.info(r"""
Note - this library is currently under development (the function `attention_heads` was uploaded on 11th December!), and I think it's still a bit buggy. These plots might work in Colab or Notebooks, but they might not work in the VSCode python interpreter.

The easiest way to solve this (if you still want to use a Python file) is replace the code above with code writing it to an HTML file:

```
html = cv.attention.attention_heads(tokens=gpt2_str_tokens, attention=attention_pattern)
with open("cv_attn_2.html", "w") as f:
    f.write(str(html))
```

Then the file should pop up in your explorer on the left of VSCode. Right click on it and select "Open in Default Browser" to view it in your browser. If you're on a mac, you might need to do this last part from your file explorer, because there doesn't seem to be a "Open in Default Browser" option.
""")
    # with open("images/cv_attn.html") as f:
    #     text = f.read()
    # st.components.v1.html(text, height=400)
    with open("images/cv_attn_2.html") as f:
        text = f.read()
    st.components.v1.html(text, height=1400)

    st.info(r"""
Second note - this graphic was produced by the function `cv.attention.attention_heads`. You can also produce a slightly different graphic with `cv.attention.attention_pattern` (same arguments `tokens` and `attention`), which presents basically the same information in a slightly different way, shown below:
""")
    with open("images/attn_patterns_2.html") as f:
        text = f.read()
    st.components.v1.html(text, height=400)

    st.markdown(r"""
## Hooks: Intervening on Activations

One of the great things about interpreting neural networks is that we have *full control* over our system. From a computational perspective, we know exactly what operations are going on inside (even if we don't know what they mean!). And we can make precise, surgical edits and see how the model's behaviour and other internals change. This is an extremely powerful tool, because it can let us eg set up careful counterfactuals and causal intervention to easily understand model behaviour. 

Accordingly, being able to do this is a pretty core operation, and this is one of the main things TransformerLens supports! The key feature here is **hook points**. Every activation inside the transformer is surrounded by a hook point, which allows us to edit or intervene on it. 

We do this by adding a **hook function** to that activation. The hook function maps `current_activation_value, hook_point` to `new_activation_value`. As the model is run, it computes that activation as normal, and then the hook function is applied to compute a replacement, and that is substituted in for the activation. The hook function can be an arbitrary Python function, so long as it returns a tensor of the correct shape.""")

    with st.expander("Relationship to PyTorch hooks"):
        st.markdown(r"""
[PyTorch hooks](https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/) are a great and underrated, yet incredibly janky, feature. They can act on a layer, and edit the input or output of that layer, or the gradient when applying autodiff. The key difference is that **Hook points** act on *activations* not layers. This means that you can intervene within a layer on each activation, and don't need to care about the precise layer structure of the transformer. And it's immediately clear exactly how the hook's effect is applied. This adjustment was shamelessly inspired by [Garcon's use of ProbePoints](https://transformer-circuits.pub/2021/garcon/index.html).

They also come with a range of other quality of life improvements, like the model having a `model.reset_hooks()` method to remove all hooks, or helper methods to temporarily add hooks for a single forward pass - it is *incredibly* easy to shoot yourself in the foot with standard PyTorch hooks!""")

    st.markdown(r"""
As a basic example, let's ablate head 7 in layer 0 on the text above. 

We define a `head_ablation_hook` function. This takes the value tensor for attention layer 0, and sets the component with `head_index==7` to zero and returns it (Note - we return by convention, but since we're editing the activation in-place, we don't strictly *need* to).

We then use the `run_with_hooks` helper function to run the model and *temporarily* add in the hook for just this run. We enter in the hook as a tuple of the activation name (also the hook point name - found with `utils.get_act_name`) and the hook function.

```python
layer_to_ablate = 0
head_index_to_ablate = 8

# We define a head ablation hook
# The type annotations are NOT necessary, they're just a useful guide to the reader
def head_ablation_hook(
    value: TT["batch", "pos", "head_index", "d_head"],
    hook: HookPoint
) -> TT["batch", "pos", "head_index", "d_head"]:
    print(f"Shape of the value tensor: {value.shape}")
    value[:, :, head_index_to_ablate, :] = 0.
    return value

original_loss = model(gpt2_tokens, return_type="loss")
ablated_loss = model.run_with_hooks(
    gpt2_tokens, 
    return_type="loss", 
    fwd_hooks=[(
        utils.get_act_name("v", layer_to_ablate), 
        head_ablation_hook
    )]
)
print(f"Original Loss: {original_loss.item():.3f}")
print(f"Ablated Loss: {ablated_loss.item():.3f}")
```

**Gotcha:** Hooks are global state - they're added in as part of the model, and stay there until removed. `run_with_hooks` tries to create an abstraction where these are local state, by removing all hooks at the end of the function. But you can easily shoot yourself in the foot if there's, eg, an error in one of your hooks so the function never finishes. If you start getting bugs, try `model.reset_hooks()` to clean things up. Further, if you *do* add hooks of your own that you want to keep, which you can do with `add_hook` on the relevant 

### Activation Patching on the Indirect Object Identification Task

For a somewhat more involved example, let's use hooks to apply **activation patching** on the **Indirect Object Identification** (IOI) task. 

The IOI task is the task of identifying that a sentence like "After John and Mary went to the store, Mary gave a bottle of milk to" with " John" rather than " Mary" (ie, finding the indirect object), and Redwood Research have [an excellent paper studying the underlying circuit in GPT-2 Small](https://arxiv.org/abs/2211.00593).

**Activation patching** is a technique from [Kevin Meng and David Bau's excellent ROME paper](https://rome.baulab.info/). The goal is to identify which model activations are important for completing a task. We do this by setting up a **clean prompt** and a **corrupted prompt** and a **metric** for performance on the task. We then pick a specific model activation, run the model on the corrupted prompt, but then *intervene* on that activation and patch in its value when run on the clean prompt. We then apply the metric, and see how much this patch has recovered the clean performance. 
(See [a more detailed explanation of activation patching here](https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/main/Exploratory_Analysis_Demo.ipynb#scrollTo=5nUXG6zqmd0f))

Here, our clean prompt is "After John and Mary went to the store, **Mary** gave a bottle of milk to", our corrupted prompt is "After John and Mary went to the store, **John** gave a bottle of milk to", and our metric is the difference between the correct logit (John) and the incorrect logit (Mary) on the final token. 

We see that the logit difference is significantly positive on the clean prompt, and significantly negative on the corrupted prompt, showing that the model is capable of doing the task!

**Exercise - before running the code below, think about what you expect the output to be and why.**
""")

    with st.expander("Output (and explanation)"):
        st.markdown(r"""
We expect the clean logit diff to be positive (because the model knows that `"Mary gave a bottle of milk to"` should be followed by `" John"`), and the corrupted logit diff to be negative (because the model knows that `"John gave a bottle of milk to"` should be followed by `" Mary"`).

This is indeed what we find:

```python
Clean logit difference: 4.276
Corrupted logit difference: -2.738
```""")


    st.markdown(r"""

```python
clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
corrupted_prompt = "After John and Mary went to the store, John gave a bottle of milk to"

clean_tokens = model.to_tokens(clean_prompt)
corrupted_tokens = model.to_tokens(corrupted_prompt)

def logits_to_logit_diff(logits, correct_answer=" John", incorrect_answer=" Mary"):
    # model.to_single_token maps a string value of a single token to the token index for that token
    # If the string is not a single token, it raises an error.
    correct_index = model.to_single_token(correct_answer)
    incorrect_index = model.to_single_token(incorrect_answer)
    return logits[0, -1, correct_index] - logits[0, -1, incorrect_index]

# We run on the clean prompt with the cache so we store activations to patch in later.
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
clean_logit_diff = logits_to_logit_diff(clean_logits)
print(f"Clean logit difference: {clean_logit_diff.item():.3f}")

# We don't need to cache on the corrupted prompt.
corrupted_logits = model(corrupted_tokens)
corrupted_logit_diff = logits_to_logit_diff(corrupted_logits)
print(f"Corrupted logit difference: {corrupted_logit_diff.item():.3f}")
```

We now setup the hook function to do **activation patching**. Here, we'll patch in the residual stream at the start of a specific layer and at a specific position. This will let us see how much the model is using the residual stream at that layer and position to represent the key information for the task. 

We want to iterate over all layers and positions, so we write the hook to take in an position parameter. Hook functions must have the input signature (activation, hook), but we can use `functools.partial` to set the position parameter before passing it to `run_with_hooks`.

```python
# We define a residual stream patching hook
# We choose to act on the residual stream at the start of the layer, so we call it resid_pre
# The type annotations are a guide to the reader and are not necessary
def residual_stream_patching_hook(
    resid_pre: TT["batch", "pos", "d_model"],
    hook: HookPoint,
    position: int
) -> TT["batch", "pos", "d_model"]:
    # Each HookPoint has a name attribute giving the name of the hook.
    clean_resid_pre = clean_cache[hook.name]
    resid_pre[:, position, :] = clean_resid_pre[:, position, :]
    return resid_pre

# We make a tensor to store the results for each patching run. We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
num_positions = len(clean_tokens[0])
ioi_patching_result = t.zeros((model.cfg.n_layers, num_positions), device=model.cfg.device)

for layer in tqdm.tqdm(range(model.cfg.n_layers)):
    for position in range(num_positions):
        # Use functools.partial to create a temporary hook function with the position fixed
        temp_hook_fn = partial(residual_stream_patching_hook, position=position)
        # Run the model with the patching hook
        patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[
            (utils.get_act_name("resid_pre", layer), temp_hook_fn)
        ])
        # Calculate the logit difference
        patched_logit_diff = logits_to_logit_diff(patched_logits).detach()
        # Store the result, normalizing by the clean and corrupted logit difference so it's between 0 and 1 (ish)
        ioi_patching_result[layer, position] = (patched_logit_diff - corrupted_logit_diff)/(clean_logit_diff - corrupted_logit_diff)
```

We can now visualize the results, and see that this computation is extremely localised within the model. Initially, the second subject (Mary) token is all that matters (naturally, as it's the only different token), and all relevant information remains here until heads in layer 7 and 8 move this to the final token where it's used to predict the indirect object.
(Note - the heads are in layer 7 and 8, not 8 and 9, because we patched in the residual stream at the *start* of each layer)

```python
# Add the index to the end of the label, because plotly doesn't like duplicate labels
token_labels = [f"{token}_{index}" for index, token in enumerate(model.to_str_tokens(clean_tokens))]
imshow(ioi_patching_result, x=token_labels, xaxis="Position", yaxis="Layer", title="Normalized Logit Difference After Patching Residual Stream on the IOI Task")
```""")

    st.plotly_chart(fig_dict["logit_diff_from_patching"], use_container_width=True)
    st.markdown(r"""

## Hooks: Accessing Activations

Hooks can also be used to just **access** an activation - to run some function using that activation value, *without* changing the activation value. This can be achieved by just having the hook return nothing, and not editing the activation in place. 

This is useful for eg extracting activations for a specific task, or for doing some long-running calculation across many inputs, eg finding the text that most activates a specific neuron. (Note - everything this can do *could* be done with `run_with_cache` and post-processing, but this workflow can be more intuitive and memory efficient.)

To demonstrate this, let's look for **[induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)** in GPT-2 Small. 

Induction circuits are a very important circuit in generative language models, which are used to detect and continue repeated subsequences. They consist of two heads in separate layers that compose together, a **previous token head** which always attends to the previous token, and an **induction head** which attends to the token *after* an earlier copy of the current token. 

To see why this is important, let's say that the model is trying to predict the next token in a news article about Michael Jordan. The token " Michael", in general, could be followed by many surnames. But an induction head will look from that occurence of " Michael" to the token after previous occurences of " Michael", ie " Jordan" and can confidently predict that that will come next.

An interesting fact about induction heads is that they generalise to arbitrary sequences of repeated tokens. We can see this by generating sequences of 50 random tokens, repeated twice, and plotting the average loss at predicting the next token, by position. We see that the model goes from terrible to very good at the halfway point.

```python
batch_size = 10
seq_len = 50
random_tokens = t.randint(1000, 10000, (batch_size, seq_len)).to(model.cfg.device)
repeated_tokens = repeat(random_tokens, "batch seq_len -> batch (2 seq_len)")
repeated_logits = model(repeated_tokens)
correct_log_probs = model.loss_fn(repeated_logits, repeated_tokens, per_token=True)
loss_by_position = reduce(correct_log_probs, "batch position -> position", "mean")
line(loss_by_position, xaxis="Position", yaxis="Loss", title="Loss by position on random repeated tokens")
```""")

    st.plotly_chart(fig_dict["line"], use_container_width=True)
    st.markdown(r"""

The induction heads will be attending from the second occurence of each token to the token *after* its first occurence, ie the token `50-1==49` places back. So by looking at the average attention paid 49 tokens back, we can identify induction heads! Let's define a hook to do this!""")

    with st.expander("Technical details"):
        st.markdown(r"""

* We attach the hook to the attention pattern activation. There's one big pattern activation per layer, stacked across all heads, so we need to do some tensor manipulation to get a per-head score. 
* Hook functions can access global state, so we make a big tensor to store the induction head score for each head, and then we just add the score for each head to the appropriate position in the tensor. 
* To get a single hook function that works for each layer, we use the `hook.layer()` method to get the layer index (internally this is just inferred from the hook names).
* As we want to add this to *every* activation pattern hook point, rather than giving the string for an activation name, this time we give a **name filter**. This is a Boolean function on hook point names, and it adds the hook function to every hook point where the function evaluates as true. 
    * `run_with_hooks` allows us to enter a list of (act_name, hook_function) pairs to all be added at once, so we could also have done this by inputting a list with a hook for each layer.
""")
        st.markdown("")

    st.markdown(r"""
```python
# We make a tensor to store the induction score for each head. We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
def induction_score_hook(
    pattern: TT["batch", "head_index", "dest_pos", "source_pos"],
    hook: HookPoint,
):
    # We take the diagonal of attention paid from each destination position to source positions seq_len-1 tokens back
    # (This only has entries for tokens with index>=seq_len)
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
    # Get an average score per head
    induction_score = reduce(induction_stripe, "batch head_index position -> head_index", "mean")
    # Store the result.
    induction_score_store[hook.layer(), :] = induction_score

# We make a boolean filter on activation names, that's true only on attention pattern names.
pattern_hook_names_filter = lambda name: name.endswith("pattern")

model.run_with_hooks(
    repeated_tokens, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        induction_score_hook
    )]
)

imshow(induction_score_store, xaxis="Head", yaxis="Layer", title="Induction Score by Head")
```""")

    st.plotly_chart(fig_dict["attn_induction_score"], use_container_width=True)
    st.markdown(r"""

Head 5 in Layer 5 scores extremely highly on this score, and we can feed in a shorter repeated random sequence, visualize the attention pattern for it and see this directly - including the "induction stripe" at `seq_len-1` tokens back.

This time we put in a hook on the attention pattern activation to visualize the pattern of the relevant head.

```python
induction_head_layer = 5
induction_head_index = 5
single_random_sequence = t.randint(1000, 10000, (1, 20)).to(model.cfg.device)
repeated_random_sequence = repeat(single_random_sequence, "batch seq_len -> batch (2 seq_len)")
def visualize_pattern_hook(
    pattern: TT["batch", "head_index", "dest_pos", "source_pos"],
    hook: HookPoint,
):
    display(
        cv.attention.attention_heads(
            tokens=model.to_str_tokens(repeated_random_sequence), 
            attention=pattern[0, induction_head_index, :, :][None, :, :] # Add a dummy axis, as CircuitsVis expects 3D patterns.
        )
    )

model.run_with_hooks(
    repeated_random_sequence, 
    return_type=None, 
    fwd_hooks=[(
        utils.get_act_name("pattern", induction_head_layer), 
        visualize_pattern_hook
    )]
)
```""")
    with open("images/attn_3.html") as f:
        text2 = f.read()
    st.components.v1.html(text2, height=1400)
    st.markdown(r"""

## Available Models

TransformerLens comes with over 40 open source models available, all of which can be loaded into a consistent(-ish) architecture by just changing the name in `from_pretrained`. You can see [a table of the available models here](https://github.com/neelnanda-io/TransformerLens/blob/main/easy_transformer/model_properties_table.md), including the hyper-parameters and the alias used to load the model in TransformerLens.

**Note:** Though TransformerLens can load in some large transformers (eg OPT-66B), it does not currently support loading a single model across multiple GPUs, so in practice loading a model (eg >7B parameters) will be impractical, depending on your GPU memory. Feel free to reach out if you need to use larger models for your research.

Notably, this means that analysis can be near immediately re-run on a different model by just changing the name - to see this, let's load in DistilGPT-2 (a distilled version of GPT-2, with half as many layers) and copy the code from above to see the induction heads in that model.

```python
distilgpt2 = HookedTransformer.from_pretrained("distilgpt2")
# We make a tensor to store the induction score for each head. We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
distilgpt2_induction_score_store = t.zeros((distilgpt2.cfg.n_layers, distilgpt2.cfg.n_heads), device=distilgpt2.cfg.device)
def induction_score_hook(
    pattern: TT["batch", "head_index", "dest_pos", "source_pos"],
    hook: HookPoint,
):
    # We take the diagonal of attention paid from each destination position to source positions seq_len-1 tokens back
    # (This only has entries for tokens with index>=seq_len)
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
    # Get an average score per head
    induction_score = reduce(induction_stripe, "batch head_index position -> head_index", "mean")
    # Store the result.
    distilgpt2_induction_score_store[hook.layer(), :] = induction_score

# We make a boolean filter on activation names, that's true only on attention pattern names.
pattern_hook_names_filter = lambda name: name.endswith("pattern")

distilgpt2.run_with_hooks(
    repeated_tokens, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        induction_score_hook
    )]
)

imshow(distilgpt2_induction_score_store, xaxis="Head", yaxis="Layer", title="Induction Score by Head in Distil GPT-2")
```""")
    st.plotly_chart(fig_dict["distil_plot"], use_container_width=True)
    st.markdown(r"""
### An overview of the important open source models in the library

* **GPT-2** - the classic generative pre-trained models from OpenAI
    * Sizes Small (85M), Medium (300M), Large (700M) and XL (1.5B).
    * Trained on ~22B tokens of internet text. ([Open source replication](https://huggingface.co/datasets/openwebtext))
* **GPT-Neo** - Eleuther's replication of GPT-2
    * Sizes 125M, 1.3B, 2.7B
    * Trained on 300B(ish?) tokens of [the Pile](https://pile.eleuther.ai/) a large and diverse dataset including a bunch of code (and weird stuff)
* **[OPT](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/)** - Meta AI's series of open source models
    * Trained on 180B tokens of diverse text.
    * 125M, 1.3B, 2.7B, 6.7B, 13B, 30B, 66B
* **GPT-J** - Eleuther's 6B parameter model, trained on the Pile
* **GPT-NeoX** - Eleuther's 20B parameter model, trained on the Pile
* **Stanford CRFM models** - a replication of GPT-2 Small and GPT-2 Medium, trained on 5 different random seeds.
    * Notably, 600 checkpoints were taken during training per model, and these are available in the library with eg `HookedTransformer.from_pretrained("stanford-gpt2-small-a", checkpoint_index=265)`.


### An overview of some interpretability-friendly models I've trained and included

(Feel free to [reach out](mailto:neelnanda27@gmail.com) if you want more details on any of these models)

Each of these models has about ~200 checkpoints taken during training that can also be loaded from TransformerLens, with the `checkpoint_index` argument to `from_pretrained`.

Note that all models are trained with a Beginning of Sequence token, and will likely break if given inputs without that! 

* **Toy Models**: Inspired by [A Mathematical Framework](https://transformer-circuits.pub/2021/framework/index.html), I've trained 12 tiny language models, of 1-4L and each of width 512. I think that interpreting these is likely to be far more tractable than larger models, and both serve as good practice and will likely contain motifs and circuits that generalise to far larger models (like induction heads):
    * Attention-Only models (ie without MLPs): attn-only-1l, attn-only-2l, attn-only-3l, attn-only-4l
    * GELU models (ie with MLP, and the standard GELU activations): gelu-1l, gelu-2l, gelu-3l, gelu-4l
    * SoLU models (ie with MLP, and [Anthropic's SoLU activation](https://transformer-circuits.pub/2022/solu/index.html), designed to make MLP neurons more interpretable): solu-1l, solu-2l, solu-3l, solu-4l
    * All models are trained on 22B tokens of data, 80% from C4 (web text) and 20% from Python Code
    * Models of the same layer size were trained with the same weight initialization and data shuffle, to more directly compare the effect of different activation functions.
* **SoLU** models: A larger scan of models trained with [Anthropic's SoLU activation](https://transformer-circuits.pub/2022/solu/index.html), in the hopes that it makes the MLP neuron interpretability easier. 
    * A scan up to GPT-2 Medium size, trained on 30B tokens of the same data as toy models, 80% from C4 and 20% from Python code. 
        * solu-6l (40M), solu-8l (100M), solu-10l (200M), solu-12l (340M)
    * An older scan up to GPT-2 Medium size, trained on 15B tokens of [the Pile](https://pile.eleuther.ai/)
        * solu-1l-pile (13M), solu-2l-pile (13M), solu-4l-pile (13M), solu-6l-pile (40M), solu-8l-pile (100M), solu-10l-pile (200M), solu-12l-pile (340M)

## Other Resources:

* [Concrete Open Problems in Mechanistic Interpretability](https://docs.google.com/document/d/1WONBzNqfKIxERejrrPlQMyKqg7jSFW92x5UMXNrMdPo/edit), a doc I wrote giving a long list of open problems in mechanistic interpretability, and thoughts on how to get started on trying to work on them. 
    * There's a lot of low-hanging fruit in the field, and I expect that many people reading this could use TransformerLens to usefully make progress on some of these!
* Other demos:
    * **[Exploratory Analysis Demo](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/Exploratory_Analysis_Demo.ipynb)**, a demonstration of my standard toolkit for how to use TransformerLens to explore a mysterious behaviour in a language model.
    * [Interpretability in the Wild](https://github.com/redwoodresearch/Easy-Transformer) a codebase from Arthur Conmy and Alex Variengien at Redwood research using this library to do a detailed and rigorous reverse engineering of the Indirect Object Identification circuit, to accompany their paper
        * Note - this was based on an earlier version of this library, called EasyTransformer. It's pretty similar, but several breaking changes have been made since. 
    * A [recorded walkthrough](https://www.youtube.com/watch?v=yo4QvDn-vsU) of me doing research with TransformerLens on whether a tiny model can re-derive positional information, with [an accompanying Colab](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/No_Position_Experiment.ipynb)
* [Neuroscope](https://neuroscope.io), a website showing the text in the dataset that most activates each neuron in some selected models. Good to explore to get a sense for what kind of features the model tends to represent, and as a "wiki" to get some info
    * A tutorial on how to make an [Interactive Neuroscope](https://github.com/neelnanda-io/TransformerLens/blob/main/Hacky-Interactive-Lexoscope.ipynb), where you type in text and see the neuron activations over the text update live.

## Transformer architecture

HookedTransformer is a somewhat adapted GPT-2 architecture, but is computationally identical. The most significant changes are to the internal structure of the attention heads: 
* The weights (W_K, W_Q, W_V) mapping the residual stream to queries, keys and values are 3 separate matrices, rather than big concatenated one.
* The weight matrices (W_K, W_Q, W_V, W_O) and activations (keys, queries, values, z (values mixed by attention pattern)) have separate head_index and d_head axes, rather than flattening them into one big axis.
    * The activations all have shape `[batch, position, head_index, d_head]`
    * W_K, W_Q, W_V have shape `[head_index, d_head, d_model]` and W_O has shape `[head_index, d_model, d_head]`

The actual code is a bit of a mess, as there's a variety of Boolean flags to make it consistent with the various different model families in TransformerLens - to understand it and the internal structure, I instead recommend reading the code in [CleanTransformerDemo](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/clean-transformer-demo/Clean_Transformer_Demo.ipynb)

### Parameter Names

Here is a list of the parameters and shapes in the model. By convention, all weight matrices multiply on the right (ie `new_activation = old_activation @ weights + bias`). 

Reminder of the key hyper-params:
* `n_layers`: 12. The number of transformer blocks in the model (a block contains an attention layer and an MLP layer)
* `n_heads`: 12. The number of attention heads per attention layer
* `d_model`: 768. The residual stream width.
* `d_head`: 64. The internal dimension of an attention head activation.
* `d_mlp`: 3072. The internal dimension of the MLP layers (ie the number of neurons).
* `d_vocab`: 50267. The number of tokens in the vocabulary.
* `n_ctx`: 1024. The maximum number of tokens in an input prompt.

**Transformer Block parameters:** 
Replace 0 with the relevant layer index.

```python
for name, param in model.named_parameters():
    if name.startswith("blocks.0."):
        print(name, param.shape)
```

**Embedding & Unembedding parameters:**

```python
for name, param in model.named_parameters():
    if not name.startswith("blocks"):
        print(name, param.shape)
```

### Activation + Hook Names

Lets get out a list of the activation/hook names in the model and their shapes. In practice, I recommend using the `utils.get_act_name` function to get the names, but this is a useful fallback, and necessary to eg write a name filter function.

Let's do this by entering in a short, 10 token prompt, and add a hook function to each activations to print its name and shape. To avoid spam, let's just add this to activations in the first block or not in a block.

Note 1: Each LayerNorm has a hook for the scale factor (ie the standard deviation of the input activations for each token position & batch element) and for the normalized output (ie the input activation with mean 0 and standard deviation 1, but *before* applying scaling or translating with learned weights). LayerNorm is applied every time a layer reads from the residual stream: `ln1` is the LayerNorm before the attention layer in a block, `ln2` the one before the MLP layer, and `ln_final` is the LayerNorm before the unembed. 

Note 2: *Every* activation apart from the attention pattern and attention scores has shape beginning with `[batch, position]`. The attention pattern and scores have shape `[batch, head_index, dest_position, source_position]` (the numbers are the same, unless we're using caching).

```python
test_prompt = "The quick brown fox jumped over the lazy dog"
print("Num tokens:", len(model.to_tokens(test_prompt)))

def print_name_shape_hook_function(activation, hook):
    print(hook.name, activation.shape)

not_in_late_block_filter = lambda name: name.startswith("blocks.0.") or not name.startswith("blocks")

model.run_with_hooks(
    test_prompt,
    return_type=None,
    fwd_hooks=[(not_in_late_block_filter, print_name_shape_hook_function)],
)
```

### Folding LayerNorm (For the Curious)

(For the curious - this is an important technical detail that's worth understanding, especially if you have preconceptions about how transformers work, but not necessary to use TransformerLens)

LayerNorm is a normalization technique used by transformers, analogous to BatchNorm but more friendly to massive parallelisation. No one *really* knows why it works, but it seems to improve model numerical stability. Unlike BatchNorm, LayerNorm actually changes the functional form of the model, which makes it a massive pain for interpretability! 

Folding LayerNorm is a technique to make it lower overhead to deal with, and the flags `center_writing_weights` and `fold_ln` in `HookedTransformer.from_pretrained` apply this automatically (they default to True). These simplify the internal structure without changing the weights.

Intuitively, LayerNorm acts on each residual stream vector (ie for each batch element and token position) independently, sets their mean to 0 (centering) and standard deviation to 1 (normalizing) (*across* the residual stream dimension - very weird!), and then applies a learned elementwise scaling and translation to each vector.

Mathematically, centering is a linear map, normalizing is *not* a linear map, and scaling and translation are linear maps. 
* **Centering:** LayerNorm is applied every time a layer reads from the residual stream, so the mean of any residual stream vector can never matter - `center_writing_weights` set every weight matrix writing to the residual to have zero mean. 
* **Normalizing:** Normalizing is not a linear map, and cannot be factored out. The `hook_scale` hook point lets you access and control for this.
* **Scaling and Translation:** Scaling and translation are linear maps, and are always followed by another linear map. The composition of two linear maps is another linear map, so we can *fold* the scaling and translation weights into the weights of the subsequent layer, and simplify things without changing the underlying computation. 

[See the docs for more details](https://github.com/neelnanda-io/TransformerLens/blob/main/further_comments.md#what-is-layernorm-folding-fold_ln)

A fun consequence of LayerNorm folding is that it creates a bias across the unembed, a `d_vocab` length vector that is added to the output logits - GPT-2 is not trained with this, but it *is* trained with a final LayerNorm that contains a bias. 

Turns out, this LayerNorm bias learns structure of the data that we can only see after folding! In particular, it essentially learns **unigram statistics** - rare tokens get suppressed, common tokens get boosted, by pretty dramatic degrees! Let's list the top and bottom 20 - at the top we see common punctuation and words like " the" and " and", at the bottom we see weird-ass tokens like " RandomRedditor":

```python
unembed_bias = model.unembed.b_U
bias_values, bias_indices = unembed_bias.sort(descending=True)
```

```python
top_k = 20
print(f"Top {top_k} values")
for i in range(top_k):
    print(f"{bias_values[i].item():.2f} {repr(model.to_string(bias_indices[i]))}")

print("...")
print(f"Bottom {top_k} values")
for i in range(top_k, 0, -1):
    print(f"{bias_values[-i].item():.2f} {repr(model.to_string(bias_indices[-i]))}")
```

This can have real consequences for interpretability - for example, this bias favours " John" over " Mary" by about 1.2, about 1/3 of the effect size of the Indirect Object Identification Circuit! All other things being the same, this makes the John token 3.6x times more likely than the Mary token.

```python
john_bias = model.unembed.b_U[model.to_single_token(' John')]
mary_bias = model.unembed.b_U[model.to_single_token(' Mary')]

print(f"John bias: {john_bias.item():.4f}")
print(f"Mary bias: {mary_bias.item():.4f}")
print(f"Prob ratio bias: {t.exp(john_bias - mary_bias).item():.4f}x")
```
""")

def section_2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#dealing-with-tokens">Dealing with tokens</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#gotcha-prepend-bos">Gotcha: <code>prepend_bos</code></a></li>
   </ul></li>
   <li><a class="contents-el" href="#factored-matrix-class">Factored Matrix Class</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#basic-examples">Basic Examples</a></li>
       <li><a class="contents-el" href="#medium-example-eigenvalue-copying-scores">Medium Example: Eigenvalue Copying Scores</a></li>
   </ul></li>
   <li><a class="contents-el" href="#generating-text">Generating Text</a></li>
   <li><a class="contents-el" href="#hook-points">Hook Points</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#toy-example">Toy Example</a></li>
   </ul></li>
   <li><a class="contents-el" href="#loading-pre-trained-checkpoints">Loading Pre-Trained Checkpoints</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#example-induction-head-phase-transition">Example: Induction Head Phase Transition</a></li>
    </ul></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Features""")

    st.error("Note - this section hasn't yet been converted to Streamlit, and there's some editing that still needs to be done.")
    st.markdown(r"""

An overview of some other important features of the library. I recommend checking out the [Exploratory Analysis Demo](https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/main/Exploratory_Analysis_Demo.ipynb) for some other important features not mentioned here, and for a demo of what using the library in practice looks like.

## Dealing with tokens

**Tokenization** is one of the most annoying features of studying language models. We want language models to be able to take in arbitrary text as input, but the transformer architecture needs the inputs to be elements of a fixed, finite vocabulary. The solution to this is **tokens**, a fixed vocabulary of "sub-words", that any natural language can be broken down into with a **tokenizer**. This is invertible, and we can recover the original text, called **de-tokenization**. 

TransformerLens comes with a range of utility functions to deal with tokenization. Different models can have different tokenizers, so these are all methods on the model.

`get_token_position`, `to_tokens`, `to_string`, `to_str_tokens`, `prepend_bos`, `to_single_token`

The first thing you need to figure out is *how* things are tokenized. `model.to_str_tokens` splits a string into the tokens *as a list of substrings*, and so lets you explore what the text looks like. To demonstrate this, let's use it on this paragraph.

Some observations - there are a lot of arbitrary-ish details in here!
* The tokenizer splits on spaces, so no token contains two words.
* Tokens include the preceding space, and whether the first token is a capital letter. `'how'` and `' how'` are different tokens!
* Common words are single tokens, even if fairly long (` paragraph`) while uncommon words are split into multiple tokens (` token|ized`).
* Tokens *mostly* split on punctuation characters (eg `*` and `.`), but eg `'s` is a single token.

```python
example_text = "The first thing you need to figure out is *how* things are tokenized. `model.to_str_tokens` splits a string into the tokens *as a list of substrings*, and so lets you explore what the text looks like. To demonstrate this, let's use it on this paragraph."
example_text_str_tokens = model.to_str_tokens(example_text)
print(example_text_str_tokens)
```

The transformer needs to take in a sequence of integers, not strings, so we need to convert these tokens into integers. `model.to_tokens` does this, and returns a tensor of integers on the model's device (shape `[batch, position]`). It maps a string to a batch of size 1.

```python
example_text_tokens = model.to_tokens(example_text)
print(example_text_tokens)
```

`to_tokens` can also take in a list of strings, and return a batch of size `len(strings)`. If the strings are different numbers of tokens, it adds a PAD token to the end of the shorter strings to make them the same length.

(Note: In GPT-2, 50256 signifies both the beginning of sequence, end of sequence and padding token - see the `prepend_bos` section for details)

```python
example_multi_text = ["The cat sat on the mat.", "The cat sat on the mat really hard."]
example_multi_text_tokens = model.to_tokens(example_multi_text)
print(example_multi_text_tokens)
```

`model.to_single_token` is a convenience function that takes in a string 

*   List item
*   List item

corresponding to a *single* token and returns the corresponding integer. This is useful for eg looking up the logit corresponding to a single token. 

For example, let's input `The cat sat on the mat.` to GPT-2, and look at the log prob predicting that the next token is ` The`. 
""")
    with st.expander("Technical notes"):
        st.markdown(r"""
Note that if we input a string to the model, it's implicitly converted to a string with `to_tokens`. 

Note further that the log probs have shape `[batch, position, d_vocab]==[1, 8, 50257]`, with a vector of log probs predicting the next token for *every* token position. GPT-2 uses causal attention which means heads can only look backwards (equivalently, information can only move forwards in the model.), so the log probs at position k are only a function of the first k tokens, and it can't just cheat and look at the k+1 th token. This structure lets it generate text more efficiently, and lets it treat every *token* as a training example, rather than every *sequence*.""")

    st.markdown(r"""
```python
cat_text = "The cat sat on the mat."
cat_logits = model(cat_text)
cat_probs = cat_logits.softmax(dim=-1)
print(f"Probability tensor shape [batch, position, d_vocab] == {cat_probs.shape}")

capital_the_token_index = model.to_single_token(" The")
print(f"| The| probability: {cat_probs[0, -1, capital_the_token_index].item():.2%}")
```

`model.to_string` is the inverse of `to_tokens` and maps a tensor of integers to a string or list of strings. It also works on integers and lists of integers.

For example, let's look up token 256 (due to technical details of tokenization, this will be the most common pair of ASCII characters!), and also verify that our tokens above map back to a string.

```python
print(f"Token 256 - the most common pair of ASCII characters: |{model.to_string(256)}|")
# Squeeze means to remove dimensions of length 1. 
# Here, that removes the dummy batch dimension so it's a rank 1 tensor and returns a string
# Rank 2 tensors map to a list of strings
print(f"De-Tokenizing the example tokens: {model.to_string(example_text_tokens.squeeze())}")
```

A related annoyance of tokenization is that it's hard to figure out how many tokens a string will break into. `model.get_token_position(single_token, tokens)` returns the position of `single_token` in `tokens`. `tokens` can be either a string or a tensor of tokens. 

Note that position is zero-indexed, it's two (ie third) because there's a beginning of sequence token automatically prepended (see the next section for details)

```python
print("With BOS:", model.get_token_position(" cat", "The cat sat on the mat"))
print("Without BOS:", model.get_token_position(" cat", "The cat sat on the mat", prepend_bos=False))
```

If there are multiple copies of the token, we can set `mode="first"` to find the first occurence's position and `mode="last"` to find the last

```python
print("First occurence", model.get_token_position(
    " cat", 
    "The cat sat on the mat. The mat sat on the cat.", 
    mode="first"))
print("Final occurence", model.get_token_position(
    " cat", 
    "The cat sat on the mat. The mat sat on the cat.", 
    mode="last"))
```

In general, tokenization is a pain, and full of gotchas. I highly recommend just playing around with different inputs and their tokenization and getting a feel for it. As another "fun" example, let's look at the tokenization of arithmetic expressions - tokens do *not* contain consistent numbers of digits. (This makes it even more impressive that GPT-3 can do arithmetic!)

```python
print(model.to_str_tokens("2342+2017=21445"))
print(model.to_str_tokens("1000+1000000=999999"))
```

I also *highly* recommend investigating prompts with easy tokenization when starting out - ideally key words should form a single token, be in the same position in different prompts, have the same total length, etc. Eg study Indirect Object Identification with common English names like ` Tim` rather than ` Ne|el`. Transformers need to spend some parameters in early layers converting multi-token words to a single feature, and then de-converting this in the late layers, and unless this is what you're explicitly investigating, this will make the behaviour you're investigating be messier.

### Gotcha: `prepend_bos`

##### Key Takeaway: **If you get weird off-by-one errors, check whether there's an unexpected `prepend_bos`!**

A weirdness you may have noticed in the above is that `to_tokens` and `to_str_tokens` added a weird `<|endoftext|>` to the start of each prompt. TransformerLens does this by default, and it can easily trip up new users. Notably, **this includes `model.forward`** (which is what's implicitly used when you do eg `model("Hello World")`). This is called a **Beginning of Sequence (BOS)** token, and it's a special token used to mark the beginning of the sequence. Confusingly, in GPT-2, the End of Sequence (EOS), Beginning of Sequence (BOS) and Padding (PAD) tokens are all the same, `<|endoftext|>` with index `50256`.

You can disable this behaviour by setting the flag `prepend_bos=False` in `to_tokens`, `to_str_tokens`, `model.forward` and any other function that converts strings to multi-token tensors. 

**Gotcha:** You only want to do this at the *start* of a prompt. If you, eg, want to input a question followed by an answer, and want to tokenize these separately, you do *not* want to prepend_bos on the answer.

```python
print("Logits shape by default (with BOS)", model("Hello World").shape)
print("Logits shape with BOS", model("Hello World", prepend_bos=True).shape)
print("Logits shape without BOS - only 2 positions!", model("Hello World", prepend_bos=False).shape)
```

`prepend_bos` is a bit of a hack, and I've gone back and forth on what the correct default here is. The reason I do this is that transformers tend to treat the first token weirdly - this doesn't really matter in training (where all inputs are >1000 tokens), but this can be a big issue when investigating short prompts! The reason for this is that attention patterns are a probability distribution and so need to add up to one, so to simulate being "off" they normally look at the first token. Giving them a BOS token lets the heads rest by looking at that, preserving the information in the first "real" token.

Further, *some* models are trained to need a BOS token (OPT and my interpretability-friendly models are, GPT-2 and GPT-Neo are not). But despite GPT-2 not being trained with this, empirically it seems to make interpretability easier.

For example, the model can get much worse at Indirect Object Identification without a BOS (and with a name as the first token):

```python
ioi_logits_with_bos = model("Claire and Mary went to the shops, then Mary gave a bottle of milk to", prepend_bos=True)
mary_logit_with_bos = ioi_logits_with_bos[0, -1, model.to_single_token(" Mary")].item()
claire_logit_with_bos = ioi_logits_with_bos[0, -1, model.to_single_token(" Claire")].item()
print(f"Logit difference with BOS: {(claire_logit_with_bos - mary_logit_with_bos):.3f}")

ioi_logits_without_bos = model("Claire and Mary went to the shops, then Mary gave a bottle of milk to", prepend_bos=False)
mary_logit_without_bos = ioi_logits_without_bos[0, -1, model.to_single_token(" Mary")].item()
claire_logit_without_bos = ioi_logits_without_bos[0, -1, model.to_single_token(" Claire")].item()
print(f"Logit difference without BOS: {(claire_logit_without_bos - mary_logit_without_bos):.3f}")
```

Though, note that this also illustrates another gotcha - when `Claire` is at the start of a sentence (no preceding space), it's actually *two* tokens, not one, which probably confuses the relevant circuit. (Note - in this test we put `prepend_bos=False`, because we want to analyse the tokenization of a specific string, not to give an input to the model!)

```
print(f"| Claire| -> {model.to_str_tokens(' Claire', prepend_bos=False)}")
print(f"|Claire| -> {model.to_str_tokens('Claire', prepend_bos=False)}")
```

## Factored Matrix Class

In transformer interpretability, we often need to analyse low rank factorized matrices - a matrix $M = AB$, where M is `[large, large]`, but A is `[large, small]` and B is `[small, large]`. This is a common structure in transformers, and the `FactoredMatrix` class is a convenient way to work with these. It implements efficient algorithms for various operations on these, such as computing the trace, eigenvalues, Frobenius norm, singular value decomposition, and products with other matrices. It can (approximately) act as a drop-in replacement for the original matrix, and supports leading batch dimensions to the factored matrix. """)

    with st.expander("Why are low-rank factorized matrices useful for transformer interpretability?"):
        st.markdown(r"""
As argued in [A Mathematical Framework](https://transformer-circuits.pub/2021/framework/index.html), an unexpected fact about transformer attention heads is that rather than being best understood as keys, queries and values (and the requisite weight matrices), they're actually best understood as two low rank factorized matrices. 

* **Where to move information from:** $W_QK = W_Q W_K^T$, used for determining the attention pattern - what source positions to move information from and what destination positions to move them to.
    * Intuitively, residual stream -> query and residual stream -> key are linear maps, *and* `attention_score = query @ key.T` is a linear map, so the whole thing can be factored into one big bilinear form `residual @ W_QK @ residual.T`
* **What information to move:** $W_{OV} = W_V W_O$, used to determine what information to copy from the source position to the destination position (weighted by the attention pattern weight from that destination to that source). 
    * Intuitively, the residual stream is a `[position, d_model]` tensor (ignoring batch). The attention pattern acts on the *position* dimension (where to move information from and to) and the value and output weights act on the *d_model* dimension - ie *what* information is contained at that source position. So we can factor it all into `attention_pattern @ residual @ W_V @ W_O`, and so only need to care about `W_OV = W_V @ W_O`

Note - the internal head dimension is smaller than the residual stream dimension, so the factorization is low rank. (here, `d_model=768` and `d_head=64`)
""")

    st.markdown(r"""

### Basic Examples

We can use the basic class directly - let's make a factored matrix directly and look at the basic operations:

```python
A = t.randn(5, 2)
B = t.randn(2, 5)
AB = A @ B
AB_factor = FactoredMatrix(A, B)
print("Norms:")
print(AB.norm())
print(AB_factor.norm())

print(f"Right dimension: {AB_factor.rdim}, Left dimension: {AB_factor.ldim}, Hidden dimension: {AB_factor.mdim}")
```

We can also look at the eigenvalues and singular values of the matrix. Note that, because the matrix is rank 2 but 5 by 5, the final 3 eigenvalues and singular values are zero - the factored class omits the zeros.

```python
print("Eigenvalues:")
print(t.linalg.eig(AB).eigenvalues)
print(AB_factor.eigenvalues)
print()
print("Singular Values:")
print(t.linalg.svd(AB).S)
print(AB_factor.S)
```

We can multiply with other matrices - it automatically chooses the smallest possible dimension to factor along (here it's 2, rather than 5)

```python
C = t.randn(5, 300)
ABC = AB @ C
ABC_factor = AB_factor @ C
print("Unfactored:", ABC.shape, ABC.norm())
print("Factored:", ABC_factor.shape, ABC_factor.norm())
print(f"Right dimension: {ABC_factor.rdim}, Left dimension: {ABC_factor.ldim}, Hidden dimension: {ABC_factor.mdim}")
```

If we want to collapse this back to an unfactored matrix, we can use the AB property to get the product:

```python
AB_unfactored = AB_factor.AB
print(t.isclose(AB_unfactored, AB).all())
```

### Medium Example: Eigenvalue Copying Scores

(This is a more involved example of how to use the factored matrix class, skip it if you aren't following)

For a more involved example, let's look at the eigenvalue copying score from [A Mathematical Framework](https://transformer-circuits.pub/2021/framework/index.html) of the OV circuit for various heads. The OV Circuit for a head (the factorised matrix $W_OV = W_V W_O$) is a linear map that determines what information is moved from the source position to the destination position. Because this is low rank, it can be thought of as *reading in* some low rank subspace of the source residual stream and *writing to* some low rank subspace of the destination residual stream (with maybe some processing happening in the middle).

A common operation for this will just be to *copy*, ie to have the same reading and writing subspace, and to do minimal processing in the middle. Empirically, this tends to coincide with the OV Circuit having (approximately) positive real eigenvalues. I mostly assert this as an empirical fact, but intuitively, operations that involve mapping eigenvectors to different directions (eg rotations) tend to have complex eigenvalues. And operations that preserve eigenvector direction but negate it tend to have negative real eigenvalues. And "what happens to the eigenvectors" is a decent proxy for what happens to an arbitrary vector.

We can get a score for "how positive real the OV circuit eigenvalues are" with $\frac{\sum \lambda_i}{\sum |\lambda_i|}$, where $\lambda_i$ are the eigenvalues of the OV circuit. This is a bit of a hack, but it seems to work well in practice.

Let's use FactoredMatrix to compute this for every head in the model! We use the helper `model.OV` to get the concatenated OV circuits for all heads across all layers in the model. This has the shape `[n_layers, n_heads, d_model, d_model]`, where `n_layers` and `n_heads` are batch dimensions and the final two dimensions are factorised as `[n_layers, n_heads, d_model, d_head]` and `[n_layers, n_heads, d_head, d_model]` matrices.

We can then get the eigenvalues for this, where there are separate eigenvalues for each element of the batch (a `[n_layers, n_heads, d_head]` tensor of complex numbers), and calculate the copying score.

```python
OV_circuit_all_heads = model.OV
print(OV_circuit_all_heads)

OV_circuit_all_heads_eigenvalues = OV_circuit_all_heads.eigenvalues 
print(OV_circuit_all_heads_eigenvalues.shape)
print(OV_circuit_all_heads_eigenvalues.dtype)

OV_copying_score = OV_circuit_all_heads_eigenvalues.sum(dim=-1).real / OV_circuit_all_heads_eigenvalues.abs().sum(dim=-1)
imshow(utils.to_numpy(OV_copying_score), xaxis="Head", yaxis="Layer", title="OV Copying Score for each head in GPT-2 Small", zmax=1.0, zmin=-1.0)
```""")

    st.plotly_chart(fig_dict["ov_copying"], use_container_width=True)
    st.markdown(r"""

Head 11 in Layer 11 (L11H11) has a high copying score, and if we plot the eigenvalues they look approximately as expected.

```python
scatter(x=OV_circuit_all_heads_eigenvalues[-1, -1, :].real, y=OV_circuit_all_heads_eigenvalues[-1, -1, :].imag, title="Eigenvalues of Head L11H11 of GPT-2 Small", xaxis="Real", yaxis="Imaginary")
```""")

    st.plotly_chart(fig_dict["scatter_evals"], use_container_width=True)
    st.markdown(r"""

We can even look at the full OV circuit, from the input tokens to output tokens: $W_E W_V W_O W_U$. This is a `[d_vocab, d_vocab]==[50257, 50257]` matrix, so absolutely enormous, even for a single head. But with the FactoredMatrix class, we can compute the full eigenvalue copying score of every head in a few seconds.""")

    st.error("This code gives a CUDA error - it will be fixed shortly.")
    st.markdown(r"""

```python
full_OV_circuit = model.embed.W_E @ OV_circuit_all_heads @ model.unembed.W_U
print(full_OV_circuit)

full_OV_circuit_eigenvalues = full_OV_circuit.eigenvalues
print(full_OV_circuit_eigenvalues.shape)
print(full_OV_circuit_eigenvalues.dtype)

full_OV_copying_score = full_OV_circuit_eigenvalues.sum(dim=-1).real / full_OV_circuit_eigenvalues.abs().sum(dim=-1)
imshow(utils.to_numpy(full_OV_copying_score), xaxis="Head", yaxis="Layer", title="OV Copying Score for each head in GPT-2 Small", zmax=1.0, zmin=-1.0)
```

Interestingly, these are highly (but not perfectly!) correlated. I'm not sure what to read from this, or what's up with the weird outlier heads!

```python
scatter(x=full_OV_copying_score.flatten(), y=OV_copying_score.flatten(), hover_name=[f"L{layer}H{head}" for layer in range(12) for head in range(12)], title="OV Copying Score for each head in GPT-2 Small", xaxis="Full OV Copying Score", yaxis="OV Copying Score")
```

```python
print(f"Token 256 - the most common pair of ASCII characters: |{model.to_string(256)}|")
# Squeeze means to remove dimensions of length 1. 
# Here, that removes the dummy batch dimension so it's a rank 1 tensor and returns a string
# Rank 2 tensors map to a list of strings
print(f"De-Tokenizing the example tokens: {model.to_string(example_text_tokens.squeeze())}")
```

## Generating Text

TransformerLens also has basic text generation functionality, which can be useful for generally exploring what the model is capable of (thanks to Ansh Radhakrishnan for adding this!). This is pretty rough functionality, and where possible I recommend using more established libraries like HuggingFace for this.

```python
model.generate("(CNN) President Barack Obama caught in embarrassing new scandal\n", max_new_tokens=50, temperature=0.7, prepend_bos=True)
```

## Hook Points

The key part of TransformerLens that lets us access and edit intermediate activations are the HookPoints around every model activation. Importantly, this technique will work for *any* model architecture, not just transformers, so long as you're able to edit the model code to add in HookPoints! This is essentially a lightweight library bundled with TransformerLens that should let you take an arbitrary model and make it easier to study. 

This is implemented by having a HookPoint layer. Each transformer component has a HookPoint for every activation, which wraps around that activation. The HookPoint acts as an identity function, but has a variety of helper functions that allows us to put PyTorch hooks in to edit and access the relevant activation. 

There is also a `HookedRootModule` class - this is a utility class that the root module should inherit from (root module = the model we run) - it has several utility functions for using hooks well, notably `reset_hooks`, `run_with_cache` and `run_with_hooks`. 

The default interface is the `run_with_hooks` function on the root module, which lets us run a forwards pass on the model, and pass on a list of hooks paired with layer names to run on that pass. 

The syntax for a hook is `function(activation, hook)` where `activation` is the activation the hook is wrapped around, and `hook` is the `HookPoint` class the function is attached to. If the function returns a new activation or edits the activation in-place, that replaces the old one, if it returns None then the activation remains as is.

### Toy Example

Here's a simple example of defining a small network with HookPoints:

We define a basic network with two layers that each take a scalar input $x$, square it, and add a constant:
$x_0=x$, $x_1=x_0^2+3$, $x_2=x_1^2-4$.

We wrap the input, each layer's output, and the intermediate value of each layer (the square) in a hook point.

```python
from transformer_lens.hook_points import HookedRootModule, HookPoint

class SquareThenAdd(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = nn.Parameter(t.tensor(offset))
        self.hook_square = HookPoint()

    def forward(self, x):
        # The hook_square doesn't change the value, but lets us access it
        square = self.hook_square(x * x)
        return self.offset + square

class TwoLayerModel(HookedRootModule):
    def __init__(self):
        super().__init__()
        self.layer1 = SquareThenAdd(3.0)
        self.layer2 = SquareThenAdd(-4.0)
        self.hook_in = HookPoint()
        self.hook_mid = HookPoint()
        self.hook_out = HookPoint()

        # We need to call the setup function of HookedRootModule to build an
        # internal dictionary of modules and hooks, and to give each hook a name
        super().setup()

    def forward(self, x):
        # We wrap the input and each layer's output in a hook - they leave the
        # value unchanged (unless there's a hook added to explicitly change it),
        # but allow us to access it.
        x_in = self.hook_in(x)
        x_mid = self.hook_mid(self.layer1(x_in))
        x_out = self.hook_out(self.layer2(x_mid))
        return x_out

model = TwoLayerModel()
```

We can add a cache, to save the activation at each hook point

(There's a custom `run_with_cache` function on the root module as a convenience, which is a wrapper around model.forward that return model_out, cache_object - we could also manually add hooks with `run_with_hooks` that store activations in a global caching dictionary. This is often useful if we only want to store, eg, subsets or functions of some activations.)

```python
out, cache = model.run_with_cache(t.tensor(5.0))
print("Model output:", out.item())
for key in cache:
    print(f"Value cached at hook {key}", cache[key].item())
```

We can also use hooks to intervene on activations - eg, we can set the intermediate value in layer 2 to zero to change the output to -5

```python
def set_to_zero_hook(tensor, hook):
    print(hook.name)
    return t.tensor(0.0)

print(
    "Output after intervening on layer2.hook_scaled",
    model.run_with_hooks(
        t.tensor(5.0), fwd_hooks=[("layer2.hook_square", set_to_zero_hook)]
    ).item(),
)
```

## Loading Pre-Trained Checkpoints

There are a lot of interesting questions combining mechanistic interpretability and training dynamics - analysing model capabilities and the underlying circuits that make them possible, and how these change as we train the model. 

TransformerLens supports these by having several model families with checkpoints throughout training. `HookedTransformer.from_pretrained` can load a checkpoint of a model with the `checkpoint_index` (the label 0 to `num_checkpoints-1`) or `checkpoint_value` (the step or token number, depending on how the checkpoints were labelled).


Available models:
* All of my interpretability-friendly models have checkpoints available, including:
    * The toy models - `attn-only`, `solu`, `gelu` 1L to 4L
        * These have ~200 checkpoints, taken on a piecewise linear schedule (more checkpoints near the start of training), up to 22B tokens. Labelled by number of tokens seen.
    * The SoLU models trained on 80% Web Text and 20% Python Code (`solu-6l` to `solu-12l`)
        * Same checkpoint schedule as the toy models, this time up to 30B tokens
    * The SoLU models trained on the pile (`solu-1l-pile` to `solu-12l-pile`)
        * These have ~100 checkpoints, taken on a linear schedule, up to 15B tokens. Labelled by number of steps.
        * The 12L training crashed around 11B tokens, so is truncated.
* The Stanford Centre for Research of Foundation Models trained 5 GPT-2 Small sized and 5 GPT-2 Medium sized models (`stanford-gpt2-small-a` to `e` and `stanford-gpt2-medium-a` to `e`)
    * 600 checkpoints, taken on a piecewise linear schedule, labelled by the number of steps.

The checkpoint structure and labels is somewhat messy and ad-hoc, so I mostly recommend using the `checkpoint_index` syntax (where you can just count from 0 to the number of checkpoints) rather than `checkpoint_value` syntax (where you need to know the checkpoint schedule, and whether it was labelled with the number of tokens or steps). The helper function `get_checkpoint_labels` tells you the checkpoint schedule for a given model - ie what point was each checkpoint taken at, and what type of label was used.

Here are graphs of the schedules for several checkpointed models: (note that the first 3 use a log scale, latter 2 use a linear scale)

```python
from transformer_lens.loading_from_pretrained import get_checkpoint_labels
for model_name in ["attn-only-2l", "solu-12l", "stanford-gpt2-small-a"]:
    checkpoint_labels, checkpoint_label_type = get_checkpoint_labels(model_name)
    line(checkpoint_labels, xaxis="Checkpoint Index", yaxis=f"Checkpoint Value ({checkpoint_label_type})", title=f"Checkpoint Values for {model_name} (Log scale)", log_y=True, markers=True)
for model_name in ["solu-1l-pile", "solu-6l-pile"]:
    checkpoint_labels, checkpoint_label_type = get_checkpoint_labels(model_name)
    line(checkpoint_labels, xaxis="Checkpoint Index", yaxis=f"Checkpoint Value ({checkpoint_label_type})", title=f"Checkpoint Values for {model_name} (Linear scale)", log_y=False, markers=True)
```

### Example: Induction Head Phase Transition

One of the more interesting results analysing circuit formation during training is the [induction head phase transition](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). They find a pretty dramatic shift in models during training - there's a brief period where models go from not having induction heads to having them, which leads to the models suddenly becoming much better at in-context learning (using far back tokens to predict the next token, eg over 500 words back). This is enough of a big deal that it leads to a visible *bump* in the loss curve, where the model's rate of improvement briefly increases. 

As a brief demonstration of the existence of the phase transition, let's load some checkpoints of a two layer model, and see whether they have induction heads. An easy test, as we used above, is to give the model a repeated sequence of random tokens, and to check how good its loss is on the second half. `evals.induction_loss` is a rough util that runs this test on a model.
(Note - this is deliberately a rough, non-rigorous test for the purposes of demonstration, eg `evals.induction_loss` by default just runs it on 4 sequences of 384 tokens repeated twice. These results totally don't do the paper justice - go check it out if you want to see the full results!)

In the interests of time and memory, let's look at a handful of checkpoints (chosen to be around the phase change), indices `[10, 25, 35, 60, -1]`. These are roughly 22M, 200M, 500M, 1.6B and 21.8B tokens through training, respectively. (I generally recommend looking things up based on indices, rather than checkpoint value!). 

```python
from transformer_lens import evals
# We use the two layer model with SoLU activations, chosen fairly arbitrarily as being both small (so fast to download and keep in memory) and pretty good at the induction task.
model_name = "solu-2l"
# We can load a model from a checkpoint by specifying the checkpoint_index, -1 means the final checkpoint
checkpoint_indices = [10, 25, 35, 60, -1]
checkpointed_models = []
tokens_trained_on = []
induction_losses = []
```

We load the models, cache them in a list, and 

```python
for index in checkpoint_indices:
    # Load the model from the relevant checkpoint by index
    model_for_this_checkpoint = HookedTransformer.from_pretrained(model_name, checkpoint_index=index)
    checkpointed_models.append(model_for_this_checkpoint)

    tokens_seen_for_this_checkpoint = model_for_this_checkpoint.cfg.checkpoint_value
    tokens_trained_on.append(tokens_seen_for_this_checkpoint)

    induction_loss_for_this_checkpoint = evals.induction_loss(model_for_this_checkpoint).item()
    induction_losses.append(induction_loss_for_this_checkpoint)
```

We can plot this, and see there's a sharp shift from ~200-500M tokens trained on (note the log scale on the x axis). Interestingly, this is notably earlier than the phase transition in the paper, I'm not sure what's up with that.

(To contextualise the numbers, the tokens in the random sequence are uniformly chosen from the first 20,000 tokens (out of ~48,000 total), so random performance is at least $\ln(20000)\approx 10$. A naive strategy like "randomly choose a token that's already appeared in the first half of the sequence (384 elements)" would get $\ln(384)\approx 5.95$, so the model is doing pretty well here.)

```python
line(induction_losses, x=tokens_trained_on, xaxis="Tokens Trained On", yaxis="Induction Loss", title="Induction Loss over training: solu-2l", markers=True, log_x=True)
```
""")

def section_3():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#introducing-our-toy-attention-only-model">Introducing Our Toy Attention-Only Model</a></li>
    <li><a class="contents-el" href="#building-interpretability-tools">Building interpretability tools</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#direct-logit-attribution">Direct Logit attribution</a></li>
        <li><a class="contents-el" href="#a-note-on-type-annotations-and-typechecking">A note on type-annotations and typechecking</a></li>
    </ul></li>
    <li><a class="contents-el" href="#visualising-attention-patterns">Visualising Attention Patterns</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#summarising-attention-patterns">Summarising attention patterns</a></li>
    </ul></li>
    <li><a class="contents-el" href="#ablations">Ablations</a></li>
    <li><a class="contents-el" href="#finding-induction-circuits">Finding Induction Circuits</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#checking-for-the-induction-capability">Checking for the induction capability</a></li>
        <li><a class="contents-el" href="#looking-for-induction-attention-patterns">Looking for Induction Attention Patterns</a></li>
        <li><ul class="contents">
            <li><a class="contents-el" href="#logit-attribution">Logit Attribution</a></li>
            <li><a class="contents-el" href="#ablations">Ablations</a></li>
        </ul></li>
    </ul></li>
    <li><a class="contents-el" href="#conclusion-do-you-understand-the-circuit">Conclusion - do you understand the circuit?</a></li>
</ul>""", unsafe_allow_html=True)
    st.markdown(r"""
# Exercises: finding induction heads

Now that we've seen some of the features of TransformerLens, let's apply them to the task of finding induction heads.

If you don't fully understand the algorithm peformed by induction heads, this diagram may prove helpful.""")

    with open("images/induction-heads-2.svg", "r") as f:
        st.download_button("Download induction heads diagram", f.read(), "induction_head_diagram.svg")

    st.markdown(r"""
The material in this section will heavily follow the Mathematical Framework for Transformer Circuits paper. Here are some notes from Neel regarding this paper:""")

    with st.expander("Tips & Insights for the Paper"):
        st.markdown(r"""

* The eigenvalue stuff is very cool, but doesn't generalise that much, it's not a priority to get your head around
* It's really useful to keep clear in your head the difference between parameters (learned numbers that are intrinsic to the network and independent of the inputs) and activations (temporary numbers calculated during a forward pass, that are functions of the input).
    * Attention is a slightly weird thing - it's an activation, but is also used in a matrix multiplication with another activation (z), which makes it parameter-y.
        * The idea of freezing attention patterns disentangles this, and lets us treat it as parameters.
* The residual stream is the fundamental object in a transformer - each layer just applies incremental updates to it - this is really useful to keep in mind throughout!
    * This is in contrast to a classic neural network, where each layer's output is the central object
    * To underscore this, a funky result about transformers is that the aspect ratio isn't *that* important - if you increase d_model/n_layer by a factor of 10 from optimal for a 1.5B transformer (ie controlling for the number of parameters), then loss decreases by <1%.
* The calculation of attention is a bilinear form (ie via the QK circuit) - for any pair of positions it takes an input vector from each and returns a scalar (so a ctx x ctx tensor for the entire sequence), while the calculation of the output of a head pre weighting by attention (ie via the OV circuit) is a linear map from the residual stream in to the residual stream out - the weights have the same shape, but are doing functions of completely different type signatures!
* How to think about attention: A framing I find surprisingly useful is that attention is the "wiring" of the neural network. If we hold the attention patterns fixed, they tell the model how to move information from place to place, and thus help it be effective at sequence prediction. But the key interesting thing about a transformer is that attention is *not* fixed - attention is computed and takes a substantial fraction of the network's parameters, allowing it to dynamically set the wiring. This can do pretty meaningful computation, as we see with induction heads, but is in some ways pretty limited. In particular, if the wiring is fixed, an attention only transformer is a purely linear map! Without the ability to intelligently compute attention, an attention-only transformer would be incredibly limited, and even with it it's highly limited in the functional forms it can represent.
    * Another angle - attention as generalised convolution. A naive transformer would use 1D convolutions on the sequence. This is basically just attention patterns that are hard coded to be uniform over the last few tokens - since information is often local, this is a decent enough default wiring. Attention allows the model to devote some parameters to compute more intelligent wiring, and thus for a big enough and good enough model will significantly outperform convolutions.
* One of the key insights of the framework is that there are only a few activations of the network that are intrinsically meaningful and interpretable - the input tokens, the output logits and attention patterns (and neuron activations in non-attention-only models). Everything else (the residual stream, queries, keys, values, etc) are just intermediate states on a calculation between two intrinsically meaningful things, and you should instead try to understand the start and the end. Our main goal is to decompose the network into many paths between interpretable start and end states
    * We can get away with this because transformers are really linear! The composition of many linear components is just one enormous matrix
* A really key thing to grok about attention heads is that the QK and OV circuits act semi-independently. The QK circuit determines which previous tokens to attend to, and the OV circuit determines what to do to tokens *if* they are attended to. In particular, the residual stream at the destination token *only* determines the query and thus what tokens to attend to - what the head does *if* it attends to a position is independent of the destination token residual stream (other than being scaled by the attention pattern).
    <p align="center">
        <img src="w2d4_Attn_Head_Pic.png" width="400" />
    </p>
* Skip trigram bugs are a great illustration of this - it's worth making sure you really understand them. The key idea is that the destination token can *only* choose what tokens to pay attention to, and otherwise not mediate what happens *if* they are attended to. So if multiple destination tokens want to attend to the same source token but do different things, this is impossible - the ability to choose the attention pattern is insufficient to mediate this.
    * Eg, keep...in -> mind is a legit skip trigram, as is keep...at -> bay, but keep...in -> bay is an inherent bug from this pair of skip trigrams
* The tensor product notation looks a lot more complicated than it is. $A \otimes W$ is shorthand for "the function $f_{A,W}$ st $f_{A,W}(x)=AxW$" - I recommend mentally substituting this in in your head everytime you read it.
* K, Q and V composition are really important and fairly different concepts! I think of each attention head as a circuit component with 3 input wires (Q,K,V) and a single output wire (O). Composition looks like connecting up wires, but each possible connection is a choice! The key, query and value do different things and so composition does pretty different things.
    * Q-Composition, intuitively, says that we want to be more intelligent in choosing our destination token - this looks like us wanting to move information to a token based on *that* token's context. A natural example might be the final token in a several token word or phrase, where earlier tokens are needed to disambiguate it, eg `E|iff|el| Tower|`
    * K-composition, intuitively, says that we want to be more intelligent in choosing our source token - this looks like us moving information *from* a token based on its context (or otherwise some computation at that token).
        * Induction heads are a clear example of this - the source token only matters because of what comes before it!
    * V-Composition, intuitively, says that we want to *route* information from an earlier source token *other than that token's value* via the current destination token. It's less obvious to me when this is relevant, but could imagine eg a network wanting to move information through several different places and collate and process it along the way
        * One example: In the ROME paper, we see that when models recall that "The Eiffel Tower is in" -> " Paris", it stores knowledge about the Eiffel Tower on the " Tower" token. When that information is routed to `| in|`, it must then map to the output logit for `| Paris|`, which seems likely due to V-Composition
* A surprisingly unintuitive concept is the notion of heads (or other layers) reading and writing from the residual stream. These operations are *not* inverses! A better phrasing might be projecting vs embedding.
    * Reading takes a vector from a high-dimensional space and *projects* it to a smaller one - (almost) any two pair of random vectors will have non-zero dot product, and so every read operation can pick up *somewhat* on everything in the residual stream. But the more a vector is aligned to the read subspace, the most that vector's norm (and intuitively, its information) is preserved, while other things are lower fidelity
        * A common reaction to these questions is to start reasoning about null spaces, but I think this is misleading - rank and nullity are discrete concepts, while neural networks are fuzzy, continuous objects - nothing ever actually lies in the null space or has non-full rank (unless it's explicitly factored). I recommend thinking in terms of "what fraction of information is lost". The null space is the special region with fraction lost = 1
    * Writing *embeds* a vector into a small dimensional subspace of a larger vector space. The overall residual stream is the sum of many vectors from many different small subspaces.
        * Every read operation can see into every writing subspace, but will see some with higher fidelity, while others are noise it would rather ignore.
    * It can be useful to reason about this by imagining that d_head=1, and that every vector is a random Gaussian vector - projecting a random Gaussian onto another in $\mathbb{R}^n$ will preserve $\frac{1}{n}$ of the variance, on average.
* A key framing of transformers (and neural networks in general) is that they engage in **lossy compression** - they have a limited number of dimensions and want to fit in more directions than they have dimensions. Each extra dimension introduces some interference, but has the benefit of having more expressibility. Neural networks will learn an optimal-ish solution, and so will push the compression as far as it can until the costs of interference dominate.
    * This is clearest in the case of QK and OV circuits - $W_QK=W_Q^TW_K$ is a d_model x d_model matrix with rank d_head. And to understand the attention circuit, it's normally best to understand $W_QK$ on its own. Often, the right mental move is to forget that $W_QK$ is low rank, to understand what the ideal matrix to learn here would be, and then to assume that the model learns the best low rank factorisation of that.
        * This is another reason to not try to interpret the keys and queries - the intermediate state of a low rank factorisations are often a bit of a mess because everything is so compressed (though if you can do SVD on $W_QK$ that may get you a meaningful basis?)
        * Rough heuristic for thinking about low rank factorisations and how good they can get - a good way to produce one is to take the SVD and zero out all but the first d_head singular values.
    * This is the key insight behind why polysemanticity (back from w1d5) is a thing and is a big deal - naturally the network would want to learn one feature per neuron, but it in fact can learn to compress more features than total neurons. It has some error introduced from interference, but this is likely worth the cost of more compression.
        * Just as we saw there, the sparsity of features is a big deal for the model deciding to compress things! Inteference cost goes down the more features are sparse (because unrelated features are unlikely to co-occur) while expressibility benefits don't really change that much.
    * The residual stream is the central example of this - every time two parts of the network compose, they will be communicating intermediate states via the residual stream. Bandwidth is limited, so these will likely try to each be low rank. And the directions within that intermediate product will *only* make sense in the context of what the writing and reading components care about. So interpreting the residual stream seems likely fucked - it's just
* The 'the residual stream is fundamentally uninterpretable' claim is somewhat overblown - most models do dropout on the residual stream which somewhat privileges that basis
    * And there are [*weird*](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/) results about funky directions in the residual stream.
* Getting your head around the idea of a privileged basis is very worthwhile! The key mental move is to flip between "a vector is a direction in a geometric space" and "a vector is a series of numbers in some meaningful basis, where each number is intrinsically meaningful". By default, it's easy to spend too much time in the second mode, because every vector is represented as a series of numbers within the GPU, but this is often less helpful!

### An aside on why we need the tensor product notation at all

Neural networks are functions, and are built up of several subcomponents (like attention heads) that are also functions - they are defined by how they take in an input and return an output. But when doing interpretability we want the ability to talk about the network as a function intrinsically and analyse the structure of this function, *not* in the context of taking in a specific input and getting a specific output. And this means we need a language that allows us to naturally talk about functions that are the sum (components acting in parallel) or composition (components acting in series) of other functions.

A simple case of this: We're analysing a network with several linear components acting in parallel - component $C_i$ is the function $x \rightarrow W_ix$, and can be represented intrinsically as $W_i$ (matrices are equivalent to linear maps). We can represent the layer with all acting in parallel as $x \rightarrow \sum_i W_ix=(\sum_i W_i)x$, and so intrinsically as $\sum_i W_i$ - this is easy because matrix notation is designed to make addition of.

Attention heads are harder because they map the input tensor $x$ (shape: `[position x d_model]`) to an output $Ax(W_OW_V)^T$ - this is a linear function, but now on a *tensor*, so we can't trivially represent addition and composition with matrix notation. The paper uses the notation $A\otimes W_OW_V$, but this is just different notation for the same underlying function. The output of the layer is the sum over the 12 heads: $\sum_i A^{(i)}x(W_O^{(i)}W_V^{(i)})^T$. And so we could represent the function of the entire layer as $\sum_i A^{(i)} x (W_O^{(i)}W_V^{(i)})$. There are natural extensions of this notation for composition, etc, though things get much more complicated when reasoning about attention patterns - this is now a bilinear function of a pair of inputs: the query and key residual streams. (Note that $A$ is also a function of $x$ here, in a way that isn't obvious from the notation.)

The key point to remember is that if you ever get confused about what a tensor product means, explicitly represent it as a function of some input and see if things feel clearer.
""")

    st.markdown(r"""

## Introducing Our Toy Attention-Only Model

Here we introduce a toy 2L attention-only transformer trained specifically for today. Some changes to make them easier to interpret:
- It has only attention blocks
- The positional embeddings are only added to each key and query vector in the attention layers as opposed to the token embeddings (meaning that the residual stream can't directly encode positional information)
  - This turns out to make it *way* easier for induction heads to form, it happens 2-3x times earlier - [see the comparison of two training runs](https://wandb.ai/mechanistic-interpretability/attn-only/reports/loss_ewma-22-08-24-11-08-83---VmlldzoyNTI0MDMz?accessToken=8ap8ir6y072uqa4f9uinotdtrwmoa8d8k2je4ec0lyasf1jcm3mtdh37ouijgdbm) here. (The bump in each curve is the formation of induction heads)
- It has no MLP layers, no LayerNorms, and no biases
- There are separate embed and unembed matrices (ie the weights are not tied)
- The activations in the attention layers $(q, k, v, z)$ have shape `[batch, position, head_index, d_head]` (i.e. not flattened into a single d_model axis)
  - Similarly $W_K, W_Q, W_V$ have shape `[head_index, d_head, d_model]`, $W_O$ has shape `[head_index, d_model, d_head]`
- Convention: All weight matrices multiply on the left (i.e. have shape `[output, input]`)

```python
MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")

cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal", # defaults to "bidirectional"
    attn_only=True, # defaults to False

    tokenizer_name="EleutherAI/gpt-neox-20b", 
    # if setting from config, set tokenizer this way rather than passing it in explicitly
    # model initialises via AutoTokenizer.from_pretrained(tokenizer_name)

    seed=398,
    use_attn_result=True,
    normalization_type=None, # defaults to "LN", i.e. use layernorm with weights and biases
    
    positional_embedding_type="shortformer" # this makes it so positional embeddings are used differently (makes induction heads cleaner to study)
)
```

You should download your model weights from [this Google Drive link](https://drive.google.com/drive/folders/1LRkK3tqNuZ6Y_UwgQKxML3980DtM0KUe), and save them under `WEIGHT_PATH` as given below.

Note - this was saved using `float16` to save data. If some of the exercises don't work for this reason, please shoot me (Callum) a message and I'll upload the `float32` version.

```python
WEIGHT_PATH = "./data/attn_only_2L_half.pth"

if MAIN:
    model = HookedTransformer(cfg)
    raw_weights = model.state_dict()
    pretrained_weights = t.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(pretrained_weights)
```

""")

    with st.expander("A guide to all the hook names in TransformerBlock:"):
        st.markdown(r"""
This is for a model with just attention layers, and no MLPs, LayerNorms, or biases. 

The names in the boxes represent the hook names (prefixed with `hook_`). For instance, you can access the attention probabilities for layer 0 with `cache["pattern", 0]`, or if you want to use the full form then:

```python
cache["blocks.0.attn.hook_pattern"]
```

These are connected by the fact that `utils.get_act_name("pattern", 0)` returns the full string used in indexing above.
""")
        st.write("""<figure style="max-width:380px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNp1UsluwjAQ_ZWRz4nUsytxSuHQQ1W1ag9NhUw8IRFxHLykLObfOya0EARWFM3yZt7zePas0BIZZ0sjugres7wFOtYvhkDO6Bti8RSNsDbDElB1bgs_tXQVf-g2SYX1snLRfLyBjhz0a7ThBiUhzhiDtpbzzmCaptkXfHPOj73TdBKElOGU19ada87Wv07h3EXTeNZjdzV2-7Eby-e20MQ2TkzHbkdANFdMs7G7G7vU0jcX4rGVN28_CZ_z17CmMUyvxlCIpjgqTMCSjSBaCUrYVbiQHYFWl06JTTippFazu1TPYRWp7uY_Qh_rr6XotscNPaVagC6hF41H6LFw2tiwA4IA1b6E4dJwFOUV6B4NVCikHSRr7-JzD-Q0EJYwhUaJWtIm7mM4Z65ChTnjZEosRZwhLeKBoL6TwuGTrImV8VI0FhMmvNNv27Zg3BmPf6CsFrQf6oQ6_ALJ2eJm" /></figure>""", unsafe_allow_html=True)
        # graph TD
        #     subgraph " "
        #         classDef empty width:0px,height:0px;
        #         classDef code color:red;

        #         resid_pre---D[ ]:::empty-->|add|resid_post
                
        #         subgraph attn
        #             q
        #             k
        #             v
        #             attn_scores
        #             F
        #             pattern
        #             G
        #             z
        #             result
        #         end
        #         resid_pre-->|W_Q|q---F[ ]:::empty-->|calc attn, scale and mask|attn_scores-->|softmax|pattern---G
        #         resid_pre-->|W_K|k---F
        #         resid_pre-->|W_V|v---G[ ]:::empty-->|convex comb of value vectors|z --> |W_O|result -->|sum over heads|attn_out---D
        #     end

    st.markdown(r"""

## Building interpretability tools

In this section, we're going to build some basic interpretability tools to decompose models and answer some questions about them.

Let's run our model on some text (feel free to write your own!)

```python
if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    str_tokens = model.to_str_tokens(text)
    tokens = model.to_tokens(text)
    tokens = tokens.to(device)
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    model.reset_hooks()
```

### Direct Logit attribution

A consequence of the residual stream is that the output logits are the sum of the contributions of each layer, and thus the sum of the results of each head. This means we can decompose the output logits into a term coming from each head and directly do attribution like this! Write a function to look at how much each head and the direct path term contributes to the correct logit.""")

    with st.expander("A concrete example"):

        st.markdown(r"""
Let's say that our model knows that the token Harry is followed by the token Potter, and we want to figure out how it does this. The logits on Harry are `W_U @ residual`. But this is a linear map, and the residual stream is the sum of all previous layers `residual = embed + attn_out_0 + attn_out_1`. So `logits = (W_U @ embed) + (W_U @ attn_out_0) + (W_U @ attn_out_1)`

We can be even more specific, and *just* look at the logit of the Potter token - this corresponds to a row of W_U, and so a direction in the residual stream - our logit is now a single number that is the sum of `(potter_U @ embed) + (potter_U @ attn_out_0) + (potter_U @ attn_out_1)`. Even better, we can decompose each attention layer output into the sum of the result of each head, and use this to get many terms.
""")

    st.markdown(r"""
Calculate the logit attributions of the following paths to the logits: direct path (via the residual connections from the embedding to unembedding); each layer 0 head (via the residual connection and skipping layer 1); each layer 1 head. To emphasise, these are not paths from the start to the end of the model, these are paths from the output of some component directly to the logits - we make no assumptions about how each path was calculated!

Note: Here we are just looking at the DIRECT effect on the logits - if heads compose with other heads and affect logits like that, or inhibit logits for other tokens to boost the correct one we will not pick up on this!

Note 2: By looking at just the logits corresponding to the correct token, our data is much lower dimensional because we can ignore all other tokens other than the correct next one (Dealing with a 50K vocab size is a pain!). But this comes at the cost of missing out on more subtle effects, like a head suppressing other plausible logits, to increase the log prob of the correct one.

Note 3: When calculating correct output logits, we will get tensors with a dimension (position - 1,), not (position,) - we remove the final element of the output (logits), and the first element of labels (tokens). This is because we're predicting the *next* token, and we don't know the token after the final token, so we ignore it.
""")

    with st.expander("Aside:"):
        st.markdown(r"""
While we won't worry about this for this exercise, logit attribution is often more meaningful if we first center W_U - ie, ensure the mean of each row writing to the output logits is zero. Log softmax is invariant when we add a constant to all the logits, so we want to control for a head that just increases all logits by the same amount. We won't do this here for ease of testing.""")

    with st.expander("Exercise: Why don't we do this to the log probs instead?"):
        st.markdown(r"""
Because log probs aren't linear, they go through log_softmax, a non-linear function.
""")
    st.markdown(r"""

```python
def to_numpy(tensor):
    '''Helper function to convert things to numpy before plotting with Plotly.'''
    return tensor.detach().cpu().numpy()

def convert_tokens_to_string(tokens, batch_index=0):
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [f"|{tokenizer.decode(tok)}|_{c}" for (c, tok) in enumerate(tokens)]

seq_len = tokens.shape[-1]
n_components = model.cfg.n_layers * model.cfg.n_heads + 1

patch_typeguard()  # must call this before @typechecked

@typechecked
def logit_attribution(
    embed: TT["seq_len": seq_len, "d_model"],
    l1_results: TT["seq_len", "n_heads", "d_model"],
    l2_results: TT["seq_len", "n_heads", "d_model"],
    W_U: TT["d_model", "d_vocab"],
    tokens: TT["seq_len"],
) -> TT[seq_len-1, "n_components": n_components]:
    '''
    We have provided 'W_U_to_logits' which is a (d_model, seq_next) tensor where each row is the unembed for the correct NEXT token at the current position.
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U: the unembedding matrix
    Returns:
        Tensor representing the concatenation (along dim=-1) of logit attributions from:
            the direct path (position-1,1)
            layer 0 logits (position-1, n_heads)
            and layer 1 logits (position-1, n_heads)
    '''
    W_U_to_logits = W_U[:, tokens[1:]]
    pass
```

### A note on type-annotations and typechecking

We've added `TensorType` annotations to the function above, and shown some examples of how it can be used. You can have the elements of a tensortype object be of the form `str`, `int`, or `str: int` (as well as other options, which you can read about [here](https://github.com/patrick-kidger/torchtyping)). Additionally, using the `@typechecked` decorator on your function will make sure to throw an error if the types don't match up to what you've specified (for instance, you have `"seq_len"` appearing twice, corresponding to two different lengths).

There are disadvantages to using type annotations in this very strict way, for instance you would have to redefine `seq_len` and `n_components` if you wanted to reuse the function above. For that reason, we recommend you remove the `int` parts of the type checking once you've got the tests below working.

These tests will check your logit attribution function is working correctly, by taking the sum of logit attributions and comparing it to the actual values in the residual stream at the end of your model.

```python
if MAIN:
    with t.inference_mode():
        batch_index = 0
        embed = cache["hook_embed"]
        l1_results = cache["result", 0] # same as cache["blocks.0.attn.hook_result"]
        l2_results = cache["result", 1]
        logit_attr = logit_attribution(embed, l1_results, l2_results, model.unembed.W_U, tokens[0])
        # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
        correct_token_logits = logits[batch_index, t.arange(len(tokens[0]) - 1), tokens[batch_index, 1:]]
        t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-2, rtol=0)
```

Once you've got the tests working, you can visualise the logit attributions for each path through the model.

```python
def plot_logit_attribution(logit_attr: TT["seq", "path"], tokens: TT["seq"]):
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(tokens[:-1])
    x_labels = ["Direct"] + [f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    imshow(to_numpy(logit_attr), x=x_labels, y=y_labels, xaxis="Term", yaxis="Position", caxis="logit", height=25*len(tokens))

if MAIN:
    embed = cache["hook_embed"]
    l1_results = cache["blocks.0.attn.hook_result"]
    l2_results = cache["blocks.1.attn.hook_result"]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.unembed.W_U, tokens[0])
    plot_logit_attribution(logit_attr, tokens)
```

## Visualising Attention Patterns

A key insight from the paper is that we should focus on interpreting the parts of the model that are intrinsically interpretable - the input tokens, the output logits and the attention patterns. Everything else (the residual stream, keys, queries, values, etc) are compressed intermediate states when calculating meaningful things. So a natural place to start is classifying heads by their attention patterns on various texts.

When doing interpretability, it's always good to begin by visualising your data, rather than taking summary statistics. Summary statistics can be super misleading! But now that we have visualised the attention patterns, we can create some basic summary statistics and use our visualisations to validate them! (Accordingly, being good at web dev/data visualisation is a surprisingly useful skillset! Neural networks are very high-dimensional object.)

A good place to start is visualising the attention patterns of the model on input text. Go through a few of these, and get a sense for what different heads are doing.

```python
if MAIN:
    for layer in range(model.cfg.n_layers):
        attention_pattern = cache["pattern", layer]
        cv.attention.attention_heads(tokens=str_tokens, attention=attention_pattern)
```
""")
    st.info(r"""Reminder: rather than plotting inline, you can do the following, and then open in your browser from the left-hand file explorer menu of VSCode:

```python
for layer in range(model.cfg.n_layers):
    attention_pattern = cache["pattern", layer]
    html = cv.attention.attention_heads(tokens=str_tokens, attention=attention_pattern)
    with open(f"layer_{layer}_attention.html", "w") as f:
        f.write(str(html))
```""")
    st.markdown(r"""

### Summarising attention patterns

Three basic patterns for attention heads are those that mostly attend to the current token, the previous token, or the first token (often used as a resting or null position for heads that only sometimes activate). Let's make detectors for those! Validate your detectors by comparing these results to the visual attention patterns above - summary statistics on their own can be dodgy, but are much more reliable if you can validate it by directly playing with the data.

Note - there's no objectively correct answer for which heads are doing which tasks, and which detectors can spot them. You should just try and come up with something plausible-seeming, which identifies the kind of behaviour you're looking for.
""")

    with st.expander("Hint"):
        st.markdown(r"""
Try and compute the average attention probability along the relevant tokens. For instance, you can get the tokens just below the diagonal by using `t.diagonal` with appropriate `offset` parameter, or by indexing a 2D array as follows:

```
arr[t.arange(1, n), t.arange(n)]
```

Remember that you should be using the object `cache["pattern", layer]` to get all the attention probabilities for a given layer, and then indexing on the 0th dimension to get the correct head.
""")
    with st.expander("Example solution for current_attn_detector (read if you're stuck)"):
        st.markdown(r"""
def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    current_attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of diagonal elements
            current_attn_score = attention_pattern[t.arange(seq_len), t.arange(seq_len)].mean()
            if current_attn_score > 0.5:
                current_attn_heads.append(f"{layer}.{head}")
    return current_attn_heads
""")

    

    st.markdown(r"""
```python
def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    pass

def prev_attn_detector(cache: ActivationCache):
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    pass

def first_attn_detector(cache: ActivationCache):
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''
    pass

if MAIN:

    print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))
```

Compare the printouts to your attention visualisations above. Do they seem to make sense?""")

    st.success("Bonus: Try inputting different text, and see how stable your results are.")
    st.markdown(r"""

## Ablations

An ablation is a simple causal intervention on a model - we pick some part of it and set it to zero. This is a crude proxy for how much that part matters. Further, if we have some story about how a specific circuit in the model enables some capability, showing that ablating *other* parts does nothing can be strong evidence of this.

You already saw examples of ablations in the TransformerLens material. Here, we'll ask you to do some more. You should write a function `head_ablation` which sets a particular head's `attn_result` (i.e. the output of the attention layer corresponding to that head, before we sum over heads) to zero. Then, you should write a function `get_ablation_scores` which returns a tensor of shape `[n_layers, n_heads]` containing the **increase** in cross entropy loss on your input sequence that results from performing this ablation (i.e. `cross_entropy_loss_with_ablation - cross_entropy_loss_without_ablation`).

A few notes, before going into these exercises:

* We've generally left strict type-checking like `@typechecked` and ints in TensorTypes out of these functions, but you should feel free to add them in if you want to (in fact we'd encourage it!).
* Remember from the TransformerLens material that you can use `functools.partial` to create a function which is a partial application of `head_ablation`, but for a particular head. You'll need to do this, because the forward hook functions you pass to `model.run_with_hooks` should take just two arguments - the tensor `attn_reuslt` and the hook.
* You can access `n_layers` and `n_neads` using `model.cfg.n_layers` and `model.cfg.n_heads` respectively.
* Remember you can use `remove_batch_dim=True` in your call to `model.run_with_cache`. This will make the cache easier to work with, because you don't have to keep indexing the zeroth element!

```python
def cross_entropy_loss(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()

def head_ablation(
    attn_result: TT["batch", "seq", "n_heads", "d_model"],
    hook: HookPoint,
    head_no: int
) -> TT["batch", "seq", "n_heads", "d_model"]:
    pass

def get_ablation_scores(
    model: HookedTransformer, 
    tokens: TT["batch", "seq"]
) -> TT["n_layers", "n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    '''
    pass

if MAIN:
    ablation_scores = get_ablation_scores(model, tokens)
    imshow(ablation_scores, xaxis="Head", yaxis="Layer", caxis="logit diff", title="Logit Difference After Ablating Heads", text_auto=".2f")
""")
    st.success("Bonus: How would you expect this to compare to your direct logit attribution scores for heads in layer 0? For heads in layer 1? Plot a scatter plot and compare these results to your predictions")
    st.markdown(r"""

## Finding Induction Circuits

(Note: I use induction *head* to refer to the head in the second layer which attends to the 'token immediately after the copy of the current token', and induction *circuit* to refer to the circuit consisting of the composition of a ***previous token head*** in layer 0 and an ***induction head*** in layer 1)

[Induction heads](https://transformer-circuits.pub/2021/framework/index.html#induction-heads) are the first sophisticated circuit we see in transformers! And are sufficiently interesting that we wrote [another paper just about them](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
""")
    with st.expander("An aside on why induction heads are a big deal"):
        st.markdown(r"""

There's a few particularly striking things about induction heads:

* They develop fairly suddenly in a phase change - from about 2B to 4B tokens we go from no induction heads to pretty well developed ones. This is a striking divergence from a 1L model [see the training curves for this model vs a 1L one](https://wandb.ai/mechanistic-interpretability/attn-only/reports/loss_ewma-22-08-24-11-08-65---VmlldzoyNTI0MDQx?accessToken=extt248d3qoxbqw1zy05kplylztjmx2uqaui3ctqb0zruem0tkpwrssq2ao1su3j) and can be observed in much larger models (eg a 13B one)
    * Phase changes are particularly interesting (and depressing) from an alignment perspective, because the prospect of a sharp left turn, or emergent capabilities like deception or situational awareness seems like worlds where alignment may be harder, and we get caught by surprise without warning shots or simpler but analogous models to test our techniques on.
* They are responsible for a significant loss decrease - so much so that there's a visible bump in the loss curve when they develop (this change in loss can be pretty comparable to the increase in loss from major increases in model size, though this is hard to make an apples-to-apples comparison)
* They seem to be responsible for the vast majority of in-context learning - the ability to use far back tokens in the context to predict the next token. This is a significant way in which transformers outperform older architectures like RNNs or LSTMs, and induction heads seem to be a big part of this.
* The same core circuit seems to be used in a bunch of more sophisticated settings, such as translation or few-shot learning - there are heads that seem clearly responsible for those *and* which double as induction heads
""")
    st.markdown(r"""
We're going to spend the next two section applying the interpretability tools we just built to hunting down induction heads - first doing feature analysis to find the relevant heads and what they're doing, and then mechanistically reverse engineering the details of how the circuit works. I recommend you re-read the inductions head section of the paper or read [this intuitive explanation from Mary Phuong](https://docs.google.com/document/d/14HY2xKDW6Pup_-XNXQBjoYbc2xFILz06uxHkk9-pYmY/edit), but in brief, the induction circuit consists of a previous token head in layer 0 and an induction head in layer 1, where the induction head learns to attend to the token immediately *after* copies of the current token via K-Composition with the previous token head.
""")
    st_image("induction_head_pic.png", 700)
    st.markdown("")
    st.markdown(r"""
**Recommended Exercise:** Before continuing, take a few minutes to think about how you would implement an induction circuit if you were hand-coding the weights of an attention-only transformer:

* How would you implement a copying head?
* How would you implement a previous token head?
* How would you implement an induction head?
""")

    with st.expander("Exercise - why couldn't an induction head form in a 1L model?"):
        st.markdown(r"""
Because this would require a head which attends a key position based on the *value* of the token before it. Attention scores are just a function of the key token and the query token, and are not a function of other tokens.
The attention pattern *does* allow other tokens because of softmax - if another key token has a high attention score, softmax inhibits this pair. But this inhibition is symmetric across positions, so can't systematically favour the token *next* to the relevant one.

Note that a key detail is that the value of adjacent tokens are (approximately) unrelated - if the model wanted to attend based on relative *position* this is easy.""")
    st.markdown(r"""

### Checking for the induction capability

A striking thing about models with induction heads is that, given a repeated sequence of random tokens, they can predict the repeated half of the sequence. This is nothing like it's training data, so this is kind of wild! The ability to predict this kind of out of distribution generalisation is a strong point of evidence that you've really understood a circuit.

To check that this model has induction heads, we're going to run it on exactly that, and compare performance on the two halves - you should see a striking difference in the per token losses.

Note - w're using small sequences (and just one sequence), since the results are very obvious and this makes it easier to visualise. In practice we'd obviously use larger ones on more subtle tasks. But it's often easiest to iterate and debug on small tasks.

```python
def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, 
    seq_len: int, 
    batch=1
) -> tuple[t.Tensor, t.Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Add a prefix token, since the model was always trained to have one.

    Outputs are:
    rep_logits: [batch, 1+2*seq_len, d_vocab]
    rep_tokens: [batch, 1+2*seq_len]
    rep_cache: The cache of the model run on rep_tokens
    '''
    prefix = t.ones((batch, 1), dtype=t.int64) * tokenizer.bos_token_id
    pass

def per_token_losses(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs[0]

if MAIN:
    seq_len = 50
    batch = 1
    (rep_logits, rep_tokens, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
    rep_str = model.to_str_tokens(rep_tokens)
    model.reset_hooks()
    ptl = per_token_losses(rep_logits, rep_tokens)
    print(f"Performance on the first half: {ptl[:seq_len].mean():.3f}")
    print(f"Performance on the second half: {ptl[seq_len:].mean():.3f}")
    fig = px.line(
        to_numpy(ptl), hover_name=rep_str[1:],
        title=f"Per token loss on sequence of length {seq_len} repeated twice",
        labels={"index": "Sequence position", "value": "Loss"}
    ).update_layout(showlegend=False, hovermode="x unified")
    fig.add_vrect(x0=0, x1=49.5, fillcolor="red", opacity=0.2, line_width=0)
    fig.add_vrect(x0=49.5, x1=99, fillcolor="green", opacity=0.2, line_width=0)
    fig.show()
```

### Looking for Induction Attention Patterns

The next natural thing to check for is the induction attention pattern.

First, go back to the attention patterns visualisation code from earlier and manually check for likely heads in the second layer. Which ones do you think might be serving as induction heads?
""")
    with st.expander("What you should see (only click after you've made your own observations):"):
        st.markdown("You should see that heads 4 and 10 are strongly induction-y, and the rest aren't.")

    st.markdown(r"""
Next, make an induction pattern score function, which looks for the average attention paid to the offset diagonal. Do this in the same style as our earlier head scorers.

Remember, the offset in your diagonal should be `- (seq_len - 1)`. This is because, if the sequence is periodic with period $k$, then then the destination token $T_{n+k}$ will attend to the source token $T_{n+1}$ (since that source token contains both the value $T_{n+1}$ itself which will be used to predict the token $T_{n+k+1}$, and the value of $T_n$ which matches $T_{n+k}$ (the latter information having been moved into $T_{n+1}$ by a prev-token head in layer 0)). See the diagram at the start of this page, if this still isn't clear.

```python
def induction_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember:
        The tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    '''
    pass

if MAIN:
    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))
```

If this function works as expected, then you should see output that matches the induction heads described in the dropdown above.

#### Logit Attribution

We can reuse our `logit_attribution` function from earlier to look at the contribution to the correct logit from each term on the first and second half of the sequence.

Gotchas:
* Remember to remove the batch dimension
* Remember to split the sequence in two, with one overlapping token (since predicting the next token involves removing the final token with no label) - your logit_attrs should both have shape [seq_len, 2*n_heads + 1] (ie [50, 25] here)

Note that the first plot will be pretty meaningless - can you see why?

```python
if MAIN:
    embed = rep_cache["hook_embed"]
    l1_results = rep_cache["blocks.0.attn.hook_result"]
    l2_results = rep_cache["blocks.1.attn.hook_result"]
    first_half_tokens = rep_tokens[0, : 1 + seq_len]
    second_half_tokens = rep_tokens[0, seq_len:]
    "TODO: YOUR CODE HERE"
    plot_logit_attribution(first_half_logit_attr, first_half_tokens)
    plot_logit_attribution(second_half_logit_attr, second_half_tokens)
```

#### Ablations

We can re-use our `get_ablation_scores` function from earlier to ablate each head and compare the change in loss.

Exercise: Before running this, what do you predict will happen? In particular, which cells will be significant?

```python
if MAIN:
    ablation_scores = get_ablation_scores(model, rep_tokens)
    imshow(ablation_scores, xaxis="Head", yaxis="Layer", caxis="logit diff", title="Logit Difference After Ablating Heads (detecting induction heads)", text_auto=".2f")
```""")
    with st.expander("Click here to see the output you should be getting:"):
        st_image("logit-ablations-2.png", 600)

    st.success("Bonus: Try ablating *every* head apart from the previous token head and the two induction heads. What does this do to performance? What if you mean ablate it, rather than zero ablating it?")
    st.markdown(r"""

## Conclusion - do you understand the circuit?

To end this section, try and summarise the induction head circuit in your own words. Your answer should reference at least one attention head in the 0th and 1st layers, and what their role in the circuit is.

You can use the dropdown below to check your understanding.
""")

    with st.expander("My summary of the algorithm"):
        st.markdown(r"""
* Head L0H7 is a previous token head (the QK-circuit ensures it always attends to the previous token).
* The OV circuit of head L0H7 writes a copy of the previous token in a *different* subspace to the one used by the embedding.
* The output of head L0H7 is used by the *key* input of head L1H4 via K-Composition to attend to 'the source token whose previous token is the destination token'.
* The OV-circuit of head L1H4 copies the *value* of the source token to the same output logit
    * Note that this is copying from the embedding subspace, *not* the L0H7 output subspace - it is not using V-Composition at all

To emphasise - the sophisticated hard part is computing the *attention* pattern of the induction head - this takes careful composition. The previous token and copying parts are fairly easy. This is a good illustrative example of how the QK circuits and OV circuits act semi-independently, and are often best thought of somewhat separately. And that computing the attention patterns can involve real and sophisticated computation!")""")

def section_4():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#mechanistic-analysis-of-induction-heads">Mechanistic Analysis of Induction Heads</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#reverse-engineering-ov-circuit-analysis">Reverse Engineering OV-Circuit Analysis</a></li>
        <li><a class="contents-el" href="#reverse-engineering-positional-embeddings--prev-token-head">Reverse Engineering Positional Embeddings + Prev Token Head</a></li>
        <li><a class="contents-el" href="#composition-analysis">Composition Analysis</a></li>
    </ul></li>
    <li><a class="contents-el" href="#further-exploration-of-induction-circuits">Further Exploration of Induction Circuits</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#composition-scores">Composition scores</a></li>
    </ul></li>
    <li><a class="contents-el" href="#looking-for-circuits-in-real-llms">Looking for Circuits in Real LLMs</a></li>
    <li><a class="contents-el" href="#training-your-own-toy-models">Training Your Own Toy Models</a></li>
    <li><a class="contents-el" href="#interpreting-induction-heads-during-training">Interpreting Induction Heads During Training</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
## Mechanistic Analysis of Induction Heads

Most of what we did above was feature analysis - we looked at activations (here just attention patterns) and tried to interpret what they were doing. Now we're going to do some mechanistic analysis - digging into the weights and using them to reverse engineer the induction head algorithm and verify that it is really doing what we think it is.

### Reverse Engineering OV-Circuit Analysis

Let's start with an easy parts of the circuit - the copying OV circuit of L1H4 and L1H10. Let's start with head 4. The only interpretable (read: **privileged basis**) things here are the input tokens and output logits, so we want to study the factored matrix $W_U W_O W_V W_E$. This is the matrix that combines with the attention pattern to get us from input to output. It might help to recall the formula for path decomposition of a transformer with a single attention layer:
""")

    st_image("math-framework-diagram.png", 800)
    st.markdown(r"")
    st.markdown(r"""
You should get a matrix `OV_circuit_full` with shape `[d_vocab, d_vocab]`.

We want to calculate this matrix, and inspect it. We should find that its diagonal values are very high, and its non-diagonal values are very low.

**Question - why should we expect this observation? (Try and come up with your own answer before reading.)**
""")

    with st.expander("Hint"):
        st.markdown(r"""
Take a repeated pattern `[A] [B] ... [A] -> [B]`. Your induction head is responsible for making sure the prediction at the second `[A]` token is `[B]`.

Try and work through each step of the matrix multiplication $(W_U W_{OV} W_E) x$, where $x$ is a one-hot encoding of the token $B$. In other words, first apply $W_E$, then $W_{OV}$, then $W_U$. What should you be left with at the end, if the induction head has done its job?
""")

    with st.expander("Answer"):
        st.markdown(r"""
The OV circuit $W_{OV}$ is meant to be a copy circuit. 

What does this mean? Well, if your repeated pattern is `[A] [B] ... [A] -> [B]`, then the OV circuit should take the embedding vector for the first `[B]`, and return a vector **which will be predicted to be `[B]`** when the unembedding matrix $W_U$ is applied. The QK circuit then makes sure this vector is moved from the first `[B]` to the second `[A]`.

If this still isn't clear, we can step through the matrix multiplications one at a time. Let $A$ and $B$ be our (1-hot encoded) tokens, then:

* $W_E B$ is the embedding vector of token $B$.
* $W_{OV} (W_E B)$ is the vector which gets moved from the first occurence of $B$ to the second occurrence of $A$ (in order to predict that $B$ will follow $A$)
* $W_U (W_{OV} W_E B)$ is the logit distribution corresponding to our prediction for the token following $A$. This should be $B$.

To make the last line more precise - the $(X, B)$th element of the matrix $W_U W_{OV} W_E$ is $X^T (W_U W_{OV} W_E) B$, which is the logit corresponding to the prediction that token $X$ will follow the second instance of $A$. We want high probability to be placed on $X=B$ and smaller probability on all other $X$ - in other words, the diagonal entry should be much larger than all other entries in the column.

(If the last part is confusing, note that we can get next-token probabilities from the matrix $W_U W_{OV} W_E$ by taking the softmax over columns. Having the largest element of each column be on the diagonal is equivalent to this probability matrix being the identity.)

---

To describe this all in much fewer words:

***The full OV circuit of a head describes how a token affects the output logits of the predicted token. This head's job is to copying - it takes as input the first instance of `[B]`, and it outputs `[B]` (which is then moved to the second `[A]`, to be used as the prediction for the following token). So it maps `[B]` $\to$ `[B]`, i.e. it is the identity map.***
""")
    st.markdown(r"")
#     st.info(r"""
# Reminder - you should think of the tensor product $A^h \otimes W_U W_O W_V W_E$ as the function that sends $t$ (our vector of tokens) to:

# $$
# A^h \;t\; (W_U W_O W_V W_E)^T = A^h \;t\; (W_E^T W_V^T W_O^T W_U^T) = A^h \;x\; W_V^T W_O^T W_U^T
# $$

# Where I've written $x = tW_E^T$ as our tensor of shape `(seq_len, hidden_size)`, which represents the embedding of each token (i.e. the initial residual stream).
# """)

    st.info(r"""
Tip: If you start running out of CUDA memory, cast everything to float16 (`tensor` -> `tensor.half()`) before multiplying - 50K x 50K matrices are large! Alternately, do the multiply on CPU if you have enough CPU memory. This should take less than a minute.

Note: on some machines like M1 Macs, half precision can be much slower on CPU - try doing a `%timeit` on a small matrix before doing a huge multiplication!

If none of this works, you might have to use LambdaLabs for these exercises (I had to!). Here are a list of `pip install`'s that you'll need to run, to save you time:

```python
!pip install git+https://github.com/neelnanda-io/TransformerLens.git@new-demo
!pip install circuitsvis
!pip install fancy_einsum
!pip install einops
!pip install plotly
!pip install torchtyping
!pip install typeguard
```
""")

    st.markdown(r"""
```python
if MAIN:
    head_index = 4
    layer = 1
    OV_circuit_full = None # replace with the matrix calculation W_U W_O W_V W_E
```

Now we want to check that this matrix is the identity. """)

    

    st.markdown(r"""This is a surprisingly big pain! It's a 50K x 50K matrix, which is far too big to visualise. And in practice, this is going to be fairly noisy. And we don't strictly need to get it to be the identity, just have big terms along the diagonal.

First, to validate that it looks diagonal-ish, let's pick 200 random rows and columns and visualise that - it should at least look identity-ish here!


```python
if MAIN:
    rand_indices = t.randperm(model.cfg.d_vocab)[:200]
    px.imshow(to_numpy(OV_circuit_full[rand_indices][:, rand_indices])).show()
```

We might  to take softmax over rows and show that the result is close to the identity matrix. Unfortunately, this is numerically unstable, because our matrix is so large. So we'll have to come up with a summary statistic that works on the matrix of logits.

Now we want to try to make a summary statistic to capture this. We might first think of just taking softmax along each column, and checking how close the result is to the identity matrix. However, in practice it won't actually be very close to the identity matrix (since this entire calculation is very noisy and involves logit outputs over a huge (size-50k) vocabulary. You can try this if you like (you'll probably have to loop over the columns, to avoid memory errors).

**Accuracy** is a good summary statistic - what fraction of the time is the largest logit in a column on the diagonal? Even if there's lots of noise, you'd probably still expect the largest logit to be on the diagonal a good deal of the time.

Bonus exercise: Top-5 accuracy is also a good metric (use `t.topk`, take the indices output)

When I run this I get about 30.8% - pretty underwhelming. It goes up to 47.72% for top-5. What's up with that?

```python
def top_1_acc(OV_circuit_full):
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    pass

if MAIN:
    print("Fraction of the time that the best logit is on the diagonal:")
    print(top_1_acc(OV_circuit_full))
```

Now we return to why we have *two* induction heads. If both have the same attention pattern, the effective OV circuit is actually $W_U(W_O^{[1,4]}W_V^{[1,4]}+W_O^{[1,10]}W_V^{[1,10]})W_E$, and this is what matters. So let's calculate this and re-run our analysis on that!""")

    with st.expander("Exercise: Why might the model want to split the circuit across two heads?"):
        st.markdown(r"""
Because $W_O W_V$ is a rank 64 matrix. The sum of two is a rank 128 matrix. This can be a significantly better approximation to the desired 50K x 50K matrix!""")

    st.markdown(r"""
```python
if MAIN:
    try:
        del OV_circuit_full
    except:
        pass
    "TODO: YOUR CODE HERE, DEFINE OV_circuit_full_both"
    print("Top 1 accuracy for the full OV Circuit:", top_1_acc(OV_circuit_full_both))
    try:
        del OV_circuit_full_both
    except:
        pass
```

### Reverse Engineering Positional Embeddings + Prev Token Head

The other easy circuit is the QK-circuit of L0H7 - how does it know to be a previous token circuit?

We can multiply out the full QK circuit via the positional embeddings: $W_\text{pos}^T W_Q^{[0,7]\,T} W_K^{[0,7]} W_\text{pos}$ to get a matrix `pos_by_pos` of shape `[max_ctx, max_ctx]` (max ctx = max context length, i.e. maximum length of a sequence we're allowing, which is set by our choice of dimensions in $W_\text{pos}$).

We can then mask it and apply a softmax, and should get a clear stripe on the lower diagonal (Tip: Click and drag to zoom in, hover over cells to see their values and indices!)

**Exercise - put in your own words, why we should expect this matrix to be a diagonal stripe.** Try to come up with an answer before looking at the answer below!
""")

    with st.expander("Answer"):
        st.markdown(r"""
The QK circuit $W_Q^T W_K$ has shape `[d_model, d_model]`. The attention patterns are created by right and left multiplying it by the source and destination embeddings respectively:

$$
(x_\text{pos}^\text{dest})^T \,W_Q^T\, W_K \,(x_\text{pos}^\text{src})
$$

where $x_\text{pos}^\text{src}$ stands for the positional embedding of the source token, etc.

We want this to be large when the position index of $x_\text{pos}^\text{src}$ is one **smaller** than the position index of $x_\text{pos}^\text{dest})$ (because the destination token needs to attend to the **previous token**). This is equivalent to saying that the $(i, j)$ th entry of the matrix $W_\text{pos}^T W_Q^T W_K W_\text{pos}$ (after applying softmax) should be close to 1 when $j = i - 1$, and close to zero everywhere else. This is equivalent to having a diagonal stripe under the major diagonal.
""")
    st.markdown(r"""
Hints:
* Remember to divide by sqrt(d_head)!
* Use the `mask_scores` function we've provided you with

(Note: If we were being properly rigorous, we'd also need to show that the token embedding wasn't important for the attention scores.)""")

    st.info(r"""If you're using VSCode, remember to periodically clear the plotly graphs from your screen; they can slow down your performance by quite a bit!""")

    st.markdown(r"""
```python
def mask_scores(
    attn_scores: TT["query_d_model", "key_d_model"]
):
    '''Mask the attention scores so that tokens don't attend to previous tokens.'''
    mask = t.tril(t.ones_like(attn_scores)).bool()
    neg_inf = t.tensor(-1.0e6).to(attn_scores.device)
    masked_attn_scores = t.where(mask, attn_scores, neg_inf)
    return masked_attn_scores

if MAIN:
    "TODO: YOUR CODE HERE"
    imshow(to_numpy(pos_by_pos_pattern[:200, :200]), xaxis="Key", yaxis="Query")
```

### Composition Analysis

We now dig into the hard part of the circuit - demonstrating the K-Composition between the previous token head and the induction head.

#### Splitting activations

We can repeat the trick from the logit attribution scores. The QK-input for layer 1 is the sum of 14 terms (2+n_heads) - the token embedding, the positional embedding, and the results of each layer 0 head. So for each head in layer 1, the query tensor (ditto key) corresponding to sequence position $i$ is:

$$
\begin{align*}
W^{[1,4]}_Q x &= W^{[1,4]}_Q (e + pe + \sum_{h=0}^{11} x^h) \\
&= W^{[1,4]}_Q e + W^{[1,4]}_Q pe + \sum_{h=0}^{11} W^{[1,4]}_Q x^h
\end{align*}
$$

where $e$ stands for the token embedding, $pe$ for the positional embedding, and $x^h$ for the output of head $h$ at sequence position $i$. All these tensors have shape `[seq, d_model]`. So we can treat the expression above as a sum of matrix multiplications of dimensions `[d_k, d_model] @ [seq, d_model] -> [seq, d_k]`. 

We can now analyse the relative importance of these 14 terms! A very crude measure is to take the norm of each term (by component and position) - when we do this here, we show clear dominance in the k from L0H7, and in the q from the embed (and pos embed).

Note that this is a pretty dodgy metric - q and k are not inherently interpretable! But it can be a good and easy-to-compute proxy.""")

    with st.expander("Question - why are Q and K not inherently interpretable? Why might the norm be a good metric in spite of this?"):
        st.markdown(r"""
They are not inherently interpretable because they operate on the residual stream, which doesn't have a **privileged basis**. You could stick a rotation matrix $R$ after all of the Q, K and V weights (and stick a rotation matrix before everything that writes to the residual stream), and the model would still behave exactly the same.

The reason taking the norm is still a reasonable thing to do is that, despite the individual elements of these vectors not being inherently interpretable, it's still a safe bet that if they are larger than they will have a greater overall effect on the residual stream. So looking at the norm doesn't tell us how they work, but it does indicate which ones are more important.
""")

    with st.expander("Help - I'm confused about how to implement this."):
        st.markdown(r"""
`decomposed_q` is a tensor with shape `[query_component, query_pos, d_head]`. We can write it as follows:

$$
\left[\begin{array}{c}
W^{[1,4]}_Q e \\
W^{[1,4]}_Q pe \\
W^{[1,4]}_Q x^0 \\
W^{[1,4]}_Q x^1 \\
\vdots \\
W^{[1,4]}_Q x^{11}\end{array}\right]
$$

where each of these 13 terms is a matrix of shape `[query_pos, d_head]`.
""")

    st.markdown(r"""

```python
def decompose_qk_input(cache: dict) -> t.Tensor:
    '''
    Output is decomposed_qk_input, with shape [2+num_heads, position, d_model]
    '''
    pass


def decompose_q(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_q with shape [2+num_heads, position, d_head] (such that sum along axis 0 is just q)
    '''
    pass


def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_k with shape [2+num_heads, position, d_head] (such that sum along axis 0 is just k) - exactly analogous as for q
    '''
    pass


if MAIN:
    ind_head_index = 4
    decomposed_qk_input = decompose_qk_input(rep_cache)
    t.testing.assert_close(decomposed_qk_input.sum(0), rep_cache["resid_pre", 1][0] + rep_cache["pos_embed"][0], rtol=0.01, atol=1e-05)
    decomposed_q = decompose_q(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(decomposed_q.sum(0), rep_cache["blocks.1.attn.hook_q"][0, :, ind_head_index], rtol=0.01, atol=0.001)
    decomposed_k = decompose_k(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(decomposed_k.sum(0), rep_cache["blocks.1.attn.hook_k"][0, :, ind_head_index], rtol=0.01, atol=0.01)
    component_labels = ["Embed", "PosEmbed"] + [f"L0H{h}" for h in range(model.cfg.n_heads)]
    imshow(to_numpy(decomposed_q.pow(2).sum([-1])), xaxis="Pos", yaxis="Component", title="Norms of components of query")
    imshow(to_numpy(decomposed_k.pow(2).sum([-1])), xaxis="Pos", yaxis="Component", title="Norms of components of key")

```""")

    with st.expander("A technical note on the positional embeddings - reading optional, feel free to skip this."):
        st.markdown(r"""
You might be wondering why the tests compare the decomposed qk sum with the sum of the `resid_pre + pos_embed`, rather than just `resid_pre`. The answer lies in how we defined the transformer, specifically in this line from the config:

```python
positional_embedding_type="shortformer"
```

The result of this is that the positional embedding isn't added to the residual stream. Instead, it's added as inputs to the Q and K calculation (i.e. we calculate `W^{[1,4]}_Q @ (resid_pre + pos_embed)` and same for `W^{[1,4]}_K`), but **not** as inputs to the V calculation (i.e. we just calculate `W_V @ resid_pre`). This isn't actually how attention works in general, but for our purposes it makes the analysis of induction heads cleaner because we don't have positional embeddings interfering with the OV circuit.

Thus, the value `resid_pre` in the cache is actually equal to the input to our Q and K matrices **subtract the positional embedding**, and we need to add it back on before comparing the two.

Don't worry too much about this though, it's only mentioned here for the sake of completeness!
""")

    st.markdown(r"""
This tells us which heads are probably important, we can do better than that. For each of our heads $h$, we can decompose the output $x^h$

We can do one better, and take the decomposed attention scores. This is a bilinear function of q and k, and so we will end up with a `decomposed_scores` tensor with shape `[query_component, key_component, query_pos, key_pos]`, where summing along BOTH of the first axes will give us the original attention scores (pre-mask).

Implement the function giving the decomposed scores (remember to scale by `sqrt(d_k)`!) For now, don't mask it.

We can now look at the standard deviation across the key and query positions for each pair of components. This is a proxy for 'how much the attention pattern depends on that component for the query and for the key. And we can plot a num_components x num_components heatmap to see how important each pair is - this again clearly shows the pair of Q=Embed, K=L0H7 dominates.

We can even plot the attention scores for that component and see a clear induction stripe.

""")
    with st.expander("Exercise: Do you expect this to be symmetric? Why/why not?"):
        st.markdown(r"""
No, because the y axis is the component in the *query*, the x axis is the component in the *key* - these are not symmetric!
""")
    with st.expander("Exercise: Why do I focus on the attention scores, not the attention pattern? (i.e. pre softmax not post softmax)"):
        st.markdown(r"""

Because the decomposition trick *only* works for things that are linear - softmax isn't linear and so we can no longer consider each component independently.
""")
# `decomposed_q` is a tensor with shape `[query_component, query_pos, d_head]`. We can write it as follows:

# $$
# \left[\begin{array}{c}
# W^{[1,4]}_Q e \\
# W^{[1,4]}_Q pe \\
# W^{[1,4]}_Q x^0 \\
# W^{[1,4]}_Q x^1 \\
# \vdots \\
# W^{[1,4]}_Q x^{11}\end{array}\right]
# $$

# where each of these 13 terms is a matrix of shape `(seq_pos, d_head)`.

# We can also write it as a matrix:

# $$
# \begin{bmatrix}
# W^{[1,4]}_Q e_0 & W^{[1,4]}_Q e_1 & ... & W^{[1,4]}_Q e_{n-1} \\
# W^{[1,4]}_Q pe_0 & W^{[1,4]}_Q pe_1 & ... & W^{[1,4]}_Q pe_{n-1} \\
# W^{[1,4]}_Q x^0_0 & W^{[1,4]}_Q x^0_1 & ... & W^{[1,4]}_Q x^0_{n-1} \\
# \vdots \\
# W^{[1,4]}_Q x^{11}_0 & W^{[1,4]}_Q x^{11}_1 & ... & W^{[1,4]}_Q x^{11}_{n-1} \\
# \end{bmatrix}
# $$

# where $n$ is the sequence length, and each element in this matrix is a vector of length `d_head`.

# We can also do the same thing with `decomposed_k`:

# $$
# \begin{bmatrix}
# W^{[1,4]}_K e_0 & W^{[1,4]}_K e_1 & ... & W^{[1,4]}_K e_{n-1} \\
# W^{[1,4]}_K pe_0 & W^{[1,4]}_K pe_1 & ... & W^{[1,4]}_K pe_{n-1} \\
# W^{[1,4]}_K x^0_0 & W^{[1,4]}_K x^0_1 & ... & W^{[1,4]}_K x^0_{n-1} \\
# \vdots \\
# W^{[1,4]}_K x^{11}_0 & W^{[1,4]}_K x^{11}_1 & ... & W^{[1,4]}_K x^{11}_{n-1} \\
# \end{bmatrix}
# $$
    with st.expander("Help - I'm confused about how to implement this."):
        st.markdown(r"""
To calculate the attention position $i$ pays to position $j$, we take $W^{[1,4]}_Q x_i = W^{[1,4]}_Q (e_i + pe_i + \sum_{h=0}^{11}x^h)$, and $W^{[1,4]}_K x_j$ (defined similarly), then take the dot product between these two vectors:

$$
\text{attn\_scores}_{i,j} = x_i^T W^{[1,4]\,T}_Q W^{[1,4]}_K x_j
$$

By writing $x_i$ and $x_j$ each as a sum of `n_components` terms, we can write this expression as a sum of `n_components ** 2` terms:

$$
\text{attn\_scores}_{i,j} = e_i^T W^{[1,4]\,T}_Q W^{[1,4]}_K e_j + e_i^T W^{[1,4]\,T}_Q W^{[1,4]}_K pe_j + e_i^T W^{[1,4]\,T}_Q W^{[1,4]}_K x^0_j + ... + (x_i^{11})^T W^{[1,4]\,T}_Q W^{[1,4]}_K x_j^{11}
$$

For instance, the interpretation of $x_{i^{h_1}}^T W^{[1,4]\,T}_Q W^{[1,4]}_K x_{j^{h_2}}$ is ***"the component of the $(i, j)$th attention scores, with the query being supplied from the output of head $h_1$ in layer 0, and the key being output from head $h_2$ in layer 0."***

---

Your matrix should have shape `[query_component, key_component, query_pos, key_pos]`, where the `[:, :, i, j]`th element is a matrix of precisely these `n_components ** 2` terms in the expression for $\text{attn\_scores}_{i,j}$.
""")
    with st.expander("Help - I'm confused about why we're doing this."):
        st.markdown(r"""
We want to show that the main contributors to the distinctive `seq_len-1`-offset pattern we see in our L1H4 and L1H10 induction heads are as follows:

* The important keys are mainly produced from the output of L0H7
* The important queries mainly just come straight from the token embeddings

since this is precisely the structure of the induction head formed via K-composition.

The first plot will show us the attention scores contribution from `[query_component, key_component] = [Embed, L0H7]`. We hope this shows us the characteristic induction head pattern of the `seq_len-1` offset. 

The second plot will show us the standard deviation over the query and key positions for each of the attention scores corresponding to a pair of query and key components. We hope this will be very large for the combination `[query_component, key_component] = [Embed, L0H7]`, and very small for the other components (because our theory is that the dominant pattern is just the `seq_len-1` offset and it gets produced by the contribution from these two components, and the other components basically just contribute a small amount of noise that doesn't change the pattern).
""")
    st.markdown(r"""

```python
def decompose_attn_scores(decomposed_q: t.Tensor, decomposed_k: t.Tensor) -> t.Tensor:
    pass


if MAIN:
    decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k)
    decomposed_stds = reduce(
        decomposed_scores, "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", t.std
    )
    # First plot: std dev over query and key positions, shown by component
    imshow(to_numpy(t.tril(decomposed_scores[0, 9])), title="Attention Scores for component from Q=Embed and K=Prev Token Head")
    # Second plot: attention score contribution from (query_component, key_component) = (Embed, L0H7)
    imshow(to_numpy(decomposed_stds), xaxis="Key Component", yaxis="Query Component", title="Standard deviations of components of scores", x=component_labels, y=component_labels)

```

#### Interpreting the K-Composition Circuit
Now we know that head L1H4 is composing with head L0H7 via K composition, we can multiply through to create a full end-to-end circuit:

$$
W_E^T W_Q^{[1, 4]T} W_K^{[1, 4]} W_O^{[0, 7]} W_V^{[0, 7]} W_E
$$

and verify that it's the identity. (Note, when we say identity here, we're again thinking about it as a distribution over logits, and so we'll be using our previous metric of `top_1_acc`.)

""")
    with st.expander("Help - I don't understand why this should be the identity."):
        st.markdown(r"""
This one is a bit more confusing, and I'd recommend reading my diagram to understand exactly what's going on here. However, I'll try to offer a brief version here.

---

Let $\color{red}A$ and $\color{red}B$ be one-hot encoded vectors, representing tokens `[A]` and `[B]` in the repeating pattern `[A] [B] ... [A] [B]`. The $(\color{red}A\color{black}, \color{red}A\color{black})$th element of the matrix above is:

$$
\color{red}A^T\color{black}W_E^T W_Q^{[1, 4]T} W_K^{[1, 4]} W_O^{[0, 7]} W_V^{[0, 7]} W_E \color{red}A
$$

The $OV^{[0, 7]}$-circuit is designed to give us the information that will then be copied one position forwards by the $QK^{[0, 7]}$ circuit. In other words, $W_O^{[0, 7]} W_V^{[0, 7]} W_E \color{red}B$ is a vector in the residual stream which gets stored in the second `[B]` token, and represents the information ***"the token before this one is `[A]`"***.

The $QK^{[1, 4]}$-circuit is designed to produce a high attention score between this token and the second instance of `[A]`. In other words, the query vector $q := W_Q^{[1, 4]} W_E \color{red}A$ and the key vector $k := W_K^{[1, 4]} (W_O^{[0, 7]} W_V^{[0, 7]} W_E\color{red}B\color{black})$ should have high attention score $q^T k$.

Thus, we've argued that the $(\color{red}A\color{black}, \color{red}A\color{black})$th element of this matrix will be large. But if we instead look at the $(\color{red}A\color{black}, \color{red}X\color{black})$th element for some arbitrary $X$, this would result in a different key vector $k' := W_K^{[1, 4]} (W_O^{[0, 7]} W_V^{[0, 7]} W_E\color{red}X\color{black})$. This stores the information ***"the token before this one is `[X]`"***, so this doesn't give us information about the token following `[A]`, and we *don't* want our second `[A]` token to attend to it. So the value $q^T k'$ should be small.

---

Note - this actually shows why the largest element of each **row** of the matrix should be the diagonal one, rather than the largest element on each column. It's important to get this the right way round - remember that logits are invariant to the addition of a constant, so it's meaningless to compare across two different logit distributions! That's why, in the code below, we've applied the test to the transpose of the function's output.
""")

# We can now reuse our `top_1_acc` code from before to check that it's identity-like, we see that half the time the diagonal is the top (goes up to 89% with top 5 accuracy) (We transpose first, because we want the argmax over the key dimension)

    st.markdown(r"""
Remember to cast to float16 (`tensor `-> `tensor.half()`) to stop your GPU getting too full!


```python
def find_K_comp_full_circuit(prev_token_head_index, ind_head_index):
    '''
    Returns a vocab x vocab matrix, with the first dimension being the query side and the second dimension being the key side (going via the previous token head)
    '''
    pass


if MAIN:
    prev_token_head_index = 7
    K_comp_circuit = find_K_comp_full_circuit(prev_token_head_index, ind_head_index)
    print("Fraction of tokens where the highest activating key is the same token", top_1_acc(K_comp_circuit.T).item())
    del K_comp_circuit
```

As a bonus - recall that we found the $OV^{[1, 4]}$ and $OV^{[1, 10]}$ circuits both did some of the job of attending to previous instances of the token. Here, we've used the previously-set value of `ind_head_index = 4`. Try changing this to `ind_head_index = 10` and see whether you get a better result. Try looking at the matrix $W_E^T (W_Q^{[1, 4]T} W_K^{[1, 4]} + W_Q^{[1, 10]T} W_K^{[1, 10]}) W_O^{[0, 7]} W_V^{[0, 7]} W_E$ corresponding to the full circuit, and see if you get a still better result!


## Further Exploration of Induction Circuits

I now consider us to have fully reverse engineered an induction circuit - by both interpreting the features and by reverse engineering the circuit from the weights. But there's a bunch more ideas that we can apply for finding circuits in networks that are fun to practice on induction heads, so here's some bonus content - feel free to skip to the later bonus ideas though.

### Composition scores

A particularly cool idea in the paper is the idea of [virtual weights](https://transformer-circuits.pub/2021/framework/index.html#residual-comms), or compositional scores. (though I came up with it, so I'm deeply biased) This is used [to identify induction heads](https://transformer-circuits.pub/2021/framework/index.html#analyzing-a-two-layer-model)

The key idea of compositional scores is that the residual stream is a large space, and each head is reading and writing from small subspaces. By defaults, any two heads will have little overlap between their subspaces (in the same way that any two random vectors have almost zero dot product in a large vector space). But if two heads are deliberately composing, then they will likely want to ensure they write and read from similar subspaces, so that minimal information is lost. As a result, we can just directly look at "how much overlap there is" between the output space of the earlier head and the K, Q, or V input space of the later head. We represent the output space with $W_OV=W_OW_V$, and the input space with $W_QK^T=W_K^TW_Q$ (for Q-composition), $W_QK=W_Q^TW_K$ (for K-Composition) or $W_OV=W_OW_V$ (for V-Composition, of the later head). Call these matrices $W_A$ and $W_B$ respectively.

How do we formalise overlap? This is basically an open question, but a surprisingly good metric is $\frac{|W_BW_A|}{|W_B||W_A|}$ where $|W|=\sum_{i,j}W_{i,j}^2$ is the Frobenius norm, the sum of squared elements. Let's calculate this metric for all pairs of heads in layer 0 and layer 1 for each of K, Q and V composition and plot it.

""")
    with st.expander("Why do we use W_OV as the output weights, not W_O? (and W_QK not W_Q or W_K, etc)"):
        st.markdown(r"""

Because W_O is arbitrary - we can apply an arbitrary invertible matrix to W_O and its inverse to W_V and preserve the product W_OV. Though in practice, it's an acceptable approximation.
""")

    st.markdown(r"""

```python
def frobenius_norm(tensor):
    '''
    Implicitly allows batch dimensions
    '''
    return tensor.pow(2).sum([-2, -1])


def get_q_comp_scores(W_QK, W_OV):
    '''
    Returns a layer_1_index x layer_0_index tensor, where the i,j th entry is the Q-Composition score from head L0Hj to L1Hi
    '''
    pass


def get_k_comp_scores(W_QK, W_OV):
    '''
    Returns a layer_1_index x layer_0_index tensor, where the i,j th entry is the K-Composition score from head L0Hj to L1Hi
    '''
    pass


def get_v_comp_scores(W_OV_1, W_OV_0):
    '''
    Returns a layer_1_index x layer_0_index tensor, where the i,j th entry is the V-Composition score from head L0Hj to L1Hi
    '''
    pass


if MAIN:
    W_O = model.blocks[0].attn.W_O
    W_V = model.blocks[0].attn.W_V
    W_OV_0 = t.einsum("imh,ihM->imM", W_O, W_V)
    W_Q = model.blocks[1].attn.W_Q
    W_K = model.blocks[1].attn.W_K
    W_V = model.blocks[1].attn.W_V
    W_O = model.blocks[1].attn.W_O
    W_QK = t.einsum("ihm,ihM->imM", W_Q, W_K)
    W_OV_1 = t.einsum("imh,ihM->imM", W_O, W_V)
    q_comp_scores = get_q_comp_scores(W_QK, W_OV_0)
    k_comp_scores = get_k_comp_scores(W_QK, W_OV_0)
    v_comp_scores = get_v_comp_scores(W_OV_1, W_OV_0)
    px.imshow(
        to_numpy(q_comp_scores),
        y=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="Q Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()
    px.imshow(
        to_numpy(k_comp_scores),
        y=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="K Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()
    px.imshow(
        to_numpy(v_comp_scores),
        y=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="V Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()

```

#### Setting a Baseline

To interpret the above graphs we need a baseline! A good one is what the scores look like at initialisation. Make a function that randomly generates a composition score 200 times and tries this. Remember to generate 4 [d_head, d_model] matrices, not 2 [d_model, d_model] matrices! This model was initialised with Kaiming Uniform Initialisation:

```python
W = t.empty(shape)
nn.init.kaiming_uniform_(W, a=np.sqrt(5))
```

(Ideally we'd do a more efficient generation involving batching, and more samples, but we won't worry about that here)


```python
def generate_single_random_comp_score() -> float:
    '''
    Write a function which generates a single composition score for random matrices
    '''
    pass


if MAIN:
    comp_scores_baseline = np.array([generate_single_random_comp_score() for i in range(200)])
    print("Mean:", comp_scores_baseline.mean())
    print("Std:", comp_scores_baseline.std())
    px.histogram(comp_scores_baseline, nbins=50).show()

```

We can re-plot our above graphs with this baseline set to white. Look for interesting things in this graph!


```python
if MAIN:
    px.imshow(
        to_numpy(q_comp_scores),
        y=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="Q Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    ).show()
    px.imshow(
        to_numpy(k_comp_scores),
        y=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="K Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    ).show()
    px.imshow(
        to_numpy(v_comp_scores),
        y=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="V Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    ).show()

```

#### Theory + Efficient Implementation

So, what's up with that metric? The key is a cute linear algebra result that the Frobenius norm is equal to the sum of the squared singular values.
""")
    with st.expander("Proof"):
        st.markdown(r"""
M = USV in the singular value decomposition. U and V are rotations and do not change norm, so |M|=|S|
""")
    st.markdown(r"""
So if $W_A=U_AS_AV_A$, $W_B=U_BS_BV_B$, then $|W_A|=|S_A|$, $|W_B|=|S_B|$ and $|W_AW_B|=|S_AV_AU_BS_B|$. In some sense, $V_AU_B$ represents how aligned the subspaces written to and read from are, and the $S_A$ and $S_B$ terms weights by the importance of those subspaces.

We can also use this insight to write a more efficient way to calculate composition scores - this is extremely useful if you want to do this analysis at scale! The key is that we know that our matrices have a low rank factorisation, and it's much cheaper to calculate the SVD of a narrow matrix than one that's large in both dimensions. See the [algorithm described at the end of the paper](https://transformer-circuits.pub/2021/framework/index.html#induction-heads:~:text=Working%20with%20Low%2DRank%20Matrices) (search for SVD). Go implement it!

Gotcha: Note that `t.svd(A)` returns `(U, S, V.T)` not `(U, S, V)`

Bonus exercise: Write a batched version of this that works for batches of heads, and run this over GPT-2 - this should be doable for XL, I think.



```python
def stranded_svd(A: t.Tensor, B: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    '''
    Returns the SVD of AB in the torch format (ie (U, S, V^T))
    '''
    pass


def stranded_composition_score(W_A1: t.Tensor, W_A2: t.Tensor, W_B1: t.Tensor, W_B2: t.Tensor):
    '''
    Returns the composition score for W_A = W_A1 @ W_A2 and W_B = W_B1 @ W_B2, with the entries in a low-rank factored form
    '''
    pass

```

#### Targeted Ablations

We can refine the ablation technique to detect composition by looking at the effect of the ablation on the attention pattern of an induction head, rather than the loss. Let's implement this!

Gotcha - by default, run_with_hooks removes any existing hooks when it runs, if you want to use caching set the reset_hooks_start flag to False


```python
def ablation_induction_score(prev_head_index: int, ind_head_index: int) -> t.Tensor:
    '''
    Takes as input the index of the L0 head and the index of the L1 head, and then runs with the previous token head ablated and returns the induction score for the ind_head_index now.
    '''

    def ablation_hook(v, hook):
        v[:, :, prev_head_index] = 0.0
        return v

    def induction_pattern_hook(attn, hook):
        hook.ctx[prev_head_index] = attn[0, ind_head_index].diag(-(seq_len - 1)).mean()

    model.run_with_hooks(
        rep_tokens,
        fwd_hooks=[("blocks.0.attn.hook_v", ablation_hook), ("blocks.1.attn.hook_attn", induction_pattern_hook)],
    )
    return model.blocks[1].attn.hook_attn.ctx[prev_head_index]


if MAIN:
    for i in range(model.cfg.n_heads):
        print(f"Ablation effect of head {i}:", ablation_induction_score(i, 4).item())

```

# Bonus

## Looking for Circuits in Real LLMs

A particularly cool application of these techniques is looking for real examples of circuits in large language models. Fortunately, there's a bunch of open source ones we can play around with! I've made a library for transformer interpretability called EasyTransformer. It loads in an open source LLMs into a simplified transformer, and gives each activation a unique name. With this name, we can set a hook that accesses or edits that activation, with the same API that we've been using on our 2L Transformer. You can see it in `w2d4_easy_transformer.py` - feedback welcome!

**Example:** Ablating the 5th attention head in layer 4 of GPT-2 medium


```python
from w2d4_easy_transformer import EasyTransformer

model = EasyTransformer("gpt2-medium")
text = "Hello world"
input_tokens = model.to_tokens(text)
head_index = 5
layer = 4


def ablation_hook(value, hook):
    value[:, :, head_index, :] = 0.0
    return value


logits = model.run_with_hooks(input_tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_v", ablation_hook)])

```

This library should make it moderately easy to play around with these models - I recommend going wild and looking for interesting circuits!

This part of the day is deliberately left as an unstructured bonus, so I recommend following your curiosity! But if you want a starting point, here are some suggestions:
- Look for induction heads - try repeating all of the steps from above. Do they follow the same algorithm?
- Look for neurons that erase info
    - Ie having a high negative cosine similarity between the input and output weights
- Try to interpret a position embedding
""")
    with st.expander("Positional Embedding Hint"):
        st.markdown(r"""
Look at the singular value decomposition `t.svd` and plot the principal components over position space. High ones tend to be sine and cosine waves of different frequencies.

**Gotcha:** The output of `t.svd` is `U, S, Vh = t.svd(W_pos)`, where `U @ S.diag() @ Vh.T == W_pos` - W_pos has shape [d_model, n_ctx], so the ith principal component on the n_ctx side is `W_pos[:, i]` NOT `W_pos[i, :]
""")
    st.markdown(r"""

- Look for heads with interpretable attention patterns: Eg heads that attend to the same word (or subsequent word) when given text in different languages, or the most recent proper noun, or the most recent full-stop, or the subject of the sentence, etc.
    - Pick a head, ablate it, and run the model on a load of text with and without the head. Look for tokens with the largest difference in loss, and try to interpret what the head is doing.
- Try replicating some of Kevin's work on indirect object vs
- Inspired by the [ROME paper](https://rome.baulab.info/), use the causal tracing technique of patching in residual stream - can you analyse how the network answers different facts?

Note: I apply several simplifications to the resulting transformer - these leave the model mathematically equivalent and doesn't change the output log probs, but does somewhat change the structure of the model and one change translates the output logits by a constant - see [Discussion](https://colab.research.google.com/drive/1_tH4PfRSPYuKGnJbhC1NqFesOYuXrir_#scrollTo=Discussion) for some discussion of these.

## Training Your Own Toy Models

A fun exercise is training models on the minimal task that'll produce induction heads - predicting the next token in a sequence of random tokens with repeated subsequences. You can get a small 2L Attention-Only model to do this.
""")
    with st.expander("Tips"):
        st.markdown(r"""
* Make sure to randomise the positions that are repeated! Otherwise the model can just learn the boring algorithm of attending to fixed positions
* It works better if you *only* evaluate loss on the repeated tokens, this makes the task less noisy.
* It works best with several repeats of the same sequence rather than just one.
* If you do things right, and give it finite data + weight decay, you *should* be able to get it to grok - this may take some hyper-parameter tuning though.
* When I've done this I get weird franken-induction heads, where each head has 1/3 of an induction stripe, and together cover all tokens.
* It'll work better if you only let the queries and keys access the positional embeddings, but *should* work either way
""")
    st.markdown(r"""

## Interpreting Induction Heads During Training

A particularly striking result about induction heads is that they consistently [form very abruptly in training as a phase change](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#argument-phase-change), and are such an important capability that there is a [visible non-convex bump in the loss curve](https://wandb.ai/mechanistic-interpretability/attn-only/reports/loss_ewma-22-08-24-22-08-00---VmlldzoyNTI2MDM0?accessToken=r6v951q0e1l4q4o70wb2q67wopdyo3v69kz54siuw7lwb4jz6u732vo56h6dr7c2) (in this model, approx 2B to 4B tokens). I have a bunch of checkpoints for this model, you can try re-running the induction head detection techniques on intermediate checkpoints and see what happens. (Bonus points if you have good ideas for how to efficiently send you a bunch of 300MB checkpoints from Wandb lol)

""")

# def section_home():
#     st.markdown(r'''Coming soon!""")

# def section_1():
#     pass

# def section_2():
#     pass

func_list = [section_home, section_1, section_2, section_3, section_4]

page_list = ["🏠 Home", "1️⃣ TransformerLens: Introduction", "2️⃣ TransformerLens: Features", "3️⃣ Finding induction heads", "4️⃣ Reverse-engineering induction heads"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()
