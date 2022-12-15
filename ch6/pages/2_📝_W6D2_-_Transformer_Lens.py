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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from torchtyping import TensorType as TT
from typing import List, Union, Optional
from functools import partial
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
torch.set_grad_enabled(False)

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)
    return px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)
    return px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)
    return px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs)
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
device = "cuda" if torch.cuda.is_available() else "cpu"

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
ioi_patching_result = torch.zeros((model.cfg.n_layers, num_positions), device=model.cfg.device)

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
random_tokens = torch.randint(1000, 10000, (batch_size, seq_len)).to(model.cfg.device)
repeated_tokens = einops.repeat(random_tokens, "batch seq_len -> batch (2 seq_len)")
repeated_logits = model(repeated_tokens)
correct_log_probs = model.loss_fn(repeated_logits, repeated_tokens, per_token=True)
loss_by_position = einops.reduce(correct_log_probs, "batch position -> position", "mean")
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
induction_score_store = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
def induction_score_hook(
    pattern: TT["batch", "head_index", "dest_pos", "source_pos"],
    hook: HookPoint,
):
    # We take the diagonal of attention paid from each destination position to source positions seq_len-1 tokens back
    # (This only has entries for tokens with index>=seq_len)
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
    # Get an average score per head
    induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
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
single_random_sequence = torch.randint(1000, 10000, (1, 20)).to(model.cfg.device)
repeated_random_sequence = einops.repeat(single_random_sequence, "batch seq_len -> batch (2 seq_len)")
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
distilgpt2_induction_score_store = torch.zeros((distilgpt2.cfg.n_layers, distilgpt2.cfg.n_heads), device=distilgpt2.cfg.device)
def induction_score_hook(
    pattern: TT["batch", "head_index", "dest_pos", "source_pos"],
    hook: HookPoint,
):
    # We take the diagonal of attention paid from each destination position to source positions seq_len-1 tokens back
    # (This only has entries for tokens with index>=seq_len)
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
    # Get an average score per head
    induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
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
print(f"Prob ratio bias: {torch.exp(john_bias - mary_bias).item():.4f}x")
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
A = torch.randn(5, 2)
B = torch.randn(2, 5)
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
print(torch.linalg.eig(AB).eigenvalues)
print(AB_factor.eigenvalues)
print()
print("Singular Values:")
print(torch.linalg.svd(AB).S)
print(AB_factor.S)
```

We can multiply with other matrices - it automatically chooses the smallest possible dimension to factor along (here it's 2, rather than 5)

```python
C = torch.randn(5, 300)
ABC = AB @ C
ABC_factor = AB_factor @ C
print("Unfactored:", ABC.shape, ABC.norm())
print("Factored:", ABC_factor.shape, ABC_factor.norm())
print(f"Right dimension: {ABC_factor.rdim}, Left dimension: {ABC_factor.ldim}, Hidden dimension: {ABC_factor.mdim}")
```

If we want to collapse this back to an unfactored matrix, we can use the AB property to get the product:

```python
AB_unfactored = AB_factor.AB
print(torch.isclose(AB_unfactored, AB).all())
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
        self.offset = nn.Parameter(torch.tensor(offset))
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
out, cache = model.run_with_cache(torch.tensor(5.0))
print("Model output:", out.item())
for key in cache:
    print(f"Value cached at hook {key}", cache[key].item())
```

We can also use hooks to intervene on activations - eg, we can set the intermediate value in layer 2 to zero to change the output to -5

```python
def set_to_zero_hook(tensor, hook):
    print(hook.name)
    return torch.tensor(0.0)

print(
    "Output after intervening on layer2.hook_scaled",
    model.run_with_hooks(
        torch.tensor(5.0), fwd_hooks=[("layer2.hook_square", set_to_zero_hook)]
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

</ul>""", unsafe_allow_html=True)
    st.markdown(r"""
# Exercises: finding induction heads

Now that we've seen some of the features of TransformerLens, let's apply them to the task of finding induction heads.

If you don't fully understand the algorithm peformed by induction heads, this diagram may prove helpful:""")

    with open("images/induction-heads.svg", "r") as f:
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
    * Q-Composition, intuitively, says that we want to be more intelligent in choosing our destination token - this looks like us wanting to move information to a token based on *that* token's context. A natural example might be the final token in a several token word or phrase, where earlier tokens are needed to disambiguate it, eg E|iff|el| Tower|
    * K-composition, intuitively, says that we want to be more intelligent in choosing our source token - this looks like us moving information *from* a token based on its context (or otherwise some computation at that token).
        * Induction heads are a clear example of this - the source token only matters because of what comes before it!
    * V-Composition, intuitively, says that we want to *route* information from an earlier source token *other than that token's value* via the current destination token. It's less obvious to me when this is relevant, but could imagine eg a network wanting to move information through several different places and collate and process it along the way
        * One example: In the ROME paper, we see that when models recall that "The Eiffel Tower is in" -> " Paris", it stores knowledge about the Eiffel Tower on the " Tower" token. When that information is routed to | in|, it must then map to the output logit for | Paris|, which seems likely due to V-Composition
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
- The activations in the attention layers $(q, k, v, z)$ have shape `[batch, position, head_index, d_head]` (ie, not flattened into a single d_model axis)
  - Similarly $W_K, W_Q, W_V$ have shape `[head_index, d_head, d_model]`, $W_O$ has shape `[head_index, d_model, d_head]`
- Convention: All weight matrices multiply on the left (i.e. have shape `[output, input]`)

```python
MAIN = __name__ == "__main__"

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

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
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
    W_U_to_logits = W_U[tokens[1:], :]
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
    px.imshow(
        to_numpy(logit_attr),
        x=x_labels,
        y=y_labels,
        labels={"x": "Term", "y": "Position", "color": "logit"},
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        height=25*len(tokens),
    ).show()

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

You already saw examples of ablations in the TransformerLens material. Here, we'll ask you to do some more. You should write a function `head_ablation` which sets a particular head's `attn_result` (i.e. the output of the attention layer corresponding to that head, before we sum over heads) to zero. Then, you should write a function `get_ablation_scores` which returns a tensor of shape `(n_layers, n_heads)` containing the **increase** in cross entropy loss on your input sequence that results from performing this ablation.

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

ablation_scores = get_ablation_scores(model, tokens)

imshow(ablation_scores, xaxis="Head", yaxis="Layer", title="Logit Difference After Ablating Heads", text_auto=".2f")
""")

def section_4():
    st.markdown(r"""
Most of what we did above was feature analysis - we looked at activations (here just attention patterns) and tried to interpret what they were doing. Now we're going to do some mechanistic analysis - digging into the weights and using them to reverse engineer the induction head algorithm and verify that it is really doing what we think it is.
""")
    st.info("These exercises will be added shortly!")


# def section_home():
#     st.markdown(r"""Coming soon!""")

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
