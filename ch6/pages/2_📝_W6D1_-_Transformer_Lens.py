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

NAMES = ["attribution_fig", "attribution_fig_2", "failure_types_fig", "failure_types_fig_2"]
def get_fig_dict():
    return {name: read_from_html(name) for name in NAMES}
if "fig_dict" not in st.session_state:
    st.session_state["fig_dict"] = {}
if NAMES[0] not in st.session_state["fig_dict"]:
    st.session_state["fig_dict"] |= get_fig_dict()
fig_dict = st.session_state["fig_dict"]

def section_home():
    st.markdown(r"""
# TransformerLens

Today is designed to get you introduced to Neel Nanda's **TransformerLens** library, which we'll be using for the rest of the interpretability chapter. The hope is that, having previously had to write our own code to do things like visualize attention heads, we'll have a better understanding of the features of TransformerLens that make this more convenient.

## Setup

This section is just for doing basic setup and installations.

First, you'll need to install TransformerLens and the visualisation library circuitsvis:

```python
pip install git+https://github.com/neelnanda-io/TransformerLens.git@new-demo
pip install circuitsvis
```

Additionally, it's useful to run the following at the top of your notebook / python file:

```python
from IPython import get_ipython
ipython = get_ipython()
# Code to automatically update the HookedTransformer code as its edited without restarting the kernel
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")
```
""")

def section_home():
    st.markdown(r"""Coming soon!""")

def section_1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#loading-and-running-models">Loading and Running Models</a></li>
   <li><a class="contents-el" href="#caching-all-activations">Caching all Activations</a></li>
   <li><a class="contents-el" href="#hooks:-intervening-on-activations">Hooks: Intervening on Activations</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#activation-patching-on-the-indirect-object-identification-task">Activation Patching on the Indirect Object Identification Task</a></li>
   </ul></li>
   <li><a class="contents-el" href="#hooks:-accessing-activations">Hooks: Accessing Activations</a></li>
   <li><a class="contents-el" href="#available-models">Available Models</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#an-overview-of-the-important-open-source-models-in-the-library">An overview of the important open source models in the library</a></li>
       <li><a class="contents-el" href="#an-overview-of-some-interpretability-friendly-models-i-ve-trained-and-included">An overview of some interpretability-friendly models I've trained and included</a></li>
   </ul></li>
   <li><a class="contents-el" href="#other-resources:">Other Resources:</a></li>
   <li><a class="contents-el" href="#transformer-architecture">Transformer architecture</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#parameter-names">Parameter Names</a></li>
       <li><a class="contents-el" href="#activation-+-hook-names">Activation + Hook Names</a></li>
       <li><a class="contents-el" href="#folding-layernorm-for-the-curious">Folding LayerNorm (For the Curious)</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
# Introduction

This is a demo notebook for [TransformerLens](https://github.com/neelnanda-io/TransformerLens), **a library I ([Neel Nanda](neelnanda.io)) wrote for doing [mechanistic interpretability](https://distill.pub/2020/circuits/zoom-in/) of GPT-2 Style language models.** The goal of mechanistic interpretability is to take a trained model and reverse engineer the algorithms the model learned during training from its weights. It is a fact about the world today that we have computer programs that can essentially speak English at a human level (GPT-3, PaLM, etc), yet we have no idea how they work nor how to write one ourselves. This offends me greatly, and I would like to solve this! Mechanistic interpretability is a very young and small field, and there are a *lot* of open problems - if you would like to help, please try working on one! **Check out my [list of concrete open problems](TODO: link) to figure out where to start.**

I wrote this library because after I left the Anthropic interpretability team and started doing independent research, I got extremely frustrated by the state of open source tooling. There's a lot of excellent infrastructure like HuggingFace and DeepSpeed to *use* or *train* models, but very little to dig into their internals and reverse engineer how they work. **This library tries to solve that**, and to make it easy to get into the field even if you don't work at an industry org with real infrastructure! The core features were heavily inspired by [Anthropic's excellent Garcon tool](https://transformer-circuits.pub/2021/garcon/index.html). Credit to Nelson Elhage and Chris Olah for building Garcon and showing me the value of good infrastructure for accelerating exploratory research!

The core design principle I've followed is to enable exploratory analysis - one of the most fun parts of mechanistic interpretability compared to normal ML is the extremely short feedback loops! The point of this library is to keep the gap between having an experiment idea and seeing the results as small as possible, to make it easy for **research to feel like play** and to enter a flow state. This notebook demonstrates how the library works and how to use it, but if you want to see how well it works for exploratory research, check out [my notebook analysing Indirect Objection Identification](TODO: link) or [my recording of myself doing research](https://www.youtube.com/watch?v=yo4QvDn-vsU)!

## Loading and Running Models

TransformerLens comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. For this demo notebook we'll look at GPT-2 Small, an 80M parameter model, see the Available Models section for info on the rest.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained("gpt2-small", device=device)
```

To try the model the model out, let's find the loss on this text! Models can be run on a single string or a tensor of tokens (shape: [batch, position], all integers), and the possible return types are: 
* "logits" (shape [batch, position, d_vocab], floats), 
* "loss" (the cross-entropy loss when predicting the next token), 
* "both" (a tuple of (logits, loss)) 
* None (run the model, but don't calculate the logits - this is faster when we only want to use intermediate activations)

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
```

```python
print("Layer 0 Head Attention Patterns:")
cv.attention.attention_patterns(tokens=gpt2_str_tokens, attention=attention_pattern)
```

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
# 
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

Here, our clean prompt is "After John and Mary went to the store, **Mary** gave a bottle of milk to", our corrupted prompt is "After John and Mary went to the store, **John** gave a bottle of milk to", and our metric is the difference between the correct logit ( John) and the incorrect logit ( Mary) on the final token. 

We see that the logit difference is significantly positive on the clean prompt, and significantly negative on the corrupted prompt, showing that the model is capable of doing the task!

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

We now setup the hook function to do activation patching. Here, we'll patch in the residual stream at the start of a specific layer and at a specific position. This will let us see how much the model is using the residual stream at that layer and position to represent the key information for the task. 

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
```

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
```

The induction heads will be attending from the second occurence of each token to the token *after* its first occurence, ie the token `50-1==49` places back. So by looking at the average attention paid 49 tokens back, we can identify induction heads! Let's define a hook to do this!""")

    with st.expander("Technical details"):
        st.markdown(r"""

* We attach the hook to the attention pattern activation. There's one big pattern activation per layer, stacked across all heads, so we need to do some tensor manipulation to get a per-head score. 
* Hook functions can access global state, so we make a big tensor to store the induction head score for each head, and then we just add the score for each head to the appropriate position in the tensor. 
* To get a single hook function that works for each layer, we use the `hook.layer()` method to get the layer index (internally this is just inferred from the hook names).
* As we want to add this to *every* activation pattern hook point, rather than giving the string for an activation name, this time we give a **name filter**. This is a Boolean function on hook point names, and it adds the hook function to every hook point where the function evaluates as true. 
    * `run_with_hooks` allows us to enter a list of (act_name, hook_function) pairs to all be added at once, so we could also have done this by inputting a list with a hook for each layer.
""")

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
```

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
        cv.attention.attention_patterns(
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
```

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
```


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

def section_1():
    pass

def section_2():
    pass

func_list = [section_home, section_1, section_2]

page_list = ["🏠 Home", "1️⃣ Introduction", "2️⃣ Features"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()
