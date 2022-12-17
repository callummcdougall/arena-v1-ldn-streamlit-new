import os
if not os.path.exists("./images"):
    os.chdir("./ch5")

from st_dependencies import *
styling()

import plotly.io as pio
import plotly.express as px
import re
import json
import numpy as np

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

def get_fig_dict():
    names = ["grid_output"]
    return {name: read_from_html(name) for name in names}

if "fig_dict" not in st.session_state:
    fig_dict = get_fig_dict()
    st.session_state["fig_dict"] = fig_dict
else:
    fig_dict = st.session_state["fig_dict"]

arr = np.load(r"images/arr.npy")
fig_dict["animation_output"] = px.imshow(arr, animation_frame=0, color_continuous_scale="gray")

st.markdown("""
<style>
.css-ffhzg2 span[style*="color: blue"] {
    color: rgb(0, 180, 240) !important;
}
.css-ffhzg2 span[style*="color: black"] {
    color: white !important;
}
.css-fg4pbf span[style*="color: orange"] {
    color: rgb(255, 130, 20) !important;
}
</style>""", unsafe_allow_html=True)

def section_home():

    st.markdown(r"""


# 1️⃣ CLIP

CLIP is a model that contains a vision model and a language model, and it is trained so that the embeddings produced by the two parts are similar when the image and the text have a similar meaning or topic. Today, we're using a vision transformer (which is exactly what it sounds like) and a GPT-style transformer, but you could also use a ResNet and a bag-of-words model or whatever you like.

For training data, the authors scraped the Internet for images that have captions under them to obtain (image, text) pairs that we assume are both "about" the same thing. For example, the image might be a guinea pig eating a cucumber and the caption says "Mr. Fluffers fueling up with some Vitamin C."

""")

    st_image("guinea-pig.jpg", 300)
    st.markdown(r"""

To do traditional supervised learning, we would start with the image, feed it in, and then try to unembed to generate text and compare the predicted text to the actual caption. Or start with the text, generate an image, and compare the image to the actual.

For either of these, it's tricky to define a differentiable loss function that captures "how close is the meaning of these two pieces of text" or "how much do these two images depict the same thing".

CLIP instead uses a **contrastive loss**, which just means that the embedding of an image and the embedding of its matching caption should be similar, while the embedding of an image and all the other captions in the batch should be dissimilar. "Similar" is efficiently calculated using the cosine similarity.

It turns out that if you use large enough batch sizes (like 32,768) and a large enough dataset (400 million pairs), this works quite well. It takes thousands of GPU-days to train a CLIP, so we will just play with the pretrained weights today.

Some cool applications of this are:

- Given an image, you can embed it and see how similar that embedding is to the embedding of the string "photo of a dog" versus the string "photo of a cat". This means you can classify images, but in a much more flexible way than a traditional supervised classifier where the set of categories is fixed up front.
- Given some text, you can search a large database of image embeddings for images that are similar to the text embedding.

## 2️⃣ Stable Diffusion

Next, we will introduce the Stable Diffusion model, a state-of-the-art architecture that integrates the text encoder from CLIP (although other text encoders can be used) into a modified diffusion model which is similar to your work from yesterday, W3D4. The primary differences between an "ordinary" diffusion model and the Stable Diffusion model:

Text encoding is done using a frozen CLIP text encoder. By frozen, we mean the encoder was pretrained separately using the contrastive loss as described yesterday, and not modified at all during the training of SD. The vision transformer part of CLIP is not used in SD.
U-Net operates in a latent space which has a lower spatial dimensionality than the pixel space of the input. The schematic below describes "LDM-8" which means that the spatial dimension is shrunk by a factor of 8 in width and height. More downsampling makes everything faster, but reduces perceptual quality.
The encoding to and decoding from the latent space are done using a Variational Autoencoder (VAE), which is trained on the reconstruction error after compressing images into the latent space and then decompressing them again. At inference time, we have no need of the encoder portion because we start with random latents, not pixels. We only need to make one call to the VAE decoder at the very end to turn our latents back into pixels.""")


def section_1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#learning-objectives">Learning Objectives</a></li>
   <li><a class="contents-el" href="#references-optional-for-part-1">References (optional) for Part 1</a></li>
   <li><a class="contents-el" href="#vision-transformers">Vision Transformers</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#patch-embedding">Patch Embedding</a></li>
       <li><a class="contents-el" href="#positional-embedding">Positional Embedding</a></li>
       <li><a class="contents-el" href="#class-embedding-replacement-for-begin-token">Class Embedding (replacement for "begin" token)</a></li>
       <li><a class="contents-el" href="#config-classes">Config Classes</a></li>
       <li><a class="contents-el" href="#positionids">position_ids</a></li>
   </ul></li>
   <li><a class="contents-el" href="#clip-mlp">CLIP MLP</a></li>
   <li><a class="contents-el" href="#self-attention">Self-Attention</a></li>
   <li><a class="contents-el" href="#clip-layer">CLIP Layer</a></li>
   <li><a class="contents-el" href="#clip-encoder">CLIP Encoder</a></li>
   <li><a class="contents-el" href="#clipvisiontransformer">CLIPVisionTransformer</a></li>
   <li><a class="contents-el" href="#cliptexttransformer">CLIPTextTransformer</a></li>
   <li><a class="contents-el" href="#clipmodel">CLIPModel</a></li>
   <li><a class="contents-el" href="#data-preparation">Data Preparation</a></li>
   <li><a class="contents-el" href="#cosine-similarities">Cosine Similarities</a></li>
   <li><a class="contents-el" href="#running-the-model">Running the Model</a></li>
   <li><a class="contents-el" href="#implementing-constrastive-loss">Implementing Constrastive Loss</a></li>
   <li><a class="contents-el" href="#bonus">Bonus</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""# CLIP""")
    st.info(r"""## Learning objectives

- Understand the basic principles behind CLIP, and contrastive loss
- Implement your own vision transformer
- Reuse old GPT code for the text transformer
- Assemble a CLIP and load pretrained weights
- Play with CLIP!
""")

    st.markdown(r"""
## References (optional) for Part 1

CLIP

- [Paper](https://arxiv.org/pdf/2103.00020.pdf)
- [Official OpenAI repo](https://github.com/openai/CLIP)
- [HuggingFace implementation](https://huggingface.co/sentence-transformers/clip-ViT-L-14)

X-CLIP - Includes experimental improvements from recent papers

- [Code repo](https://github.com/lucidrains/x-clip)


```python
import glob
import os
import sys
from typing import Callable, Union, cast
import pandas as pd
import torch as t
from einops import rearrange, repeat
from fancy_einsum import einsum
from IPython.display import display
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from transformers.models.clip import modeling_clip
import sentence_transformers # You might need to pip install this
import w5d4_tests
from w5d4_utils import (
    CLIPConfig,
    CLIPOutput,
    CLIPTextConfig,
    CLIPVisionConfig,
    get_reference_model,
    get_reference_clip_model,
)

MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")

```

## Vision Transformers

Our first task for today is to implement a vision transformer.

Using transformers on image data is actually easier than using it on text data. Because our input data is already continuous, we don't need a special tokenizer and we don't need a `nn.Embedding` to translate from discrete token ids to a continuous representation. (Technically the RGB pixel values are discrete in the range [0, 256), but we won't worry about this).

The only issue is that for our images (which for today we'll assume to be exactly 224 x 224px) treating each pixel as a sequence element would result in a sequence length around 50K. Since self-attention is quadratic in the sequence length, we'd prefer to decrease this sequence length to something more manageable. This is analogous to how we won't model text as individual characters, but as slightly larger chunks.

The original [Vision Transformers paper](https://arxiv.org/pdf/2010.11929.pdf) used 14x14 patches of 16x16 pixels each, but in our implementation of CLIP (matching CLIP ViT-L/14) the patch size specified in `CLIPVisionConfig` is 14x14 pixels, which means there are 16x16 = 256 total patches.

The rest of the vision transformer is going to look extremely similar to what you've seen with GPT.

### Patch Embedding

There are a couple equivalent ways to obtain an embedding vector for each patch. For example, you could use `einops.rearrange` and a `Linear(patch_pixels, hidden_size)`. Instead, we're going to follow HuggingFace and use a `nn.Conv2d` with appropriate stride and kernel size (and no bias).

### Positional Embedding

When first learning vision transformers, I expected the positional embedding would work best by indicating (x, y) coordinates for each patch so that the model can easily understand the 2D spatial relationships.

However, the Vision Transformers paper found no difference between this 2D method and just simply numbering the patches (see appendix D.4). This means that the model has to learn itself that patch 16 is correlated with patches 0 and 32 (because they are vertically adjacent), but this doesn't seem to be a problem. They speculate that there are so few patches that it's just very easy to memorize these patterns.

### Class Embedding (replacement for "begin" token)

When using text, it's common practice to have the tokenizer prepend a special placeholder token called the "begin token". When we train the model for sequence classification, we use the final layer's embedding at this sequence position for the representation of the entire sequence, so attention heads learn to copy relevant data to this position.

Since there's no separate tokenizer for vision, we're going to initialize a random normal embedding vector of `embedding_size` and prepend that to every sequence. This embedding is called the class embedding because it's used for classification.

### Config Classes

Config dataclasses for CLIP have been defined and imported from `w3d5_globals.py` These will be a helpful reference as you build the CLIP components.


```python
def print_class_attrs(cls: type) -> None:
    print(f"\n\n{cls.__name__}\n---")
    for (k, v) in ((k, v) for (k, v) in vars(cls).items() if k[0] != "_"):
        print(f"{k}: {v}")


if MAIN:
    print_class_attrs(CLIPVisionConfig)
    print_class_attrs(CLIPTextConfig)
    print_class_attrs(CLIPConfig)

```


### position_ids

Register a buffer called `position_ids` which just contains `arange(0, (self.image_size // self.patch_size) ** 2 + 1)`. The extra index is for the class embedding in addition to the standard patches. This avoids redundantly allocating the `arange` on the target device on every forward pass.


```python
class CLIPVisionEmbeddings(nn.Module):
    config: CLIPVisionConfig
    patch_size: int
    image_size: int
    embed_dim: int
    num_patches: int
    class_embedding: nn.Parameter
    patch_embedding: nn.Conv2d
    position_embedding: nn.Embedding
    position_ids: t.Tensor

    def __init__(self, config: CLIPVisionConfig):
        '''Assign values from input config to class member variables as appropriate,
        e.g. self.patch_size = config.patch_size'''
        super().__init__()
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the patch embeddings and the positional embeddings and return their sum.

        x: shape (batch, channels=3, height=224, width=224)
        out: shape (batch, sequence, hidden)
        '''
        pass


if MAIN:
    w5d4_utils.test_vision_embeddings(CLIPVisionEmbeddings)
```

## CLIP MLP

The remaining layers of CLIP operate on embedding vectors of `hidden_size`, so they're independent of whether the input was text or images.

The MLP uses a faster approximation to the [GELU](https://arxiv.org/pdf/1606.08415.pdf) nonlinearity. Note that as of PyTorch 1.11, `nn.GELU` and `F.gelu` compute the exact equation for GELU.

Use the equation from the paper and implement the sigmoid approximation from section 2 yourself. Plot the absolute difference on the interval [-5, 5] and check how different the approximation is from the exact. Then implement the MLP using the approximation.

The MLP looks the same as in a standard transformer: a Linear layer that goes from hidden size to an intermediate size 4 times larger, a GELU, and a second Linear back down to the hidden size.


```python
def gelu_sigmoid_approximation(x: t.Tensor) -> t.Tensor:
    '''Return sigmoid approximation of GELU of input tensor x with same shape.'''
    pass


def plot_gelu_approximation(x: t.Tensor):
    (fig, (ax0, ax1)) = plt.subplots(nrows=2, figsize=(12, 12))
    actual = F.gelu(x)
    approx = gelu_sigmoid_approximation(x)
    diff = (actual - approx).abs()
    x_cpu = x.cpu()
    ax0.plot(x_cpu, diff.cpu(), label="absolute error")
    ax0.legend()
    ax1.plot(x_cpu, actual.cpu(), label="exact", alpha=0.5)
    ax1.plot(x_cpu, approx.cpu(), label="sigmoid", alpha=0.5)
    ax1.legend()
    ax1.set(xlabel=f"x ({x.dtype})")


if MAIN:
    x = t.linspace(-5, 5, 400)
    plot_gelu_approximation(x)
    if t.cuda.is_available():
        x16 = t.linspace(-5, 5, 400, dtype=t.float16, device=device)
        plot_gelu_approximation(x16)


class CLIPMLP(nn.Module):
    fc1: nn.Linear
    fc2: nn.Linear

    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        '''Initialize parent class, then assign fully-connected layers based
        on shape in input config'''
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Run forward pass of MLP, including fully-connected layers and non-linear
        activations where appropriate'''
        pass


if MAIN:
    w5d4_utils.test_mlp(CLIPMLP)

```

## Self-Attention

For the vision transformer, the authors don't use masked attention. You should be able to copy and paste from your `BertSelfAttention` class you wrote previously and fix up the variable names. Or try writing it from memory for the practice.


```python
class CLIPAttention(nn.Module):
    num_heads: int
    head_size: int
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    out_proj: nn.Linear
    dropout: nn.Dropout

    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        '''Assign values from input config to class member variables as appropriate'''
        pass

    def attention_pattern_pre_softmax(self, x: t.Tensor) -> t.Tensor:
        '''Return the attention pattern after scaling but before softmax.

        pattern[batch, head, q, k] should be the match between a query at sequence position q and a key at sequence position k.
        '''
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Perform forward pass through attention layer, computing attention pattern and value projections
        to combine into output. Remember to apply dropout.'''
        pass


if MAIN:
    w5d4_utils.test_vision_attention(CLIPAttention)

```

## CLIP Layer

Identical to GPT (besides calling our slightly different MLP), so this is provided for you. Make sure to read through and understand
the operations being performed.


```python
class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x

```

## CLIP Encoder

This is also provided as it's trivial. Note that a full-fledged implementation this would have more code in it for things like checkpointing.


```python
class CLIPEncoder(nn.Module):
    layers: nn.ModuleList[CLIPEncoderLayer]

    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        super().__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x: t.Tensor) -> t.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

```

## CLIPVisionTransformer

This is the last class to implement before we can load pretrained weights for the vision transformer!

The output will consist of only the first sequence position corresponding to the prepended "class embedding". Do the slice before the final layer norm to avoid unnecessary computation.

We've made all the variable names identical so far with the idea that the state dict should exactly match. However, the pretrained weights have spelled `pre_layrnorm` incorrectly. Sad! If this really bothers you, you can fix it in your version and adjust the weight loading code to adapt.


```python
class CLIPVisionTransformer(nn.Module):
    config: CLIPVisionConfig
    embeddings: CLIPVisionEmbeddings
    pre_layrnorm: nn.LayerNorm
    encoder: CLIPEncoder
    post_layernorm: nn.LayerNorm

    def __init__(self, config: CLIPVisionConfig):
        '''Assign values from input config to class member variables as appropriate'''
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Perform forward pass through vision transformer: embedding, layer norm, encoder, layer norm
        Return output corresponding to prepended class_embedding'''
        pass


if MAIN:
    w5d4_utils.test_vision_transformer(CLIPVisionTransformer)

```


## CLIPTextTransformer

The text transformer looks a lot like BERT, except it does have the causal attention mask like GPT.

It supports sequences of varying lengths with padding at the end, and padding tokens are also masked out during attention. We won't bother re-implementing the code, since this is very similar to what you've done before.

We do need a tokenizer for the text stuff, and again we'll use the provided one since it works the same as you've seen previously.


```python
if MAIN:
    tokenize = get_reference_model().tokenize

```

## CLIPModel

Now we're ready to put together the full model. In general, since we allow mixing and matching any models, the embedding of the image and text aren't going to have the same dimension.

The CLIPModel has two linear projections that take the individual model outputs to a common hidden size of `config.projection_dim`.

CLIPModel also takes care of normalizing each unit vector to have a L2 norm of 1. This is because cosine similarity can be calculated as a dot product of two unit vectors. Finally, the two embeddings are packaged into a tuple.

The scalar parameter `logit_scale` is only used during training, where it's used to multiply the similarity scores before computing the contrastive loss.

```mermaid

graph TD
    subgraph CLIPModel

    Image --> ImageTransformer --> VisualProjection --> Normalize1[Normalize] --> CLIPOutput
    Text --> TextTransformer --> TextProjection --> Normalize2[Normalize] --> CLIPOutput

    end
```


```python
class CLIPModel(nn.Module):
    config: CLIPConfig
    text_config: CLIPTextConfig
    vision_config: CLIPVisionConfig
    projection_dim: int
    text_embed_dim: int
    vision_embed_dim: int
    text_model: modeling_clip.CLIPTextTransformer
    vision_model: CLIPVisionTransformer
    visual_projection: nn.Linear
    text_projection: nn.Linear
    logit_scale: nn.Parameter

    def __init__(self, config: CLIPConfig):
        '''Assign values from input config to class member variables as appropriate.

        The typechecker will complain when passing our CLIPTextConfig to CLIPTextTransformer, because the latter expects type transformers.models.clip.configuration_clip.CLIPTextConfig. You can ignore this as our type is in fact compatible.
        '''
        pass

    def forward(self, input_ids, attention_mask, pixel_values) -> CLIPOutput:
        '''
        Perform forward pass through CLIP model, applying text and vision model/projection.

        input_ids: (batch, sequence)
        attention_mask: (batch, sequence). 1 for visible, 0 for invisible.
        pixel_values: (batch, channels, height, width)
        '''
        pass


if MAIN:
    w5d4_utils.test_clip_model(CLIPModel)

```

## Data Preparation

The data preparation is the same as you've seen before. The ImageNet normalization constants are used. Feel free to supply some of your own text and/or images here.


```python
def get_images(glob_fnames: str) -> tuple[list[str], list[Image.Image]]:
    filenames = glob.glob(glob_fnames)
    images = [Image.open(filename).convert("RGB") for filename in filenames]
    image_names = [os.path.splitext(os.path.basename(filename))[0] for filename in filenames]
    for im in images:
        display(im)
    return (image_names, images)


if MAIN:
    preprocess = cast(
        Callable[[Image.Image], t.Tensor],
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
    texts = [
        "A guinea pig eating a cucumber",
        "A pencil sketch of a guinea pig",
        "A rabbit eating a carrot",
        "A paperclip maximizer",
    ]
    out = tokenize(texts)
    input_ids = out["input_ids"]
    attention_mask = out["attention_mask"]
    (image_names, images) = get_images("./clip_images/*")
    pixel_values = t.stack([preprocess(im) for im in images], dim=0)

```

## Cosine Similarities

Since the model already normalizes each embedding to be a unit vector, this function becomes a one-liner.


```python
def cosine_similarities(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    '''Return cosine similarities between all pairs of embeddings.

    Each element of the batch should be a unit vector already.

    a: shape (batch_a, hidden_size)
    b: shape (batch_b, hidden_size)
    out: shape (batch_a, batch_b)
    '''
    pass


if MAIN:
    w5d4_utils.test_cosine_similarity(cosine_similarities)

```

## Running the Model

Run the model and compute the cosine similarities between each image and each piece of text. Visualize the results and see if they match what you expect.


```python
def load_trained_model(config: CLIPConfig):
    model = CLIPModel(config)
    full_state_dict = get_reference_clip_model().state_dict()
    model.load_state_dict(full_state_dict)
    return model


if MAIN:
    config = CLIPConfig(CLIPVisionConfig(), CLIPTextConfig())
    model = load_trained_model(config).to(device)
    with t.inference_mode():
        out = model(input_ids.to(device), attention_mask.to(device), pixel_values.to(device))
    similarities = cosine_similarities(out.text_embeds, out.image_embeds)
    df = pd.DataFrame(similarities.detach().cpu().numpy(), index=texts, columns=image_names).round(3)
    display(df)

```

## Implementing Constrastive Loss

We're not going to train today, but we'll implement the contrastive loss to make sure we understand it.

There's a nice trick to implement the contrastive loss using of the average of two `F.cross_entropy` terms. See if you can find it.

""")

    with st.expander("Spoiler - Contrastive Loss Calculation"):
        st.markdown(r"""

First compute the matrix `similarities[text_index][image_index]`, of shape (batch, batch).

Since the ith text corresponds to the ith image in the training data, `similarities[i]` should have a value near 1 at index i, and be low otherwise. This is just like cross entropy where the target is class i.

The same holds for `similarities[:, i]`, so that's the second cross entropy term. Each value in our matrix contributed to each term, so taking the average prevents double-counting.""")

    st.markdown(r"""
```python
def contrastive_loss(text_embeds: t.Tensor, image_embeds: t.Tensor, logit_scale: t.Tensor) -> t.Tensor:
    '''Return the contrastive loss between a batch of text and image embeddings.

    The embeddings must be in order so that text_embeds[i] corresponds with image_embeds[i].

    text_embeds: (batch, output_dim)
    image_embeds: (batch, output_dim)
    logit_scale: () - log of the scale factor to apply to each element of the similarity matrix

    Out: scalar tensor containing the loss
    '''
    pass


if MAIN:
    w5d4_utils.test_contrastive_loss(contrastive_loss)

```

# On to Part 2

In the following part of this day's exercises, we will finally get to play with the exciting and *very* state-of-the-art Stable Diffusion model, with ideas for bonus tasks after you complete the implementation of the model.

If you would like to continue working on CLIP-related models, here are some bonus tasks that you can return to after completing Part 2.

## Bonus

### Prompt Engineering and Zero Shot Classification

Thinking back to Part 1, CLIP can be used as a classifier by comparing the unknown image's embedding with the embedding of a prompt like "a photo of [class name]". Implement this idea and see how good the results are, then try to improve them by finding a better prompt. Or, use several prompts and ensemble the outputs together.

### GELU approximations

In the CLIP model, could we have "gotten away" with using PyTorch's GELU instead of the approximation the authors used? Or are the pretrained weights precisely adapted to the approximation? Try running the pretrained weights using the PyTorch exact implementation and see how different the results are.

### X-CLIP

Read through the code at the [X-CLIP](https://github.com/lucidrains/x-clip) repo and try to understand some of the modifications and improvements.


""")

def section_2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#learning-objectives">Learning Objectives</a></li>
   <li><a class="contents-el" href="#setting-up-your-instance">Setting up your instance</a></li>
   <li><a class="contents-el" href="#introducing-the-model">Introducing the model</a></li>
   <li><a class="contents-el" href="#schematic">Schematic</a></li>
   <li><a class="contents-el" href="#a-final-product">A final product</a></li>
   <li><a class="contents-el" href="#references-optional-for-part-2">References (optional) for Part 2</a></li>
   <li><a class="contents-el" href="#preparation">Preparation</a></li>
   <li><a class="contents-el" href="#text-encoder">Text encoder</a></li>
   <li><a class="contents-el" href="#getting-pretrained-models">Getting pretrained models</a></li>
   <li><a class="contents-el" href="#tokenization">Tokenization</a></li>
   <li><a class="contents-el" href="#assembling-the-inference-pipeline">Assembling the inference pipeline</a></li>
   <li><a class="contents-el" href="#trying-it-out">Trying it out!</a></li>
   <li><a class="contents-el" href="#implementing-interpolation">Implementing interpolation</a></li>
   <li><a class="contents-el" href="#prompt-interpolation">Prompt Interpolation</a></li>
   <li><a class="contents-el" href="#saving-a-gif">Saving a GIF</a></li>
   <li><a class="contents-el" href="#speeding-up-interpolation">Speeding up interpolation</a></li>
   <li><a class="contents-el" href="#bonus">Bonus</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#multiple-prompt-image-generation">Multiple prompt image generation</a></li>
       <li><a class="contents-el" href="#stylistic-changes">Stylistic changes</a></li>
   </ul></li>
   <li><a class="contents-el" href="#acknowledgements">Acknowledgements</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("# Stable Diffusion")
    st.info(r"""## Learning objectives

- Understand how Stable Diffusion connects together previous parts of this chapter (VAEs, diffusion models, CLIP)
- Complete the implementation of a Stable Diffusion inference pipeline
- Run inference on your model
- Play with other things you can do with Stable Diffusion, such as animations""")

    st.markdown(r"""
## Setting up your instance

In order to download the stable-diffusion models from HuggingFace, there is some setup required:

1. Make a [HuggingFace account](https://huggingface.co/join) and confirm your email address.
1. Visit [here](https://huggingface.co/CompVis/stable-diffusion-v1-4) and click `yes` to the terms and conditions (after thoroughly reading them, of course) and then click `access repository`.
1. Generate a [HuggingFace token](https://huggingface.co/settings/tokens) with a `read` role.
1. Run `huggingface-cli login` in your VSCode terminal and paste the token you generated above (ignore the warning text). This will allow the Python module to download the pretrained models we will be using.

You should now be able to load the pretrained models in this notebook.

## Introducing the model

Before moving on to integrate CLIP into the Stable Diffusion (SD) model, it's worth briefly reviewing what we've built in Part 1. CLIP provides a text encoder and image encoder that are trained together to minimize contrastive loss, and therefore allows for embedding arbitrary text sequences in a latent space that has some relevance to images.

Now, we will introduce the Stable Diffusion model, a state-of-the-art architecture that integrates the text encoder from CLIP (although other text encoders can be used) into a modified diffusion model which is similar to your work from yesterday, W3D4. The primary differences between an "ordinary" diffusion model and the Stable Diffusion model:

* Text encoding is done using a frozen CLIP text encoder. By frozen, we mean the encoder was pretrained separately using the contrastive loss as described yesterday, and not modified at all during the training of SD. The vision transformer part of CLIP is not used in SD.
* U-Net operates in a latent space which has a lower spatial dimensionality than the pixel space of the input. The schematic below describes "LDM-8" which means that the spatial dimension is shrunk by a factor of 8 in width and height. More downsampling makes everything faster, but reduces perceptual quality.
* The encoding to and decoding from the latent space are done using a Variational Autoencoder (VAE), which is trained on the reconstruction error after compressing images into the latent space and then decompressing them again. At inference time, we have no need of the encoder portion because we start with random latents, not pixels. We only need to make one call to the VAE decoder at the very end to turn our latents back into pixels.

## Schematic""")

    st_image("stable_diffusion.png", 450)
    st.markdown(r"""

## A final product

Before getting into it, here's a *very rough* example of the sort of interpolation-based animations that you will (hopefully) be generating by the end of the day with only a few minutes of runtime! Ideally, yours will be smoother and more creative given longer inference time to generate more frames. :)

""")
    st.info(r"""
Note - more than any other section of this whole course, you will **definitely** need a better GPU to run this model than can be found in your laptop, e.g. using Lambda Labs.
""")
    st_image("stablediff_animation.gif", 400)
    st.markdown(r"""

## References (optional) for Part 2

- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [HuggingFace implementation](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- [HuggingFace diffusers module](https://github.com/huggingface/diffusers/tree/71ba8aec55b52a7ba5a1ff1db1265ffdd3c65ea2)
- [Classifier-Free Diffusion Guidance paper](https://arxiv.org/abs/2207.12598)

# Implementation

Now, we will work on implementing Stable Diffusion according to the above schematic as each of the parts have already been implemented. Furthermore, due to the significant training time of a model, we will use pretrained models from HuggingFace: [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4). This pretrained Stable Diffusion pipeline includes weights for the text encoder, tokenizer, variational autoencoder, and U-Net that have been trained together (again, with a fixed pretrained text encoder).

## Preparation

First, we import the necessary libraries, define a config class, and provide a helper function to assist you in your implementation. This function gets the pretrained models for the tokenizer, text encoder, VAE, and U-Net. As always, it is worth reading through this code to make sure you understand what it does.

```python
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Union, cast
import numpy as np
import torch as t
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip import modeling_clip
from transformers.tokenization_utils import PreTrainedTokenizer
from w3d5_globals import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from w3d5_part1_clip_solution import CLIPModel

MAIN = __name__ == "__main__"
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

@dataclass
class StableDiffusionConfig:
    '''
    Default configuration for Stable Diffusion.

    guidance_scale is used for classifier-free guidance.

    The sched_ parameters are specific to LMSDiscreteScheduler.
    '''

    height = 512
    width = 512
    num_inference_steps = 100
    guidance_scale = 7.5
    sched_beta_start = 0.00085
    sched_beta_end = 0.012
    sched_beta_schedule = "scaled_linear"
    sched_num_train_timesteps = 1000

    def __init__(self, generator: t.Generator):
        self.generator = generator

T = TypeVar("T", CLIPTokenizer, CLIPTextModel, AutoencoderKL, UNet2DConditionModel)

def load_model(cls: type[T], subfolder: str) -> T:
    model = cls.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder=subfolder, use_auth_token=True)
    return cast(T, model)

def load_tokenizer() -> CLIPTokenizer:
    return load_model(CLIPTokenizer, "tokenizer")

def load_text_encoder() -> CLIPTextModel:
    return load_model(CLIPTextModel, "text_encoder").to(DEVICE)

def load_vae() -> AutoencoderKL:
    return load_model(AutoencoderKL, "vae").to(DEVICE)

def load_unet() -> UNet2DConditionModel:
    return load_model(UNet2DConditionModel, "unet").to(DEVICE)

```

Now that the pretrained models we will be using are in an accessible format, try printing one of them out to examine its architecture. For example:

```python
if MAIN:
    vae = load_vae()
    print(vae)
    del vae
```

## Text encoder

Next, we provide a function to initialize a text encoder model from our implementation of CLIP and load the weights from the pretrained CLIP text encoder. This uses the `load_state_dict` function, which loads the variables from an `OrderedDict` that maps parameter names to their values (typically tensors) into another model with identically named parameters.

In this case, the pretrained `state_dict` of the `CLIPTextModel` instance contains keys prepended with `text_model.`, as the `CLIPTextModel` encapsulates the `CLIPTextTransformer` model, i.e. `type(CLIPTextModel.text_model) == CLIPTextTransformer`. Therefore, to match the input dictionary keys to the parameter names in our `CLIPModel.text_model` class, we need to modify the dictionary from `pretrained.state_dict()` to remove the `text_model.` from each key.

**Note:** You may have noticed that by using the `text_model` member of our `CLIPModel` class from Part 1, we depend on the implementation of `CLIPTextTransformer` imported from `modeling_clip`. This is the same class as that used by the pretrained text model in `CLIPTextModel`. Therefore, this function effectively initializes a `CLIPTextTransformer` class with the pretrained weights just to copy its weights to a second `CLIPTextTransformer` class. However, if we later choose to modify or re-implement the `text_model` in our `CLIPModel` class, maintaining the same parameter names, this function will serve to initialize its weights using the pretrained model weights.

```python
def clip_text_encoder(pretrained: CLIPTextModel) -> modeling_clip.CLIPTextTransformer:
    pretrained_text_state_dict = OrderedDict([(k[11:], v) for (k, v) in pretrained.state_dict().items()])
    clip_config = CLIPConfig(CLIPVisionConfig(), CLIPTextConfig())
    clip_text_encoder = CLIPModel(clip_config).text_model
    clip_text_encoder.to(DEVICE)
    clip_text_encoder.load_state_dict(pretrained_text_state_dict)
    return clip_text_encoder
```

## Getting pretrained models

Now, we're ready to start building the model. There are only a few parts to instantiate and connect into a pipeline that can transform our text prompt into an image. First, we initialize the pretrained models as well as our `CLIPModel.text_model` with pretrained text encoder weights.

```python
@dataclass
class Pretrained:
    tokenizer = load_tokenizer()
    vae = load_vae()
    unet = load_unet()
    pretrained_text_encoder = load_text_encoder()
    text_encoder = clip_text_encoder(pretrained_text_encoder)

if MAIN:
    pretrained = Pretrained()
```

## Tokenization

We provide part of a helper function that uses our `PreTrainedTokenizer` to tokenize prompt strings, embed the tokens, and concatenate embeddings for the empty padding token for "classifier-free guidance" ([see paper for details](https://arxiv.org/abs/2207.12598)).

Please implement the `uncond_embeddings` used for classifier-free guidance below, based on the format of `text_embeddings`. Note that `uncond_embeddings` should be of the same shape as `text_embeddings`, and `max_length` has already been assigned for you. Return the concatenated tensor with `uncond_embeddings` and `text_embeddings`, in that order.

```python
def tokenize(pretrained: Pretrained, prompt: list[str]) -> t.Tensor:
    text_input = pretrained.tokenizer(
        prompt,
        padding="max_length",
        max_length=pretrained.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pretrained.text_encoder(text_input.input_ids.to(DEVICE))[0]
    max_length = text_input.input_ids.shape[-1]
    pass
```

## Assembling the inference pipeline

Using the scheduler parameters defined in the config at the beginning (`sched_`), instantiate and return the `LMSDiscreteScheduler` in `get_scheduler()`. The scheduler defines the noise schedule during training and/or inference, and will be used later in our inference process.

```python
def get_scheduler(config: StableDiffusionConfig) -> LMSDiscreteScheduler:
    pass

```

Now, we will implement the missing parts of the inference pipeline in `stable_diffusion_inference()` below. The intended behavior of this function is as follows:

1. Initialize the scheduler, batch_size, multiply latent random Gaussian noise by initial scheduler noise term $\sigma_0$
2. If prompt strings are provided, compute text embeddings
3. In the inference loop, for each timestep defined by the scheduler:
    1. Expand/repeat latent embeddings by 2 for classifier-free guidance, divide the result by $\sqrt{\sigma_i^2 + 1}$ using $\sigma_i$ from the scheduler
    2. Compute concatenated noise prediction using U-Net, feeding in latent input, timestep, and text embeddings
    3. Split concatenated noise prediction $N_c = [N_u, N_t]$ into the unconditional $N_u$ and text $N_t$ portion. You can use the `torch.Tensor.chunk()` function for this.
    4. Compute the total noise prediction $N$ with respect to the guidance scale factor $g$: $N = N_u + g * (N_t - N_u)$
    5. Step to the previous timestep using the scheduler to get the next latent input
4. Rescale latent embedding and decode into image space using VAE decoder
5. Rescale resulting image into RGB space
6. Permute dimensions and convert to `PIL.Image.Image` objects for viewing/saving

Examine the existing implementation, identify which parts are missing, and implement these by referring to the surrounding code and module implementations as necessary.

```python
def stable_diffusion_inference(
    pretrained: Pretrained, config: StableDiffusionConfig, prompt: Union[list[str], t.Tensor], latents: t.Tensor
) -> list[Image.Image]:
    scheduler = get_scheduler(config)
    if isinstance(prompt, list):
        text_embeddings = None
        text_embeddings = tokenize(pretrained, prompt)
    elif isinstance(prompt, t.Tensor):
        text_embeddings = prompt
    scheduler.set_timesteps(config.num_inference_steps)
    latents = latents * scheduler.sigmas[0]
    with t.autocast("cuda"):
        for (i, ts) in enumerate(scheduler.timesteps):
            latent_input = None
            "TODO: YOUR CODE HERE"
            with t.no_grad():
                "TODO: YOUR CODE HERE"
            "TODO: YOUR CODE HERE"
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]
    images = pretrained.vae.decode(latents / 0.18215)
    images = (images * 255 / 2 + 255 / 2).clamp(0, 255)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy().round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images
```

## Trying it out!

Finally, after implementing a function to compute our latent noise (provided for you below), we can use our Stable Diffusion inference pipeline by passing in the pretrained models, config, and a prompt of strings.


```python
def latent_sample(config: StableDiffusionConfig, batch_size: int) -> t.Tensor:
    latents = t.randn(
        (batch_size, cast(int, pretrained.unet.in_channels), config.height // 8, config.width // 8),
        generator=config.generator,
    ).to(DEVICE)
    return latents


if MAIN:
    SEED = 1
    config = StableDiffusionConfig(t.manual_seed(SEED))
    prompt = ["A digital illustration of a medieval town"]
    latents = latent_sample(config, len(prompt))
    images = stable_diffusion_inference(pretrained, config, prompt, latents)
    images[0].save("./w3d5_image.png")

```

# Fun with animations!

Finally, let's close off MLAB with some interpolation-based animation fun. The idea is relatively straightforward: the continuous text embedding space from which our Stable Diffusion pipeline generates an image means that we can interpolate between two or more text embeddings to build a set of images generated from the interpolation path. In other words, this means that we can generate a relatively sensible (to the extent that the denoising model works as we expect) image "between" any other two images.

## Implementing interpolation

Given that we've already built our Stable Diffusion inference pipeline, the only thing we need to add is interpolation. Here, we create a function to handle the interpolation of tensors, `interpolate_embeddings()`, and use this function in `run_interpolation()` to loop over each embedded prompt, feeding it into the Stable Diffusion inference pipeline.

Please complete the implementation of `interpolate_embeddings()` as described. However, as this is the last day of MLAB content, if you would prefer to play around with generating images/animations feel free to use the solution code implementation.


```python
def interpolate_embeddings(concat_embeddings: t.Tensor, scale_factor: int) -> t.Tensor:
    '''
    Returns a tensor with `scale_factor`-many interpolated tensors between each pair of adjacent
    embeddings.
    concat_embeddings: t.Tensor - Contains uncond_embeddings and text_embeddings concatenated together
    scale_factor: int - Number of interpolations between pairs of points
    out: t.Tensor - shape: [2 * scale_factor * (concat_embeddings.shape[0]/2 - 1), *concat_embeddings.shape[1:]]
    '''
    "TODO: YOUR CODE HERE"
    assert out.shape == (2 * scale_factor * (num_prompts - 1), *text_embeddings.shape[1:])
    return out


def run_interpolation(prompts: list[str], scale_factor: int, batch_size: int, latent_fn: Callable) -> list[Image.Image]:
    SEED = 1
    config = StableDiffusionConfig(t.manual_seed(SEED))
    concat_embeddings = tokenize(pretrained, prompts)
    (uncond_interp, text_interp) = interpolate_embeddings(concat_embeddings, scale_factor).chunk(2)
    split_interp_emb = t.split(text_interp, batch_size, dim=0)
    interpolated_images = []
    for t_emb in tqdm(split_interp_emb):
        concat_split = t.concat([uncond_interp[: t_emb.shape[0]], t_emb])
        config = StableDiffusionConfig(t.manual_seed(SEED))
        latents = latent_fn(config, t_emb.shape[0])
        interpolated_images += stable_diffusion_inference(pretrained, config, concat_split, latents)
    return interpolated_images

```

## Prompt Interpolation

Finally, if you've implemented Stable Diffusion correctly, you're ready to play with prompt interpolation. Go ahead and fiddle with the prompts and interpolation scaling factor below, and be sure to share your favorite results on Slack!

`scale_factor` indicates the number of images between each consecutive prompt.


```python
if MAIN:
    prompts = [
        "a photograph of a cat on a lawn",
        "a photograph of a dog on a lawn",
        "a photograph of a bunny on a lawn",
    ]
    interpolated_images = run_interpolation(prompts, scale_factor=2, batch_size=1, latent_fn=latent_sample)

```

## Saving a GIF

Save your list of images as a GIF by running the following:


```python
def save_gif(images: list[Image.Image], filename):
    images[0].save(filename, save_all=True, append_images=images[1:], duration=100, loop=0)


if MAIN:
    save_gif(interpolated_images, "w3d5_animation1.gif")

```

## Speeding up interpolation

Consider how you might speed up the interpolation inference process above. Note that batching multiple prompts (making sure to concatenate their correspondings unconditional embeddings as expected) tends to speed up the per-prompt generation time. However, this also affects the random generation of Gaussian noise fed into the U-Net as the noise is different for each sample, which in practice tends to result in images that don't always "fit" together or play smoothly in an animation. Think about how you can modify the latent noise generation step to batch prompts without affecting the randomness relative to individually feeding prompts into the model, and try implementing this change.

Here, a new function `latent_sample_same()` is created which uses the same inputs as `latent_sample()` and is intended to output the same noise for a batch size of 1. For larger batches, it should use the same noise for each image in the batch. Implement this quick change, looking back at `latent_sample()` if needed, and try testing whether a larger interpolation batch size with this sampling function improves performance on your system. This will depend on your maximum batch size usually constrained by GPU memory size as well as other minor factors.


```python
def latent_sample_same(config: StableDiffusionConfig, batch_size: int) -> t.Tensor:
    '''TODO: YOUR CODE HERE'''
    return latents

```

For example, here is a call to `run_interpolation()` that uses a batch size of 2 and passes in your modified `latent_sample_same()` function to generate random noise.


```python
if MAIN:
    prompts = [
        "a photograph of a cat on a lawn",
        "a photograph of a dog on a lawn",
        "a photograph of a bunny on a lawn",
    ]
    interpolated_images = run_interpolation(prompts, scale_factor=2, batch_size=2, latent_fn=latent_sample_same)
    save_gif(interpolated_images, "w3d5_animation2.gif")

```

If you've gotten this far, you're all done with today's content as well as the standard MLAB content. Congratulations!

## Bonus

Here are a few bonus tasks as inspiration. However, feel free to play with the Stable Diffusion model to find your own idea!

### Multiple prompt image generation

What does it mean to combine prompts within a single image? Can this be done by modifying the Stable Diffusion inference process to condition on two or more text embeddings, or for parts of an image to condition on different embeddings?

### Stylistic changes

Try to identify changes in prompts that induce stylistic changes in the resulting image. For example, a painting as opposed to a photograph, or a greyscale photograph as opposed to a color photograph.

## Acknowledgements

- [HuggingFace blog post on Stable Diffusion](https://huggingface.co/blog/stable_diffusion>https://huggingface.co/blog/stable_diffusion), a great resource for introducing the model and implementing an inference pipeline.
""")

func_list = [section_home, section_1, section_2]

page_list = ["🏠 Home", "1️⃣ CLIP", "2️⃣ Stable Diffusion"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()


# %%
import numpy as np
import plotly.express as px



# %%