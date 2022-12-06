import os
if not os.path.exists("./images"):
    os.chdir("./ch5")

from st_dependencies import *
styling()

import plotly.io as pio
import re
import json

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
    return {"gan_output": read_from_html(f"gan_output")}

if "fig_dict" not in st.session_state:
    fig_dict = get_fig_dict()
    st.session_state["fig_dict"] = fig_dict
else:
    fig_dict = st.session_state["fig_dict"]

def section_home():

    st.markdown("""
## 1️⃣ Introduction
This section includes some reading material on GANs and transposed convolutions.
## 2️⃣ Transposed convolutions
In this section, you'll implement the transposed convolution operation. This is similar to a regular convolution, but designed for upsampling rather than downsampling (i.e. producing an image from a latent vector rather producing output from an image). These are very important in many generative algorithms.
## 3️⃣ GANs
Here, you'll actually implement and train your own GANs, to generate celebrity pictures. By the time you're done, you'll hopefully have produced positively sexy-looking output like this:
""")
    # cols = st.columns([1, 10, 4])
    # with cols[1]:
    #     st_image("gan_output.png", 700)
    st.plotly_chart(fig_dict["gan_output"], use_container_width=True)

def section1():

    st.success("""
A quick note here - the data that we'll be using later today to train our model is from [Celeb-A Faces dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), which can be downloaded from the link provided. There's a lot of data (1.3GB), so it might take a long time to download and unzip. For that reason, **you're recommended to download it now rather than when you get to that section!**
You should find the folder icon that says **Align&Cropped Images**, which will then direct you to a Google Drive folder containing three directories. Enter the one called **Img**, and you should see two more directories plus a zip file called `img_align_celeba.zip`. Download this zip file, and create a file in your working directory called `celebA` and unzip it into there. (Note, it is important not to just unzip it to your working directory, for reasons which will become clear later).
""")

    st.markdown(r"""
### Imports
Note, you might have to add filenames to `sys.path` to make these imports work, depending on how your directory is structured.
```python
import torch as t
from typing import Union, Optional, Tuple
from torch import nn
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from fancy_einsum import einsum
import os
import sys
from tqdm.auto import tqdm
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataclasses import dataclass
import wandb

# Add to your path here, so you can import the appropriate functions
import w5d1_utils
import w5d1_tests
from w0d2_chapter0_convolutions.solutions import pad1d, pad2d, conv1d_minimal, conv2d_minimal, Conv2d, Linear, ReLU
from w0d3_chapter0_resnets.solutions import BatchNorm2d
```
## Reading
The following are readings that you may find helpful today. You can read through them now, although you might also find it helpful to get stuck into some of the exercises and return here if/when you get confused.
* Google Machine Learning Education, [Generative Adversarial Networks](https://developers.google.com/machine-learning/gan) \*\*
    * This is a very accessible introduction to the core ideas behind GANs
    * You should read at least the sections in **Overview**, and the sections in **GAN Anatomy** up to and including **Loss Functions**
* [Unsupervised representation learning with Deep Convolutional Generative Adversarial Networks](https://paperswithcode.com/method/dcgan)
    * This paper introduced the DCGAN, and describes an architecture very close to the one we'll be building today.
    * It's one of the most cited ML papers of all time!
* [Transposed Convolutions explained with… MS Excel!](https://medium.com/apache-mxnet/transposed-convolutions-explained-with-ms-excel-52d13030c7e8) \*
    * It's most important to read the first part (up to the highlighted comment), which gives a high-level overview of why we need to use transposed convolutions in generative models and what role they play.
    * [These visualisations](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) may also help.
""")

def section2():
    st.sidebar.markdown("""
## Table of Contents
<ul class="contents">
    <li><a class="contents-el" href="#learning-objectives">Learning Objectives</a></li>
    <li><a class="contents-el" href="#introduction">Introduction</a></li>
    <li><a class="contents-el" href="#minimal-1d-transposed-convolutions">Minimal 1D transposed convolutions</a></li>
    <li><a class="contents-el" href="#1d-transposed-convolutions">1D transposed convolutions</a></li>
    <li><a class="contents-el" href="#2d-transposed-convolutions">2D transposed convolutions</a></li>
    <li><a class="contents-el" href="#making-your-own-modules">Making your own modules</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown("""
# Transposed convolutions

In this section, we'll build all the modules required to implement our DCGAN. Mainly, this involves transposed convolutions.

The material here should be pretty reminiscent of the module-building you did back in week 0!""")

    st.info(r"""
## Learning Objectives
* Understand what a transposed convolution is, and why they are important in generative networks.
    * Importantly, understand that a transposed convolution is not the same as the inverse of a convolution!
* Implement your own transposed convolutions, by treating them as a kind of modified convolution (using your code from week 0).
""")

    st.markdown(r"""
## Introduction
Why do we care about transposed convolutions? One high-level intuition goes something like this: generators are basically decoders in reverse. We need something that performs the reverse of a convolution - not literally the inverse operation, but something reverse in spirit, which uses a kernel of weights to project up to some array of larger size.
**Importantly, a transposed convolution isn't literally the inverse of a convolution**. A lot of confusion can come from misunderstanding this!
You can describe the difference between convolutions and transposed convolutions as follows:
* In convolutions, you slide the kernel around inside the input. At each position of the kernely, you take a sumproduct between the kernel and that section of the input to calculate a single element in the output.
* In transposed convolutions, you slide the kernel around what will eventually be your output, and at each position you add some multiple of the kernel to your output.
Below is an illustration of both for comparison, in the 1D case (where $*$ stands for the 1D convolution operator, and $*^T$ stands for the transposed convolution operator). Note the difference in size between the output in both cases. With standard convolutions, our output is smaller than our input, because we're having to fit the kernel inside the input in order to produce the output. But in our transposed convolutions, the output is actually larger than the input, because we're fitting the kernel inside the output.""")

    st_image("img12.png", 800)
    st.caption("Above: convolution. Below: transposed convolution.")
    st.markdown("")
    st.markdown("""
##### **Question - what do you think the formula is relating `input_size`, `kernel_size` and `output_size` in the case of 1D convolutions (with no padding or stride)?**
""")

    with st.expander("Answer"):
        st.markdown("""The formula is `output_size = input_size + kernel_size - 1`. For instance, with the example above this becomes 6 = 4 + 3 - 1.
        
Note how this exactly mirrors the equation in the convolutional case; it's identical if we swap around `output_size` and `input_size`.""")

    st.markdown("")

    st.markdown("""
Consider the elements in the output of the transposed convolution: `x`, `y+4x`, `z+4y+3x`, etc. Note that these look like convolutions, just using a version of the kernel where the element order is reversed (and cropped at the edges). This observation leads nicely into why transposed convolutions are called transposed convolutions - because they can actually be written as convolutions, just with a slightly modified input and kernel.
##### **Question - how do you think this operation can be cast as a convolution? In other words, exactly what arrays `input` and `kernel` would produce the same output as the transposed convolution above, if we performed a standard convolution on them?**
""")

    with st.expander("Hint"):
        st.markdown("Try padding `input` with zeros.")

#     with st.expander("Hint"):
#         st.markdown("""

# Let `input_mod` and `kernel_mod` be the modified versions of the input and kernel, to be used in the convolution. 

# You should be able to guess what `kernel_mod` is by looking at the diagram.

# Also, from the formula for transposed convolutions, we must have:

# ```
# output_size = input_mod_size + kernel_mod_size - 1
# ```

# But we currently have:

# ```
# output_size = input_size - kernel_size + 1
# ```

# which should help you figure out what size `input_mod` needs to be, relative to `input`.
# """)

#     with st.expander("Hint 2"):
#         st.markdown("""
# `kernel_mod` should be the same size as kernel (but altered in a particular way). `input_mod` should be formed by padding `input`, so that its size increases by `2 * (kernel_size - 1)`.
# """)

    with st.expander("Answer"):
        st.markdown("""
If you create `input_modified` by padding `input` with exactly `kernel_size - 1` zeros on either side, and reverse your kernel to create `kernel_modified`, then the convolution of these modified arrays equals your original transposed convolution output.
""")
        st_image("img3.png", 850)

    st.markdown(r"""
## Minimal 1D transposed convolutions
Now, you should implement the function `conv_transpose1d_minimal`. You're allowed to call functions like `conv1d_minimal` and `pad1d` which you wrote during `w0d2_chapter0_convolutions` (remember that, by the observation above, we can treat transposed convolutions as a kind of modified convolution).
One important note - in our convolutions we assumed the kernel had shape `(out_channels, in_channels, kernel_width)`. Here, PyTorch has a different convention: `in_channels` comes before `out_channels`.
```python
def conv_transpose1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv_transpose1d using bias=False and all other keyword arguments left at their default values.
    x: shape (batch, in_channels, width)
    weights: shape (in_channels, out_channels, kernel_width)
    Returns: shape (batch, out_channels, output_width)
    '''
    pass

w5d1_tests.test_conv_transpose1d_minimal(conv_transpose1d_minimal)
```
So far, the discussions and diagrams above have assumed a single input and output channel. But now that you've seen how a transposed convolution can be written as a convolution on modified inputs, you should be able to guess how to generalize it to multiple channels.
""")

    st.markdown(r"""
Now we add in the extra parameters `padding` and `stride`, just like we did for our convolutions back in week 0. The basic idea here is that both parameters mean the inverse of what they did in for convolutions.
* In convolutions, `padding` tells you how much to pad the input by.
    * But in **transposed convolutions**, we pad the input by `kernel_size - 1 - padding` (recall that we're already padding by `kernel_size - 1` by default). So padding decreases our output size rather than increasing it.
* In convolutions, `stride` tells you how much to step the kernel by, as it's being moved around inside the input.
    * In **transposed convolutions**, stride does something different: you space out all your input elements by an amount equal to `stride` before performing your transposed convolution.
    * This might sound strange, but **it's actually equivalent to performing strides as you're moving the kernel around inside the output.** This diagram should help show why:""")

    st_image("img4.png", 900)

    st.markdown("""
For this reason, transposed convolutions are also referred to as **fractionally strided convolutions**, since a stride of 2 over the output is equivalent to a 1/2 stride over the input (i.e. every time the kernel takes two steps inside the spaced-out version of the input, it moves one stride with reference to the original input).
##### Question - what is the formula relating `output_size`, `input_size`, `kernel_size`, `stride` and `padding`?""")

    with st.expander("Hint"):
        st.markdown("""
First take the original formula above, with no stride or padding:
```
output_size = input_size + kernel_size - 1
```
Next, consider the effect of adding in padding, then adding in stride.
""")

    with st.expander("Answer"):
        st.markdown("""
Without any padding, we had:
```
output_size = input_size + kernel_size - 1
```
Twice the `padding` parameter gets subtracted from the right-hand side (since we pad by the same amount on each side), so this gives us:
```
output_size = input_size + kernel_size - 1
```
Finally, consider `stride`. As mentioned above, we can consider stride here to have the same effect as "spacing out" elements in the input. Each non-zero element will be `stride - 1` positions apart (for instance, `stride = 2` turns `[1, 2, 3]` into `[1, 0, 2, 0, 3]`). You can check that the number of zeros added between elements equals `(input_size - 1) * (stride - 1)`. When you add this to the right hand side, and simplify, you are left with:
```
output_size = (input_size - 1) * stride + kernel_size - 2 * padding
```
""")

    st.markdown("""
Padding should be pretty easy for you to implement on top of what you've already done. For strides, you will need to construct a strided version of the input which is "spaced out" in the way described above, before performing the transposed convolution. It might help to write a `fractional_stride` function; we've provided the code for you to do this.
""")

    st.markdown("""
```python
def fractional_stride_1d(x, stride: int = 1):
    '''Returns a version of x suitable for transposed convolutions, i.e. "spaced out" with zeros between its values.
    This spacing only happens along the last dimension.
    x: shape (batch, in_channels, width)
    Example: 
        x = [[[1, 2, 3], [4, 5, 6]]]
        stride = 2
        output = [[[1, 0, 2, 0, 3], [4, 0, 5, 0, 6]]]
    '''
    pass

w5d1_tests.test_fractional_stride_1d(fractional_stride_1d)

def conv_transpose1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    '''Like torch's conv_transpose1d using bias=False and all other keyword arguments left at their default values.
    x: shape (batch, in_channels, width)
    weights: shape (in_channels, out_channels, kernel_width)
    Returns: shape (batch, out_channels, output_width)
    '''
    pass

w5d1_tests.test_conv_transpose1d(conv_transpose1d)
```""")

    with st.expander("Help - I'm not sure how to implement fractional_stride."):
        st.markdown("""The easiest way is to initialise an array of zeros with the appropriate size, then slicing to set its elements from `x` (e.g. the slice `[::2]` returns every second element along that dimension).
Warning - if you do it this way, **make sure the output has the same device as `x`**. This gave me a bug in my code that took a while to find!""")

    with st.expander("Help - I'm not sure how to implement conv_transpose1d."):
        st.markdown("""
There are three things you need to do:
* Modify `x` by "spacing it out" with `fractional_stride_1d` and padding it the appropriate amount
* Modify `weights` (just like you did for `conv_transpose1d_minimal`)
* Use `conv1d_minimal` on your modified `x` and `weights` (just like you did for `conv_transpose1d_minimal`)
""")
        st.markdown("")

    st.info(r"""
Another fun fact about transposed convolutions - they are also called **backwards strided convolutions**, because they are equivalent to taking the gradient of Conv2d with respect to its output.
Optional bonus - can you show this mathematically?""")

    st.markdown(r"""
## 2D transposed convolutions
Finally, we get to 2D transposed convolutions! Since there's no big conceptual difference between this and the 1D case, we'll jump straight to implementing the full version of these convolutions, with padding and strides. A few notes:

* You'll need to make `fractional_stride_2d`, which performs spacing along the last two dimensions rather than just the last dimension.
* Defining the modified version of your kernel will involve reversing on more than one dimension. You'll still need to perform the same rearrangement flipping the output and input channel dimensions though.
* We've provided you with the `force_pair` functions again.

```python
IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

def fractional_stride_2d(x, stride_h: int, stride_w: int):
    '''Same as fractional_stride_1d, except we apply it along the last 2 dims of x (width and height).
    '''
    pass

def conv_transpose2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    '''Like torch's conv_transpose2d using bias=False
    x: shape (batch, in_channels, height, width)
    weights: shape (in_channels, out_channels, kernel_height, kernel_width)
    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    pass

w5d1_tests.test_conv_transpose2d(conv_transpose2d)
```
## Making your own modules
Now that you've written a function to calculate the convolutional transpose, you should implement it as a module just like you've done for `Conv2d` previously. Your weights should be initialised with the uniform distribution $U(-\sqrt{k}, \sqrt{k})$, where $k = 1 / (\text{out\_channels} \times \text{kernel\_width} \times \text{kernel\_height})$ (this is PyTorch's standard behaviour for convolutional transpose layers). 
Don't worry too much about this though, because we'll use our own initialisation anyway.""")

    st.markdown("""
```python
class ConvTranspose2d(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.ConvTranspose2d with bias=False.
        Name your weight field `self.weight` for compatibility with the tests.
        '''
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        pass

w5d1_tests.test_ConvTranspose2d(ConvTranspose2d)
```
You'll also need to implement a few more modules, which have docstrings provided below. They are:
* [`Tanh`](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html) which is an activation function used by the DCGAN you'll be implementing.
* [`LeakyReLU`](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html) which is an activation function used by the DCGAN you'll be implementing. This function is popular in tasks where we we may suffer from sparse gradients (GANs are a primary example of this).
* [`Sigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html), for converting the single logit output from the discriminator into a probability.
```python
class Tanh(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        pass

class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        pass
    def forward(self, x: t.Tensor) -> t.Tensor:
        pass
    def extra_repr(self) -> str:
        pass

class Sigmoid(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        pass

w5d1_tests.test_Tanh(Tanh)
w5d1_tests.test_LeakyReLU(LeakyReLU)
w5d1_tests.test_Sigmoid(Sigmoid)
```
""")

def section3():
    st.sidebar.markdown("""
## Table of Contents
<ul class="contents">
    <li><a class="contents-el" href="#learning-objectives">Learning Objectives</a></li>
    <li><a class="contents-el" href="#how-gans-work">How GANs work</a></li>
    <li><a class="contents-el" href="#dcgan-paper">DCGAN paper</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#discriminator-architecture">Discsriminator architecture</a></li>
        <li><a class="contents-el" href="#convolutions">Convolutions</a></li>
        <li><a class="contents-el" href="#activation-functions">Activation functions</a></li>
        <li><a class="contents-el" href="#batchnorm">BatchNorm</a></li>
        <li><a class="contents-el" href="#weight-initialisation">Weight initialisation</a></li>
        <li><a class="contents-el" href="#optimizers">Optimizers</a></li>
        <li><a class="contents-el" href="#misc-points">Misc. points</a></li>
    </ul></li>
    <li><a class="contents-el" href="#building-your-generator-and-discriminator">Building your generator and discriminator</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#if-you-re-stuck">If you're stuck...</a></li>
    </ul></li>
    <li><a class="contents-el" href="#loading-data">Loading data</a></li>
    <li><a class="contents-el" href="#training-loop">Training loop</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#training-the-discriminator">Training the discriminator</a></li>
        <li><a class="contents-el" href="#training-the-generator">Training the generator</a></li>
        <li><a class="contents-el" href="#logging-images-to-wandb">Logging images to wandb</a></li>
        <li><a class="contents-el" href="#training-the-generator">Implementing your training loop</a></li>
        <li><a class="contents-el" href="#debugging-and-improvements">Debugging and Improvements</a></li>
    </ul></li>
    <li><a class="contents-el" href="#final-words">Final words</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown("""
# GANs""")
    st.info(r"""
## Learning Objectives
* Read and understand the important parts of the [DCGAN paper](https://arxiv.org/abs/1511.06434v2).
* Understand the loss function used in GANs, and why it can be expected to result in the generator producing realistic outputs.
* Implement the DCGAN architecture from the paper, with relatively minimal guidance.
* Learn how to identify and fix bugs in your GAN architecture, to improve convergence properties.
""")

    st.markdown(r"""
## How GANs work

The basic idea behind GANs is as follows: you have two networks, the **generator** and the **discriminator**. The generator's job is to produce output realistic enough to fool the discriminator, and the discriminator's job is to try and tell the difference between real and fake output. The idea is for both networks to be trained simultaneously, in a positive feedback loop: as the generator produces better output, the discriminator's job becomes harder, and it has to learn to spot more subtle features distinguishing real and fake images, meaning the generator has to work harder to produce images with those features.

The discriminator works by taking an image (either real, or created by the generator), and outputting a single value between 0 and 1, which is the probability that the discriminator puts on the image being real. The discriminator sees the images, but not the labels (i.e. whether the images are real or fake), and it is trained to distinguish between real and fake images with maximum accuracy. The architecture of discriminators in a GAN setup is generally a mirror image of the generator, with transposed convolutions swapped out for convolutions. This is the case for the DCGAN paper we'll be reading (which is why they only give a diagram of the generator, not both). The discriminator's loss function is the cross entropy between its probability estimates ($D(x)$ for real images, $D(G(z))$ for fake images) and the true labels ($1$ for real images, $0$ for fake images).

The generator works by taking in a vector $z$, whose elements are all normally distributed with mean 0 and variance 1. We call the space $z$ is sampled from the **latent dimension** or **latent space**, and $z$ is a **latent vector**. The formal definition of a latent space is *an abstract multi-dimensional space that encodes a meaningful internal representation of externally observed events.* We'll dive a little deeper into what this means and the overall significance of latent spaces later on, but for now it's fine to understand this vector $z$ as a kind of random seed, which causes the generator to produce different outputs. After all, if the generator only ever produced the same image as output then the discriminator's job would be pretty easy (just subtract the image $g$ always produces from the input image, and see if the result is close to zero!). The generator's objective function is an increasing function of $D(G(z))$, in other words it tries to produce images $G(z)$ which have a high chance of fooling the discriminator (i.e. $D(G(z)) \approx 1$).

The ideal outcome when training a GAN is for the generator to produce perfect output indistringuishable from real images, and the discriminator just guesses randomly. However, the precise nature of the situations when GANs converge is an ongoing area of study (in general, adversarial networks have very unstable training patterns). For example, you can imagine a situation where the discriminator becomes almost perfect at spotting fake outputs, because of some feature that the discriminator spots and that the generator fails to capture in its outputs. It will be very difficult for the generator to get a training signal, because it has to figure out what feature is missing from its outputs, and how it can add that feature to fool the discriminator. And to make matters worse, maybe marginal steps in that direction will only increase the probability of fooling the discriminator from almost-zero to slightly-more-than-almost-zero, which isn't much of a training signal! Later on we will see techniques people have developed to overcome problems like this and others, but in general they can't be solved completely.
""")

    with st.expander("Optional exercise - what conditions must hold for the discriminator's best strategy to be random guessing with probability 0.5?"):
        st.markdown(r"""
It is necessary for the generator to be producing perfect outputs, because otherwise the discriminator could do better than random guessing.

If the generator is producing perfect outputs, then the discriminator never has any ability to distinguish real from fake images, so it has no information. Its job is to minimise the cross entropy between its output distribution $(D(x), 1-D(x))$, and the distribution of real/fake images. Call this $(p, 1-p)$, i.e. $p$ stands for the proportion of images in training which are real. Note how we just used $p$ rather than $p(x)$, because there's no information in the image $x$ which indicates whether it is real or fake. Trying to minimize the cross entropy between $(p, 1-p)$ and $(D(x), 1-D(x))$ gives us the solution $D(x) = p$ for all $x$. In other words, our discriminator guesses real/fake randomly with probability equal to the true underlying frequency of real/fake images in the data. This is 0.5 if and only if the data contains an equal number of real and fake images.

To summarize, the necessary and sufficient conditions for $(\all x) \; D(x) = 0.5$ being the optimal strategy are:

* The generator $G$ produces perfect output
* The underlying frequency of real/fake images in the data is 50/50
""")
        st.markdown("")
    st.markdown("")

    st_excalidraw("gans", 1500)
    # st_image("gans-light.png", 1500)

    st.markdown(r"""
## DCGAN paper

Now, you're ready to implement and train your own DCGAN! You should do this only using the [DCGAN paper](https://arxiv.org/abs/1511.06434v2) (implementing architectures based on descriptions in papers is an incredibly valuable skill for any would-be research engineer). There are some hints we've provided below if you get stuck while attempting this.

All the details you'll need to build the architecture can be found on page 4 of the paper, or before (in particular, the diagram on page 4 and the description on page 3 should be particularly helpful).
""")

    st.markdown("""
A few other notes, to make this task slightly easier or resolve ambiguities in the paper's description:
### Discriminator architecture
The paper includes only a diagram of the generator, not the discriminator. You can assume that the discriminator's architecture is the mirror image of the generator - in other words, you start from an image and keep applying convolutions to downsample. Rather than applying a projection and reshape at the end, we instead flatten and apply a fully connected linear layer which maps the output of size `(1024, 4, 4)` to a single value, which we then calculate the sigmoid of to use as our probability that the image is real.
### Convolutions
The diagram labels the kernels in the convolutional transpose layers as having size 5, **but they actually have size 4**. I don't actually know why this is - it either seems like a mistake, or the thing they're labelling is not actually the kernel!
The stride is 2, and given that the width and height are doubled at each step, you should be able to deduce what the padding is.""")

    with st.expander("Click to reveal the padding of each of the transposed convolutions."):
        st.markdown("""
Our formula from earlier is:
```
output_size = (input_size - 1) * stride + kernel_size - 2 * padding
```
The stride is always 2, the kernel size is always 4, and the output size is always double the input size (it goes up in powers of two, from 4 to 64). Substituting this into the formula above, we find `padding=1`. This is true for every layer.""")

    st.markdown(r"""
The convolutional kernels in the discriminator also have size 4 and stride 2, and similarly since they halve the input size you should be able to deduce the padding.
### Activation functions
Pay attention to the paper's instructions on activation functions. There are three you'll have to use: Tanh, ReLU and LeakyReLU.
Note that the two fully connected layers (at the start of the generator, and end of the discriminator) will also be getting activation functions.
### BatchNorm
When the paper says BatchNorm is used in a layer, they mean it is used between the convolution and the activation function. So a convolutional block in the generator goes `ConvTransposed -> BatchNorm -> ActivationFn`, and a convolutional block in the discriminator goes `Conv -> BatchNorm -> ActivationFn`.
The generator's first layer does include a BatchNorm, but the discriminator's last layer doesn't, meaning you should have 4 BatchNorm layers in the generator but just 3 in the discriminator.
### Weight initialisation
They mention at the end of page 3 that all weights were initialized from a $N(0, 0.02)$ distribution. This applies to the convolutional and convolutional transpose layers' weights, but the BatchNorm layers' weights should be initialised from $N(1, 0.02)$ (since 1 is their default value). The BatchNorm biases should all be set to zero (which they are by default).
You should fill in the following code to get a function that initialises weights appropriately (inplace). The function `nn.init.normal_` might be useful here.
```python
def initialize_weights(model) -> None:
    pass
```
### Optimizers
You should use the paper's recommended optimizers and hyperparameters. However, if their batch size is too large, you're recommended to do one of two things:
1. Scale down the learning rate from the one recommended in the paper. Theory suggests you should scale your learning rate down by the square root of the ratio between your batch size and theirs (all else being equal, this means the variance of each gradient descent step is the same).
2. Only perform an optimizer step every `n` batches (where `n` is equal to the ratio between their batch size and your batch size). In this case, you should be careful about where gradients are accumulated, since you'll be running computations on both `netG` and `netD` within your training loop. You should also divide the loss by `n` (so it's equivalent to taking a mean across a larger batch).
### Misc. points
None of the fully connected layers or convolutional layers have biases.
The paper recommends having your noise $Z$ input into the generator be uniformly distributed. Instead, you should have $Z$ be a standard multivariate normal distribution (i.e. $Z \sim N(\mathbf{0}, \mathbf{I})$).
## Building your generator and discriminator
You should implement your code below.
I've provided the initialisation parameters I used when building this architecture, but this just represents one possible design choice, and you should feel free to design your GAN in whichever way makes most sense to you.
```python
class Generator(nn.Module):

    def __init__(
        self,
        latent_dim_size: int,           # size of the random vector we use for generating outputs
        img_size: int,                  # size of the images we're generating
        img_channels: int,              # indicates RGB images
        generator_num_features: int,    # number of channels after first projection and reshaping
        n_layers: int,                  # number of CONV_n layers
    ):
        pass

    def forward(self, x: t.Tensor):
        pass

class Discriminator(nn.Module):

    def __init__(
        self,
        img_size: int,
        img_channels: int,
        generator_num_features: int,
        n_layers: int,
    ):
        pass

    def forward(self, x: t.Tensor):
        pass

class DCGAN(nn.Module):
    netD: Discriminator
    netG: Generator
```""")

    st.info("""
If it's straining your computer's GPU (or just resulting in extremely long epochs), you can reduce the model size by halving the number of channels at each intermediate step (e.g. the first shape is `(512, 4, 4)` rather than `(1024, 4, 4)`). This will reduce the cost of forward/backward passes by a factor of 4 (can you see why?). Also, consider terminating your training loop before you see the entire dataset (if you aren't using a service like Lambda Labs to access a more powerful GPU, it can take upwards of 2 hours to get through a single epoch for this dataset).
""")

    st.markdown("""
### If you're stuck...
...you can import the generator and discriminator from the solutions, and compare it with yours. `celeb_DCGAN` is the full architecture, while `celeb_mini_DCGAN` corresponds to the choice to halve each of the channel sizes (see the note above).

```python
from solutions import celeb_DCGAN
w5d1_utils.print_param_count(my_Generator, celeb_DCGAN.netG)
```
Also, a good way to test your model's architecture if you don't have access to the real thing is to run input through it and check you don't get any errors and the output is the size you expect - this can catch a surprisingly large fraction of all bugs! 
Lastly, remember that `torchinfo` is a useful library for inspecting the architecture of your model.
""")


    st.markdown("""
## Loading data
You should have already downloaded the Celeb-A data and unzipped it to your working directory (return to the note at the top of this page if you haven't). Your next step should be to transform this output into tensors and construct a dataset from it.
To load in the data, we'll need to use the `ImageFolder` function. You might have come across this before if you went through the week 0 material on finetuning ResNet. This function's first two arguments are `root` (specifying the filepath for the root directory containing the data), and `transform` (which is a transform object that gets applied to each image).
Note, this function requires the root directory to contain subdirectories, which themselves contain the actual data. These subdirectories tell PyTorch what the data labels are. This is why you needed to unzip the file to a directory within your working directory, rather than just into your working directory. You only need one folder, because we don't have any labels (or rather, the label represents "real/fake" and so all data in this folder has the same label "real").

```python
from torchvision import transforms, datasets

    transform = transforms.Compose([
        transforms.Resize((image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

trainset = ImageFolder(
    root="data",
    transform=transform
)

w5d1_utils.show_images(trainset, rows=3, cols=5)
```
""")

    st.markdown(r"""
## Training loop
Recall, the goal of training the discriminator is to maximize the probability of correctly classifying a given input as real or fake. The goal of the generator is to produce images to fool the discriminator. This is framed as a **minimax game**, where the discriminator and generator try to solve the following:
$$
\min_G \max_D V(D, G)=\mathbb{E}_x[\log (D(x))]+\mathbb{E}_z[\log (1-D(G(z)))]
$$
where $D$ is the discriminator function mapping an image to a probability estimate for whether it is real, and $G$ is the generator function which produces an image from latent vector $z$.
Since we can't know the true distribution of $x$, we instead estimate the expression above by calculating it over a back of real images $x$ (and some random noise $z$). This gives us a loss function to train against (since $D$ wants to maximise this value, and $G$ wants to minimise this value). For each batch, we perform gradient descent on the discriminator and then on the generator.
What this looks like in practice:
### Training the discriminator
We take the following steps:
* Zero the gradients of $D$.
    * This is important because if the last thing we did was evaluate $D(G(z))$ (in order to update the parameters of $G$), then $D$ will have stored gradients from that backward pass.
* Generate random noise $z$, and compute $D(G(z))$. Take the average of $\log(1 - D(G(z)))$, and we have the first part of our loss function.
* Take the real images  $x$ in the current batch, and use that to compute $\log(D(x))$. This gives us the second part of our loss function.
* We now add the two terms together, and perform gradient ascent (since we're trying to maximise this expression).
    * You can perform gradient ascent by either flipping the sign of the thing you're doing a backward pass on, or passing the keyword argument `maximize=True` when defining your optimiser (all optimisers have this option).
Tip - when calculating $D(G(z))$, for the purpose of training the discriminator, it's best to first calculate $G(z)$ then call `detach` on this tensor before passing it to $D$. This is because you then don't need to worry about gradients accumulating for $G$.
### Training the generator
We take the following steps:
* Zero the gradients of $G$.
* Generate random noise $z$, and compute $D(G(z))$.
* We **don't** use $\log(1 - D(G(z)))$ to calculate our loss function, instead we use $\log(D(G(z)))$ (and gradient ascent).
""")

    with st.expander("Question - can you explain why we use log(D(G(z))? (The Google reading material mentions this but doesn't really explain it.)"):
        st.markdown(r"""
Early in learning, when the generator is really bad at producing realistic images, it will be easy for the discriminator to distinguish between them. So $\log(1 - D(G(z)))$ will be very close to $\log(1) = 0$. The gradient of $\log$ at this point is quite flat, so there won't be a strong gradient with which to train $G$. To put it another way, a marginal improvement in $G$ will have very little effect on the loss function. On the other hand, $\log(D(G(z)))$ tends to negative infinity as $D(G(z))$ gets very small. So the gradients here are very steep, and a small improvement in $G$ goes a long way.
It's worth emphasising that these two functions are both monotonic in opposite directions, so maximising one is equivalent to minimising the other. We haven't changed anything fundamental about how the GAN works; this is just a trick to help with gradient descent.
""")
    st.info("""Note - PyTorch's [`BCELoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) clamps its log function outputs to be greater than or equal to -100. This is because in principle our loss function could be negative infinity (if we take log of zero). You might find you need to employ a similar trick if you're manually computing the log of probabilities.
    
You might also want to try using [`nn.utils.clip_grad_norm`](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) in your model. Using a value of 1.0 usually works fine for this function.
However, you should probably only try these if your model doesn't work straight away. I found I was able to get decent output on the celebrity database without either of these tricks.""")

    st.markdown("""
### Implementing your training loop
Again, we've provided a possible template below, which you're welcome to ignore!
It can be hard to check your model is working as expected, because the interplay between the loss functions of the discriminator and the generator isn't always interpretable. A better method is to display output from your generator at each step. We've provided you with a function to do this, called `w5d1_utils.display_generator_output`. It takes `netG` and `latent_dim_size` as its first arguments, and `rows` and `cols` as keyword arguments. You can write your own version of this function if you wish. If you do, remember to **set a random seed before creating your latent vectors**.
""")
    with st.expander("Question - why do you think it's important to set a random seed?"):
        st.markdown("""
So that we can compare our outputs across different stages of our model's evolution. It becomes less meaningful if each set of output is being produced from completely different random vectors.""")

    st.markdown(r"""
### Logging images to `wandb`
Weights and biases provides a nice feature allowing you to log images! This requires you to use the function `wandb.Image`. The first argument is `data_or_path`, which can take the following forms (to name a few):
* A numpy array in shape `(height, width)` or `(height, width, 1)` -> interpreted as monochrome image
* A numpy array in shape `(height, width, 3)` -> interpreted as RGB image
* A PIL image (can be RGB or monochrome)
When it comes to logging, you can log a list of images rather than a single image. However, it's often better to use `einops` to reshape your batched images into a single row of images, since it will appear with a better aspect ratio in the Weights and Biases site. 
Here is some example code, and the output it produces from my GAN. The object `arr` has shape `(20, 3, 28, 28)`, i.e. it's an array of 20 RGB images.

```python
arr_rearranged = einops.rearrange(arr, "b c h w -> h (b w) c")
images = wandb.Image(arr_rearranged, caption="Top: original, Bottom: reconstructed")
wandb.log({"images": images}, step=n_examples_seen)
```""")

    st_image("gan_output_2.png", 650)
    st.markdown(r"""
You should now implement your training loop below. We've provided a suggested template for you, which uses dataclasses in a similar way to your DQN and PPO implementations in the previous chapter. However, you're free to structure this part of your code however you like. At this point in the course, you should hopefully be developing a good sense for the most useful way to structure training loops in a way that allows you to iterate quickly.

```python
@dataclass
class DCGANargs():
    latent_dim_size: int
    img_size: int
    img_channels: int
    generator_num_features: int
    n_layers: int
    trainset: datasets.ImageFolder
    lr: float
    betas: Tuple[float]
    batch_size: int = 8
    epochs: int = 1
    track: bool = True
    cuda: bool = True
    seconds_between_image_logs: int = 40

def train_DCGAN(args: DCGANargs) -> DCGAN:
    pass
```
If your training works correctly, you should see your discriminator loss consistently low, while your generator loss will start off high (and will be very jumpy) but will slowly come down over time. The details of convergence speed will vary depending on details of the hardware, but I would recommend that if your generator's output doesn't resemble anything like a face after 2 minutes, then something's probably going wrong in your code.
### Debugging and Improvements
GANs are notoriously hard to get exactly right. I ran into quite a few bugs myself building this architecture, and I've tried to mention them somewhere on this page to help particpiants avoid them. If you run into a bug and are able to fix it, please send it to me and I can add it here, for the benefit of everyone else!
* Make sure you apply the batch normalisation (mean 0, std dev 0.02) to your linear layers as well as your convolutional layers.
    * More generally, in your function to initialise the weights of your network, make sure no layers are being missed out. The easiest way to do this is to inspect your model afterwards (i.e. loop through all the params, printing out their mean and std dev).
Also, you might find [this page](https://github.com/soumith/ganhacks) useful. It provides several tips and tricks for how to make your GAN work. Some of them we've already covered (or are assumed from the model architecture we're using), for instance:
* (1) Normalize your inputs
* (3) Use normally distributed noise, not uniformly distributed
* (4) Avoid activation functions with sparse gradients (e.g. ReLU)
But there are others we haven't discussed here and which might improve training, such as:
* (7) Use stability tracks from RL
    * For instance, in our DQN implementation we kept a buffer of past experiences (also called an **experience replay**). You can try keeping checkpointed versions of your discriminator and generator networks (i.e. the discriminator is trained on a staggered version of the generator, and vice-versa).
* (13) Add noise to inputs, decay over time
    * e.g. adding noise to the inputs to your discriminator, or to each layer of your generator
    * These can help solve the problem of DCGAN **overconfidence**
## Final words
Hopefully, these two days of exercises showed you some of the difficulties involved in training GANs. They're notoriously unstable, and the cases in which they do / do not converge have been the source of much study. In the next few days, we'll build up to studying **diffusion models**, a more recent type of generative algorithm which have absolutely blown GANs out of the water.
Additionally, the ideas of GANs live on in other architectures, such as VQGANs (Vector-Quantized GANs) which can be combined with CLIP (Contrastive Language-Image Pre-training) to create **VQGAN+CLIP**, an open-source text-to-image model which was published around the same time as DALLE.
""")

def section4():
    st.sidebar.markdown("""
## Table of Contents
<ul class="contents">
    <li><a class="contents-el" href="#smooth-interpolation">Smooth interpolation</a></li>
    <li><a class="contents-el" href="#mnist">MNIST</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown("""
## Smooth interpolation
Suppose you take two vectors in the latent space. If you use your generator to create output at points along the linear interpolation between these vectors, your image will change continuously (because it is a continuous function of the latent vector), but it might look very different at the start and the end. Can you create any cool animations from this?
Instead of linearly interpolating between two vectors, you could try applying a [rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix) to a vector. There are certain subtle reasons you might expect the performance to be better here than with linear interpolation, can you guess what these are? (Hint - it has to do with the $l_2$ norm of the latent vector).
## MNIST
You can try training your GAN to produce MNIST images, rather than faces. You might actually find this harder than for the celebrity faces (my guess as to why is that the features of numbers are much easier for the discriminator to understand and recognise, meaning the generator is often stuck in a situation where it has no good training gradients). 
If you do manage to get this working, can you find a latent vector that produces each of the digits from 0 to 9? What happens if you smoothly interpolate between them, like in the task suggested above?
""")

func_list = [section_home, section1, section2, section3, section4]

page_list = ["🏠 Home", "1️⃣ Introduction", "2️⃣ Transposed convolutions", "3️⃣ GANs", "4️⃣ Bonus"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()
