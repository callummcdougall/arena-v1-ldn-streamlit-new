import os
if not os.path.exists("./images"):
    os.chdir("./ch5")

from st_dependencies import *
styling()

def section_home():

    st.markdown("""
## 1️⃣ Introduction

This section includes some reading material on GANs and transposed convolutions.

## 2️⃣ Transposed convolutions

In this section, you'll implement the transposed convolution operation. This is similar to a regular convolution, but designed for upsampling rather than downsampling (i.e. producing an image from a latent vector rather producing output from an image). These are very important in many generative algorithms.

## 3️⃣ GANs

Here, you'll actually implement and train your own GANs, to generate celebrity pictures. By the time you're done, you'll hopefully have produced output like this:
""")
    cols = st.columns([1, 10, 4])
    with cols[1]:
        st_image("gan_output.png", 600)
    st.markdown("""
---

This material is expected to take approximately two days. At the end of that time, you'll be able to decide whether you want to keep studying it, or change to the **training at scale** track.
""")

def section1():

    st.info("""
A quick note here - the data that we'll be using later today to train our model is from [Celeb-A Faces dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), which can be downloaded from the link provided. There's a lot of data (1.3GB), so it might take a long time to download and unzip. For that reason, **you're recommended to download it now rather than when you get to that section!**

You should find the folder icon that says **Align&Cropped Images**, which will then direct you to a Google Drive folder containing three directories. Enter the one called **Img**, and you should see two more directories plus a zip file called `img_align_celeba.zip`. Download this zip file, and create a file in your working directory and unzip it into there. (Note, it is important not to just unzip it to your working directory, for reasons which will become apparent later).
""")

    st.markdown("""
### Imports

```python
import torch as t
from typing import Union
from torch import nn
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from fancy_einsum import einsum
import os
from tqdm.auto import tqdm
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, TensorDataset
import wandb
import utils
```

## Reading

* Google Machine Learning Education, [Generative Adversarial Networks](https://developers.google.com/machine-learning/gan) \*\*
    * This is a very accessible introduction to the core ideas behind GANs
    * You should read at least the sections in **Overview**, and the sections in **GAN Anatomy** up to and including **Loss Functions**
* [Unsupervised representation learning with deep convolutional generative adversarial networks](https://paperswithcode.com/method/dcgan)
    * This paper introduced the DCGAN, and describes an architecture very close to the one we'll be building today.
    * It's one of the most cited ML papers of all time!
* [Transposed Convolutions explained with… MS Excel!](https://medium.com/apache-mxnet/transposed-convolutions-explained-with-ms-excel-52d13030c7e8) \*
    * It's most important to read the first part (up to the highlighted comment), which gives a high-level overview of why we need to use transposed convolutions in generative models and what role they play.
    * You can read beyond this (and there are some very helpful visualisations), although I've also provided some illustrations below of 1D transposed convolutions which might be easier to follow if you haven't come across this operation before.
    * [These visualisations](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) may also help.
""")

def section2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#introduction">Introduction</a></li>
    <li><a class="contents-el" href="#minimal-1d-transposed-convolutions">Minimal 1D transposed convolutions</a></li>
    <li><a class="contents-el" href="#1d-transposed-convolutions">1D transposed convolutions</a></li>
    <li><a class="contents-el" href="#2d-transposed-convolutions">2D transposed convolutions</a></li>
    <li><a class="contents-el" href="#making-your-own-modules">Making your own modules</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown("""
## Transposed convolutions (and other modules)

In this section, we'll build all the modules required to implement our DCGAN. 

### Introduction

Why do we care about transposed convolutions? One high-level intuition goes something like this: generators are basically decoders in reverse. We need something that performs the reverse of a convolution - not literally the inverse operation, but something reverse in spirit, which uses a kernel of weights to project up to some array of larger size.

**Importantly, a transposed convolution isn't literally the inverse of a convolution**. A lot of confusion can come from misunderstanding this!

You can describe the difference between convolutions and transposed convolutions as follows:

* In convolutions, you slide the kernel around inside the input. At each position of the kernely, you take a sumproduct between the kernel and that section of the input to calculate a single element in the output.
* In transposed convolutions, you slide the kernel around what will eventually be your output, and at each position you add some multiple of the kernel to your output.

Below is an illustration of both for comparison, in the 1D case (where $*$ stands for the 1D convolution operator, and $*^T$ stands for the transposed convolution operator). Note the difference in size between the output in both cases. With standard convolutions, our output is smaller than our input, because we're having to fit the kernel inside the input in order to produce the output. But in our transposed convolutions, the output is actually larger than the input, because we're fitting the kernel inside the output.""")

    st_image("img12.png", 500)
    st.markdown("")
    st.markdown("""
**Question - what do you think the formula is relating `input_size`, `kernel_size` and `output_size` in the case of 1D convolutions (with no padding or stride)?**
""")

    with st.expander("Answer"):
        st.markdown("""The formula is `output_size = input_size + kernel_size - 1`. 
        
Note how this exactly mirrors the equation in the convolutional case; it's identical if we swap around `output_size` and `input_size`.""")

    st.markdown("")

    st.markdown("""
Consider the elements in the output of the transposed convolution: `x`, `y+4x`, `z+4y+3x`, etc. Note that these look like convolutions, just using a version of the kernel where the element order is reversed (and sometimes cropped). This observation leads nicely into why transposed convolutions are called transposed convolutions - because they can actually be written as convolutions, just with a slightly modified input and kernel.

**Question - how can this operation be cast as a convolution? In other words, exactly what arrays `input` and `kernel` would produce the same output as the transposed convolution above, if we performed a standard convolution on them?**
""")

    with st.expander("Hint"):
        st.markdown("""

Let `input_mod` and `kernel_mod` be the modified versions of the input and kernel, to be used in the convolution. 

You should be able to guess what `kernel_mod` is by looking at the diagram.

Also, from the formula for transposed convolutions, we must have:

```
output_size = input_mod_size + kernel_mod_size - 1
```

But we currently have:

```
output_size = input_size - kernel_size + 1
```

which should help you figure out what size `input_mod` needs to be, relative to `input`.
""")

    with st.expander("Hint 2"):
        st.markdown("""
`kernel_mod` should be the same size as kernel (but altered in a particular way). `input_mod` should be formed by padding `input`, so that its size increases by `2 * (kernel_size - 1)`.
""")

    with st.expander("Answer"):
        st.markdown("""
If you create `input_mod` by padding `input` with exactly `kernel_size - 1` zeros on either side, and reverse your kernel to create `kernel_mod`, then the convolution of these modified arrays equals your original transposed convolution output.
""")
        st_image("img3.png", 500)

    st.markdown("""
### Minimal 1D transposed convolutions

Now, you should implement the function `conv_transpose1d_minimal`. You're allowed to call functions like `conv1d_minimal` and `pad1d` which you wrote on week 0, day 2.

One important note - in our convolutions we assumed the kernel had shape `(out_channels, in_channels, kernel_width)`. Here, the order is different: `in_channels` comes before `out_channels`.

```python
def conv_transpose1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv_transpose1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (in_channels, out_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    pass

utils.test_conv_transpose1d_minimal(conv_transpose1d_minimal)
```

So far, the discussions and diagrams above have assumed a single input and output channel. But now that you've seen how a transposed convolution can be written as a convolution on modified inputs, you should be able to guess how to generalize it to multiple channels.

""")

    st.markdown("""
Now we add in the extra parameters `padding` and `stride`, just like we did for our convolutions back in week 0.

The basic idea is that both parameters mean the inverse of what they did in for convolutions.

In convolutions, `padding` tells you how much to pad the input by. But in transposed convolutions, we pad the input by `kernel_size - 1 - padding` (recall that we're already padding by `kernel_size - 1` by default). So padding decreases our output size rather than increasing it.

In convolutions, `stride` tells you how much to step the kernel by, as it's being moved around inside the input. In transposed convolutions, stride does something different: you space out all your input elements by an amount equal to `stride` before performing your transposed convolution. This might sound strange, but **it's actually equivalent to performing strides as you're moving the kernel around inside the output.** This diagram should help show why:""")

    st_image("img4.png", 500)
    
    st.markdown("""
For this reason, transposed convolutions are also referred to as **fractionally strided convolutions**, since a stride of 2 over the output is equivalent to a 1/2 stride over the input (i.e. every time the kernel takes two steps inside the spaced-out version of the input, it moves one stride with reference to the original input).""")

    with st.expander("Question - what is the formula relating output size, input size, kernel size, stride and padding? (note, you shouldn't need to refer to this explicitly in your functions)"):
        st.markdown("""
Without any padding, we had:

```
output_size = input_size + kernel_size - 1
```

Twice the `padding` parameter gets subtracted from the RHS (since we pad by the same amount on each side), so this gives us:

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

utils.test_fractional_stride_1d(fractional_stride_1d)

def conv_transpose1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    '''Like torch's conv_transpose1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (in_channels, out_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    pass

utils.test_conv_transpose1d(conv_transpose1d)
```""")

    with st.expander("Help - I'm not sure how to implement fractional_stride."):
        st.markdown("""The easiest way is to initialise an array of zeros with the appropriate size, then slicing to set its elements from `x`.

Warning - if you do it this way, **make sure the output has the same device as `x`**. This gave me a bug in my code that took a while to find!""")

    with st.expander("Help - I'm not sure how to implement conv_transpose1d."):
        st.markdown("""
There are three things you need to do:
* Modify `x` by "spacing it out" with `fractional_stride_1d` and padding it the appropriate amount
* Modify `weights` (just like you did for `conv_transpose1d_minimal`)
* Use `conv1d_minimal` on your modified `x` and `weights` (just like you did for `conv_transpose1d_minimal`)
""")

    st.info(r"""
Another fun fact about transposed convolutions - they are also called **backwards strided convolutions**, because they are equivalent to taking the gradient of Conv2d with respect to its output.

Optional bonus - can you show this mathematically?""")

    st.markdown(r"""
### 2D transposed convolutions

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
    '''
    Same as fractional_stride_1d, except we apply it along the last 2 dims of x (width and height).
    '''
    pass

def conv_transpose2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    '''Like torch's conv_transpose2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (in_channels, out_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    pass

utils.test_conv_transpose2d(conv_transpose2d)
```

### Making your own modules

Now that you've written a function to calculate the convolutional transpose, you should implement it as a module just like you've done for `Conv2d` previously. Your weights should be initialised with the uniform distribution $U(-\sqrt{k}, \sqrt{k})$, where $k = 1 / (\text{out\_channels} \times \text{kernel\_width} \times \text{kernel\_height})$ (this is PyTorch's standard behaviour for convolutional transpose layers). Don't worry too much about this though, because we'll use our own initialisation anyway).""")

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

utils.test_ConvTranspose2d(ConvTranspose2d)
```

You'll also need to implement a few more modules, which have docstrings provided below. They are:
* [`Tanh`](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html) which is an activation function used by the DCGAN you'll be implementing.
* [`LeakyReLU`](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html) which is an activation function used by the DCGAN you'll be implementing. This function is popular in tasks where we we may suffer from sparse gradients (GANs are a primary example of this).
* [`Sigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html), for converting the single logit output from the discriminator into a probability.

```python
class Tanh(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        pass

utils.test_Tanh(Tanh)
```

```python
class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        pass
    def forward(self, x: t.Tensor) -> t.Tensor:
        pass
    def extra_repr(self) -> str:
        pass

utils.test_LeakyReLU(LeakyReLU)
```

```python
class Sigmoid(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        pass

utils.test_Sigmoid(Sigmoid)
```
""")

def section3():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#gans">GANs</a></li>
    <li><a class="contents-el" href="#loading-data">Loading data</a></li>
    <li><a class="contents-el" href="#training-loop">Training loop</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#training-the-discriminator">Training the discriminator</a></li>
        <li><a class="contents-el" href="#training-the-generator">Training the generator</a></li>
        <li><a class="contents-el" href="#logging-images-to-wandb">Logging images to wandb</a></li>
        <li><a class="contents-el" href="#training-the-generator">Implementing your training loop</a></li>
        <li><a class="contents-el" href="#fixing-bugs">Fixing bugs</a></li>
    </ul></li>
    <li><a class="contents-el" href="#final-words">Final words</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown("""
## GANs

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

### Building your generator and discriminator

You should implement your code below. I've provided the initialisation parameters I used when building this architecture, but this just represents one possible design choice, and you should feel free to design your GAN in whichever way makes most sense to you.

```python
class Generator(nn.Module):

    def __init__(
        self,
        latent_dim_size: int,           # size of the random vector we use for generating outputs
        img_size = int,                 # size of the images we're generating
        img_channels = int,             # indicates RGB images
        generator_num_features = int,   # number of channels after first projection and reshaping
        n_layers = int,                 # number of CONV_n layers
    ):
        pass

    def forward(self, x: t.Tensor):
        pass

class Discriminator(nn.Module):

    def __init__(
        self,
        img_size = 64,
        img_channels = 3,
        generator_num_features = 1024,
        n_layers = 4,
    ):
        pass

    def forward(self, x: t.Tensor):
        pass
```""")

    st.info("""
If it's straining your computer's GPU, you can reduce the model size by halving the number of channels at each intermediate step (e.g. the first shape is `(512, 4, 4)` rather than `(1024, 4, 4)`). This will reduce the cost of forward/backward passes by a factor of 4 (can you see why?).

This is one reason I chose to implement the generators and discriminators as I did above - just one parameter has to be changed in order to reduce the model size in this way.
""")

    st.markdown("""
### If you're stuck...

...you can import the generator and discriminator from the solutions, and compare it with yours. `netG_celeb` is the full architecture, while `netG_celeb_mini` corresponds to the choice to halve each of the channel sizes (see the note above). The discriminators can be imported in the same way (with `netD`).

```python
from solutions import netG_celeb_mini
utils.print_param_count(my_Generator, netG_celeb_mini)
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
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = ImageFolder(
    root="data",
    transform=transform
)

utils.show_images(trainset, rows=3, cols=5)
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

It can be hard to check your model is working as expected, because the interplay between the loss functions of the discriminator and the generator isn't always interpretable. A better method is to display output from your generator at each step. We've provided you with a function to do this, called `utils.display_generator_output`. It takes `netG` and `latent_dim_size` as its first arguments, and `rows` and `cols` as keyword arguments. You can write your own version of this function if you wish. If you do, remember to **set a random seed before creating your latent vectors**.
""")
    with st.expander("Question - why do you think it's important to set a random seed?"):
        st.markdown("""
So that we can compare our outputs across different stages of our model's evolution. It becomes less meaningful if each set of output is being produced from completely different random vectors.

### Logging images to `wandb`

Weights and biases provides a nice feature allowing you to log images! This requires you to use the function `wandb.Image`. The first argument is `data_or_path`, which can take the following forms (to name a few):

* A numpy array in shape `(height, width)` or `(height, width, 1)` -> interpreted as monochrome image
* A numpy array in shape `(height, width, 3)` -> interpreted as RGB image
* A PIL image (can be RGB or monochrome)

When it comes to logging, you can log a list of images rather than a single image. Example code, and the output it produces from my GAN:

```python
# arr has shape (20, 28, 28), i.e. it's an array of 20 monochrome images

arr_rearranged = einops.rearrange(arr, "(b1 b2) h w -> (b1 h) (b2 w)", b1=2)

images = wandb.Image(arr_rearranged, caption="Top: original, Bottom: reconstructed")
wandb.log({"images": images}, step=n_examples_seen)
```""")

    st_image("gan_output_2.png", 500)
    st.markdown("""
You should now implement your training loop below.

```python
def train_generator_discriminator(
    netG: Generator, 
    netD: Discriminator, 
    optG,
    optD,
    trainloader,
    epochs: int,
    max_epoch_duration: Optional[Union[int, float]] = None,           # Each epoch terminates after this many seconds
    log_netG_output_interval: Optional[Union[int, float]] = None,     # Generator output is logged at this frequency
    use_wandb: bool = True
):
    pass
```

If your training works correctly, you should see your discriminator loss consistently low, while your generator loss will start off high (and will be very jumpy) but will slowly come down over time.

This varies depending on details of the hardware, but I would recommend that if your generator's output doesn't resemble anything like a face after 2 minutes, then something's probably going wrong in your code.

### Fixing bugs

GANs are notoriously hard to get exactly right. I ran into quite a few bugs myself building this architecture, and I've tried to mention them somewhere on this page to help particpiants avoid them. If you run into a bug and are able to fix it, please send it to me and I can add it here, for the benefit of everyone else!

* Make sure you apply the layer normalisation (mean 0, std dev 0.02) to your linear layers as well as your convolutional layers.
* More generally, in your function to initialise the weights of your network, make sure no layers are being missed out. The easiest way to do this is to inspect your model afterwards (i.e. loop through all the params, printing out their mean and std dev).

Also, you might find [this page](https://github.com/soumith/ganhacks) useful. It provides several tips and tricks for how to make your GAN work (many of which we've already mentioned on this page).

## Final words

Hopefully, these two days of exercises showed you some of the difficulties involved in training GANs. They're notoriously unstable, and the cases in which they do / do not converge have been the source of much study. In the next few days, we'll build up to studying **diffusion models**, a more recent type of generative algorithm which have absolutely blown GANs out of the water. 

We may also see GANs return later on, in the form of VQGANs (Vector-Quantized GANs).
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
