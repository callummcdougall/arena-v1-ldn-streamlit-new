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
    st.session_state["fig_dict"] = {}

if "grid_output" not in st.session_state["fig_dict"]:
    st.session_state["fig_dict"] |= get_fig_dict()

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

    st.markdown("""
## 1️⃣ Introduction to diffusion models

In this section, we'll read up on diffusion models, and try to understand some of the maths behind them. Don't worry if you don't follow all of it though - it's not absolutely essential to grokking how they work, and you should be able to get through the material regardless (although you might get less out of it).

## 2️⃣ Training a basic diffusion model

We'll apply our knowledge from section 1, and train a very simple diffusion model to produce color gradients.

## 3️⃣ The DDPM Architecture

The 2020 paper [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf) (on which most of our work during this section will be based) describes the architecture it uses. This will be more complicated than any model you've designed so far in this course (and probably more complicated than any you will design), but the skills should be transferrable from your experiences with transformers in chapter 1.

## 4️⃣ Training U-Net on FashionMNIST

Finally, you'll train your model on [https://www.kaggle.com/datasets/zalando-research/fashionmnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist).
""")

def section_1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#introduction">Introduction</a></li>
    <li><a class="contents-el" href="#reading">Reading</a></li>
    <li><a class="contents-el" href="#what-even-is-diffusion?">What Even Is Diffusion?</a></li>
    <li><a class="contents-el" href="#the-connection-to-vaes">The connection to VAEs</a></li>
    <li><a class="contents-el" href="#forward-and-backward-processes">Forward and backward processes</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#forward-process">Forward process</a></li>
        <li><a class="contents-el" href="#backward-process">Backward process</a></li>
    </ul></li>
    <li><a class="contents-el" href="#loss-functions">Loss functions</a></li>
    <li><a class="contents-el" href="#simplifications">Simplifications</a></li>
    <li><a class="contents-el" href="#gradient-descent">Gradient descent</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
## Introduction

Today you're going to implement and train a tiny diffusion model from scratch on FashionMNIST. Specifically, we'll be following the [2020 paper **Denoising Diffusion Probabilistic Models**](https://arxiv.org/pdf/2006.11239.pdf), which was an influential early paper in the field of realistic image generation. Understanding this paper will give you a solid foundation to understand state of the art diffusion models. I personally believe that diffusion models are an exciting research area and will make other methods like GANs obsolete in the coming years. To get a sense of how diffusion works, you'll first implement and train an even tinier model to generate images of color gradients.

The material is divided into three parts. In Part 1 we'll implement the actual equations for diffusion and train a basic model to generate color gradient images. In Part 2 we'll implement the U-Net architecture, which is a spicy mix of convolutions, MLPs, and attention all in one network. In Part 3, we'll train the U-Net architecture on FashionMNIST.

You're getting to be experts at implementing things, so today will be a less guided experience where you'll have to refer to the paper as you go. Don't worry about following all the math - the math might look intimidating but it's actually either less complicated than it sounds, or can safely be skipped over for the time being.

Diffusion models are an area of active research with rapid developments occurring - our goal today is to conceptually understand all the moving parts and how they fit together into a working system. It's also to understand all the different names for things - part of the difficulty is that diffusion models can be understood mathematically from different perspectives and since these are relatively new, different people use different terminology to refer to the same thing.

Once you understand today's material, you'll have a solid foundation for understanding state of the art systems involving diffusion models like:

- [GLIDE](https://arxiv.org/abs/2112.10741)
- [DALL-E 2](https://openai.com/dall-e-2/) by OpenAI
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion) by the University of Heidelberg
- [ImageGen](https://imagen.research.google/) by Google Brain.

## Reading

* [Denoising Probabilistic Diffusion Models](https://arxiv.org/pdf/2006.11239.pdf) \*
    * This is the paper that all of our implementations will be based on over the next few days.
    * You can see a snapshot of the most important parts of the paper [here](https://hojonathanho.github.io/diffusion/).
* [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) \*
    * This is a (slightly) more accessible guide to some of the maths in the diffusion papers below. In particular, reading up to the diagram in the **Reverse diffusion process** section (before the maths gets too heavy!) is strongly recommended.
* [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/pdf/2208.11970.pdf)
    * The first chapters should provide a recap of the VAE material from last week. The following chapters transition smoothly into diffusion models (pun intended). Up to page 17 will be relevant (although you might want to skip some of the more intense maths which comes up near the end).

If you find these readings confusing, you can pick the one or two that seem most readable and go through them at the same time as you complete the exercises below. This will allow you to get a hands-on experience applying the formulae as you learn about them.

Also, I've tried to provide my own explanations below which liberally sample from all the resources above (as well as some material from MLAB), so you might want to try that first!

## What Even Is Diffusion?

We're going to start by thinking about the input distribution of images. [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) is a dataset of 60K training examples and 10K test examples that belong to one of 10 different classes like "t-shirt" or "sandal". Each image is 28x28 pixels and in 8-bit grayscale. We think of those dataset examples as being samples drawn IID (independent and identically distributed) from some larger input distribution "the set of all FashionMNIST images".

One way to think about the input distribution is a mapping from each of the $256^{28*28}$ grayscale images to the probability that the image would be collected if we collected more training examples via an identical process as was used to obtain the 60K training examples.

Our goal in generative modeling is to take this input distribution and learn a very rough estimate of the probability in various regions. It should be near zero for images that look like random noise, and also near zero for a picture of a truck since that isn't part of the concept of "the set of all FashionMNIST images".

For our training examples, the fact that they were already sampled is evidence that their probability should be pretty high, but we only have information on 60K examples which is really not a lot compared to $256^{28*28}$. To have any hope of mapping out this space, we need to make some assumptions.

**The assumption behind the forward process is that if we add Gaussian noise to an image from the distribution, on average this makes the noised image less likely to belong to the distribution**. This isn't guaranteed - there exists some random noise that you could sample with positive probability that is exactly what's needed to turn your sandal into a stylish t-shirt.

The claim is that this is an empirical fact about the way the human visual system perceives objects - a sandal with a small splotch on it still looks like a sandal to us. As long as this holds most of the time, then we've successfully generated an additional training example. In addition, we know something about how the new example relates to the original example.

Note that this is similar but not the same as data augmentation in traditional supervised learning. In that setup, we make a perturbation to the original image and claim that the class label is preserved - that is, we would tell the model via the loss function that our noised sandal is exactly as much a sandal as the original sandal is, for any level of noise up to some arbitrary maximum. In today's setup, we're claiming that the noised sandal is less of a FashionMNIST member in proportion to the amount of noise involved.

Now that we know how to generate as much low-probability data as we want, in theory we could learn a reverse function that takes an image and returns one that is *more* likely to belong to the distribution.

Then we could just repeatedly apply the reverse function to "hill climb" and end up with a final image that has a relatively large probability. We know that deep neural networks are a good way to learn complicated functions, if you can define a loss function suitable for gradient descent and if you can find a suitable parameterization so that learning is smooth.""")

    st.info(r"""
Diffusion models are insired by a branch of maths/physics called **non-equilibrium thermodynamics**. A core idea here is the **second law of thermodynamics**, which is the tendency for an isolated system to increase in entropy (or disorder) over time. 

One pithy (albeit entirely accurate) way of describing diffusion models is as attempts to **reverse the second law of thermodynamics**, i.e. working against entropy. They model the **forward process** of adding Gaussian noise to an image (increasing the entropy), and then use this to learn the **backward process** which allows them to generate images from a random seed. It still seems incredible to me that something like this actually works!
""")

    st.markdown(r"""
## The connection to VAEs

If you understand VAEs, then you're well on the way to understanding diffusion models too. In fact, we can view a diffusion model as just a more advanced VAE with a few special additions.

Recall the following diagram we had for VAEs in the mathematical derivation section:""")

    st_excalidraw("vae-graphical-2", 500)

    st.markdown(r"""
and we had the loss function derived from the ELBO:
$$
\mathbb{E}_{z \sim q_\phi(z \mid x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right] =\underbrace{\mathbb{E}_{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})\right]}_{\text {reconstruction loss }}-\underbrace{D_{\mathrm{KL}}\left(q_\phi(\boldsymbol{z} \mid \boldsymbol{x}) \| p(\boldsymbol{z})\right)}_{\text {regularisation term }}
$$
We can also extend the concept of VAEs into **hierarchical VAEs**, where you learn multiple processes which encode the input as a probability distribution and then sample from that distribution.""")
    st_excalidraw("vae-graphical-3", 650)
    st.markdown(r"""
What would our loss function be in this case? Again, we can start with what would be the equivalent of the ELBO in this situation:
$$
\begin{aligned}
p(x)&=\iint q_\phi\left(z_1, z_2 \mid x\right) \frac{p_\theta\left(x, z_1, z_2\right)}{q_\phi\left(z_1, z_2 \mid x\right)} \\
&=\mathbb{E}_{z_1, z_2 \sim q_\phi\left(z_1, z_2 \mid x\right)}\left[\frac{p_\theta\left(x, z_1, z_2\right)}{q_\phi\left(z_1, z_2 \mid x\right)}\right]\\
\\
\log p(x) &\geq \mathbb{E}_{z_1, z_2 \sim q_\phi\left(z_1, z_2 \mid x\right)}\left[\log \frac{p_\theta\left(x, z_1, z_2\right)}{q_\phi\left(z_1, z_2 \mid x\right)}\right]
\end{aligned}
$$
and then use this lower bound expression to derive our loss function, in a way which looks similar to the last one:
$$
\mathbb{E}_{q_\phi\left(z_1, z_2 \mid x\right)}\left[\log \frac{p_\theta\left(x, z_1, z_2\right)}{q_\phi\left(z_1, z_2 \mid x\right)}\right] =\underbrace{\mathbb{E}_{q_\phi\left(z_1 \mid x\right)}\left[\log p_{\theta}(x \mid z_1)\right]}_{\text {reconstruction loss }}-\underbrace{D_{\mathrm{KL}}\left(q_\phi(z_1 \mid x) \,\|\, p_\theta(z_1 \mid z_2)\right)}_{\text{consistency term}} - \underbrace{D_{\mathrm{KL}}\left(q_\phi(z_2 \mid z_1) \,\|\, p(z_2)\right)}_{\text {regularisation term }}
$$
Here we still have one reconstruction loss term (the log-probability of reconstructing $x$ from the first-step latent), and the regularisation term which makes sure that the final latent vector $z_2$ has the distribution we want. But we also have a middle term, which we can interpret as the **consistency term** - it tries to make the distribution at $z_1$ consistent in both the processes $q_\phi$ (left to right) and $p_\theta$ (right to left).

Finally, we come to diffusion models. These can be viewed as a special kind of hierarchical VAE, where the process $q_\phi$ (which we refer to as the **forward process**) is predefined, not learned. We say that our forward process is adding noise to the image, and our **backward process** is where we learn to denoise images.
""")
    st_excalidraw("vae-graphical-4", 750)

# It might seem a bit of a stretch to equate latent vectors and noised images. After all, we thought about our latent vectors as a compressed representation of the features of our image; is it reasonable to think about noised images as a kind of latent vector? Well, sort of. 
    st.markdown(r"""
Just like for the hierarchical VAE, we can decompose our loss into three terms (the learned parameters are coloured $\color{orange}\text{orange}$):

$$
\begin{align*}
L(\color{orange}\theta\color{black}) &=\mathbb{E}_q\bigg[-\log \color{orange}p_\theta(\mathbf{x}_0 | \mathbf{x}_1)\color{black} + \sum_{t=2}^T D_{\mathrm{KL}}\left(q\left(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0\right) \| \,\color{orange}p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)\color{black}\right) + D_{\mathrm{KL}}\left(q\left(\mathbf{x}_T | \mathbf{x}_0\right) \|\, p\left(\mathbf{x}_T\right)\right)\bigg]\\
&=\underbrace{L_0}_{\text{reconstruction loss}} + \underbrace{\sum_{t=2}^T L_t}_{\text{consistency terms}} + \underbrace{L_T}_{\text{regularisation term}}
\end{align*} 
$$

We assume that our process runs for long enough that $x_T$ is complete random noise (i.e. no trace of the original image is left), so $L_T$ evaluates to basically zero and we can ignore it. We called the other terms consistency and reconstruction loss, but they can all be described as a kind of consistency term, which tries to make sure that the model leans to reverse the random noise process at each step.""")

    st.info(r"""
**Note about latent spaces and noised images**

It might seem a bit of a stretch to equate latent vectors and noised images. After all, we thought about our latent vectors as a compressed representation of the features of our image; is it reasonable to think about noised images as a kind of latent vector? Well, sort of. Adding noise to the input images in a diffusion model can be thought of as adding random perturbations to the data, which forces the model to learn the underlying structure of the data in the same way that adding noise to our latent vectors forced the VAEs to learn the underlying structure. The main difference between these two cases is that the forward process of our diffusion model always stays in the same basis as the original image, rather than learning a mapping to a latent space with different (usually fewer) dimensions.
""")
    st.markdown(r"""

Now that we've broadly sketched out the connection, let's get into some more specifics like how the noise process is represented, and how our parameters are actually learned.

### Forward process

To start with, we have $x_0$ which is a sample of data, and $q(x_0)$ which is a probability distribution. The interpretation of $q(x_0)$ is as the probability that we would choose $x_0$ if we were sampling from the idealised process which generated all our input images. $x_0$ has `channels * height * width` elements (where `channels=3` in the case of RGB images). Rather than thinking of $x_0$ as a 3D tensor, we can also view it as a vector with this number of elements elements.

The **forward process** is given by:
$$
\begin{equation}
q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right):=\prod_{t=1}^T q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right), \quad q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right):=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)
\end{equation}
$$

Let's break down this notation. $\mathcal{N}$ stands for the multivariate normal distribution. This function has three arguments - the second and third represent the mean and variance, and the first is a point at which the corresponding probability density function will be evaluated. So we could also write this as:
$$
\begin{equation}
\mathbf{x}_t \sim q(\, \cdot \, | \mathbf{x}_{t-1}) = \mathcal{N}\left(\sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I} \right)
\end{equation}
$$
which is terminology you may be more familiar with. For the rest of this section, we'll assume $\mathcal{N}$ is the distribution if it has two arguments (comma-separated), and the function if it has three arguments (semicolon-comma-separated). This is the same convention used by the DDPM paper.

Rather than writing this as a distribution, we can also write it as an exact equation: $\;\mathbf{x}_t=\sqrt{1-\beta_t} \mathbf{x}_{t-1}+\sqrt{\beta_t} \boldsymbol{\delta}_t\;\,$ where $\boldsymbol{\delta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

The forward process represents the adding of Gaussian noise to our image. At each step, we construct $\mathbf{x}_t$ from $\mathbf{x}_{t-1}$ by shifting it towards zero (the mean is a less-than-1 scalar multiple of $\mathbf{x}_{t-1}$) and adding some IID random noise with variance $\beta_t$.""")

    with st.expander("Why do you think we use this value as the mean (i.e. multiplying by the square root), rather than just using x_t-1 ?"):
        st.markdown(r"""
In the long-run equilibrium of this process, we want every element of $\mathbf{x}_t$ to be IID with $\mathcal{N}(\mathbf{0}, \mathbf{I}$ distribution. Without scaling down the mean in this way, we would just be adding random noise progressively, and $\mathbf{x}_t$ would get arbirarily large (since a high-dimensional random walk will always diverge). 

Doing things this way, we can continually "pull $\mathbf{x}_t$ back to zero" as we add random noise.
""")

    with st.expander("What is the interpretation of beta in this formula?"):
        st.markdown(r"""
$\beta_t$ is the variance of the noise we add at each step. If it was 0 then we'd never change our image; if it was 1 then we would immediately lose our image and jump straight to the $\mathcal{N}(\mathbf{0}, \mathbf{I})$ distribution.

We are somewhere between these two extremes, and the smaller $\beta_t$ is the closer we are to $\mathbf{x}_{t-1}$ (i.e. having no noise). 
""")

    st.markdown(r"""
Theoretically $\beta_t$ can be a learned parameter, but in practice it is often fixed rather than learned. We will have it increase linearly over time (intuitively, this is because adding noise will have a very degrading effect at first with the image $x_0$ being completely clean, but the more noise is added the less noticable more noise will be).

$\mathbf{x_{1:T}}$ means the vector containing $(\mathbf{x}_1, ..., \mathbf{x}_T)$. The first part of the formula above just says that the distribution of the sequence of images can be factorised into a product of probability density functions, with each $\mathbf{x}_t$ only dependent on the image immediately before it. This is why we call the sequence a **Markov chain** - the probabilistic evolution rule of an image at a point in time only depends on its current state.

### Backward process

The thing we really want to do is construct the backward process by reversing our forward process. We write our **backward process** (also called the **reverse process**) as:
$$
\begin{equation}
\color{orange}p_\theta(\mathbf{x}_{0: T})\color{black}:=p(\mathbf{x}_T) \prod_{t=1}^T \color{orange}p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)\color{black}, \quad \color{orange}p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)\color{black}:=\mathcal{N}\left(\mathbf{x}_{t-1} ; \color{orange}\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)\color{black}, \color{orange}\mathbf{\Sigma}_\theta(\mathbf{x}_t, t)\color{black}\right) 
\end{equation}
$$
A note on terminology - any time you see $\theta$ appearing in an expression, this represents the parameters of a neural network. In other words, we are trying to learn the parameters of this backward process distribution so that we can recover the original image. I will use the color $\color{orange}\textbf{orange}$ to indicate learned parameters.

## Simplifications

At the end of page 2 of the DDPM paper, we see that $q(\mathbf{x}_t | \mathbf{x}_0)$ admits a nice closed-form solution:
$$
\begin{equation}
q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\bar{\alpha}_t} \mathbf{x}_0,\left(1-\bar{\alpha}_t\right) \mathbf{I}\right)
\end{equation}
$$
where $\alpha_t=1-\beta_t$, and $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$.

This formula is relatively easy to derive - we just write $\mathbf{x}_t=\sqrt{\alpha_t} \mathbf{x}_{t-1}+\sqrt{1-\alpha_t} \boldsymbol{\delta}_t$ (where $\boldsymbol{\delta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$), then keep substituting in this formula on the right hand side (and using the fact that a sum of $\mathcal{N}(\mathbf{0}, \mathbf{I})$ terms is also normally distributed, with variances summing). The implication is that a sequence of steps of the form "scale your image and add random noise" is equivalent to a single step of "scale your image and add random noise", where the scaling factor & amount of random noise is larger. This helps us when we're calculating $\mathbf{x}_t$, because we don't have to add $t$ different random variables to $\mathbf{x}_0$ (you'll implement this in the next section).

Another nice trick - we can get a formula for the distribution of $\mathbf{x}_{t-1}$, conditional on $\mathbf{x}_t$ ***and*** $x_0$:
$$
\begin{equation}
q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_{t-1} ; \color{blue}{\tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), }\color{red}{\tilde{\beta}_t \mathbf{I}}\color{black}\right)
\end{equation}
$$
where we have:
$$
\begin{aligned}
\color{red}{\tilde{\beta}}_t&=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \cdot \beta_t\\
    \\
\color{blue}{\boldsymbol{\tilde{\mu}}_t}&=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_t\right)
\end{aligned}
$$
and $\boldsymbol{\epsilon}_t$ is the (normalised) noise term which gets us from $\mathbf{x}_0$ to $\mathbf{x}_t$, i.e. we have $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon_t}$, and $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

How should we interpret this expression, intuitively? The basic idea is this: we know a lot of noise gets added as we go from $\mathbf{x}_0 \to \mathbf{x}_{t-1}$, and a bit of noise gets added as we go from $\mathbf{x}_{t-1} \to \mathbf{x}_t$. Furthermore, we actually know what the noise ratio is between these two steps, because we know the value of the parameters $\beta_t$. So we have an idea of how far along the path from $\mathbf{x}_0$ to $\mathbf{x}_t$ we are likely to find $\mathbf{x}_{t-1}$. As an extreme case, if $\beta_t = 1 - \alpha_t \approx 0$, this means almost no noise is added going from $\mathbf{x}_{t-1}$ to $\mathbf{x}_t$, so we expect to find $\mathbf{x}_{t-1}$ much closer to $\mathbf{x}_t$ than $\mathbf{x}_0$, and the variance of $\mathbf{x}_{t-1}$ is very small.""")

    st_image("noise-sketch.png", 500)
    st.markdown(r"""
**Why does this trick help us?** Well, let's return to our loss function from earlier:
$$
\begin{align}
L(\color{orange}\theta\color{black}) &=\mathbb{E}_q\bigg[-\log \color{orange}p_\theta(\mathbf{x}_0 | \mathbf{x}_1)\color{black} + \sum_{t=2}^T D_{\mathrm{KL}}\left(q\left(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0\right) \| \,\color{orange}p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)\color{black}\right) + D_{\mathrm{KL}}\left(q\left(\mathbf{x}_T | \mathbf{x}_0\right) \|\, p\left(\mathbf{x}_T\right)\right)\bigg]\\
&=L_0 + \sum_{t=2}^T L_t + \cancel{L_T}
\end{align}
$$

Consider $L_{t-1}$ for $2 \leq t \leq T$. This is the KL divergence between two normal distributions. Recall in the section on VAEs that we measured the KL divergence between two normal distributions as part of our loss function:
$$
D_{K L}\big(\mathcal{N}(\mu, \sigma^2)\,||\, \mathcal{N}(0,1)\big)=\frac{1}{2}(\mu^2+\sigma^2-1)-\log \sigma
$$
Here instead we are measuring the multi-dimensional KL divergence:
$$
D_{KL} \big(\, \mathcal{N}\big(\color{blue}\tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}\tilde{\beta}_t \mathbf{I}\,\color{black} \big) \;||\;  \mathcal{N}\left(\color{orange}\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)\color{black}, \color{orange}\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)\right)\color{black} \big)
$$

This is obviously a bit more complicated! However, we can simplify this a lot. Firstly, we assume $\color{orange}\boldsymbol{\Sigma}_\theta$ is known rather than learned. The most common choice is $\color{orange}\boldsymbol{\Sigma}_\theta\color{black}=\color{red}\tilde{\beta}_t \mathbf{I}$, to match variances (although $\color{orange}\boldsymbol{\Sigma}_\theta\color{black}=\color{red}\beta_t \mathbf{I}\color{black}$ is sometimes used instead). This allows us to simplify the expression a lot. Since we only care about the contribution to the loss function from our mean $\boldsymbol{\mu}_\theta$, we are eventually left with (a scalar multiple of):
$$
\|\color{blue}{\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) }\color{black}-\color{orange}\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)\color{black}\|^2
$$

Note that in $(7)$, we wrote $\color{blue}{\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) }$ in terms of $\mathbf{x}_t$ and a noise term $\boldsymbol{\epsilon}_t$. If we already know $\mathbf{x}_t$, then knowing the noise term is equivalent to knowing $\color{blue}{\tilde{\boldsymbol{\mu}}_t}$. It turns out that it's much easier for our function to learn this noise term rather than learning the mean, so we can rewrite our estimated mean as:
$$
\begin{align}
\color{orange}\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)\color{black}&=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \color{orange}\epsilon_\theta\left(\mathbf{x}_t, t\right)\color{black}\right)\\
&=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \color{orange}\boldsymbol{\epsilon}_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_t, t\right)\color{black}\right)
\end{align}
$$
and finally, rewrite our loss function (up to a scalar multiple) as:
$$
\begin{equation}
L_t^{\text{simple}} = \left\|\boldsymbol{\epsilon}_t-\color{orange}\boldsymbol{\epsilon}_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_t, t\right)\color{black}\right\|^2
\end{equation}
$$
Finally, $L_0$ uses a different derivation, but also ends up popping out as a scalar multiple of this expression (for $t=1$), so we don't have to treat this term any differently.""")
    st.info(r"""
Let's step back, and review what we're really doing here at a high level. Our **forward process** adds noise to an image according to a variance schedule which we control. Our neural network learns the **backward process** as a normal distribution with fixed variance and learned mean. We reparameterize in terms of the Gaussian error $\boldsymbol{\epsilon}_t$ between the original image $\mathbf{x}_0$ and the current image $\mathbf{x}_t$, and try to learn this quantity. In other words, we train our network to take in a noised image $\mathbf{x}_t$ and return its best estimate for the noise term which produced this image from the original image.

If we successfully learn this quantity, then we can use this estimate of noise to reconstruct an estimate of the original image, by reversing the equation we used to add noise.""")
    st.markdown(r"""
## Gradient descent

So, we have all our loss functions - how do we perform gradient descent? The answer is pretty interesting - rather than performing descent on all the functions at once, we actually sample a random interger $t \sim \text{Unif}\{1, 2, ..., T\}$, and then we perform gradient descent on the loss function $L^{\text{simple}}_{t-1}$. In other words:
$$
\begin{equation}
L_{\text{simple}}(\color{orange}\theta\color{black}) = \mathbb{E}\left[ \left\|\boldsymbol{\epsilon}_t-\color{orange}\boldsymbol{\epsilon}_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_t, t\right)\color{black}\right\|^2\; | \;\;t \sim \text{Unif}\{1, 2, ..., T\}\,\right]
\end{equation}
$$
The expectation is taken over $\mathbf{x}_0 \sim q(\mathbf{x}_0)$, $t \sim \text{Unif}\{1, \ldots, T\}$ and $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. So our actual gradient descent algorithm involves sampling all of these values and taking gradient descent on the result.""")

    st_image("alg.png", 400)
    st.markdown("""

---

That's the end of the maths in this section! In the Section 2️⃣, you'll be translating these formulas into code, and by the end you'll have created your own toy networks to de-noise an image consisting of a simple gradient between two colors.
""")

def section_2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#implementing-the-forward-process">Implementing the forward process</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#image-processing">Image Processing</a></li>
       <li><a class="contents-el" href="#normalization">Normalization</a></li>
       <li><a class="contents-el" href="#variance-schedule">Variance Schedule</a></li>
       <li><a class="contents-el" href="#forward-(q)-function---equation-2">Forward (q) function - Equation 2</a></li>
       <li><a class="contents-el" href="#forward-(q)-function---equation-4">Forward (q) function - Equation 4</a></li>
       <li><a class="contents-el" href="#training-loop">Training Loop</a></li>
   </ul></li>
   <li><a class="contents-el" href="#sampling-from-the-model">Sampling from the Model</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
```python
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List
import torch as t
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.data import DataLoader
import wandb
from torchvision import transforms
import torchinfo
from torch import nn
import plotly.express as px
from einops.layers.torch import Rearrange
from torch.utils.data import TensorDataset
from tqdm import tqdm
from torchvision import datasets
from pathlib import Path
from fancy_einsum import einsum

MAIN = __name__ == "__main__"

device = "cuda" if t.cuda.is_available() else "cpu"

import sys, os
p = r"my/path" # CHANGE THIS TO YOUR PATH, TO IMPORT FROM WEEK 0
sys.path.append(p)

from w0d2_chapter0_convolutions.solutions import Linear, conv2d, force_pair, IntOrPair
from w0d3_chapter0_resnets.solutions import Sequential
from w1d1_chapter1_transformer_reading.solutions import GELU, PositionalEncoding
from w5d1_solutions import ConvTranspose2d
import w5d3_tests
```

## Implementing the forward process

Let's start by implementing the forward process. Recall, this is the function we called $q$, which adds noise to our images in a way parameterised by our **variance schedule** $\beta_t$.

A quick note on terminology - when I refer to equation numbers, these are the numbers in the [DDPM paper](https://arxiv.org/pdf/2006.11239.pdf). If the equations are also present in the last Streamlit page and they have a different number, I will specify this.

### Image Processing

We'll first generate a toy dataset of random color gradients, and train the model to be able to recover them. This should be an easy task because the structure in the data is simple.

We've also provided you with a bunch of functions below for visualising your plots. These functions should work with both RGB and grayscale images (you will see examples of them being used below).

```python
def gradient_images(n_images: int, img_size: tuple[int, int, int]) -> t.Tensor:
    '''Generate n_images of img_size, each a color gradient
    '''
    (C, H, W) = img_size
    corners = t.randint(0, 255, (2, n_images, C))
    xs = t.linspace(0, W / (W + H), W)
    ys = t.linspace(0, H / (W + H), H)
    (x, y) = t.meshgrid(xs, ys, indexing="xy")
    grid = x + y
    grid = grid / grid[-1, -1]
    grid = repeat(grid, "h w -> b c h w", b=n_images, c=C)
    base = repeat(corners[0], "n c -> n c h w", h=H, w=W)
    ranges = repeat(corners[1] - corners[0], "n c -> n c h w", h=H, w=W)
    gradients = base + grid * ranges
    assert gradients.shape == (n_images, C, H, W)
    return gradients / 255

def plot_img(img: t.Tensor, title: Optional[str] = None) -> None:
    '''Plots a single image, with optional title.
    '''
    img = rearrange(img, "c h w -> h w c").clip(0, 1)
    img = (255 * img).to(t.uint8)
    fig = px.imshow(img, title=title)
    fig.update_layout(margin=dict(t=70 if title else 40, l=40, r=40, b=40))
    fig.show()

def plot_img_grid(imgs: t.Tensor, title: Optional[str] = None, cols: Optional[int] = None) -> None:
    '''Plots a grid of images, with optional title.
    '''
    b = imgs.shape[0]
    imgs = (255 * imgs).to(t.uint8).squeeze()
    if imgs.ndim == 3:
        imgs = repeat(imgs, "b h w -> b 3 h w")
    imgs = rearrange(imgs, "b c h w -> b h w c")
    if cols is None: cols = int(b**0.5) + 1
    fig = px.imshow(imgs, facet_col=0, facet_col_wrap=cols, title=title)
    for annotation in fig.layout.annotations: annotation["text"] = ""
    fig.show()

def plot_img_slideshow(imgs: t.Tensor, title: Optional[str] = None) -> None:
    '''Plots slideshow of images.
    '''
    imgs = (255 * imgs).to(t.uint8).squeeze()
    if imgs.ndim == 3:
        imgs = repeat(imgs, "b h w -> b 3 h w")
    imgs = rearrange(imgs, "b c h w -> b h w c")
    fig = px.imshow(imgs, animation_frame=0, title=title)
    fig.show()

if MAIN:
    print("A few samples from the input distribution: ")
    image_shape = (3, 16, 16)
    n_images = 5
    imgs = gradient_images(n_images, image_shape)
    for i in range(n_images):
        plot_img(imgs[i])
```

### Normalization

Each input value is from [0, 1] right now, but it's easier for the neural network to learn if we center them so they have mean 0. Here are some helper functions to do that, and to recover our original image.

All our computations will operate on normalized images, and we'll denormalize whenever we want to plot the result.


```python
def normalize_img(img: t.Tensor) -> t.Tensor:
    return img * 2 - 1

def denormalize_img(img: t.Tensor) -> t.Tensor:
    return ((img + 1) / 2).clamp(0, 1)

if MAIN:
    plot_img(imgs[0], "Original")
    plot_img(normalize_img(imgs[0]), "Normalized")
    plot_img(denormalize_img(normalize_img(imgs[0])), "Denormalized")
```

### Variance Schedule

The amount of noise to add at each step is called $\beta$.

Compute the vector of $\beta$ according to Section 4 of the paper (page 5). They use 1000 steps, but when reproducing a paper it's a good idea to start by making everything smaller in order to have faster feedback cycles, so we're using only 200 steps here.

```python
def linear_schedule(max_steps: int, min_noise: float = 0.0001, max_noise: float = 0.02) -> t.Tensor:
    '''
    Return the forward process variances as in the paper.

    max_steps: total number of steps of noise addition
    out: shape (step=max_steps, ) the amount of noise at each step
    '''
    pass

if MAIN:
    betas = linear_schedule(max_steps=200)
```

### Forward (q) function - Equation 2

Implement Equation $(2)$ in code (i.e. the forward function for adding noise, which is equation $(1)$ on the Streamlit page). Literally use a for loop to implement the $q$ function iteratively and visualize your results. After 50 steps, you should barely be able to make out the colors of the gradient. After 200 steps it should look just like random Gaussian noise.

```python
def q_forward_slow(x: t.Tensor, num_steps: int, betas: t.Tensor) -> t.Tensor:
    '''Return the input image with num_steps iterations of noise added according to schedule.
    x: shape (channels, height, width)
    betas: shape (T, ) with T >= num_steps

    out: shape (channels, height, width)
    '''
    pass

if MAIN:
    x = normalize_img(gradient_images(1, (3, 16, 16))[0])
    for n in [1, 10, 50, 200]:
        xt = q_forward_slow(x, n, betas)
        plot_img(denormalize_img(xt), f"Equation 2 after {n} step(s)")
    plot_img(denormalize_img(t.randn_like(xt)), "Random Gaussian noise")
```""")

    with st.expander("Help - I can still see the gradient pretty well after 200 steps."):
        st.markdown("""
The beta value indicates the variance of the normal distribution - did you forget a square root to convert it to standard deviation?""")

    st.markdown(r"""
### Forward (q) function - Equation 4

The equation we used above is very slow, and would be even slower if we went to 1000 steps. Conveniently, the authors chose to use Gaussian noise and a nice closed form expression exists to go directly to step t without needing a for loop. Implement Equation $(4)$ from the paper (also equation $(4)$ in the Streamlit page) and verify it looks visually similar to the previous equation.

```python
def q_forward_fast(x: t.Tensor, num_steps: int, betas: t.Tensor) -> t.Tensor:
    '''Equivalent to Equation 2 but without a for loop.'''
    pass

if MAIN:
    for n in [1, 10, 50, 200]:
        xt = q_forward_fast(x, n, betas)
        plot_img(denormalize_img(xt), f"Equation 4 after {n} steps")
```""")
    with st.expander("Help - I'm not sure where to start."):
        st.markdown(r"Start by constructing a vector of $\alpha_s$ terms from our $\beta_s$, then find $\bar{\alpha}_s$ by taking a product.")
    st.markdown(r"""
Our image reconstruction process will depend on the noise schedule we use during training. So that we can save our noise schedule with our model later, we'll define a `NoiseSchedule` class that subclasses `nn.Module`.

Note that we've indicated the type of `betas`, `alphas` and `alpha_bars` at the start of the class - this means you should define objects `self.betas`, etc (the practical purpose is that it tells the type checker what type these objects are).

Also, **make sure you save these objects as buffers**, since being able to save them is the reason we're subclassing `nn.Module` in the first place. Recall that using `self.register_buffer(name, tensor)` is equivalent to calling `self.name = tensor` ***and*** saving `tensor` in the model's buffers (and state dict).

Finally, note that we've given `device` as an argument for `NoiseSchedule`. This way, you can call `self.to(device)` in your intialisation, and you don't have to worry about moving the object to a different device once you start training.

```python
class NoiseSchedule(nn.Module):
    betas: t.Tensor
    alphas: t.Tensor
    alpha_bars: t.Tensor

    def __init__(self, max_steps: int, device: Union[t.device, str]) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.device = device
        pass

    @t.inference_mode()
    def beta(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        '''
        Returns the beta(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        '''
        pass

    @t.inference_mode()
    def alpha(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        '''
        Returns the alphas(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        '''
        pass

    @t.inference_mode()
    def alpha_bar(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        '''
        Returns the alpha_bar(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        '''
        pass

    def __len__(self) -> int:
        return self.max_steps

    def extra_repr(self) -> str:
        return f"max_steps={self.max_steps}"
```

Now we'll use this noise schedule to apply noise to our generated images. This will be the batched version of `q_forward_fast`.

```python
def noise_img(
    img: t.Tensor, noise_schedule: NoiseSchedule, max_steps: Optional[int] = None
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    '''
    Adds a uniform random number of steps of noise to each image in img.

    img: An image tensor of shape (B, C, H, W)
    noise_schedule: The NoiseSchedule to follow
    max_steps: if provided, only perform the first max_steps of the schedule

    Returns a tuple composed of:
    num_steps: an int tensor of shape (B,) of the number of steps of noise added to each image
    noise: the unscaled, standard Gaussian noise added to each image, a tensor of shape (B, C, H, W)
    noised: the final noised image, a tensor of shape (B, C, H, W)
    '''
    pass

if MAIN:
    noise_schedule = NoiseSchedule(max_steps=200, device="cpu")
    img = gradient_images(1, (3, 16, 16))
    (num_steps, noise, noised) = noise_img(normalize_img(img), noise_schedule, max_steps=10)
    plot_img(img[0], "Gradient")
    plot_img(noise[0], "Applied Unscaled Noise")
    plot_img(denormalize_img(noised[0]), "Gradient with Noise Applied")
```

You should get something like:""")
    cols = st.columns(3)
    for i, col in enumerate(cols, 1):
        with col:
            st_image(f"plot{i}.png", 500)

    st.markdown("""
Later, we'd like to reconstruct images for logging purposes. If we pass the true noise to this function, it will compute the inverse of `noise_img()` above.

During training, we'll pass the predicted noise and we'll be able to visually see how close the prediction is.


```python
def reconstruct(noisy_img: t.Tensor, noise: t.Tensor, num_steps: t.Tensor, noise_schedule: NoiseSchedule) -> t.Tensor:
    '''
    Subtract the scaled noise from noisy_img to recover the original image. We'll later use this with the model's output to log reconstructions during training. We'll use a different method to sample images once the model is trained.

    Returns img, a tensor with shape (B, C, H, W)
    '''
    pass

if MAIN:
    reconstructed = reconstruct(noised, noise, num_steps, noise_schedule)
    denorm = denormalize_img(reconstructed)
    plot_img(img[0], "Original Gradient")
    plot_img(denorm[0], "Reconstruction")
    t.testing.assert_close(denorm, img)
```""")

    with st.expander("Help - I'm not sure how to calculate this."):
        st.markdown(r"""Recall that we've defined noise terms by:
$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_t
$$
You need to rearrange this formula, so that $\mathbf{x}_0$ is written in terms of the other variables. You can then use this to calculate the reconstruction.
""")


    st.markdown(r"""
Now, we'll create a tiny model to use as our diffusion model. We'll use a simple two-layer MLP.

Note that we setup our `DiffusionModel` class to subclass `nn.Module` and the abstract base class (ABC). All ABC does for us is raise an error if subclasses forget to implement the abstract method `forward`. Later, we can write our training loop to work with any `DiffusionModel` subclass.

You should add code in the `TinyDiffuser` class, in the lines that say `pass`. 

Note - the timestep $t$ is also an input to our model. Here, we will handle this by scaling `num_steps` down to [0, 1] and concatenating it to the flattened image.""")

    with st.expander("Question - how many in_features should your first linear layer have?"):
        st.markdown("""
The image gives us `3 * height * width * in features`, and then we add 1 for the `num_steps` array.

So we have:

```
in_features = 3 * height * width + 1
```
""")

    st.markdown(r"""
```python
@dataclass
class DiffusionArgs():
    lr: float = 0.001
    image_shape: tuple = (3, 4, 5)
    epochs: int = 10
    max_steps: int = 100
    batch_size: int = 128
    seconds_between_image_logs: int = 10
    n_images_per_log: int = 3
    n_images: int = 50000
    n_eval_images: int = 1000
    cuda: bool = True
    track: bool = True

class DiffusionModel(nn.Module, ABC):
    image_shape: tuple[int, ...]
    noise_schedule: Optional[NoiseSchedule]

    @abstractmethod
    def forward(self, images: t.Tensor, num_steps: t.Tensor) -> t.Tensor:
        ...

@dataclass(frozen=True)
class TinyDiffuserConfig:
    max_steps: int
    image_shape: Tuple[int, ...] = (3, 4, 5)
    hidden_size: int = 128

class TinyDiffuser(DiffusionModel):
    def __init__(self, config: TinyDiffuserConfig):
        '''
        A toy diffusion model composed of an MLP (Linear, ReLU, Linear)
        '''
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.image_shape = config.image_shape
        self.noise_schedule = None
        self.max_steps = config.max_steps
        pass

    def forward(self, images: t.Tensor, num_steps: t.Tensor) -> t.Tensor:
        '''
        Given a batch of images and noise steps applied, attempt to predict the noise that was applied.
        images: tensor of shape (B, C, H, W)
        num_steps: tensor of shape (B,)

        Returns
        noise_pred: tensor of shape (B, C, H, W)
        '''
        pass

if MAIN:
    image_shape = (3, 4, 5)
    n_images = 5
    imgs = gradient_images(n_images, image_shape)
    n_steps = t.zeros(imgs.size(0))
    model_config = TinyDiffuserConfig(image_shape, 16, 100)
    model = TinyDiffuser(model_config)
    out = model(imgs, n_steps)
    plot_img(out[0].detach(), "Noise prediction of untrained model")
```

### Training Loop

After a pile of math, the authors arrive at Equation $(14)$ for the loss function (equation $(11)$ in the Streamlit page) and $\text{Algorithm 1}$ for the training procedure. We're going to skip over the derivation for now and implement the training loop at the top of Page 4.""")

    st_image("alg.png", 400)
    st.markdown("")
    st.markdown(r"""

Exercise: go through each line of Algorithm 1, explain it in plain English, and describe the shapes of each thing (in terms of `batch`, `height`, `width` and `channel`).""")

    with st.expander("Line 2"):
        st.markdown("""
The $x_0$ is just the original training data distribution, so we're just going to draw a minibatch from the training data of shape `(batch, channels, height, width)`.""")

    with st.expander("Line 3"):
        st.markdown("""
We need to draw the number of steps of noise to add for each element of the batch, so the $t$ here will have shape `(batch,)` and be an integer tensor. Both 1 and T are inclusive here. Each element gets a different number of steps of noise added.""")

    with st.expander("Line 4"):
        st.markdown(r"""
$\epsilon$ is the sampled noise, not scaled by anything. It's going to add to the image, so its shape also has to be `(batch, channel, height, width)`.""")

    with st.expander("Line 5"):
        st.markdown(r"""
$\epsilon_\theta$ is our neural network. It takes two arguments: the image with noise applied in one step of `(batch, channel, height, width)`, and the number of steps `(batch,)`, normalized to the range [0, 1].""")

    st.markdown(r"""

In Line 6, it's unspecified how we know if the network is converged. We're just going to go until the loss seems to stop decreasing.

Now implement the training loop on minibatches of examples, using Adam as the optimizer (with default parameters). Log your results to Weights and Biases. We've given you a function to return a list of images for logging, to help with this.

I recommend starting with a previous training loop you've written, e.g. for your variational autoencoder.

```python
def log_images(
    img: t.Tensor, noised: t.Tensor, noise: t.Tensor, noise_pred: t.Tensor, reconstructed: t.Tensor, num_images: int = 3
) -> list[wandb.Image]:
    '''
    Convert tensors to a format suitable for logging to Weights and Biases. Returns an image with the ground truth in the upper row, and model reconstruction on the bottom row. Left is the noised image, middle is noise, and reconstructed image is in the rightmost column.
    '''
    actual = t.cat((noised, noise, img), dim=-1)
    pred = t.cat((noised, noise_pred, reconstructed), dim=-1)
    log_img = t.cat((actual, pred), dim=-2)
    images = [wandb.Image(i) for i in log_img[:num_images]]
    return images

def train(
    model: DiffusionModel, 
    args: DiffusionArgs, 
    trainset: TensorDataset,
    testset: Optional[TensorDataset] = None
) -> DiffusionModel:
    pass

if MAIN:
    args = DiffusionArgs(epochs=2) # This shouldn't take long to train
    model_config = TinyDiffuserConfig(args.max_steps)
    model = TinyDiffuser(model_config).to(device).train()
    trainset = TensorDataset(normalize_img(gradient_images(args.n_images, args.image_shape)))
    testset = TensorDataset(normalize_img(gradient_images(args.n_eval_images, args.image_shape)))
    model = train(model, args, trainset, testset)
```

## Sampling from the Model

Our training loss went down, so maybe our model learned something. Implement sampling from the model according to Algorithm 2 so we can see what the images look like. This can be found on the top of page 4:""")

    st_image("alg2.png", 400)
    st.markdown("")

    st.markdown(r"""
Recall that we use $\sigma_t = \sqrt{\beta_t}$. If your output is coming out a bit noisy, then you can try using $\sigma_t = 0$ for the particular task of gradient denoising. I don't know why this is the case for this particular task (and neither did the folks at MLAB!). One possible theory is that, by omitting the noise term $\sigma_t \mathbb{z}$, we're actually performing maximum likelihood estimation at each step (i.e. setting $\mathbb{x}_{t-1}$ to be its posterior mean rather than sampling it from the distribution), and this works much better when all of your original images are perfectly regular. Adding noise in the later stages is strictly counterproductive, because once we've de-noised our image to the point where it's smooth, adding noise will just make it look worse.""")

    st.markdown(r"""
```python
def sample(model: DiffusionModel, n_samples: int, return_all_steps: bool = False) -> Union[t.Tensor, list[t.Tensor]]:
    '''
    Sample, following Algorithm 2 in the DDPM paper

    model: The trained noise-predictor
    n_samples: The number of samples to generate
    return_all_steps: if true, return a list of the reconstructed tensors generated at each step, rather than just the final reconstructed image tensor.

    out: shape (B, C, H, W), the denoised images
            or (T, B, C, H, W), if return_all_steps=True (where ith element is batched result of (i+1) steps of sampling)
    '''
    schedule = model.noise_schedule
    assert schedule is not None
    pass


if MAIN:
    print("Generating multiple images")
    assert isinstance(model, DiffusionModel)
    with t.inference_mode():
        samples = sample(model, 6)
        samples_denormalized = denormalize_img(samples).cpu()
    plot_img_grid(samples_denormalized, title="Sample denoised images", cols=3)
if MAIN:
    print("Printing sequential denoising")
    assert isinstance(model, DiffusionModel)
    with t.inference_mode():
        samples = sample(model, 1, return_all_steps=True)[::10, 0, :]
        samples_denormalized = denormalize_img(samples).cpu()
    plot_img_slideshow(samples_denormalized, title="Sample denoised image slideshow")
```

Now that we've got the training working, on to the next part!

""")

def section_3():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#the-u-net">The U-Net</a></li>
   <li><a class="contents-el" href="#the-ddpm-model">The DDPM Model</a></li>
   <li><a class="contents-el" href="#the-downblock">The DownBlock</a></li>
   <li><a class="contents-el" href="#the-midblock">The MidBlock</a></li>
   <li><a class="contents-el" href="#the-upblock">The UpBlock</a></li>
   <li><a class="contents-el" href="#residual-block">Residual Block</a></li>
   <li><a class="contents-el" href="#group-normalization">Group Normalization</a></li>
   <li><a class="contents-el" href="#sinusoidal-positional-encodings">Sinusoidal Positional Encodings</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""
The DDPM paper uses a custom architecture which is based on the PixelCNN++ paper with some modifications, which is in turn based on a couple other papers (U-Net and Wide ResNet) which are in turn modified versions of other papers.

You have been given some reasonably strict tests which will measure parameter count, shape of your output and in some cases the value of your output. However, in general don't worry if you don't match the architecture exactly. This is going to be the most complicated architecture you've done so far, but the good news is that if you don't do it exactly right, it'll probably still work fine.
""")

    st.markdown(r"""

## The U-Net

At the high level, the shape of this network resembles the U-Net architecture, pictured below. Like ResNet, U-Net is mostly composed of convolutional layers, but whereas ResNets were developed for classifying an image as a whole, U-Net was developed for medical segmentation tasks where the goal is to predict an output class such as "this is part of a tumour" for each input pixel in the image. This means that the network has to both get an understanding of the global structure of the image like ResNet does, but also have the ability to make fine-grained predictions at the pixel level.""")

    st_image("unet.png", 650)
    st.markdown("""
In the diagram, the grey rectangles represent tensors with the height of the rectangle being the height (and width) of the image tensor and the width of the rectangle being the number of channels.

The network is conceptually divided into three parts: the downsampling part, the middle part, and the upsampling part. In the downsampling part starting on the left of the diagram, we do some convolutions and then the yellow downsampling operation halves the width and height. The number of channels increases throughout the downsampling part, but since the spatial dimensions are shrinking, the compute per layer stays similar.

The middle section is just a couple more convolutions, and then we go into the upsampling part. In these layers, we double the spatial dimensions using transposed convolutions (which should be familiar to you now, from your GAN and VAE models).

At the end, a final convolution takes us down to the desired number of output channels. In the medical segmentation case, this might be one channel for each class of tumour that you want to detect. In our case, we're going to have three output channels to predict a RGB image.

## The DDPM Model

The model used in the DDPM is shown below and has the same three part structure as the U-Net: at first the spatial dimensions half and the channels double, and then the spatial dimensions double and channels are concatenated. It's common to still call this a U-Net and name the class `Unet` because it has this basic shape, even though the majority of components have been modified from the original U-Net.

We've got some 2D self-attention in there, new nonlinearities, group normalization, and sinusoidal position embeddings. We'll implement these from scratch so you understand what they are. Once you've done that, assembling the network will be routine work for you at this point.

One complication is that in addition to taking a batch of images, for each image we also have a single integer representing the number of steps of noise added. In the paper, this ranges from 0 to 1000, so the range is too wide to directly pass this as an integer. Instead, these get embedded into a tensor of shape `(batch, emb)` where `emb` is some embedding dimension and passed into the blocks.""")

    # ```mermaid
    # graph TD
    #     subgraph DDPM Architecture
    #         subgraph Overview
    #             MTime[Num Noise Steps] --> MTimeLayer[SinusoidalEmbedding<br/>Linear: Steps -> 4C</br>GELU<br/>Linear: 4C -> 4C]
    #             MTimeLayer -->|emb|DownBlock0 & DownBlock1 & DownBlock2 & MidBlock & UpBlock0 & UpBlock1 & OutBlock
    #             Image -->|3, H| InConv[7x7 Conv<br/>Padding 3] -->|C, H| DownBlock0 -->|C, H/2| DownBlock1 -->|2C,H/4| DownBlock2 -->|4C,H/4| MidBlock -->|4C,H/4| UpBlock0 -->|2C,H/2| UpBlock1 -->|C,H| OutBlock[Residual Block] -->|C,H| FinalConv[1x1 Conv] -->|3,H| Output
    #             DownBlock2 -->|4C,H/4| UpBlock0
    #             DownBlock1 -->|2C,H/2| UpBlock1
    #         end
    # end
    # ```

    st.write("""<figure style="max-width:550px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNp1UstuwjAQ_BXLh55AlIeEFCGkNqEFiZcKnAgHJ17AauxEjs1DpP9ex4EQJPAh2p2dzcxYvuAwpoAdvJMk2aOl5wtkTqqDAvC8-QR9yHDPFIRKSyjmD5zZAeSBwfE-ys9kyTisp5qjacxSQAsFSbpB9Xq_GI3JGeR6wYROY0ZJNOABUMrErhfIRn_MBBDpFFvI7HTcXiOQ_e_BePVA6LjFdPNE3UrkihnwIPPio_iM4vD3Hb2hsmlWm5ZpJoza2pSrpORfy5w908rWj4IjTnZgtdo1NMzQSLixOKy7py7KC-t5TmxA1LbXkLmWWLF1AxutrGowh1tubdjoZFWrOdy5wqXpKljaL3_Qyu5BCjFj4JZn_QMpo5pEyLabO-OLCRLZOM1T08bZXJMW64lWj5fxwuXN0Aty86nROxkE9YX54BrmIDlh1LzbSz73sdoDBx87pqSwJTpSPvbFn6HqhBIFA8pULLGzJVEKNUy0ihdnEWJHSQ03kseIedH8yvr7B6LL9GQ" /></figure>""", unsafe_allow_html=True)

    st.markdown("""
A few notes here:

* When the model has more than one arrow going into the same block, it means that this block takes more than one input. Similarly, multiple arrows going out of the same block means that the block has multiple outputs. Exactly which arrows correspond to which inputs/outputs can be inferred from the diagram.
* In the diagram, `H` is a standin for `(H, W)`, which are assumed to be the same.

## The DownBlock

This block takes some input height `h` and returns two things: a skip output of height `h` that connects to a later UpBlock, and a downsampled output of height `h//2`. We are going to assume (and it's good practice to assert inside the code) that `h` is always going to be divisible by 2.

If `downsample=True` then we perform the operations shown in the diagram below. If `downsample=False` then the convolution at the end isn't applied, i.e. we just return two copies of the output of the Attention Block. So when `downsample=False` we don't need to assert that `h` is even.
""")

    st.write("""<figure style="max-width:300px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNplkctugzAQRX_F8qIroigoK1RFaksX2bRVoSuIKgdPwAq2kRn3oZB_rzFJk8Cs5l6deWkOtNAcaERLw5qKpHGuiIvWbgcj1t_qsdbFfvD7eLEyQWhaMputOpDbjsTv0CpAzy3I3Y0OL4XxWrISsrVqLG58dfEpVECqUYfMCcEtq4nXZHGGtcUJHY7pcEI_IIJCoVX2nw3shHzS6ivk2fJnSfr0fmvmqwSN4EBCL94Y50KV45XmYUdeLbq7rq69DLudkuxFk-qPxq-Q9Yqk2nc_eZuhCShOAyrBSCa4-9Cht3OKFUjIaeRSDjtma8xpro4OtQ1nCM9coDY02rG6hYAyizr5VQWN0Fg4Q7Fg7r_yRB3_AJ4MrGg" /></figure>""", unsafe_allow_html=True)
    
    # ```mermaid
    # graph TD
    #     subgraph DownBlock
    #         NumSteps -->|emb| DResnetBlock1 & DResnetBlock2
    #         DImage[Input] -->|c_in, h| DResnetBlock1[Residual Block 1] -->|c_out, h| DResnetBlock2[Residual Block 2] -->|c_out, h| DAttention[Attention Block] -->|c_out, h| DConv2d[4x4 Conv<br/>Stride 2<br/>Padding 1] -->|c_out, h/2| Output
    #         DAttention -->|c_out, h| SkipToUpBlock[Skip To<br/>UpBlock]
    #     end
    # ```
    st.markdown("""
## The MidBlock

After the DownBlocks, the image is passed through a MidBlock which doesn't change the number of channels.
""")
    st.write("""<figure style="max-width:300px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNp9kL1uwzAMhF9F4JDJGeLRQ4EW6dAhLdA0kxQUjMXEQi3JsMmhSPLuVaymyA9QDtLx8PFAcA91tAQV7HrsGvUxN0GlGmSTjYWzT22sv7J9qtWr-CVTN-izWKvp9OFAfnNQq3caAvE4MlOTq768yHjxuCM9vnm6_vTOFqq5idCpcVawVWOvZnf0IzMFdjHoP5XZ_3LL29zyjn4T7oR1_tZ5dQoWCvDUe3Q23Wx_sg1wQ54MVEla2qK0bMCEY0Kls8j0bB3HHqottgMVgMJx-R1qqLgXOkNzh-ni_pc6_gBFGYnP" /></figure>""", unsafe_allow_html=True)
    # ```mermaid
    # graph TD
    #     subgraph MidBlock
    #         UNumSteps[NumSteps] -->|emb| UResnetBlock1 & UResnetBlock2
    #         UImage[Image] -->|c_mid, h| UResnetBlock1[Residual Block 1] -->|c_mid, h| UAttention[Attention Block] -->|c_mid, h| UResnetBlock2[Residual Block 2] -->|c_mid, h| UOutput[Output]
    #     end
    # ```
    st.markdown("""
## The UpBlock

Note here that the first upsampling block takes a skip connection from the last downsampling block, and the second upsampling block takes a skip connection from the second last downsampling block. In your implementation, pushing and popping a stack is a clean way to handle this.""")

    st.write("""<figure style="max-width:380px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNp1kk1PAyEQhv8K4eDBtDElPW1ME3U18aLGbU9sY-gybUmXj7BD1bT977Ksrf2IJMC8w8MwDGxoZSXQjC68cEsyzktDYmvCrHNM3H1tq1XnbdvkJegCwTV8b0xJvz_agp5tyeQdGgOYtgzI1YlmfzGKlXK8HciTt_p25m9Guf00CUuqC1l9KNMjyy15sKYSCCb2o0SetVgAT-P_fFpg14elkwx5FEoGUZOkyWAfxwa8oNk5zS7oO4yHorKGH6yOPSfzmOF67IVpnG2AST78GpKDlu0F1qkQBXolgbAk3oSUyizO0mTtya8BXUDeTdOuSGAk7VENXgsl4wtvWndJcQkaSppFU8JchBpLWppdRIOTsWKPUqH1NJuLuoEeFQFt8W0qmqEPsIdyJeL_0L_U7gexzsQB" /></figure>""", unsafe_allow_html=True)
    # ```mermaid
    # graph TD
    #     subgraph UpBlock
    #         UNumSteps[NumSteps] -->|emb| UResnetBlock1 & UResnetBlock2
    #         Skip[Skip From<br/>DownBlock<br/>] -->|c_in, h| Concatenate
    #         UImage[Image] -->|c_in, h| Concatenate -->|2*c_in, h| UResnetBlock1[Residual Block 1] -->|c_out, h| UResnetBlock2[Residual Block 2] -->|c_out, h| UAttention[Attention Block] -->|c_out, h| DConvTranspose2d[4x4 Transposed Conv<br/>Stride 2<br/>Padding 1] -->|c_out, 2h| UOutput[Output]
    #     end
    # ```
    st.markdown("""
## Residual Block

These are called residual blocks because they're derived from but not identical to the ResNet blocks. You can see the resemblance with a main branch and a residual branch. When the input dimensions don't match the output dimensions, the residual branch uses a 1x1 convolution to keep them consistent.""")

    # ```mermaid
    # graph TD
    # subgraph ResidualBlock
    # Image -->|c_in, h| ResConv[OPTIONAL<br/>Conv 1x1] -->|c_out, h| Out
    #         Image -->|c_in, h| Conv1[Conv 3x3, pad 1<br/>GroupNorm<br/>SiLU] -->|c_out, h| AddTimeEmbed[Add] -->|c_out, h| Conv2[Conv 3x3, pad 1<br/>Group Norm</br>SiLU] -->|c_out, h| Out
    #         NumSteps[Num Steps<br/>Embedding] -->|emb| TimeLayer[SiLU<br/>Linear] -->|c_out| AddTimeEmbed
    #         end
    # ```

    st.write("""<figure style="max-width:420px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNp1kVFrgzAQgP9KyLOluL7JKHRrGQXRMd2TGSOaq4aZRGIyWmr_-0zqWFu6PN3lvvvu4I64UgxwhGtNuwblayJ7W56TN-g5s7R9alX1ReRW0BrQbLYcqk8uA9QMjnhW8rtIX_Ntmqzix1LPl-4HhfvwY2KVNR5OrSESTe-OzPWFhe9e7BcB6ihDoTe-aGW7RGnhs4zH77fuFWM5F7ARJbBiTG7rzvrwvxt5-bzUd-VXiydWZAa6vhgD5COv8ZMZl_W5GUQ5ILdRTA-gC2f1WMwlUH0x4Hr1vzEgGQ6wAC0oZ-N5jq5EsGlAAMHRGDLYUdsagok8jajtGDWwYdwojaMdbXsIMLVGZQdZ4choC7_QmtPxwGKiTj9LDqo6" /></figure>""", unsafe_allow_html=True)

    st.markdown(r"""
---

## Group Normalization

In Layer Normalization, we computed a mean and standard deviation for each training example, across all channels (channels are the same as embedding dimensions in a transformer).

Group Normalization means we divide our channels into some number of groups, and we have a mean and standard devication for each training example AND group. When the number of groups is 1, GroupNorm can be expressed as a LayerNorm. The main difference is that GroupNorm expects the channel dimension right after the batch dimension, as is conventional in PyTorch for image data. LayerNorm expects the channel (embedding) dimension to be last, as is conventional in PyTorch for NLP.
""")

    st_image("groupnorm.png", 700)
    st.markdown("""
The pixels in blue are normalized by the same mean and standard deviation. For group norm, two groups are depicted.

For more intuition behind why this could be a good alternative to other normalization schemes, see the [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf) paper.

Implement `GroupNorm` so it behaves identically to `torch.nn.GroupNorm` given a `(batch, channels, height, width)` tensor. While [`torch.nn.GroupNorm`](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html) supports more than 2 spatial dimensions, you don't need to worry about this.

Note - `num_groups` should always evenly divide `num_channels`. As good practice, you should raise an error if this isn't the case.

Like with BatchNorm, you should use `unbiased=False` in your variance calculation.
""")
    with st.expander("Help - I'm not sure how to implement GroupNorm."):
        st.markdown(r"""
Use `rearrange` to introduce a 5th group dimension and then compute the mean and variance over the appropriate dimensions. After you subtract and divide the mean and variance, rearrange again back into BCHW before `applying` the learnable parameters.
""")

    st.markdown(r"""
## Sinusoidal Positional Encodings

We use sinusoidal positional encodings, to embed the num noise steps object. 

You can reuse much of your previous sinusoidal positional encodings code, but you may have to rewrite some of it. In previous exercises, your encoding worked on `x` which had been token-encoded (i.e. the input had shape `(batch, seq_len, embedding_dim)`) but here you need to apply it directly to your `num_steps` object (which is an array of size `(batch,)`, with each element being an integer between `0` and `max_steps-1`).

```python
class PositionalEncoding(nn.Module):

    def __init__(self, max_steps: int, embedding_dim: int):
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch,) - for each batch element, the number of noise steps
        Out: shape (batch, embedding_dim)
        '''
        pass
```

## Sigmoid Linear Unit

The Sigmoid Linear Unit (SiLU) nonlinearity is just elementwise `x * sigmoid(x)`. Confusingly, this function is also called Swish in the literature - these two names refer to exactly the same thing. Implement the function and plot it on the interval [-5, 5]. Like every other new non-linearity published, its authors claim that it has superior performance on benchmarks, but we don't fully understand why.

For more on this activation function, see [Swish: A Self-Gated Activation Function](https://arxiv.org/pdf/1710.05941v1.pdf).

```python
def swish(x: t.Tensor) -> t.Tensor:
    pass

class SiLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return swish(x)

if MAIN:
    "TODO: YOUR CODE HERE, TO PLOT FUNCTION"
```

## Self-Attention with Two Spatial Dimensions

In the transformer, we had one spatial (sequence) dimension, but now we have image data with both height and width (which we're assuming to be equal). Implement the code for this - feel free to refer to your previous self-attention implementation.

Note - for consistency with the tests, you should define your weight matrices as `W_QKV` and `W_O` (and they should have biases). This isn't absolutely essential, but the tests which compare your output to the solution's output won't be fully reliable otherwise.
""")

    with st.expander("Question - what kind of masking (if any) should you apply to your attention here?"):
        st.markdown("""
No masking is necessary here. There's no concept of "looking ahead in the sequence" which we're trying to prevent from happening.

This also means the attention is bidirectional, rather than unidirectional.
""")

    with st.expander("Help - I'm not sure how to implement this."):
        st.markdown("""
The easiest solution is to flatten the width and height dims into a single dimension at the start, and use the same attention function as you did for your language transformers.

There are other options. For instance:
* Implement your matrices `W_QKV` and `W_O` as `nn.Parameter` objects which have `width` and `height` dimensions, and which you perform `einsum` operations on.
* Implement your matrices `W_QKV` and `W_O` as `Conv2d` with kernel size of 1x1.
""")

    st.markdown("""
```python
class SelfAttention(nn.Module):
    W_QKV: Linear
    W_O: Linear

    def __init__(self, channels: int, num_heads: int = 4):
        '''
        Self-Attention with two spatial dimensions.

        channels: the number of channels. Should be divisible by the number of heads.
        '''
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        out: shape (batch, channels, height, width)
        '''
        b, c, h, w = x.shape
        assert c == self.channels
        pass

if MAIN:
    w5d3_tests.test_self_attention(SelfAttention)
```

## Assembling the UNet

Implement the various blocks according to the diagram.

Note that all the downblocks perform downsampling except for the last one. The first one does perform downsampling (we get `H` going to `H/2`), and we discard the residual output (unlike all the following downblocks, where the residual output is fed into one of the upblocks).

All of the upblocks perform upsampling.

```python
class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        pass

if MAIN:
    w5d3_tests.test_attention_block(SelfAttention)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, step_dim: int, groups: int):
        '''
        input_channels: number of channels in the input to foward
        output_channels: number of channels in the returned output
        step_dim: embedding dimension size for the number of steps
        groups: number of groups in the GroupNorms

        Note that the conv in the left branch is needed if c_in != c_out.
        '''
        pass

    def forward(self, x: t.Tensor, time_emb: t.Tensor) -> t.Tensor:
        '''
        Note that the output of the (silu, linear) block should be of shape (batch, c_out). Since we would like to add this to the output of the first (conv, norm, silu) block, which will have a different shape, we need to first add extra dimensions to the output of the (silu, linear) block.
        '''
        pass

if MAIN:
    w5d3_tests.test_residual_block(ResidualBlock)


class DownBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, time_emb_dim: int, groups: int, downsample: bool):
        pass

    def forward(self, x: t.Tensor, step_emb: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        '''
        x: shape (batch, channels, height, width)
        step_emb: shape (batch, emb)
        Return: (downsampled output, full size output to skip to matching UpBlock)
        '''
        pass

if MAIN:
    w5d3_tests.test_downblock(DownBlock, downsample=True)
    w5d3_tests.test_downblock(DownBlock, downsample=False)


class UpBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, time_emb_dim: int, groups: int, upsample: bool):
        '''
        IMPORTANT: arguments are with respect to the matching DownBlock.

        '''
        pass

    def forward(self, x: t.Tensor, step_emb: t.Tensor, skip: t.Tensor) -> t.Tensor:
        pass

if MAIN:
    w5d3_tests.test_upblock(UpBlock, upsample=True)
    w5d3_tests.test_upblock(UpBlock, upsample=False)


class MidBlock(nn.Module):
    def __init__(self, mid_dim: int, time_emb_dim: int, groups: int):
        pass

    def forward(self, x: t.Tensor, step_emb: t.Tensor):
        pass

if MAIN:
    w5d3_tests.test_midblock(MidBlock)


@dataclass(frozen=True)
class UnetConfig():
    '''
    image_shape: the input and output image shape, a tuple of (C, H, W)
    channels: the number of channels after the first convolution.
    dim_mults: the number of output channels for downblock i is dim_mults[i] * channels. Note that the default arg of (1, 2, 4, 8) will contain one more DownBlock and UpBlock than the DDPM image above.
    groups: number of groups in the group normalization of each ResnetBlock (doesn't apply to attention block)
    max_steps: the max number of (de)noising steps. We also use this value as the sinusoidal positional embedding dimension (although in general these do not need to be related).
    '''
    image_shape: Tuple[int, ...] = (1, 28, 28)
    channels: int = 128
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8)
    groups: int = 4
    max_steps: int = 600

class Unet(DiffusionModel):
    def __init__(self, config: UnetConfig):
        self.noise_schedule = None
        self.image_shape = config.image_shape
        pass

    def forward(self, x: t.Tensor, num_steps: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        num_steps: shape (batch, )

        out: shape (batch, channels, height, width)
        '''
        pass

if MAIN:
    w5d3_tests.test_unet(Unet)
```

Now, you're ready to move on to the final section!
""")

def section_4():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#dataset-loading">Dataset Loading</a></li>
    <li><a class="contents-el" href="#model-creation">Model Creation</a></li>
    <li><a class="contents-el" href="#troubleshooting)">Troubleshooting</a></li>
    <li><a class="contents-el" href="#bonus">Bonus</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#improving-the-diffusion-model">Improving the Diffusion Model</a></li>
        <li><a class="contents-el" href="#breaking-the-is-and-fid-metrics">Breaking the IS and FID Metrics</a></li>
        <li><a class="contents-el" href="#implement-your-own-is-and-fid-metrics">Implement your own IS and FID Metrics</a></li>
    </ul></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
We've already got most of the pieces in place, so setting up this training should be straightforward.

We're going to start by thinking about the input distribution of images. [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) is a dataset of 60K training examples and 10K test examples that belong to one of 10 different classes like "t-shirt" or "sandal". Each image is 28x28 pixels and in 8-bit grayscale. We think of those dataset examples as being samples drawn IID from some larger input distribution "the set of all FashionMNIST images".

## Dataset Loading

```python
def get_fashion_mnist(train_transform, test_transform) -> Tuple[TensorDataset, TensorDataset]:
    '''Return MNIST data using the provided Tensor class.'''
    mnist_train = datasets.FashionMNIST("../data", train=True, download=True)
    mnist_test = datasets.FashionMNIST("../data", train=False)
    print("Preprocessing data...")
    train_tensors = TensorDataset(
        t.stack([train_transform(img) for (img, label) in tqdm(mnist_train, desc="Training data")])
    )
    test_tensors = TensorDataset(t.stack([test_transform(img) for (img, label) in tqdm(mnist_test, desc="Test data")]))
    return (train_tensors, test_tensors)


if MAIN:
    train_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.RandomHorizontalFlip(), 
        transforms.Lambda(lambda t: t * 2 - 1)
    ])
    data_folder = Path("./data/fashion_mnist")
    data_folder.mkdir(exist_ok=True, parents=True)
    DATASET_FILENAME = data_folder / "generative_models_dataset_fashion.pt"
    if DATASET_FILENAME.exists():
        (trainset, testset) = t.load(str(DATASET_FILENAME))
    else:
        (trainset, testset) = get_fashion_mnist(train_transform, train_transform)
        t.save((trainset, testset), str(DATASET_FILENAME))
```

## Model Creation

Now, set up training, using your implemenations of UNet and calling the training loop from part 1. Note that we are using a smaller number of channels and dim_mults than the default UNetConfig. This is because the default UNetConfig is designed for 256x256 images, and we are using 28x28 images. You can experiment with different values for these hyperparameters (and you might also want to try working with datasets containing larger images - see the bonus exercises below).

```python
if MAIN:
    model_config = UnetConfig(
        channels = 28,
        dim_mults = (1, 2, 4), # Using smaller channels and dim_mults than default
    )
    args = DiffusionArgs(
        image_shape = model_config.image_shape, 
        max_steps = model_config.max_steps,
    )
    model = Unet(model_config)

if MAIN:
    model = train(model, args, trainset, testset)
```

## Troubleshooting""")

    with st.expander("Help - my generated images have a bunch of random white pixels on them!"):

        st.markdown("If they look like this:")

        st_image("diffusion_highvar.png", 120)
        st.markdown("")
        st.markdown("""
This could indicate that you're adding too much noise near the end of the diffusion process. Check your equations again, in particular that you're not missing a square root anywhere - $\sigma^2$ represents variance and you might be needing a standard deviation instead.

Also, check your `max_steps` argument - for these examples, you should have this somewhere in the range of 500-1000 at least.""")

    with st.expander("Ok, what are they 'supposed' to look like?"):
        st.markdown("""
There might be some variance of quality between your samples. Here is a grid of mine:""")
        st.plotly_chart(fig_dict["grid_output"], use_container_width=True)
        st.markdown("And here is an example animation:")
        st.plotly_chart(fig_dict["animation_output"], use_container_width=True)
        st.markdown("Note - in general, your outputs on `wandb` will look much better than your reconstructions, because even when $t$ is pretty large, $x_t$ won't be fully IID normal, and it'll be easier to reconstruct the original image than it is to come up with a realistic image from pure random noise.")

    with st.expander("What values should I expect for my loss function?"):
        st.markdown("""
While training on a `gpu_1x_a100_sxm4` instance ($1 per hour on Lambda Labs), I got the following results:""")

        st_image("loss-diffusion.png", 650)
        st.markdown("")

        st.markdown("""

This was after a single epoch of training, which took about 90 seconds. At the end, my model was able to produce samples at the level of those shown above (and judging from the loss curve, I imagine it could have generated images like those even earlier).

The loss was calculated using `F.mse_loss` on the noise and predicted noise.
""")

    st.markdown("""
```python
if MAIN:
    print("Generating multiple images")
    assert isinstance(model, DiffusionModel)
    with t.inference_mode():
        samples = sample(model, 6)
        samples_denormalized = denormalize_img(samples).cpu()
    plot_img_grid(samples_denormalized, title="Sample denoised images", cols=3)
    print("Printing sequential denoising")
    assert isinstance(model, DiffusionModel)
    with t.inference_mode():
        samples = sample(model, 1, return_all_steps=True)[::30, 0, :]
        samples_denormalized = denormalize_img(samples).cpu()
    plot_img_slideshow(samples_denormalized, title="Sample denoised image slideshow")
```""")

    st.markdown(r"""
## Bonus

Congratulations on completing this chapter's content! Here are a few suggested bonus exercises for you to try.

### Improving the Diffusion Model

As you've seen, it's challenging to determine if a change to your generative model actually is an improvement to the "thing we care about". Try to improve your model nonetheless using more recent advances in diffusion models. Some ideas are:

- Do a hyperparameter search to find improved sample quality
- Try using different loss functions (l1_loss, smooth_l1_loss) and see how this changes the samples.
- Try training on datasets with larger & more complicted images, like CIFAR10 or celebA. Post your results in the Slack channel!
- Try to find a more valid automated metric to measure sample quality.
- [Some papers suggest](https://arxiv.org/abs/2102.09672) cosine annealing the learning rate, rather than a constant learning rate. Try this out and see if it improves your samples.
- Try to implement the [DDPM+](https://arxiv.org/abs/2102.09672) model, which suggests some improvements to the original DDPM.

A few of these points are discussed in the second half of [this post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/), which you might find more accessible than reading the papers.

### Implement IS and FID Metrics

The DDPM paper mentions the [Inception scoore](https://arxiv.org/pdf/1801.01973.pdf) (IS). This is a popular metric claimed to measure visual quantity, in two different ways:

* The generated images contain clear, identifiable objects, and don't have noise or blurriness
* The generated images cover all possible clusers in the training set (e.g. in MNIST, all digits can be generated)

The metric works by generating a bunch of examples $x$, passing them through the **Inception v3** CNN, then measuring the [KL divergence](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence) between two distributions:

* $p(y|x)$, the probability distribution over the 1000 ImageNet classes given some particular input $x$
* $p(y)$, the probability distribution over the classes when we average logit outputs over all $x$

The idea is that we want our samples to cover a diverse range of objects (meaning $p(y)$ has high entropy), but each item contains very clear and well-defined objects which ImageNet assigns confident predictions to (meaning $(p(y|x)$ has low entropy). So in this case, $D_{KL}(p(y|x)||p(y))$ will be large.

FID (Frechet Inception Distance) follows the same principle, but we look at the distribution of activations just before the final layer, rather than the final layer.

Implement both of these metrics. Try running them on images generated at various points during training. Do the metrics trend in the direction you would expect if they did correlate with sample quality?

What do you think might be some problems with these metrics? Do you expect them to work well on the Fashion MNIST dataset?
""")

func_list = [section_home, section_1, section_2, section_3, section_4]

page_list = ["🏠 Home", "1️⃣ Introduction to diffusion models", "2️⃣ Training a basic diffusion model", "3️⃣ The DDPM Architecture", "4️⃣ Training U-Net on FashionMNIST"]
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