import os
if not os.path.exists("./images"):
    os.chdir("./ch5")

from st_dependencies import *
styling()

import plotly.io as pio
import json
import re
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
    return {name: read_from_html(name) for name in ["autoencoder_interpolation", "vae_interp"]}

if "fig_dict" not in st.session_state:
    fig_dict = get_fig_dict()
    st.session_state["fig_dict"] = fig_dict
else:
    fig_dict = st.session_state["fig_dict"]

def home():
    st.markdown("""
Yesterday, we looked at GANs: an adversarial setup which trains a generator and discriminator network in parallel. Today, we'll turn our attention to autoencoders and variational autoencoders (VAEs), which operate on a very different paradigm. You'll be pleased to know that they are generally much easier to train!

## 1️⃣ Autoencoders

Autoencoders are a pretty simple architecture: you learn a compressed representation of your data (mainly using linear layers and convolutions), then reconstruct it back into an image (with linear layers and transposed convolutions).

## 2️⃣ Variational Autoencoders

Although autoencoders can learn some interesting low-dimensional representations, they are less good for generating images because their latent spaces aren't generally meaningful. This leads to VAEs, which solve this problem by having their encoders map to a distribution over latent vectors, rather than a single latent vector. This incentivises the latent space to be more meaningful, and we can more easily generate images from sample vectors in this space.

## 3️⃣ Bonus exercises

If you get to the end of the material, there are a few bonus exercises to try. Some of these might be particularly important if you want to get to grips with diffusion models later on!
""")
    st.info("""
Note - in the last section we strongly encouraged you to use the modules you built yourself, because this was a good way to get a feel for how transposed convolutions work. From this section onwards, you can feel free to use PyTorch's built-in modules instead. The focus of these next sections will be on the concepts and intuitions behind these different architectures, not on low-level implementational details.""")

def section_AE():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#reading">Reading</a></li>
   <li><a class="contents-el" href="#autoencoders">Autoencoders</a></li>
    <li><a class="contents-el" href="#write-your-own-autoencoder,-for-mnist">Write your own autoencoder</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#encoder">Encoder</a></li>
       <li><a class="contents-el" href="#decoder">Decoder</a></li>
       <li><a class="contents-el" href="#training-loop">Training loop</a></li>
       <li><a class="contents-el" href="#generating-images-from-an-encoder">Generating images from an encoder</a></li>
       <li><a class="contents-el" href="#exercise---plot-your-embeddings">Exercise - plot your embeddings</a></li>
    </li></ul>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""
## Reading

* [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) \*\*
    * This is a very good introduction to the basic mechanism of autoencoders and VAEs. 
    * Read up to the section on the reparameterization trick (inclusive); the rest is optimal.
    * Don't worry if you don't follow all the maths; we'll go through some of it below.
* [Six (and a half) intuitions for KL divergence](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence) \*
    * As the architectures we look at in this chapter get more heavily mathematical, it will become important to have good intuitions for the basics of information theory. In particular, KL divergence will show up a **lot**!

## Autoencoders

We'll start by looking at **Autoencoders**, which are much conceptually simpler than VAEs. These are simply systems which learn a compressed representation of the input, and then reconstruct it. There are two parts to this:

* The **encoder** learns to compress the output into a latent space which is lower-dimensional than the original image.
* The **decoder** learns to uncompress the encoder's output back into a faithful representation of the original image.
""")
    st_image("autoencoder-architecture.png", 600)

    st.markdown(r"""
Our loss function is simply some metric of the distance between the input and the reconstructed input, e.g. the $l_2$ loss.

## Write your own autoencoder

This should be relatively straightforward. Use an encoder with just two fully connected linear layers, with one activation function in the middle. Your first linear layer should have `out_features=100`, and you should try a small number for your second linear layer (e.g. between 2 and 10). Your decoder architecture should mirror your encoder, just like how the generator and discriminator mirrored each other in the GAN you created yesterday. 

You can create your autoencoder from the code below.

```python
class Autoencoder(nn.Module):
    encoder: nn.Sequential
    decoder: nn.Sequential

    def __init__(self, *args):
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_compressed = self.encoder(x)
        x_reconstructed = self.decoder(x_compressed)
        return x_reconstructed
```

Once you've done this, you should write a training loop which works with $l_2$ loss between the original and reconstructed data. The standard Adam optimiser with default parameters should suffice. You can reuse code from the first week to fetch MNIST data. 

Much like for the generator, you might find it helpful to display output while your model is training. There are actually two ways you can do this:

1. Take an MNIST image and feed it into your autoencoder - your result should look similar to the input.
2. Take a random vector in the latent space, and run just the decoder on it in order to produce an output.

The problem with the second option (which we'll discuss more when we get to VAEs) is that your latent space might not actually be meaningful. In other words, it's unclear exactly how to sample from it to get output which will look like an MNIST image. For that reason, you're recommended to try the first of these two approaches. You should also do standard things like printing out the loss as your model trains (or logging the result to Weights and Biases).

You should be able to get a result like this (which was produced using `out_features=100`, `latent_dim_size=5`, and after ten epochs).""")
    st_image("autoencoder_1.png", 700)
    st.markdown(r"""
That's pretty good, but it's messing some things up. For instance, it seems confused about whether the 2 is actually a 3 when it performs reconstruction. Some degree of this is inevitable because we can't completely faithfully represent an image in just a five-dimensional space. But here, we can still make some significant improvements without increasing `latent_dim_size`.

We've only used linear layers so far, but it's much more natural to use convolutions instead, because these presever the spatial structure. Also, this way we can add more layers without signficiantly increasing our parameter count. Below, we've provided some guidance for building an improved architecture.

### Encoder

Your encoder should consist of two convolutional layers, a flatten, then two linear layers. After each convolution or linear layer (except for the last one) you should have an activation function. The convolutions should have kernel size 4, stride 2, padding 0 (recall from yesterday that this exactly halves the width and height). The number of output channels can be 16 and 32 respectively.

After the convolutional layers, you flatten then apply linear layers. Your flattened size will be $32 \times 7 \times 7$. Your first linear layer should have `out_features=128`; the second is up to you (again we recommend playing around with values between 2 and 10).

### Decoder

Again, your decoder should be a mirror image of your encoder. Reverse the order of all the layers, and replace the convolutions with transposed convolutions. Your transposed convolutions should have kernel size 4, stride 2, and padding 1 (this results in a doubling of the height and width of the image, so you can use the same shapes for the linear layers as for the encoder, but in reverse).

### Training loop

Implement your training loop below. The implementational details are left up to you.

```python
def train_autoencoder(*args):
    pass
```

After ten epochs, I was able to get the following output:
""")
    st_image("autoencoder_2.png", 700)
    st.markdown("""
This is a very faithful representation; much better than the first version. Note how it's mixing up features for some of the numbers - for instance, the 5 seems to have been partly reproduced as a 9. But overall, it seems pretty accurate!

## Generating images from an encoder

We'll now return to the issue we mentioned briefly earlier - how to generate output? This was easy for our GAN; the only way we ever produced output was by putting random noise into the generator. But how should we interpret the latent space between our encoder and decoder?

We can try and plot the outputs produced by the decoder over a range, e.g. using code like this (the details might vary slightly for your model depending on how you defined your layers):

```python
# Choose number of interpolation points
n_points = 11

# Constructing latent dim data by making two of the dimensions vary independently between 0 and 1
latent_dim_data = t.zeros((n_points, n_points, latent_dim_size), device=device)
x = t.linspace(-1, 1, n_points)
latent_dim_data[:, :, 0] = x.unsqueeze(0)
latent_dim_data[:, :, 1] = x.unsqueeze(1)
# Rearranging so we have a single batch dimension
latent_dim_data = rearrange(latent_dim_data, "b1 b2 latent_dim -> (b1 b2) latent_dim")

# Getting model output, and normalising & truncating it in the range [0, 1]
output = model.decoder(latent_dim_data).detach().cpu().numpy()
output_truncated = np.clip((output * 0.3081) + 0.1307, 0, 1)
output_single_image = rearrange(output_truncated, "(b1 b2) 1 height width -> (b1 height) (b2 width)", b1=n_points)

# Plotting results
fig = px.imshow(output_single_image, color_continuous_scale="greys_r")
fig.update_layout(
    title_text="Decoder output from varying first two latent space dims", title_x=0.5,
    coloraxis_showscale=False, 
    xaxis=dict(tickmode="array", tickvals=list(range(14, 14+28*n_points, 28)), ticktext=[f"{i:.2f}" for i in x]),
    yaxis=dict(tickmode="array", tickvals=list(range(14, 14+28*n_points, 28)), ticktext=[f"{i:.2f}" for i in x])
)
fig.show()
```

This generates images from a vector in the latent space which has all elements set to zero except the first two. The output should look like:
""")
    st.plotly_chart(fig_dict["autoencoder_interpolation"])

    st.markdown("""
This is ... pretty underwhelming actually. Although some of these shapes seem legible (e.g. the bottom-right look like 9s, and some of the right images look like 7s), much of the space doesn't look like any recognisable number.

Why is this? Well unfortunately, the model has no reason to treat the latent space in any meaningful way. It might be the case that almost all the images are embedded into a particular subspace of the latent space, and so the encoder only gets trained on inputs in this subspace. You can explore this idea further in the exercise below.

## Exercise - plot your embeddings

The output above generated images from embeddings in the latent space, using `model.decoder`. Now, you should try and do the opposite - feed MNIST data into your encoder, and plot its embedding projected along the first two dimensions of the latent space.

Before you do this, think about what you expect to see in this plot, based on the comments above. A few questions you might want to ask yourself:

* Do you expect the entire space to be utilised, or will the density of points be uneven?
* You can color-code the points of your scatter graph according to their true label. Do you expect same-colored points to be clustered together?

Note - you might see very different results depending on how many dimensions there are in your latent space.
""")

def section_VAE():
    st.sidebar.markdown("""
# Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#introduction">Introduction</a></li>
   <li><a class="contents-el" href="#building-a-vae">Building a VAE</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#probabilistic-encoder">Probabilistic encoder</a></li>
       <li><a class="contents-el" href="#new-loss-function">New loss function</a></li>
       <li><a class="contents-el" href="#training-loop">Training loop</a></li>
    </li></ul>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
## Introduction

Variational autoencoders try and solve the problem posed by autoencoders: how to actually make the latent space meaningful, such that you can generate output by feeding a $N(0, 1)$ random vector into your model's decoder?

The key perspective shift is this: **rather than mapping the input into a fixed vector, we map it into a distribution**. The way we learn a distribution is very similar to the way we learn our fixed inputs for the autoencoder, i.e. we have a bunch of linear or convolutional layers, our input is the original image, and our output is the tuple of parameters $(\mu(x), \Sigma(x))$ (as a trivial example, our VAE learning a distribution $\mu(x)=z(x)$, $\Sigma(x)=0$ is equivalent to our autoencoder learning the function $z(x)$ as its encoder).

From this [TowardsDataScience](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) article:
""")
    cols = st.columns([1, 6, 1])
    with cols[1]:
        st.markdown("""
**Due to overfitting, the latent space of an autoencoder can be extremely irregular (close points in latent space can give very *different* decoded data, some point of the latent space can give *meaningless* content once decoded) and, so, we can’t really define a *generative* process that simply consists to sample a point from the *latent space* and make it go through the decoder to get new data. *Variational autoencoders* (VAEs) are autoencoders that tackle the problem of the latent space irregularity by making the encoder return a *distribution over the latent space* instead of a single point and by adding in the loss function a *regularisation* term over that returned distribution in order to ensure a better *organisation* of the latent space.**""")

    st.markdown("""Or, in fewer words:""")

    cols = st.columns([1, 6, 1])
    with cols[1]:
        st.markdown("""
**A variational autoencoder can be defined as being an autoencoder whose training is regularised to avoid overfitting and ensure that the latent space has good properties that enable generative process.**
""")

    st.markdown("""

The diagram below might help clear up exactly what our model is learning:""")

    st_image("vae_diagram.png", 800)

    st.markdown("""

From this point on, we'll really be getting into the mathematical weeds! Now might be a good time to go back and review both of the reading materials - if you're still confused by anything, you can messge on `#technical-questions`. Below are a few questions designed to test your understanding (based on material from the [VAE section](https://lilianweng.github.io/posts/2018-08-12-vae/#vae-variational-autoencoder) onwards, as well as the [KL divergence LessWrong post](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence)).

""")
   
    with st.expander("Summarize in one sentence why we need the reparameterization trick in order to train our network. **"):
        st.markdown(r"""
One sentence summary:

We can't backpropagate through random (stochastic) processes like the ones we used to sample our latent vectors $z$ from our params $\phi$ and input $x$, but if we instead write $z$ as a deterministic function of our params and input (plus some auxiliary random variable $\epsilon$) then we can backpropagate.

Longer summary:

Our encoder works by generating parameters and then using those parameters to sample latent vectors $z$ (i.e. a **stochastic process**). Our decoder is deterministic; it just maps our latent vectors $z$ to fixed outputs $x'$. The stochastic part is the problem; we can't backpropagate gradients through random functions. However, instead of writing $z \sim q_{\phi}$, we can write $z$ as a deterministic function of its inputs: $z = g(\phi, x, \epsilon)$, where $\phi$ are the parameters of the distribution, $x$ is the input, and $\epsilon$ is a randomly sampled value. We can then backpropagate through the network.
""")

    with st.expander("Summarize in one sentence what concept we're capturing when we measure the KL divergence D(P||Q). **"):
        st.markdown(r"""
Any of the following would work - $D(P||Q)$ is...

* How much information is lost if the distribution $Q$ is used to represent $P$.
* The quality of $Q$ as a probabilistic model for $P$ (where lower means $Q$ is a better model).
* How close $P$ and $Q$ are, with $P$ as the actual ground truth.

---

A quick note here - it might seem confusing why we measure $D(q_{\phi}(z|x)||p_{\theta}(z|x))$ rather than $D(p_{\theta}(z|x)||q_{\phi}(z|x))$. After all, we are trying to learn a distribution $q_{\phi}(z|x)$ to approximate the distribution $p_{\theta}(z|x)$, and above I described $Q$ as being the model for $P$. What gives?

The key point here is that **we never actually see $p_{\theta}(z|x)$, but we do see $q_{\phi}(z|x)$**. So $q_{\phi}(z|x)$ is the "actual ground truth" in the sense of it being the thing that generates the latent vectors we actually see.

I think [Section 2 of this post](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence#2__Hypothesis_Testing) gives the best intuitions for what's going on here. Translating this section to our current use-case, we see that $D(q_{\phi}(z|x)||p_{\theta}(z|x))$ being small is equivalent to the following: **when we're generating samples $z$ via our encoder (with distribution $z \sim q_{\phi}(z|x)$), we find this process hard to distinguish from reality**. 
""")

    st.markdown(r"""
## Building a VAE

For your final exercise of today, you'll build a VAE and run it to produce the same kind of output you did in the previous section. Luckily, this won't require much tweaking from your encoder architecture. The decoder can stay unchanged; there are just two big changes you'll need to make:

### Probabilistic encoder

Rather than your encode outputting a latent vector $z$, it should output a mean $\mu$ and standard deviation $\sigma$; both vectors of dimension `latent_dim_size`. We then sample our latent vector $z$ using $z_i = \mu_i + \sigma_i \cdot \epsilon_i$. 

Note that this is equivalent to $z = \mu + \Sigma \epsilon$ as shown in the diagram above, but where we assume $\Sigma$ is a diagonal matrix (i.e. the auxiliary random variables $\epsilon$ which we're sampling are independent). This is the most common approach taken in situations like these.

How exactly should this work in your model's architecture? Very simply - take the final linear layer in your embedding (which should have had `in_channels=128` and `out_channels=latent_dim_size`) and replace it with two linear layers with the same input and output sizes, one to produce `mu` and one to produce `sigma`. Then use these to produce `z` using the method above, and have that be the output of your encoder. (If you prefer, you can just have one linear layer with `out_channels=2*latent_dim_size` then use e.g. `torch.split` on the output to get `mu` and `sigma`.) One extra subtlety - your output `sigma` should be the standard deviation of your distribution, meaning it should always be positive. The easiest way to enforce this is to return `mu` and `logsigma`, then calculate `sigma = t.exp(logsigma)`.

You should also return the parameters `mu` and `logsigma` in your VAE's forward function - the reasons for this will become clear below.

### New loss function

We're no longer calculating loss simply as the reconstruction error between $x$ and our decoder's output $x'$. Recall from the **Loss Function: ELBO** section of the VAE paper that we have loss function:
$$
\begin{aligned}
L_{\mathrm{VAE}}(\theta, \phi) &=-\log p_\theta(\mathbf{x})+D_{\mathrm{KL}}\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x})\right) \\
&=-\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \log p_\theta(\mathbf{x} \mid \mathbf{z})+D_{\mathrm{KL}}\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z})\right) \\
\theta^*, \phi^* &=\arg \min _{\theta, \phi} L_{\mathrm{VAE}}
\end{aligned}
$$

Let's look at both of the two terms on the second line, and see what they actually mean.

**The first term** is playing the role of reconstruction loss. This might not be obvious at first, but notice that minimising it is equivalent to *maximising the probability of the decoder reconstructing $x$ from $z$, given that $z$ was itself the encoding of input $x$*. In fact, we can just swap this term out for any reasonable reconstruction loss (for instance, the $l_2$ loss that we used in the last section).

**The second term** is the KL divergence between $q_{\phi}(z|x)$ (the distribution of latent vectors produced by your VAE when given inputs $x$) and $p_{\theta}(z)$ (the true generative distribution of latent vectors $z$). Note that both of these distributions are known to us - the first is normal with mean $\mu(x)$ and variance $\sigma(x)^2$, and the second is just the standard normal distribution with mean 0, variance 1 (see [Figure 6](https://lilianweng.github.io/posts/2018-08-12-vae/#beta-vae:~:text=.-,Fig.%206.,-The%20graphical%20model) in the blog post). The KL divergence of these two distributions has a closed form expression, which is given by:
$$
D_{KL}(N(\mu, \sigma^2) || N(0, 1)) = \sigma^2 + \mu^2 - \log{\sigma} - \frac{1}{2}
$$
This is why it was important to output `mu` and `logsigma` in our forward functions, so we could compute this expression! (It's easier to use `logsigma` than `sigma` when evaluating the expression above, for stability reasons).

We won't ask you to derive this formula, because it requires understanding of **differential entropy**. However, it is worth doing some sanity checks, e.g. plot some graphs and convince yourself that this expression is larger as $N(\mu, \sigma^2)$ is further away from the standard normal distribution.

One can interpret this as the penalty term to make the latent space meaningful. If all the latent vectors $z$ you generate have each component $z_i$ normally distributed with mean 0, variance 1 (and we know they're independent because our $\epsilon_i$ we used to generate them are independent), then there will be no gaps in your latent space where you produce weird-looking output (like we saw in our autoencoder plots from the previous section). You can try training your VAE without this term, and it might do okay at reproducing $x$, but it will perform much worse when it comes time to use it for generation. Again, you can quantify this by encoding some input data and projecting it onto the first two dimensions. When you include this term you should expect to see a nice regular cloud surrounding the origin, but without this term you might see some irregular patterns or blind spots:
""")
    st_image("vae_latent_space.png", 600)

    st.markdown(r"""
Once you've computed both of these loss functions, you should add them together and perform gradient descent on them. **You might need to change the coefficients of the sum before you get your VAE working** (read the section on Beta-VAEs for more on this). For instance, I got very poor results when just taking the sum of the mean of the two loss functions above, but much better results when using a weighting of 0.1 for the mean of the KL-div loss.

You can build your VAE below. Again, the implementational details are mostly left up to you.

```python
class Autoencoder(nn.Module):

    def __init__(self, *args):
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        pass
```

### Training loop

You should write and run your training loop below. 

Note that you might need to use a different optimizer than for your autoencoder. I found Adam with standard learning rate and `weight_decay=1e-5` worked well (for the autoencoder I didn't need the weight decay).

```python
def train_vae(*args):
    pass
```
""")

    with st.expander("Help - my KL divergence is close to zero, and my reconstruction loss isn't decreasing."):
        st.markdown(r"""
This is likely because your $\beta$ is too large. In this case, your model prioritizes having its latent vectors' distribution equal to the standard normal distribution.

Your model still wants to reduce the reconstruction loss, if it can find a way to do this without changing the distribution of the latent vectors. But this might not be possible, if your model has settled into a local minimum.

---

In general, when you're optimizing two different loss functions, it's important to test out different values for their weighted average. Getting the balance wrong can lead to local minima where the model prioritizes reducing one of your loss functions, and basically ignores the other one.

Weights and biases hyperparameter searches are a good tool for this.
""")

    st.markdown("""
Once you've got your VAE working, you should go back through the exercises from your encoder (i.e. the ones where you produced plots). How different are they this time? Are your outputs any more or less faithful?""")

    with st.expander("Click here to see a decoder output from varying two of its latent space dimensions. (Before you do, have a think about what you expect to see, relative to the autoencoder's output from the previous section.)"):
        st.plotly_chart(fig_dict["vae_interp"])
        st.markdown("""
Note how we have a smooth continuum over all parts of the generation space! The top-left is clearly a 9, the top-right is clearly a 0, the bottom-right is an 8, and the bottom-left seems to be merging into a 1. There are some unidentifiable shapes, but these are mostly just linear interpolations between two shapes which *are* identifiable. It certainly looks much better than our autoencoder's generative output!
""")

    st.markdown(r"""
Note - don't be disheartened if your reconstructions don't look as faithful for your VAE than they did for your encoder. Remember the goal of these architectures isn't to reconstruct images faithfully, it's to generate images from samples in the latent dimension. This is the basis on which you should compare your models to each other.
""")

def section_bonus():
    st.markdown(r"""
# Bonus exercises

## Beta-VAEs

Read the section on [Beta-VAEs](https://lilianweng.github.io/posts/2018-08-12-vae/#beta-vae), if you haven't already. Can you use a Beta-VAE to get better performance?

To decide on an appropriate parameter $\beta$, you can look at the distribution of your latent vector. For instance, if your latent vector looks very different to the standard normal distribution when it's projected onto one of its components (e.g. maybe that component is very sharply spiked around some particular value), this is a sign that you need to use a larger parameter $\beta$. You can also just use hyperparameter searches to find an optimal $\beta$. See [the paper](https://openreview.net/pdf?id=Sy2fzU9gl) which introduced Beta-VAEs for more ideas.

## CelebA database

Try to build an autoencoder for the CelebA database. You shouldn't need to change the architecture much from your MNIST VAE. You should find the training much easier than with your GAN (as discussed yesterday, GANs are notoriously unstable when it comes to training). Can you get better results than you did for your GAN?

## Hierarchical VAEs

Hierarchical VAEs are ones which stack multiple layers of parameter-learning and latent-vector-sampling, rather than just doing this once. Read the section of [this paper](https://arxiv.org/pdf/2208.11970.pdf) for a more thorough description.""")

    st_image("hierarchical_vae_diagram.png", 500)

    st.markdown("""
Try to implement your own hierarchical VAE.

Note - understanding these is pretty crucial if you want to move on to diffusion models! In fact, you might want to read the section immediately after that one, which is on variational diffusion models, and explains them by relating them to hierarchical VAEs.

## Denoising and sparse autoencoders

The reading material on VAEs talks about [denoising](https://lilianweng.github.io/posts/2018-08-12-vae/#denoising-autoencoder) and [sparse](https://lilianweng.github.io/posts/2018-08-12-vae/#sparse-autoencoder) autoencoders. Try changing the architecture of your autoencoder (not your VAE) to test out one of these two techniques. Do does your decoder output change? How about your encoder scatter plot?

If you're mathematically confident and feeling like a challenge, you can also try to implement [contractive autoencoders](https://lilianweng.github.io/posts/2018-08-12-vae/#contractive-autoencoder)!
""")

func_list = [home, section_AE, section_VAE, section_bonus]

page_list = ["🏠 Home", "1️⃣ Autoencoders", "2️⃣ Variational Autoencoders", "3️⃣ Bonus"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()
