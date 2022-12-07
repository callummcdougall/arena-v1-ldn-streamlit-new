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
    st.session_state["fig_dict"] = {}

if "vae_interp" not in st.session_state["fig_dict"]:
    st.session_state["fig_dict"] |= get_fig_dict()

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
    # st_image("autoencoder-architecture.png", 600)
    st_excalidraw("autoencoder-diagram", 850)

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

Your encoder should consist of two convolutional layers, a flatten, then two linear layers. After each convolution or linear layer (except for the last one) you should have an activation function. The convolutions should have kernel size 4, stride 2, padding 1 (recall from yesterday that this exactly halves the width and height). The number of output channels can be 16 and 32 respectively.

After the convolutional layers, you flatten then apply linear layers. Your flattened size will be $32 \times 7 \times 7$. Your first linear layer should have `out_features=128`; the second is up to you (again we recommend playing around with values between 2 and 10).""")

    st_excalidraw("vae-guide-ex-1", 650)
    st.markdown(r"""

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
# Choose number of interpolation points, and interpolation range (you might need to adjust these)
n_points = 11
interpolation_range = (-10, 10)

# Constructing latent dim data by making two of the dimensions vary independently between 0 and 1
latent_dim_data = t.zeros((n_points, n_points, latent_dim_size), device=device)
x = t.linspace(*interpolation_range, n_points)
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

The key perspective shift is this: **rather than mapping the input into a fixed vector, we map it into a distribution**, from which we sample to get our latent vector.

From this [TowardsDataScience](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) article:
""")
    # cols = st.columns([1, 6, 1])
    # with cols[1]:
    st.success("""
Due to overfitting, the latent space of an autoencoder can be extremely irregular (close points in latent space can give very **different** decoded data, some point of the latent space can give **meaningless** content once decoded) and, so, we can’t really define a **generative** process that simply consists to sample a point from the **latent space** and make it go through the decoder to get new data. **Variational autoencoders** (VAEs) are autoencoders that tackle the problem of the latent space irregularity by making the encoder return a **distribution over the latent space** instead of a single point and by adding in the loss function a **regularisation** term over that returned distribution in order to ensure a better **organisation** of the latent space.""")

    st.markdown("""Or, in fewer words:""")

    # cols = st.columns([1, 6, 1])
    # with cols[1]:
    st.success("""
**A variational autoencoder can be defined as being an autoencoder whose training is regularised to avoid overfitting and ensure that the latent space has good properties that enable generative process.**
""")

    # st_image("vae_diagram.png", 800)
    st_excalidraw("vae-diagram-ex", 850)

    st.markdown(r"""
At first, this idea of mapping to a distribution sounds like a crazy hack - why on earth does it work? This diagram should help convey some of the intuitions:
""")

    st_excalidraw("vae-scatter", 800)
    st.markdown(r"""
With our encoder, there was nothing incentivising us to make full and meaningful use of the latent space. It's hypothetically possible that our network was mapping all the inputs to some very small subspace and reconstructing them with perfect fidelity. This wouldn't have required numbers with different features to be far apart from each other in the latent space, because even if they are close together no information is lost. See the first image above.

But with our variational autoencoder, each MNIST image produces a **sample** from the latent space, with a certain mean and variance. This means that, when two numbers look very different, their latent vectors are forced apart - if the means were close together then the decoder wouldn't be able to reconstruct them.

Another nice property of using random latent vectors is that the entire latent space will be meaningful. For instance, there is no reason why we should expect the linear interpolation between two points in the latent space to have meaningful decodings. The decoder output *will* change continuously as we continuously vary the latent vector, but that's about all we can say about it. However, if we use a variational autoencoder, we don't have this problem. The output of a linear interpolation between the cluster of $2$s and cluster of $7$s will be *"a symbol which pattern-matches to the family of MNIST digits, but has equal probability to be interpreted as a $2$ and a $7$ respectively, and this is indeed what we find.
""")
    st_excalidraw("vae-scatter-2", 800)
    st.markdown(r"""
### Reparameterisation trick

One question that might have occurred to you - how can we do backward passes through our network? We know how to differentiate with respect to the inputs to a function, but how can we differentiate wrt the parameters of a probability distribution from which we sample our vector? The solution is to convert our random sampling into a function, by introducing an extra parameter $\epsilon$. We sample $\epsilon$ from the standard normal distribution, and then express $z$ as a deterministic function of $\mu$, $\sigma$ and $\epsilon$:
$$
z = \mu + \sigma \odot \epsilon
$$
where $\odot$ is a notation meaning pointwise product, i.e. $z_i = \mu_i + \sigma_i \epsilon_i$. Intuitively, we can think of this as follows: when there is randomness in the process that generates the output, there is also randomness in the derivative of the output wrt the input, so **we can get a value for the derivative by sampling from this random distribution**. If we average over enough samples, this will give us a valid gradient for training.""")

    st_excalidraw("vae-diagram-reparam-ex", 1000)

    st.markdown(r"""

Note that if we have $\sigma_\theta(x)=0$ for all $x$, the VAE reduces to an autoencoder (since the latent vector $z = \mu_\theta(x)$ is once again a deterministic function of $x$). This is why it's important to add a KL-divergence term to the loss function, to make sure this doesn't happen. It's also why, if you print out the average value of $\sigma(x)$ while training, you'll probably see it stay below 1 (it's being pulled towards 1 by the KL-divergence loss, **and** pulled towards 0 by the reconstruction loss).

---

Before you move on to implementation details, there are a few questions below designed to test your understanding. They are based on material from this section, as well as the [KL divergence LessWrong post](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence). You might also find [this post](https://lilianweng.github.io/posts/2018-08-12-vae/#vae-variational-autoencoder) on VAEs from the readings helpful.

""")
   
    with st.expander("State in your own words why we need the reparameterization trick in order to train our network."):
        st.markdown(r"""
One sentence summary:

We can't backpropagate through random processes like $z_i \sim N(\mu_i(x), \sigma_i(x)^2)$, but if we instead write $z$ as a deterministic function of $\mu_i(x)$ and $\sigma_i(x)$ (plus some auxiliary random variable $\epsilon$) then we can differentiate our loss function wrt the inputs, and train our network.

Longer summary:

Our encoder works by generating parameters and then using those parameters to sample latent vectors $z$ (i.e. a **stochastic process**). Our decoder is deterministic; it just maps our latent vectors $z$ to fixed outputs $x'$. The stochastic part is the problem; we can't backpropagate gradients through random functions. However, instead of just writing $z \sim N(\mu_\theta(x), \sigma_\theta(x)^2I)$, we can write $z$ as a deterministic function of its inputs: $z = g(\theta, x, \epsilon)$, where $\theta$ are the parameters of the distribution, $x$ is the input, and $\epsilon$ is a randomly sampled value. We can then backpropagate through the network.
""")

    with st.expander("Summarize in one sentence what concept we're capturing when we measure the KL divergence D(P||Q) between two distributions."):
        st.markdown(r"""
Any of the following would work - $D(P||Q)$ is...

* How much information is lost if the distribution $Q$ is used to represent $P$.
* The quality of $Q$ as a probabilistic model for $P$ (where lower means $Q$ is a better model).
* How close $P$ and $Q$ are, with $P$ as the actual ground truth.
* How much evidence you should expect to get for hypothesis $P$ over $Q$, when $P$ is the actual ground truth.

---

This last interpretation is (in my opinion) the most useful for understanding **Loss #2** from the diagram above. We're trying to minimise the KL divergence $D_{KL}(N(\mu, \sigma) || N(0, I))$ between our model and the standard normal distribution. Equivalently, given that our discriminator $f_\theta$ is being fed inputs with distribution $N(\mu, \sigma)$ (this is the "ground truth" for the discriminator during training), we want the discriminator to find it hard to distinguish between these and the standard normal distribution (since then, in the process of training our model on these inputs, we will also be training it to perform well on $N(0, I)$ inputs).

Read [section 2](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence#2__Hypothesis_Testing) of the KL divergence post for more on this.""")
# A quick note here - it might seem confusing why we measure $D(q_{\phi}(z|x)||p_{\theta}(z|x))$ rather than $D(p_{\theta}(z|x)||q_{\phi}(z|x))$. After all, we are trying to learn a distribution $q_{\phi}(z|x)$ to approximate the distribution $p_{\theta}(z|x)$, and above I described $Q$ as being the model for $P$. What gives?

# The key point here is that **we never actually see $p_{\theta}(z|x)$, but we do see $q_{\phi}(z|x)$**. So $q_{\phi}(z|x)$ is the "actual ground truth" in the sense of it being the thing that generates the latent vectors we actually see.

# I think [Section 2 of this post](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence#2__Hypothesis_Testing) gives the best intuitions for what's going on here. Translating this section to our current use-case, we see that $D(q_{\phi}(z|x)||p_{\theta}(z|x))$ being small is equivalent to the following: **when we're generating samples $z$ via our encoder (with distribution $z \sim q_{\phi}(z|x)$), we find this process hard to distinguish from reality**. 

    st.markdown(r"""
## Building a VAE

For your final exercise of today, you'll build a VAE and run it to produce the same kind of output you did in the previous section. Luckily, this won't require much tweaking from your encoder architecture. The decoder can stay unchanged; there are just two big changes you'll need to make:

### Probabilistic encoder

Rather than your encoder outputting a latent vector $z = z(x)$, it should output a mean $\mu(x)$ and standard deviation $\sigma(x)$; both vectors of dimension `latent_dim_size`. We then sample our latent vector $z$ using $z_i = \mu_i + \sigma_i \epsilon_i$.

How exactly should this work in your model's architecture? Very simply - take the final linear layer in your encoder (which should have had `in_channels=128` and `out_channels=latent_dim_size`) and replace it with two linear layers with the same input and output sizes, one to produce `mu` and one to produce `sigma`. Then use these to produce `z` using the method above, and have that be the output of your encoder. One extra subtlety - we are interpreting your output `sigma` as the standard deviation of your distribution, meaning it should always be positive. The easiest way to enforce this is to return `mu` and `logsigma`, then calculate `sigma = t.exp(logsigma)`.""")

    st_excalidraw("vae-guide-ex", 1450)

    st.markdown(r"""

You should also return the parameters `mu` and `logsigma` in your VAE's forward function - the reasons for this will become clear below.

```python
class Autoencoder(nn.Module):

    def __init__(self, *args):
        pass

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        '''
        Returns a tuple of (mu, logsigma, z), where:
            mu and logsigma are the outputs of your encoder module
            z is the sampled latent vector taken from distribution N(mu, sigma**2)
        '''
        pass
```

### New loss function

We're no longer calculating loss simply as the reconstruction error between $x$ and our decoder's output $x'$ - now we have a new loss function. Recall from the diagram at the top of the page that our loss function was the sum of two different loss functions: the reconstruction loss between $x$ and $x'$, and the KL-divergence between the distribution of our latent vectors $N(\mu(x), \sigma(x)^2)$ and the target distribution of $N(0, I)$.""")

#     with st.expander(r"""Dropdown"""):
#         # Let's denote $\theta$ as the parameters of our encoder, and $\phi$ as the parameters of our decoder (following the convention in [this post](https://lilianweng.github.io/posts/2018-08-12-vae/)). 
#         st.markdown(r"""
# First, we're going to flip the problem on its head. Rather than thinking of $z$ as being a compressed version of $x$, we're going to pretend there's some random process $p$ which first samples the latent vector $z$ from the distribution $p(z)$, then samples $x$ from the conditional distribution $p(x \mid z)$. We use the following graphical model to describe this:
# """)
#         st_excalidraw("graphical-model", 500)
#         st.markdown(r"""
# We assume $p(z)$ is a known distribution. Typically, this is assumed to be the standard normal distribution, and we'll adopt this convention going forwards.

# We are trying to learn a **generative model**, in the form of the **decoder** $p_\theta(x \mid z)$. This is generative because we can sample $z \sim N(0, I)$ then sample $x \sim p_\theta(x \mid z)$ to generate output according to our model.

# Our goal is **likelihood-maximisation** - if $x$ is real MNIST data, we want to maximize $p_\theta(x)$ (or equivalently, maximize $\log{p_\theta(x)}$). This would mean that our model had a high probability of generating real data, when fed a random latent vector $z$.

# The problem? It's hard to maximize $p_\theta(x)$ directly. This is because computing $p_\theta(x_i)$ for a single sample $x_i$ is very computationally expensive. We'd have to integrate over all possible values of the latent vectors $z$:
# $$
# p_\theta\left(x_i\right)=\int p_\theta\left(x_i \mid \mathbf{z}\right) p_\theta(\mathbf{z}) d \mathbf{z}
# $$
# For a fixed value of $\mathbf{z}$, computing $p_\theta\left(x_i \mid \mathbf{z}\right)$ is equivalent to doing a forward pass in our neural network. To do this enough times to approximate a high-dimensional integral would be completely impossible!

# This is where our **encoder** comes in. We don't actually need to evaluate this integral everywhere, if we manage to learn some notion of what latent vectors $\mathbf{z}$ are "good" for inputs $x$. We write our encoder in the form of the probability distribution $q_\phi(z \mid x)$, and add it to our graphical model as follows:
# """)
#         st_excalidraw("vae-graphical-2", 500)
#         st.markdown(r"""
# Ideally, we want our approximation function $Q_\phi(\mathbf{z} \mid x)$ to be close to the true posterior distribution $P(z \mid x)$. Why is this? Because
# """)
# We can write the joint distribution as $p(x, z) = p(x \mid z) p(z)$.


# We want to learn the **reverse** of this distribution, i.e. $p_\theta(z \mid x)$. Why does this help us? Because it tells our encoder how to produce good latent vectors $z$ from inputs $x$. Once we know how to do that, we will be able to train our discriminator



# First, we're going to flip the problem on its head. Rather than thinking of $z$ as being a compressed version of $x$, we're going to pretend there's some random process that does what our encoder does, but in reverse. It first samples a latent vector $z$ according to the distribution $p_\theta(z)$, then samples an MNIST image $x$ from the distribution $p_\theta(x \mid z)$. Our decoder can be thought of as finding the value of $x$ that maximises the probability $q_\phi(z \mid x)$.

# We want to recover the distribution $p(x)$ of MNIST images. Unfortunately, this is pretty hard.

# If we could learn the **posterior distribution** $p(z \mid x)$, then we would be done. Why? Because it would then be easy to learn the forward distribution $p(x \mid z)$, using the following method:

# If we knew how to compute $p(x \mid z)$, then we could generate MNIST images ourselves in exactly the same way. 
# We can see our decoder model as learning a probability distribution $p_\theta(x \mid z)$, which represents the probability that a given latent vector $z$ came from the input image $x$. We can then find the value of $x$ that maximises this PDF, and choose that to be our decoder's output.
# """)

    st.markdown(r"""
Our full loss function is:
$$
L(\theta) = \mathbb{E}\left[\|x-x'\|^2 + D_{KL}\left(N(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)\, \| \,N(\mathbf{0},I)\right)\right]
$$
where the expectation is taken over random inputs $x$, and random noise $\boldsymbol{\epsilon}$ (since this is used to produce $x'$).

**The first term** is the standard reconstruction loss, just like we saw for autoencoders last section.

**The second term** is the KL divergence between $N(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ (the distribution of latent vectors produced by your VAE when given inputs $x$) and $N(\mathbf{0},I)$ (the true generative distribution of latent vectors $z$). The formula for the KL divergence between two multinomial normal distributions can get a bit messy, but luckily for us both of these distributions are independent (the covariance matrices are diagonal), and the KL divergence decomposes into the sum of KL divergences for each component:
$$
D_{KL}(\,N(\mu, \sigma^2) \,||\, N(0, 1)\,) = \frac{1}{2}(\mu^2 + \sigma^2 - 1) - \log{\sigma}
$$
This is why it was important to output `mu` and `logsigma` in our forward functions, so we could compute this expression! (It's easier to use `logsigma` than `sigma` when evaluating the expression above, for stability reasons).

We won't ask you to derive this formula, because it requires understanding of **differential entropy**. However, it is worth doing some sanity checks, e.g. plot some graphs / do some calculus and convince yourself that this expression is larger as $N(\mu, \sigma^2)$ is further away from the standard normal distribution.

As we discussed earlier, this second term can be seen as a **regulariser** or **penalty term** designed to make the latent space meaningful. If you train your VAE without this term, it will result in $\sigma$ being pulled down to zero (because reconstruction loss will always be smaller when there's less noise added), and you'll just be left with an autoencoder.
""")
    # st_image("vae_latent_space.png", 600)

    st.markdown(r"""
Once you've computed both of these loss functions, you should perform gradient descent on their sum. **You might need to change the coefficients of the sum before you get your VAE working** (read the section on Beta-VAEs for more on this). For instance, I got very poor results when just taking the sum of the mean of the two loss functions above, but much better results when using a weighting of 0.1 for the mean of the KL-div loss.

### A deeper dive into the maths of VAEs

If you're happy with the loss function as described in the section above, then you can move on from here. If you'd like to take a deeper dive into the mathematical justifications of this loss function, you can read the dropdown below. (Note that this may prove useful when you get to diffusion models tomorrow.)

""")

# *(Eventually we will treat $p_\theta$ as a **probabilistic decoder** and parameterize it as $p_\theta(x \mid z)$, but for now assume this is just a process which we are handed, which we're allowed to evaluate. In other words, given some $z$ and $x$ we can compute the probability $p_\theta(x \mid z)$, and so given some $z$ we can sample $x$ according to this distribution.)*
    with st.expander("Maths"):
        st.markdown(r"""
Firstly, let's flip the model we currently have on its head. Rather than having some method of sampling images $x$ from our image set, then having a function mapping images to latent vectors $z$, we will start with the decoder $p_\theta$ which:
* first generates the latent vector $z$ from distribution $p(z)$ (which we assume to be the standard normal distribution),
* then generates $x$ from the conditional distribution $p_\theta(x \mid z)$.

It may help to imagine $z$ as being some kind of efficient encoding of all the information in our image distribution (e.g. if our images were celebrity faces, we could imagine the components of the vector $z$ might correspond to features like gender, hair color, etc).

We can recover the probability distribution of $x$ by integrating over all possible values of the latent vector $z$:
$$
\begin{aligned}
p(x)&=\int_z p_\theta(x \mid z) p(z) \\
&= \mathbb{E}_{z \sim p(z)}[p_\theta(x \mid z)]
\end{aligned}
$$
We can interpret this as the probability that our decoder produces the image $x$ when we feed it some noise sampled from the standard normal distribution. So all we have to do is maximize the expected value of this expression over our sample of real images $x_i$, then we're training our decoder to produce images like the ones in our set of real images, right?

Unfortunately, it's not that easy. Evaluating this would be computationally intractible, because we would have to sample over all possible values for the latent vectors $z$:
$$
\theta^*=\underset{\theta}{\operatorname{argmax}}\; \mathbb{E}_{x \sim \hat{p}(x), z \sim p(z)}\left[\log p_\theta(x \mid z)\right]
$$
where $\hat{p}(x)$ denotes our distribution over samples of $x$. This problem gets exponentially harder as we add more latent dimensions, because our samples of $z$ will only cover a tiny fraction of the entire possible latent space. 

Imagine now that we had a function $q_\phi(z \mid x)$, which is high when **the latent vector $z$ is likely to have been produced by $x$**. This function would be really helpful, because for each possible value of $x$ we would have a better idea of where to sample $z$ from. We can represent this situation with the following **graphical model**:
""")
        st_excalidraw("vae-graphical-2", 400)
        st.markdown(r"""

The seemingly intractible optimization problem above is replaced with a much easier one:
$$
\begin{aligned}
p(x) &=\int q_\phi(z \mid x) \frac{p_\theta(x \mid z) p(z)}{q_\phi(z \mid x)} \\
\theta^*&=\underset{\theta}{\operatorname{argmax}}\; \mathbb{E}_{x \sim \hat{p}(x), z \sim q_\phi(z\mid x)}\left[\frac{p_\theta(x \mid z)p(z)}{q_\phi(z \mid x)}\right]
\end{aligned}
$$
Note, we've written $\log{p_\theta(x)}$ here because it's usually easier to think about maximizing the log probability than the actual probability.

Why is this problem easier? Because in order to estimate the quantity above, we don't need to sample a huge number of latent vectors $z$ for each possible value of $x$. The probability distribution $q_\phi(z \mid x)$ already concentrates most of our probability mass for where $z$ should be, so we can sample according to this instead.

We now introduce an important quantity, called the **ELBO**, or **evidence lower-bound**. It is defined as:
$$
\mathbb{E}_{z \sim q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log \frac{p_\theta(\boldsymbol{x} \mid z)p(\boldsymbol{z})}{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}\right]
$$
This is called the ELBO because it's a lower bound for the quantity $p(x)$, which we call the **evidence**. The proof for this being a lower bound comes from **Jensen's inequality**, which states that $\mathbb{E}[f(X)] \geq f(\mathbb{E}[X])$ for any convex function $f$ (and $f(x)=-\log(x)$ is convex). In fact, we can prove the following identity holds:
$$
\log{p(x)}=\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log \frac{p(\boldsymbol{z}) p_\theta(\boldsymbol{x} \mid \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\right]+D_{\mathrm{KL}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \,\|\, p_\theta(\boldsymbol{z} \mid \boldsymbol{x})\right)
$$
So the evidence minus the ELBO is equal to the KL divergence between the distribution $q_\phi$ and the **posterior distribution** $p_\theta(z \mid x)$ (the order of $z$ and $x$ have been swapped).

---

Finally, this brings us to VAEs. With VAEs, we treat $p_\theta(x \mid z)$ as our decoder, $q_\phi(z \mid x)$ as our encoder, and we train them jointly to minimise the ELBO. Using the previous identity, we see that maximizing the ELBO is equivalent to maximizing the following:
$$
\log{p(x)}-D_{\mathrm{KL}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \,\|\, p_\theta(\boldsymbol{z} \mid \boldsymbol{x})\right)
$$
In other words, we are maximizing the log-likelihood, and **at the same time** penalising any difference between our approximate posterior $q_\phi(\boldsymbol{z} \mid \boldsymbol{x})$ and our true posterior $p_\theta(\boldsymbol{z} \mid \boldsymbol{x})$.

---

We can rewrite the ELBO in a different way:
$$
\begin{aligned}
\mathbb{E}_{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log \frac{p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z}) p(\boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\right] & =\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})\right]+\mathbb{E}_{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log \frac{p(\boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}\right] \\
& =\underbrace{\mathbb{E}_{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})\right]}_{\text {reconstruction loss}}-\underbrace{D_{\mathrm{KL}}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \| p(\boldsymbol{z})\right)}_{\text {regularisation term}}
\end{aligned}
$$
which is now starting to look a lot more like the loss function we used earlier! The **regularisation term** is recognised as exactly equal to the KL divergence term in our loss function (because our encoder $q_\phi(z \mid x)$ learns a normal distribution with mean $\mu(x)$ and variance $\sigma(x)^2$, and our latent vector $z$ has a standard normal distribution). It might not be immediately obvious why the first term serves as reconstruction loss, but in fact that is what it's doing. We can describe this term as ***"the expected log-likelihood of our decoder reconstructing $x$ from latent vector $z$, given that $z$ was itself a latent vector produced by our encoder on input $x$".***

The decoder used in our VAE isn't actually probabilistic, it's deterministic, but we can pretend that our decoder is actually outputting a probability distribution $p_\theta(\cdot \mid z)$ with mean $\mu_\theta(z)$, and then we take this mean as our final output. This reconstruction loss will be smallest when the decoder's output $\mu_\theta(z)$ is closest to the original input $x$ (since then the probability distribution $p_\theta(\cdot \mid z)$ will be centered on $x$). Although the formula here and the reconstruction loss aren't exactly the same, it turns out that we can swap out the formula for something which also works as reconstruction loss (although the fact that these two things aren't exactly the same might help to motivate why we need to use a different coefficient for the KL divergence loss - see $\beta$-VAEs later).
""")

    st.markdown(r"""

### Training loop

You should write and run your training loop below. Again, most implementational details are left up to you.

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
