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

NAMES = []
def get_fig_dict():
    return {name: read_from_html(name) for name in NAMES}
if "fig_dict" not in st.session_state:
    st.session_state["fig_dict"] = {}
if NAMES and NAMES[0] not in st.session_state["fig_dict"]:
    st.session_state["fig_dict"] |= get_fig_dict()
fig_dict = st.session_state["fig_dict"]

def section_home():
    st.markdown(r"""
## 1️⃣ Adversarial attacks on vision models

In this section, we'll look at the **Fast Gradient Sign Method** (FGSM), which is a simple and effective method for generating adversarial examples. We'll also look at **Projected Gradient Descent** (PGD). We'll end by training our own model to be robust to adversarial examples.

## 2️⃣ Bonus

If you complete the main exercises, there are a few other interesting options to explore!
""")

def section_1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#first-adversarial-attack-using-fgsm">First Adversarial Attack using FGSM</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#untargeted-fgsm">Untargeted FGSM</a></li>
       <li><a class="contents-el" href="#targeted-fgsm">Targeted FGSM</a></li>
   </ul></li>
   <li><a class="contents-el" href="#additional-adversarial-attacks">Additional Adversarial Attacks</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#implementing-l2l-2l2-normalization-and-clamping">Implementing L2 normalization and clamping</a></li>
   </ul></li>
   <li><a class="contents-el" href="#adversarial-training">Adversarial Training</a></li>
   <li><a class="contents-el" href="#attacks-on-adversarially-trained-models">Attacks on Adversarially Trained Models</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#attacking-normally-trained-models">Attacking normally trained models</a></li>
       <li><a class="contents-el" href="#attacking-adversarially-trained-models">Attacking Adversarially Trained Models</a></li>
       <li><a class="contents-el" href="#comparing-adversarial-attacks-against-different-models">Comparing Adversarial Attacks against different models</a></li>
   </ul></li>
   <li><a class="contents-el" href="#train-your-own-adversarially-trained-model">Train your own adversarially trained model</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
Adversarial examples are examples designed in order to cause an machine learning system to malfunction. Here, an adversary is taking a real image of a panda and adds some adversarially generated noise to get the adversarial example. The adversarial noise is designed to have small distance from the original image, so it still looks like a panda for humans. However, the model now believes its a gibbon with 99.3\% confidence. 

![picture](https://drive.google.com/uc?export=view&id=1kvJRRUDssx8ZarAH71-nxv2c2_RBNz4G)

```python
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchvision import models

from w7_chapter7_adversarial_training import w7d3_utils, w7d3_tests
```

You'll also need to install the `robustness` library. robustness is a package created to make training, evaluating, and exploring neural networks flexible and easy. It is built on top of PyTorch.

```
pip install robustness
```

## First Adversarial Attack using FGSM

### Untargeted FGSM

The first method we look at is the untargeted Fast Gradient Sign Method (FGSM) proposed by [Goodfellow et al.](https://arxiv.org/pdf/1412.6572.pdf). The attack constructs adversarial examples as follows:

$$x_\text{adv} = x + \epsilon\cdot\text{sign}(\nabla_xJ(\theta, x, y))$$

where 

*   $x_\text{adv}$ : Adversarial image.
*   $x$ : Original input image.
*   $y$ : Original input label.
*   $\epsilon$ : Multiplier to ensure the perturbations are small.
*   $\theta$ : Model parameters.
*   $J$ : Loss.

The current attack formulation is considered 'untargeted' because it only seeks to maximize loss rather than to trick the model into predicting a specific label. """)

    with st.expander("Exercise - try and explain in words what the FGSM attack is doing."):
        st.markdown(r"""
Answer:

The FGSM attack is trying to find the direction of the gradient of the loss function with respect to the input image. It then multiplies this direction by a small number $\epsilon$ and adds it to the original image. For small enough $\epsilon$, this is the perturbation which will increase the model's loss the most.""")

    st.markdown(r"""

Try implementing the untargeted FGSM method for a batch of images yourself!

```python
def untargeted_FGSM(x_batch, true_labels, network, normalize, eps=8/255., **kwargs):
    '''Generates a batch of untargeted FGSM adversarial examples

    x_batch (torch.Tensor): the batch of unnormalized input examples.
    true_labels (torch.Tensor): the batch of true labels of the example.
    network (nn.Module): the network to attack.
    normalize (function): a function which normalizes a batch of images 
        according to standard imagenet normalization.
    eps (float): the bound on the perturbations.
    '''
    loss_fn = nn.CrossEntropyLoss(reduce="mean")
    x_batch.requires_grad = True

    pass

w7d3_tests.test_untargeted_attack(untargeted_FGSM, eps=8/255.)
```

If things go well, the model should switch from predicting 'giant panda' to predicting 'brown bear' or some other class. Additionally, try increasing the epsilon to see the noise more clearly.

### Targeted FGSM

In addition to the untargeted FGSM which simply seeks to maximize loss, we can also create targeted adversarial attacks. We do this using the following equation:

$$x_{adv} = x - \epsilon\cdot\text{sign}(\nabla_xJ(\theta, x, y_{target}))$$

where 

* $x_{adv}$ : Adversarial image.
* $x$ : Original input image.
* $y_{target}$ : The target label.
* $\epsilon$ : Multiplier to ensure the perturbations are small.
* $\theta$ : Model parameters.
* $J$ : Loss.

Try implementing the targeted FGSM method for a batch of images yourself!

```python
def targeted_FGSM(x_batch, target_labels, network, normalize, eps=8/255., **kwargs):
    '''Generates a batch of targeted FGSM adversarial examples

    x_batch (torch.Tensor): the unnormalized input example.
    target_labels (torch.Tensor): the labels the model will predict after the attack.
    network (nn.Module): the network to attack.
    normalize (function): a function which normalizes a batch of images 
        according to standard imagenet normalization.
    eps (float): the bound on the perturbations.
    '''
    loss_fn = nn.CrossEntropyLoss(reduce="mean")
    x_batch.requires_grad = True

    pass

w7d3_tests.test_targeted_attack(targeted_FGSM, target_idx=8, eps=8/255.)
```

**Note that even if the implementation is perfect, FGSM is not able to generate effective targeted attacks, so don't expect the output image to assign a high probability to the target label.**

## Additional Adversarial Attacks

### Implementing $L_2$ normalization and clamping

**Projected Gradient Descent** is a variant of gradient descent, which involves taking the normal gradient descent step and then projecting it onto some set (often a [ball](https://en.wikipedia.org/wiki/Ball_(mathematics)#In_normed_vector_spaces)). In the case of L2 (the most common type), we project onto the L2 ball.

Like FGSM, the PGD is classified as a 'white-box' attack, meaning that it requires access to the model's parameters. This is in contrast to 'black-box' approaches, where only the model's outputs are observed.
""")

    st_image("robin.png", 600)
    st.markdown("")
    st.caption("Left: a robin. Right: a waffle iron.")
    st.markdown(r"""

We will implement some helper functions that we can use for the **Projected Gradient Descent** (PGD) L2 method below.

For the `normalize_l2` function we will be returning the following value:

$$
\frac{x}{||x||_{2}}
$$

For the `tensor_clamp_l2` function we will compute and return the following value

$$
\begin{equation}
    X=
    \begin{cases}
      clamp(x), & \text{if}\ ||x-c||_2 > r \\
      x, & \text{otherwise}
    \end{cases}
  \end{equation}
$$

where 
$ \text{clamp}(x) = c + \frac{x-c}{||x-c||_2} ⋅ r$, X is the return value, x is the input, c (center) is a tensor of the same shape as x, and r (radius) is a scalar value.

Try implementing the batched version of `normalize_l2` and `tensor_clamp_l2` below.

```python
def normalize_l2(x_batch):
    '''
    Expects x_batch.shape == [N, C, H, W]
    where N is the batch size, 
    C is the channels (or colors in our case),
    H, W are height and width respectively.

    Note: To take the l2 norm of an image, you will want to flatten its dimensions (be careful to preserve the batch dimension of x_batch).
    '''
    
    pass

def tensor_clamp_l2(x_batch, center, radius):
    '''Batched clamp of x into l2 ball around center of given radius.'''

    pass

def PGD_l2(x_batch, true_labels, network, normalize, num_steps=20, step_size=3./255, eps=128/255., **kwargs):
        '''
        Returns perturbed batch of images
        '''
        # Initialize our adversial image
        x_adv = x_batch.detach().clone()
        x_adv += torch.zeros_like(x_adv).uniform_(-eps, eps)

        for _ in range(num_steps):
            x_adv.requires_grad_()
            
            # Calculate gradients
            with torch.enable_grad():
              logits = network(normalize(x_adv))
              loss = F.cross_entropy(logits, true_labels, reduction='sum')

            # Normalize the gradients with your L2
            grad = normalize_l2(torch.autograd.grad(loss, x_adv, only_inputs=True)[0])

            # Take a step in the gradient direction.
            x_adv = x_adv.detach() + step_size * grad
            # Project (by clamping) the adversarial image back onto the hypersphere
            # around the image.
            x_adv = tensor_clamp_l2(x_adv, x_batch, eps).clamp(0, 1)

        return x_adv
```

Try out the helper functions you wrote. Note how the hyperparameters differ depending on the attack that one is using. You can see more examples below.

```python
w7d3_tests.test_untargeted_attack(PGD_l2, eps=128/255.)
```

In addition to your implementations of FGSM, we will provide you with an implementation of PGD by [Madry et al.](https://arxiv.org/pdf/1706.06083.pdf). We provide both targeted and untargeted versions.

```python
def untargeted_PGD(x_batch, true_labels, network, normalize, num_steps=10, step_size=0.01, eps=8/255., **kwargs):
    '''Generates a batch of untargeted PGD adversarial examples

    x_batch (torch.Tensor): the batch of unnormalized input examples.
    true_labels (torch.Tensor): the batch of true labels of the example.
    network (nn.Module): the network to attack.
    normalize (function): a function which normalizes a batch of images 
        according to standard imagenet normalization.
    num_steps (int): the number of steps to run PGD.
    step_size (float): the size of each PGD step.
    eps (float): the bound on the perturbations.
    '''
    x_adv = x_batch.detach().clone()
    x_adv += torch.zeros_like(x_adv).uniform_(-eps, eps)

    for i in range(num_steps):
    x_adv.requires_grad_()

    # Calculate gradients
    with torch.enable_grad():
        logits = network(normalize(x_adv))
        loss = F.cross_entropy(logits, true_labels, reduction='sum')
    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

    # Perform one gradient step
    x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())

    # Project the image to the ball.
    x_adv = torch.maximum(x_adv, x_batch - eps)
    x_adv = torch.minimum(x_adv, x_batch + eps)

    return x_adv

w7d3_tests.test_untargeted_attack(untargeted_PGD, eps=8/255.)
```

And targeted PGD:

```python
def targeted_PGD(x_batch, target_labels, network, normalize, num_steps=100, step_size=0.01, eps=8/255., **kwargs):
    '''Generates a batch of untargeted PGD adversarial examples

    Args:
    x_batch (torch.Tensor): the batch of preprocessed input examples.
    target_labels (torch.Tensor): the labels the model will predict after the attack.
    network (nn.Module): the network to attack.
    normalize (function): a function which normalizes a batch of images 
        according to standard imagenet normalization.
    num_steps (int): the number of steps to run PGD.
    step_size (float): the size of each PGD step.
    eps (float): the bound on the perturbations.
    '''
    x_adv = x_batch.detach().clone()
    x_adv += torch.zeros_like(x_adv).uniform_(-eps, eps)

    for i in range(num_steps):
    x_adv.requires_grad_()

    # Calculate gradients
    with torch.enable_grad():
        logits = network(normalize(x_adv))
        loss = F.cross_entropy(logits, target_labels, reduction='sum')
    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

    # Perform one gradient step
    # Note that this time we use gradient descent instead of gradient ascent
    x_adv = x_adv.detach() - step_size * torch.sign(grad.detach())

    # Project the image to the ball
    x_adv = torch.maximum(x_adv, x_batch - eps)
    x_adv = torch.minimum(x_adv, x_batch + eps)

    return x_adv

# Try changing the target_idx around!
w7d3_tests.test_targeted_attack(targeted_PGD, target_idx=1, eps=8/255.)
```

## Adversarial Training

Now that we’ve learned about attacks like FGSM or PGD, one natural question might be to ask: “how might we train models to be robust to adversarial attacks?” One common approach is adversarial training. During adversarial training, we expose our model to adversarial examples and penalize our model if the model is decieved. In particular, an adversarial training loss might be as follows:

$$
\operatorname{Loss}(f, \mathcal{D})=\underset{x, y \sim \mathcal{D}}{\mathbb{E}}\left[\operatorname{CrossEntropy}(f(x), y)+\lambda \cdot \operatorname{CrossEntropy}\left(f\left(g_{\text {adv }}(x)\right), y\right)\right]
$$

where $λ$ is some hyperparameter determining how much we emphasize the adversarial training. This often reduces accuracy, but increases robustness to the specific adversarial attack which your model is trained on. However, many have shown that robustness towards one type of adversarial attack does not provide robustness to other adversarial attacks. For instance, a model trained on FGSM will not perform well on PGD attacks with n=10. PGD was significant because PGD showed that models trained on it were actually robust to a whole class of adversarial examples.

## Attacks on Adversarially Trained Models

We devote this section to attacking an adversarially trained model. As a reminder, a model which has been "adversarially trained" means that it has been exposed to a load of adversarial examples over training and has specifically trained to recognize them properly.

In this section, we hope to demonstrate that adversarial attacks look a lot different if you're attacking an adversarially trained model.

The model we use is an $L_\infty$ robust ResNet18 trained with adversarial examples of $ϵ=8/255$.

### Attacking normally trained models

```python
# Attack a normal model (we only support targeted methods)
w7d3_utils.attack_normal_model(
    targeted_PGD, 
    target_idx=10, 
    eps=8/255., 
    num_steps=10, 
    step_size=0.01
)
```

### Attacking Adversarially Trained Models

```python
# Attack an adversarially trained model (we only support targeted methods)
w7d3_utils.attack_adversarially_trained_model(
    targeted_PGD, 
    target_idx=10, 
    eps=8/255., 
    num_steps=10, 
    step_size=0.01
)
```

### Comparing Adversarial Attacks against different models

```python
Take a few minutes to play around with the previous code. Jot down three observations about how attacking an adversarially trained model differs from attacking a normal model.

Example responses:

1. The confidence of typical models is higher than adversarially trained models
2. [Fill in Observation]
3. [Fill in Observation]
```

## Train your own adversarially trained model

Now that you've seen how to attack an adversarially trained model, you might be interested in training your own. Here, you can have a chance to do just that! We'll provide you with a few recommendations:

* Use a relatively simple and small model, e.g. your MNIST from week 0, or your resnet with CIFAR10 data.
* You can use the loss function we mentioned above:
    $$
    \operatorname{Loss}(f, \mathcal{D})=\underset{x, y \sim \mathcal{D}}{\mathbb{E}}\left[\operatorname{CrossEntropy}(f(x), y)+\lambda \cdot \operatorname{CrossEntropy}\left(f\left(g_{\text {adv }}(x)\right), y\right)\right]
    $$
    where $g_\text{adv}$ is your adversarial attack of choice, e.g. FGSM, PGD, etc.
* If you're feeling stuck, [this GitHub](https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR) might be useful.
""")

def section_2():
    st.markdown(r"""
## Bonus

Here are a few suggested bonus exercises. You can do as many or as few as you'd like.

### Defensive Distillation

Defensive distillation is an adversarial training technique that adds flexibility to an algorithm’s classification process so the model is less susceptible to exploitation.

Implement the defensive distillation algorithm from [this paper](https://arxiv.org/pdf/1511.04508.pdf), and compare this to your previous results. You can use the CIFAR10 dataset for this exercise.

### Features, not Bugs

Read the paper [Adversarial Examples Are Not Bugs, They Are Features](https://gradientscience.org/adv/). Download the datasets used in that paper, and try and replicate their results.

### Diffusion Models for Adversarial Purification

Adversarial purification refers to a class of defense methods that remove adversarial perturbations using a generative model. 

Read [the paper](https://arxiv.org/abs/2205.07460?context=cs.CV), and see if you can replicate their **DiffPure** algorithm, and their results. You might want to use your diffusion model from chapter 5. Also, [this GitHub](https://github.com/NVlabs/DiffPure) might be useful!

### Robust Feature-Level Adversaries are Interpretability Tools

The literature on adversarial attacks in computer vision typically focuses on pixel-level perturbations. [This paper](https://arxiv.org/abs/2110.03605) explores the idea of using "feature-level" adversarial attacks, which is to say attacks that manipulate the latent space of image generators rather than the pixels themselves.

A notebook to work through is also available [here](https://github.com/thestephencasper/feature_level_adv).

---

(Note - some of these would also make decent capstone projects!)
""")

func_list = [section_home, section_1, section_2]

page_list = ["🏠 Home", "1️⃣ Adversarial attacks on vision models", "2️⃣ Bonus"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()
