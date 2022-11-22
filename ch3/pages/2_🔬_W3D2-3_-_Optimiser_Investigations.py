import os
if not os.path.exists("./images"):
    os.chdir("./ch3")
from st_dependencies import *
styling()
import numpy as np
import plotly.express as px
if "fig" not in st.session_state:
    img = np.random.random(size=(15, 10, 10, 3))
    fig = px.imshow(img, animation_frame=0)
    st.session_state["fig"] = fig
else:
    fig = st.session_state["fig"]


st.markdown("""
<style>
table {
    width: calc(100% - 30px);
    margin: 15px
}
[data-testid="stDecoration"] {
    background-image: none;
}
div.css-fg4pbf [data-testid="column"] {
    box-shadow: 7px 7px 14px #aaa;
    padding: 15px;
}
div.css-ffhzg2 [data-testid="column"] {
    box-shadow: 7px 7px 14px #aaa;
    background: #333;
    padding: 15px;
}
[data-testid="column"] a {
    text-decoration: none;
}
</style>""", unsafe_allow_html=True)

def section_home():
    st.markdown("""
## 1️⃣ Suggested exercises

Today, we just have a collection of possible exercises that you can try today. The first one is more guided, and closely related to the material you covered yesterday. Many of the rest are based on material from Jacob Hilton's curriculum. They have relatively little guidance, and leave many of the implementation details up to you to decide. Some are more mathematical, others are more focused on engineering and implementation skills. If you're struggling with any of them, you can try speaking with your teammates in the Slack channel, or messaging on `#technical-questions`.

All the exercises here are optional, so you can attempt as many / few as you like. It's highly possible you'll only have time to do one of them, since each of them can take you down quite a long rabbit hole!
""")

def section1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#1-benchmark-different-optimizers-and-learning-rates">1. Benchmark different optimizers and learning rates</a></li>
    <li><a class="contents-el" href="#2-noisy-quadratic-model">2. Noisy Quadratic Model</a></li>
    <li><a class="contents-el" href="#3-shampoo">3. Shampoo</a></li>
    <li><a class="contents-el" href="#4-the-colorization-problem">4. The Colorization Problem</a></li>
    <li><a class="contents-el" href="#5-extra-reading">5. Extra reading</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown("""
## 1. Benchmark different optimizers and learning rates

Now that you've learned about different optimizers and learning rates, and you've used `wandb` to run hyperparameter sweeps, you now have the opportunity to combine the two and run your own experiments. A few things which might be interesting to investigate:

* How does the training loss for different optimizers behave when training your ConvNet or ResNets from chapter 0, or your decoder-only transformer from chapter 1?
* It was mentioned yesterday that PyTorch applies weight decay to all parameters equally, rather than only to weights and not to biases. What happens when you run experiments on your ConvNet or ResNet with weight decay varying across weights and biases?
    * Note - you'll need to use **parameter groups** for this task; see exercise 1 above. You can find all the biases by iterating through `model.named_parameters()`, and checking whether the name contains the string `"bias"`.
* You might want to go back to your BERT implementation from last section, and run some of these experiments on that.

## 2. Noisy Quadratic Model

As was mentioned yesterday in the discussion of gradient descent, a large bach generally means that the estimate of the gradient is closer to that of the true gradient over the entire dataset (because it is an aggregate of many different datapoints). But empirically, we tend to observe a [critical batch size](https://arxiv.org/pdf/1812.06162.pdf), above which training becomes less-data efficient.

The NQM is the second-order Taylor expansion of the loss discussed in the critical batch size paper, and accounts for surprisingly many deep learning phenomena. [This paper](https://arxiv.org/abs/1907.04164) uses this model to explain the effect of curvature and preconditioning on the critical batch size.

You can try running your own set of noisy quadratic model experiments, based on the NQM paper:

* Set up a testbed using the setup from the NQM paper, where the covariance matrix of the gradient and the Hessian are both diagonal. You can use the same defaults for these matrices as in the paper, i.e., diagonal entries of 1, 1/2, 1/3, ... for both (in the paper they go up to 10^4, you can reduce this to 10^3 if experiments are taking too long to run). Implement both SGD with momentum and Adam.
* Create a method for optimizing learning rate schedules. You can either use dynamic programming using equation (3) as in the paper (see footnote on page 7), or a simpler empirical method such as black-box optimization (perhaps with simpler schedule).
* Check that at very small batch sizes, the optimal learning rate scales with batch size as expected: proportional to the batch size for SGD, proportional to the square root of the batch size for Adam.
* Look at the relationship between the batch size and the number of steps to reach a target loss. Study the effects of momentum and using Adam on this relationship.""")

    st.info("""
Note - if you're confused by the concepts of **preconditioned gradient descent** and the **conditioning number** (which come up quite a lot in the NQM paper), you might find [this video](https://www.youtube.com/watch?v=zjzOYL4fhrQ) helpful, as well as the accompanying [Colab notebook](https://colab.research.google.com/drive/1lBu2aprYsOq5wj73Avaf0klRv47GfrOH#scrollTo=eo3DiaJmZ9B9). The latter gives an interactive walkthrough of these concepts, with useful visualisations. Additionally, [this page of lecture notes](https://www.cs.princeton.edu/courses/archive/fall18/cos597G/lecnotes/lecture5.pdf) discusses preconditioning, although it's somewhat more technical than the Colab notebook.

""")

    st.markdown("""
## 3. Shampoo

We briefly mentioned Shampoo yesterday, a special type of structure-aware preconditioning algorithm. Try to implement it, based on algorithm 1 from [this paper](https://arxiv.org/pdf/1802.09568.pdf).""")
    st_image("shampoo.png", 540)
    st.markdown("")

    st.markdown("""
You can do this by defining an optimizer just like your `SGD`, `RMSprop` and `Adam` implementations yesterday. Try using your optimizer on image classifiers; either your MNIST classifier something more advanced like CIFAR-10. How does it compare to the other algorithms? Do your results appear similar to **Figure 2** in the paper?

## 4. Neural Tangent Kernel

Read the blog post [Understanding the Neural Tangent Kernel](https://rajatvd.github.io/NTK/). This is an accessible introduction to the [NTK paper](https://arxiv.org/pdf/1806.07572.pdf). You might also find [this post](https://lilianweng.github.io/posts/2022-09-08-ntk/#neural-tangent-kernel) helpful.

The NTK paper's critical proposition is:""")

    cols = st.columns([1])
    with cols[0]:
        st.markdown(r"""
**When $n_1, ..., n_L \to \infty$ (network with infinite width), the NTK:**

1. **Converges to a deterministic limit, meaning that the kernel's limit is irrelevant to the initialization values and only determined by the model architecture.**
2. **Stays essentially constant during training.**
""")

    st.markdown(r"""
Investigate these claims in your own 2-layer MNIST CNN. You can try the following:

* Test out the **lazy training** phenomenon. Make an animated graph of how the weights between your hidden layers change over time. Do you observe them to be approximately constant for large networks?
* Estimate the quantity $\color{blue}\kappa(\boldsymbol{\omega_0})$, defined in the paper as:

    $\color{blue}\kappa(\boldsymbol{\omega_0}) = \left\|\left(\boldsymbol{y}\left(\boldsymbol{w}_0\right)-\overline{\boldsymbol{y}}\right)\right\| \frac{\left\|\nabla_w^2 \boldsymbol{y}\left(\boldsymbol{w}_{\mathbf{0}}\right)\right\|}{\left\|\nabla_w \boldsymbol{y}\left(\boldsymbol{w}_{\mathbf{0}}\right)\right\|^2}$

    Note that you can calculate the Hessian and Jacobian using the PyTorch functions:

    ```python
    torch.autograd.functional.jacobian(func, x)
    torch.autograd.functional.hessian(func, x)
    ```

    Plot $\color{blue}\kappa(\boldsymbol{\omega_0})$ against the number of neurons in the hidden layers. Do you get it decaying to near zero, as expected?
    
    (Make sure you properly initialise the weights, as [described in the post](https://rajatvd.github.io/NTK/#:~:text=independent%20zero%2Dmean%20Gaussian%20random%20variables%20with%20a%20variance%20that%20goes%20inversely%20with%20the%20size%20of%20the%20input%20layer.).)

""")

    with st.expander("Example code for producing animations with Plotly:"):
        st.markdown("""
```python
import numpy as np
import plotly.express as px

# img is a list of 10 images
img = np.random.random(size=(15, 10, 10, 3))
fig = px.imshow(img, animation_frame=0)
fig.show()
```""")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""

## 5. Extra reading

If none of the experiments above seem all that exciting to you, then you can read some more about optimisation instead. As well as yesterday's resources, a few you might enjoy are:

* [Deep Double Descent](https://openai.com/blog/deep-double-descent/) - A revision of the classical bias-variance trade-off for deep learning. Further investigation of the phenomenon can be found [here](https://arxiv.org/abs/2002.11328).
* [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) - A well-known counterintuitive result about pruning.
""")

func_list = [section_home, section1]

page_list = ["🏠 Home", "1️⃣ Suggested exercises"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()
