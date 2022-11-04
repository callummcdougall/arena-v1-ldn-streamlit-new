import streamlit as st

st.set_page_config(layout="wide")

# code > span.string {
#     color: red !important;
# }

st.markdown("""
<style>
label.effi0qh3 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 15px;
}
p {
    line-height:1.48em;
}
.streamlit-expanderHeader {
    font-size: 1em;
    color: darkblue;
}
.css-ffhzg2 .streamlit-expanderHeader {
    color: lightblue;
}
header {
    background: rgba(255, 255, 255, 0) !important;
}
code {
    font-size: 13px;
    color: red;
    white-space: pre-wrap !important;
}
a.contents-el > code {
    color: black;
    background-color: rgb(248, 249, 251);
}
.css-ffhzg2 a.contents-el > code {
    color: orange;
    background-color: rgb(26, 28, 36);
}
.css-ffhzg2 code:not(pre code) {
    color: orange;
}
.css-ffhzg2 .contents-el {
    color: white !important;
}
pre code {
    font-size:13px !important;
}
.katex {
    font-size:17px;
}
h2 .katex, h3 .katex, h4 .katex {
    font-size: unset;
}
ul.contents {
    line-height:1.3em; 
    list-style:none;
    color-black;
    margin-left: -10px;
}
ul.contents a, ul.contents a:link, ul.contents a:visited, ul.contents a:active {
    color: black;
    text-decoration: none;
}
ul.contents a:hover {
    color: black;
    text-decoration: underline;
}
</style>""", unsafe_allow_html=True)

def section_home():
    st.markdown("""
# Transformer exercises

Today will contain some exercises that should help deepen your understanding of some of the material you were exposed to yesterday.

All exercises today are strongly recommended, although there aren't very many of them. If you finish them, we recommend you go back over the questions from yesterday and complete more of them.

Note - from now on, it's fine to directly use layers from `torch.nn`, rather than using your implementations from last week (although you're welcome to still use these if you'd like!).
""")

def section1():
    st.markdown(r"""

# Positional Encoding \*\*

If you didn't already implement the positional embedding from Monday's exercises, you should do so now.

The link is [here](https://arena-w1d1.streamlitapp.com/Positional_encoding).

You might also want to try implementing this as an `nn.Module` instance, e.g. using the following template:

```python
class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, embedding_dim: int):
        pass

    def forward(self, x: Tensor) -> Tensor:
        '''
        x: shape (batch, seq_len, embedding_dim)
        '''
        pass
```

Note, we have used `max_seq_len` rather than `seq_len`. One of the advantages of positional encoding (as described in section 3.5 of the [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) paper) is the ability to " extrapolate to sequence lengths longer than the ones encountered during training". When we initialise the positional embedding we should do so using `max_seq_len` rows, and when we add it to `x` in `forward` we will just be taking a slice of it.
""")

    with st.expander("Why might we need to use register_buffer in the positional encoding? (you might want to look at the implementation of BatchNorm last week to remind yourself what register_buffer does)"):
        st.markdown("""`register_buffer` is used for objects which shouldn't be considered a model parameter, e.g. BatchNorm's `running_mean`. Here, we are defining an embedding, but it isn't learned (because we're fixing its weights to be equal to the sinusoidal weights).""")

def section2():
    st.markdown(r"""

# Attention Mechanism \*\*

The second and third pieces of [Jacob Hilton's suggested exercise](https://github.com/jacobhilton/deep_learning_curriculum/blob/master/1-Transformers.md) are:

* Implement the function which calculates attention, given (Q,K,V) as arguments.
* Implement the masking function.

You should do both of these now. We've provided the template for these functions. There are no solutions or test functions uploaded for these exercises; we encourage you to get feedback either from uploading your solutions to GitHub or speaking to your fellow participants.

First, you should just implement single-head attention, without worrying about masking or multiple heads:

```python
def single_head_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    '''
    Should return the results of self-attention (see the "Self-Attention in Detail" section of the Illustrated Transformer).

    With this function, you can ignore masking.

    Q: shape (FILL THIS IN!)
    K: shape (FILL THIS IN!)
    V: shape (FILL THIS IN!)

    Return: shape (FILL THIS IN!)
    '''
    pass
```

Next, you should use masking:

```python
def single_head_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    '''
    Should return the results of masked self-attention.

    See "The Decoder Side" section of the Illustrated Transformer for an explanation of masking.

    Q: shape (FILL THIS IN!)
    K: shape (FILL THIS IN!)
    V: shape (FILL THIS IN!)

    Return: shape (FILL THIS IN!)
    '''
    pass
```

""")

    with st.expander("Question - why do we use masking for decoder blocks?"):
        st.markdown("""It prevents our model from 'looking into the future'. The self attention layer is only allowed to attend to earlier positions in the output sequence, and all information from future positions is blocked.""")

func_list = [section_home, section1, section2]

page_list = ["🏠 Home", "1️⃣ Positional Encoding", "2️⃣ Attention Mechanism"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

with st.sidebar:

    radio = st.radio("Section", page_list)

    st.markdown("---")

func_list[page_dict[radio]]()
