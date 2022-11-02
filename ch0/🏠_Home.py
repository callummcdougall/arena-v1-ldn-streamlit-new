import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
<style>
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
    color:red;
    white-space: pre-wrap !important;
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

st.markdown(r"""
# Further investigations

Hopefully, last week you were able to successfully implement a transformer last week (or get pretty close!). If you haven't done that yet, then this should be your first priority going forwards with this week. **If you are struggling with getting your transformer to work, please send me (Callum) a link to your GitHub repo and I will be able to help troubleshoot.**

The rest of this week will involve continuing to iterate on your transformer architecture, as well as doing some more experiments with transformers. In the following pages, we'll provide a few suggested exercises. These range from highly open-ended (with no testing functions or code template provided) to highly structured (in the style of last week's exercises). 

All of the material here is optional, so you can feel free to do whichever exercises you want - or just go back over the transformers material that we've covered so far. **You should only implement them once you've done last week's tasks (in particular, building a transformer and training it on the Shakespeare corpus). 

Below, you can find a description of each of the set of exercises on offer. You can do them in any order, as long as you make sure to do exercises 1 and 2 at some point. Note that you can do e.g. 3B before 3A, but this is not advised since you'd have to import the solution from 3A and work with it, possibly without fully understanding the architecture.

---

### 1. Build and sample from GPT-2

As was mentioned in yesterday's exercises, you've already built something that was very close to GPT-2. In this task, you'll be required to implement an exact copy of GPT-2, and load in the weights just like you did last week for ResNet. Just like last week, this might get quite fiddly!

We will also extend last week's work by looking at some more advanced sampling methods, such as **beam search**.

### 2. Use your own modules

In week 0 we built a ResNet using only our own modules, which all inherited from `nn.Module`. Here, you have a chance to take this further, and build an entire transformer using only your own modules! This starts by having you define a few new blocks (`LayerNorm`, `Embedding` and `Dropout`), and then put them all together into your own transformer, which you can train on the Shakespeare corpus as in the task from last week.

### 3. Build and finetune BERT

BERT is an encoder-only transformer, which has a different kind of architecture (and a different purpose) than GPT-2. In this task, you'll build a copy of BERT and load in weights, then train it on 

Once you've built BERT, you'll be able to train it to perform well on tasks like classification and sentiment analysis.

Test.

""")
