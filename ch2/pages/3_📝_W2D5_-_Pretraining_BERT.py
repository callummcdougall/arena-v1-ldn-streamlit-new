import streamlit as st
import base64
st.set_page_config(layout="wide")
import os
if os.path.exists(os.getcwd() + "/images"):
    is_local = True
else:
    is_local = False
    os.chdir("./ch2")
st.write(is_local)
def img_to_html(img_path, width):
    with open("images/" + img_path, "rb") as file:
        img_bytes = file.read()
    encoded = base64.b64encode(img_bytes).decode()
    return f"<img style='width:{width}px;max-width:100%;margin-bottom:25px' src='data:image/png;base64,{encoded}' class='img-fluid'>"
def st_image(name, width):
    st.markdown(img_to_html(name, width=width), unsafe_allow_html=True)

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
    color: red;
    white-space: pre-wrap !important;
}
code:not(h1 code):not(h2 code):not(h3 code):not(h4 code) {
    font-size: 13px;
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
Today, we'll apply what we learned about BERT at the end of the transformers chapter, and the training at scale material from the current chapter, and train BERT from scratch on a GPU.

## 1️⃣ Wikitext

In this section, you'll load in your data for finetuning, and also run some of your own experiments.

## 2️⃣ Pretraining

In this section, you'll actually pretrain your BERT transformer.

## 3️⃣ Bonus exercises

To conclude, we've suggested a few bonus exercises if you've managed to get through the first two sections.
""")
 
def section_1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#introduction">Introduction</a></li>
   <li><a class="contents-el" href="#reading">Reading</a></li>
   <li><a class="contents-el" href="#data-preparation">Data Preparation</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#vocab-size">Vocab Size</a></li>
       <li><a class="contents-el" href="#context-length-experimentation">Context Length Experimentation</a></li>
       <li><a class="contents-el" href="#data-inspection">Data Inspection</a></li>
       <li><a class="contents-el" href="#use-of-zipfile-library">Use of zipfile library</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown("""
```python
import hashlib
import os
import sys
import zipfile
import torch as t
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import transformers
from einops import rearrange
from torch.nn import functional as F
from tqdm import tqdm
import requests
import utils

MAIN = __name__ == "__main__"
DATA_FOLDER = "./data"
DATASET = "2"
BASE_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/"
DATASETS = {"103": "wikitext-103-raw-v1.zip", "2": "wikitext-2-raw-v1.zip"}
TOKENS_FILENAME = os.path.join(DATA_FOLDER, f"wikitext_tokens_{DATASET}.pt")

if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)
```

## Introduction

Now we'll prepare text data to train a BERT from scratch! The largest BERT would require days of training and a large training set, so we're going to train a tiny BERT on a small training set: [WikiText](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/). This comes in a small version WikiText-2 (the 2 means 2 million tokens in the train set) and a medium version WikiText-103 with 103 million tokens. For the sake of fast feedback loops, we'll be using the small version but in the bonus you'll be able to use the medium version with the same code by changing the DATASET variable below. Both versions consist of text taken from Good and Featured articles on Wikipedia.

## Reading

* [BERT Paper](https://arxiv.org/pdf/1810.04805.pdf) - focus on the details of pretraining, found primarily in Section 3.1 and Appendix A.

## Data Preparation

Since we aren't using pretrained weights, we don't have to match the tokenizer like we did when fine-tuning. We're free to use any tokenization strategy we want.

### Vocab Size

For example, we could use a smaller vocabulary in order to save memory on the embedding weights. It's straightforward to train a new tokenizer, but for the sake of time we'll continue using the existing tokenizer and its vocabulary size.

```python
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
```

### Context Length

We're also free to use a shorter or longer context length, and this doesn't require training a new tokenizer. The only thing that this really affects is the positional embeddings. For a fixed compute budget, it's not obvious whether we should decrease the context length or increase it.

The computational cost of attention is quadratic in the context length, so decreasing it would allow us to use more compute elsewhere, or just finish training earlier. Increasing it would allow for longer range dependencies; for example, our model could learn that if a proper noun appears early in a Wikipedia article, it's likely to appear again.

The authors pretrain using a length of 128 for 90% of the steps, then use 512 for the rest of the steps. The idea is that early in training, the model is mostly just learning what tokens are more or less frequent, and isn't able to really take advantage of the longer context length until it has the basics down. Since our model is small, we'll do the simple thing to start: a constant context length of 128.

### Data Inspection

Run the below cell and inspect the text. It is one long string, so don't try to print the whole thing. What are some things you notice?""")

    with st.expander("Things to notice"):
        st.markdown("""
There is still some preprocessing done even though this is allegedly "raw" text. For example, there are spaces before and after every comma.

There are Japanese characters immediately at the start of the training set, which in a real application we might want to do something with depending on our downstream use case.

There is some markup at least for section headings. Again, this might be something we'd want to manually handle.""")

    st.markdown("""
### Use of zipfile library

It's important to know that the `zipfile` standard library module is written in pure Python, and while this makes it portable it is extremely slow as a result. It's fine here, but for larger datasets, definitely don't use it - it's better to launch a subprocess and use an appropriate decompression program for your system like `unzip` or `7-zip`.

```
path = os.path.join(DATA_FOLDER, DATASETS[DATASET])
maybe_download(BASE_URL + DATASETS[DATASET], path)
expected_hexdigest = {"103": "0ca3512bd7a238be4a63ce7b434f8935", "2": "f407a2d53283fc4a49bcff21bc5f3770"}
with open(path, "rb") as f:
    actual_hexdigest = hashlib.md5(f.read()).hexdigest()
    assert actual_hexdigest == expected_hexdigest[DATASET]

print(f"Using dataset WikiText-{DATASET} - options are 2 and 103")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

z = zipfile.ZipFile(path)

def decompress(*splits: str) -> str:
    return [
        z.read(f"wikitext-{DATASET}-raw/wiki.{split}.raw").decode("utf-8").splitlines()
        for split in splits
    ]

train_text, val_text, test_text = decompress("train", "valid", "test")
```

### Preprocessing

To prepare data for the next sentence prediction task, we would want to use a library like [spaCy](https://spacy.io/) to break the text into sentences - it's tricky to do this yourself in a robust way. We'll ignore this task and just do masked language modelling today.

Right now we have a list of lines, but we need (batch, seq) of tokens. We could use padding and truncation as before, but let's play with a different strategy:

- [Call the tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__) on the list of lines with `truncation=False` to obtain lists of tokens. These will be of varying length, and you'll notice some are empty due to blank lines.
- Build one large 1D tensor containing all the tokens in sequence
- Reshape the 1D tensor into (batch, sequence).

You should run your tokenizer on the entire `lines` object rather than using a for loop over `lines` (this is faster, for reasons related to the implementational details of the tokenizer). However, this removes the possibility of using a progress bar. A compromise would be to tokenize `lines` e.g. 1% at once.

Instead of padding, we'll just discard tokens at the very end that would form an incomplete sequence. This will only discard up to (max_seq - 1) tokens, so it's negligible.

This is nice because we won't waste any space or compute on padding tokens, and we don't have to truncate long lines. Some fraction of sequences will contain both the end of one article and the start of another, but this won't happen too often and there will be clues the model can use, like the markup for a heading appearing.

Note we won't need the attention mask, because we're not using padding. We'll also not need the `token_type_ids` or the special tokens CLS or SEP. You can pass arguments into the tokenizer to prevent it from returning these.

Don't shuffle the tokens here. This allows us to change the context length at load time, without having to re-run this preprocessing step.

You can ignore a warning about 'Token indices sequence length is longer than the specified maximum sequence length' - this is expected.

For WikiText-2 (the default setting here), this function should run pretty much immediately. Wikitext-103 might take a bit longer.

```python
def tokenize_1d(tokenizer, lines: list[str], max_seq: int) -> t.Tensor:
    '''Tokenize text and rearrange into chunks of the maximum length.

    Return (batch, seq) and an integer dtype.
    '''
    pass

if MAIN:
    max_seq = 128
    print("Tokenizing training text...")
    train_data = tokenize_1d(tokenizer, train_text, max_seq, 100)
    print("Training data shape is: ", train_data.shape)
    print("Tokenizing validation text...")
    val_data = tokenize_1d(tokenizer, val_text, max_seq)
    print("Tokenizing test text...")
    test_data = tokenize_1d(tokenizer, test_text, max_seq)
    print("Saving tokens to: ", TOKENS_FILENAME)
    t.save((train_data, val_data, test_data), TOKENS_FILENAME)
```

### Masking

Implement `random_mask`, which we'll call during training on each batch. This function should apply masking to the tokens based on the material in the [BERT Paper](https://arxiv.org/pdf/1810.04805.pdf) - specifically in Section 3.1 and Appendix A.

Note - it's not enough just to get the right mask in expectation (e.g. by constructing a mask directly from `t.rand() < threshold_value`), you should try to mask close to the exact right number of tokens. When sampling a random token, sample uniformly at random from [0..vocabulary size).

Make sure that any tensors that you create are on the same device as `input_ids` - you'll find this helpful later on.
""")

    with st.expander("Help - I'm not sure how to implement the indexing / I'm failing the test by a very small amount."):
        st.markdown("""
Probably the easiest way to approach this task is to construct boolean masks, and apply them using `t.where`.

Using `t.randperm` is probably the easiest way to construct your masks.
""")

    with st.expander("Is there anything special or optimal about the numbers 15%, 80%, and 10%?"):
        st.markdown("No, these are just some ad-hoc numbers that the BERT authors chose. The paper [Should You Mask 15% in Masked Language Modelling](https://arxiv.org/pdf/2202.08005.pdf) suggests that you can do better.")

    st.markdown("""
```python
def random_mask(
    input_ids: t.Tensor, mask_token_id: int, vocab_size: int, select_frac=0.15, mask_frac=0.8, random_frac=0.1
) -> tuple[t.Tensor, t.Tensor]:
    '''Given a batch of tokens, return a copy with tokens replaced according to Section 3.1 of the paper.

    input_ids: (batch, seq)

    Return: (model_input, was_selected) where:

    model_input: (batch, seq) - a new Tensor with the replacements made, suitable for passing to the BertLanguageModel. Don't modify the original tensor!

    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise
    '''
    pass

if MAIN:
    utils.test_random_mask(random_mask, input_size=10000, max_seq=max_seq)

```

### Loss Function

Exercise: what should the loss be if the model is predicting tokens uniformly at random? Use the [formula for discrete cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy), with $q$ as your estimates and $p$ as the true frequencies.
""")
    with st.expander("Solution - Random Cross-Entropy Loss for uniform"):
        st.markdown(r"""
Let $k$ be the vocabulary size. For each token to be predicted, the expected probability assigned by the model to the true token is $1/k$. Plugging this into the cross entropy formula gives an expected loss of $log(k)$, which for $k=28996$ is about 10.2.

Importantly, this is the loss per predicted token, and we have to decide how to aggregate these over the batch and sequence dimensions.

For a batch, we can aggregate the loss per token in any way we want, as long as we're clear and consistent about what we're doing. Taking the mean loss per predicted token has the nice property that we can compare models with a different number of predicted tokens per batch.""")

    st.markdown("""
Now, find the cross-entropy loss of the distribution of unigram frequencies. This is the loss you'd see when predicting words purely based on word frequency without the context of other words. During pretraining, your model should reach this loss very quickly, as it only needs to learn the final unembedding bias to predict this unigram frequency.

You should use `train_data` to calculate this entropy.

```python
if MAIN:
    "TODO: YOUR CODE HERE, TO CALCULATE CROSS ENTROPY OF UNIGRAM FREQUENCIES"
```""")

    with st.expander("Solution - Random Cross-Entropy Loss for unigrams"):
        st.markdown(r"""
```python
# Find the word frequencies
word_frequencies = t.bincount(train_data.flatten())
# Drop the words with occurrence zero (because these contribute zero to cross entropy)
word_frequencies = word_frequencies[word_frequencies > 0]
# Get probabilities
word_probabilities = word_frequencies / word_frequencies.sum()
# Calculate the cross entropy
cross_entropy = (- word_probabilities * word_probabilities.log()).sum()
print(cross_entropy)
# ==> 7.3446
```""")

    st.markdown("""

### Cross Entropy of MLM

For our loss function, we only want to sum up the loss at the tokens that were chosen with probability `select_frac`. As a reminder, when a token is selected, that input token could be replaced with either `[MASK]`, a random token, or left as-is and the target is the original, unmodified input token.

Write a wrapper around [torch.nn.functional.cross_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy) that only looks at the selected positions. It should output the total loss divided by the number of selected tokens.

`torch.nn.functional.cross_entropy` divides by the batch size by default, which means that the magnitude of the loss will be larger if there are more predictions made per batch element. We will want to divide by the number of tokens predicted: this ensures that we can interpret the resulting value and we can compare models with different sequence lengths.

```python
def cross_entropy_selected(pred: t.Tensor, target: t.Tensor, was_selected: t.Tensor) -> t.Tensor:
    '''
    pred: (batch, seq, vocab_size) - predictions from the model
    target: (batch, seq, ) - the original (not masked) input ids
    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise

    Out: the mean loss per predicted token
    '''
    pass

if MAIN:
    utils.test_cross_entropy_selected(cross_entropy_selected)

    batch_size = 8
    seq_length = 512
    batch = t.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
    pred = t.rand((batch_size, seq_length, tokenizer.vocab_size))
    (masked, was_selected) = random_mask(batch, tokenizer.mask_token_id, tokenizer.vocab_size)
    loss = cross_entropy_selected(pred, batch, was_selected).item()
    print(f"Random MLM loss on random tokens - does this make sense? {loss:.2f}")

```
""")

def section_2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#introduction">Introduction</a></li>
   <li><a class="contents-el" href="#data-&-configs">Data & configs</a></li>
   <li><a class="contents-el" href="#learning-rate-schedule">Learning Rate Schedule</a></li>
   <li><a class="contents-el" href="#weight-decay">Weight Decay</a></li>
   <li><a class="contents-el" href="#training-loop">Training Loop</a></li>
   <li><a class="contents-el" href="#model-evaluation">Model Evaluation</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown("""
## Introduction

Here, you'll write a training loop that loads the data you saved in part 1, and makes use of your implementations of `random_mask` and `cross_entropy_selected`.

We can't exactly follow the hyperparameters from section A.2 of the paper because they are training a much larger model on a much larger dataset using multiple GPUs. We might return to this later in the ARENA programme, if we find time to cover distributed computing.

Don't expect too much out of WikiText-2: deep learning is extremely sample inefficient, and we are orders of magnitude away from the amount of compute needed to pretrain something as good as the real BERT.

For comparison, the [Induction Heads paper](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#results-loss) observed induction heads forming around 2 billion tokens seen, which would require 1000 epochs on our 2 million token dataset. To train a model like PaLM, they used 780 billion tokens for one epoch.

Today, if your model can beat the baseline of just predicting the token frequencies, you can consider it a success. Training the model to do better than predicting token frequencies may take longer than you have available. For reference, to see whether your training run is off-track, here is a plot of the loss for a successful training run:
""")
    st_image("trainloss.png", 600)
    st.markdown("""
## Data & configs

Run the code block below.

You'll have to import the code from your BERT implementation. You may also have to change the paths in the code below.

```python
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

hidden_size = 512
bert_config_tiny = TransformerConfig(
    num_layers = 8,
    num_heads = hidden_size // 64,
    vocab_size = 28996,
    hidden_size = hidden_size,
    max_seq_len = 128,
    dropout = 0.1,
    layer_norm_epsilon = 1e-12
)

config_dict = dict(
    lr=0.0002,
    epochs=40,
    batch_size=128,
    weight_decay=0.01,
    mask_token_id=tokenizer.mask_token_id,
    warmup_step_frac=0.01,
    eps=1e-06,
    max_grad_norm=None,
)

(train_data, val_data, test_data) = t.load("./data/wikitext_tokens_103.pt")
print("Training data size: ", train_data.shape)

train_loader = DataLoader(
    TensorDataset(train_data), shuffle=True, batch_size=config_dict["batch_size"], drop_last=True
)
```

## Learning Rate Schedule

The authors used learning rate warmup from an unspecified value and an unspecified shape to a maximum of 1e-4 for the first 10,000 steps out of 1 million, and then linearly decayed to an unspecified value.

It's very common that authors will leave details out of the paper in the interests of space, and the only way to figure it out is hope that they published source code. The source code doesn't always match the actual experimental results, but it's the best you can do other than trying to contact the authors.

From the repo, we can see in [optimization.py](https://github.com/google-research/bert/blob/master/optimization.py) that AdamW is used for the optimizer, that the warmup is linear and that the epsilon used for AdamW is 1e-6.

Assume that the initial learning rate and the final learning rate are both 1/10th of the maximum, and that we want to warm-up for 1% of the total number of steps.
""")

    with st.expander("Click to see the expected LR Schedule."):
        st_image("lr_schedule.png", 700)

    st.markdown(r"""
```python
def lr_for_step(step: int, max_step: int, max_lr: float, warmup_step_frac: float):
    '''Return the learning rate for use at this step of training.'''
    pass


if MAIN:
    max_step = int(len(train_loader) * config_dict["epochs"])
    lrs = [
        lr_for_step(step, max_step, max_lr=config_dict["lr"], warmup_step_frac=config_dict["warmup_step_frac"])
        for step in range(max_step)
    ]
    # TODO: YOUR CODE HERE, PLOT `lrs` AND CHECK IT RESEMBLES THE GRAPH ABOVE
```

## Weight Decay

The BERT paper specifies that a "L2 weight decay" of 0.01 is used, but leaves unspecified exactly which parameters have weight decay applied. Recall that weight decay is an inductive bias, which in the case of linear regression is exactly equivalent to a prior that each weight is Gaussian distributed.

For modern deep learning models, weight decay is much harder to analyze, having interactions with adaptive learning rate methods and normalization layers. Papers on weight decay feature phrases like "The effect of weight decay remains poorly understood" [1](https://arxiv.org/pdf/1810.12281.pdf) or "despite its ubiquity, its behavior is still an area of active research" [2](https://www.cs.cornell.edu/gomes/pdf/2021_bjorck_aaai_wd.pdf), and you'll see different implementations do different things.

Today we're going to use weight decay conservatively, and only apply it to the weight (and not the bias) of each `Linear` layer. I didn't find much effect, but it provides an opportunity to learn how to use [parameter groups](https://pytorch.org/docs/stable/optim.html#per-parameter-options) in the optimizer.

```python
def make_optimizer(model: BertLanguageModel, config_dict: dict) -> t.optim.AdamW:
    '''
    Loop over model parameters and form two parameter groups:

    - The first group includes the weights of each Linear layer and uses the weight decay in config_dict
    - The second has all other parameters and uses weight decay of 0
    '''
    pass

if MAIN:
    test_config = TransformerConfig(
        num_layers = 3,
        num_heads = 1,
        vocab_size = 28996,
        hidden_size = 1,
        max_seq_len = 4,
        dropout = 0.1,
        layer_norm_epsilon = 1e-12,
    )

    optimizer_test_model = BertLanguageModel(test_config)
    opt = make_optimizer(
        optimizer_test_model, 
        dict(weight_decay=0.1, lr=0.0001, eps=1e-06)
    )
    expected_num_with_weight_decay = test_config.num_layers * 6 + 1
    wd_group = opt.param_groups[0]
    actual = len(wd_group["params"])
    assert (
        actual == expected_num_with_weight_decay
    ), f"Expected 6 linear weights per layer (4 attn, 2 MLP) plus the final lm_linear weight to have weight decay, got {actual}"
    all_params = set()
    for group in opt.param_groups:
        all_params.update(group["params"])
    assert all_params == set(optimizer_test_model.parameters()), "Not all parameters were passed to optimizer!"
```

## Training Loop

Write your training loop here, logging to Weights and Biases. Tips:

- Log your training loss and sanity check that it looks reasonable. At initialization it should be around the value of random prediction; if it's much higher, then you probably have a bug in your weight initialization code.
- By 10M tokens, if your model hasn't basically figured out the unigram frequencies then you probably have a bug, or you changed your hyperparameters to bad ones.
- Log your learning rate as well, as it's easy to not apply the learning rate schedule properly.
- If you use gradient clipping, note that `clip_grad_norm_` returns the norm of the gradients. It's useful to know if your clipping is actually doing anything.


```python
def bert_mlm_pretrain(model: BertLanguageModel, config_dict: dict, train_loader: DataLoader) -> None:
    '''Train using masked language modelling.'''
    pass


if MAIN:
    model = BertLanguageModel(bert_config_tiny)
    num_params = sum((p.nelement() for p in model.parameters()))
    print("Number of model parameters: ", num_params)
    bert_mlm_pretrain(model, config_dict, train_loader)
```

## Model Evaluation

You can test the model's predictions, but they're going to be underwhelming given our limited computational budget.


```python
if MAIN:
    model = BertLanguageModel(bert_config_tiny)
    model.load_state_dict(t.load(config_dict["filename"]))
    your_text = "The Answer to the Ultimate Question of Life, The Universe, and Everything is [MASK]."
    predictions = predict(model, tokenizer, your_text)
    print("Model predicted: \n", "\n".join(map(str, predictions)))
```
""")

def section_3():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#context-length-experimentation">Context Length Experimentation</a></li>
   <li><a class="contents-el" href="#whole-word-masking">Whole Word Masking</a></li>
   <li><a class="contents-el" href="#improved-versions-of-bert">Improved Versions of BERT</a></li>
   <li><a class="contents-el" href="#next-sentence-prediction">Next Sentence Prediction</a></li>
   <li><a class="contents-el" href="#applying-scaling-laws">Applying Scaling Laws</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
## Context Length Experimentation

We used a fixed context length of 128 in the exercises.

Play with a shorter context length and observe the difference in training speed. Does it decrease performance, or was the small model unable to make much use of the longer context length anyway?

In section A2, the [BERT paper](https://arxiv.org/pdf/1810.04805.pdf) suggestes pretraining with a sequence length of 128 for 90% of the steps, then 512 for the remaining steps. Try this and see what happens. Why do you think it might improve performance, relative to other possible strategies (e.g. training with seq length of 512 for only the first 10%)?

## Whole Word Masking

The official BERT repo has a README section on a different way of computing the mask, called **Whole Word Masking**. Rather than masking tokens randomly like this:

```
Input Text: We demand rigid ##ly defined areas of doubt and uncertainty

Original Masked Input: We demand [MASK] ##ly defined [MASK] of doubt and uncertainty
```

you might instead mask like this:

```
Whole Word Masked Input: We demand [MASK] [MASK] defined areas of doubt and uncertainty
```

while keeping the total proportions of masked tokens the same.

Try implementing this masking method, and see if you get any benefit. Why do you think you might expect to see benefit from this?

## Improved Versions of BERT

Read about one of the improved versions of BERT and try to replicate it.

[DistilBERT](https://arxiv.org/abs/1910.01108v4) would be a good one to try. It is 40% smaller than the original BERT-base model, is 60% faster than it, and retains 97% of its functionality. It is trained using a teacher-student model called **knowledge distillation**, It also uses a more complicated loss function combining language modeling, distillation and cosine-distance losses.""")
    st_image("distilbert.png", 600)
    st.markdown("""
If you're feeling especially ambitious, two other models you could try are:

- [ELECTRA](https://arxiv.org/pdf/2003.10555.pdf%27)
- [DEBERTA](https://arxiv.org/pdf/2006.03654.pdf)

## Next Sentence Prediction

Try to pretrain on the next sentence prediction task.

You can see [this blog post](https://towardsdatascience.com/bert-for-next-sentence-prediction-466b67f8226f) for an idea of how NSP works in practice, when you're performing inference with BERT.

## Applying Scaling Laws

The paper [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf) attempts to find the optimal model and dataset sizes for a given amount of compute. If you were to extrapolate their curves down to our very limited amount of compute, what does the paper suggest would be the ideal model and dataset size?

Note - we will be exploring this more during the chapter on scaling laws, later on in the course.
""")

func_list = [section_home, section_1, section_2, section_3]

page_list = ["🏠 Home", "1️⃣ Wikitext", "2️⃣ Pretraining BERT", "3️⃣ Bonus exercises"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:

        radio = st.radio("Section", page_list)

        st.markdown("---")

    func_list[page_dict[radio]]()

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if is_local or check_password():
    page()
