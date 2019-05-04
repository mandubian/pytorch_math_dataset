# Mathematical Dataset for Pytorch

> You can contact me on twitter as [@mandubian](http://twitter.com/mandubian)

First of all, the current project provides a few Numpy/Pytorch helpers to help playing with Mathematical Reasoning Dataset in Pytorch based on this very cool paper:

> Analysing Mathematical Reasoning Abilities of Neural Models
>
> David Saxton, Edward Grefenstette, Felix Hill, Pushmeet Kohli
>
> (Submitted on 2 Apr 2019)
>
>http://arxiv.org/abs/1904.01557

The main purpose of the authors is to provide a playground to people who want to study how neural networks can learn to solve mathematical problems and maybe learn math abstractions. 

My aim is to explore if models can unravel math laws by themselves... or not :D

The dataset is not extraordinary huge but in v1.0, it contains 10s of millions of questions/answers and despite the small size of each pair, millions of them will consume all your RAM. So, those little helpers try to manage data in a streamed and lazy-loading way (as much as Python allows it) and then allow to mix different parts of the dataset to explore multiple ideas.

The idea of the paper is to provide a robust toolkit (https://github.com/deepmind/mathematics_dataset) to randomly and heterogenously generate mathematical datasets among multiple math problem categories:

    - algebra
    - numbers
    - polynomials
    - arithmetic
    - measurement
    - comparison
    - probability
    - calculus
    
All problem are constituted of a textual question like `what is 30 + 535?` (max 160 chars) and a textual response `565` (max 30 chars). So it's not an abstract representation of math problem but a very human one and it mixes Natural Language Processing with math concepts.

For each of those categories, it provides multiple operation modules. For ex, in `algebra` category:

    - mul
    - add_or_sub_in_base
    - simplify_surd
    - mul_div_multiple
    - mixed
    - nearest_integer_root
    - div
    - add_or_sub
    - add_sub_multiple
    - add_sub_multiple_longer
    - mul_div_multiple_longer
    - div_big
    - mul_big
    - mixed_longer
    - add_or_sub_big

Problem can be generated with different difficulties:

    - train-easy
    - train-medium
    - train-hard

It also provides test datasets for `interpolation` tests mixing all kinds of problem per category and `extrapolation` tests to measure generalization capabilities of models.

If you want to check the different features provided in this github and some samples, you can check:

### Original Transformer experiments

- [math_ds_train.ipynb](./math_ds_train.ipynb) showing how to use MathDatasetManager to build custom math datasets from full dataset and how to train a Transformer model on it.

- [math_ds_predict.ipynb](./math_ds_predict.ipynb) playing with a pre-trained transformer model on arithmetic addition/subtraction in easy or hard difficulty

### Transformer implemented as a DGL Graph Neural Network experiments

- [math_ds_dgl_transformer.ipynb](./math_ds_dgl_transformer.ipynb) demonstrates the implementation of a Transformer as a DGL Graph Neural Network with some considerations on the nature of Transformer seen as Graph Neural Network. Then we train this DGL-Transformer on a sub-sample of Mathematical Reasoning Dataset to check preliminary results.


If you want to use current code (and maybe contribute), you should:

1. Clone this repository

2. Retrieve v1.0 of math dataset at https://github.com/deepmind/mathematics_dataset (or generate your own dataset with code provided there)

3. Play with notebooks listed above


> All code is licensed under Apache 2.0 License