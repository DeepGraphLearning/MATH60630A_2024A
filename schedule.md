---
layout: default
title: MATH 60630A - Machine Learning II<br>Deep Learning
permalink: /schedule

schedule:
  - date: Aug.<br>26, 30
    topics:
      - name: Introduction
        url: assets/slides/Week1-Intro.pdf
      - name: Mathematics
        url: assets/slides/Week1-Maths.pdf
      - name: Machine Learning Basics
        url: assets/slides/Week1-ML.pdf
    readings:
      - name: Deep Learning Book
      - name: Chap. 2
        url: http://www.deeplearningbook.org/contents/linear_algebra.html
      - name: Chap. 3
        url: http://www.deeplearningbook.org/contents/prob.html
      - name: Chap. 5
        url: http://www.deeplearningbook.org/contents/ml.html
  - date: Sep.<br>9, 6
    topics:
      - name: Feedforward Neural Networks & Optimization Tricks
    #      url: https://www.dropbox.com/s/zv4920r75ek7u4u/Week2-FFN%26Regularization.pdf?dl=0
    readings:
      - name: Deep Learning Book
      - name: Chap. 6
        url: http://www.deeplearningbook.org/contents/mlp.html
      - name: Chap. 7
        url: http://www.deeplearningbook.org/contents/regularization.html
      - name: Chap. 8
        url: http://www.deeplearningbook.org/contents/optimization.html
  - date: Sep.<br>16, 13
    topics:
      - name: PyTorch part 1
      #        url: https://www.dropbox.com/s/xpd4fjisk3n08vx/Deep%20Learning%20Frameworks%20part%201.pdf?dl=0
      - name: PyTorch Part 2
    #        url: https://www.dropbox.com/s/2mzbdnfgah9yimw/Deep%20Learning%20Frameworks%20part%202.pdf?dl=0
    readings:
      - name: Python Numpy Tutorial
        url: http://cs231n.github.io/python-numpy-tutorial/
      - name: Pytorch's official tutorial
        url: https://pytorch.org/tutorials/beginner/basics/intro.html
      - name: Neural Network from Scratch
        url: https://medium.com/dair-ai/a-simple-neural-network-from-scratch-with-pytorch-and-google-colab-c7f3830618e0
      - name: Dive into Deep Learning
        url: https://github.com/dsgiitr/d2l-pytorch
    homeworks:
      - name: HW1 (to be announced)
  #      - name: Instruction
  #        url: https://www.dropbox.com/s/7s19xya7yck4nul/HWK1.pdf?dl=0
  #      - name: Colab
  #        url: https://colab.research.google.com/drive/1FjRjNlBqPVz7SrEPvqrL10Q76NeHhvJW?usp=sharing

  # <<<<<<<<<<<<<<< Jianan >>>>>>>>>>>>>>>>>>

  - date: Sep.<br>23, 20
    topics:
      - name: Convolutional Neural Networks & Recurrent Neural Networks
    #        url: https://www.dropbox.com/s/9vnrcjo4ykhdj9l/Week4-CNN%26RNN.pdf?dl=0
    readings:
      - name: Deep Learning Book
      - name: Chap. 9
        url: http://www.deeplearningbook.org/contents/convnets.html
      - name: Chap. 10
        url: http://www.deeplearningbook.org/contents/rnn.html
    references:
      - name: ResNet
        url: http://arxiv.org/abs/1512.03385
      - name: GRU
        url: https://arxiv.org/abs/1412.3555
      - name: DenseNet
        url: https://arxiv.org/abs/1608.06993
  - date: Oct.<br>1, 27
    topics:
      - name: NLP Basis
    #        url: https://www.dropbox.com/s/bjxvre1d3w30iqj/Week5-DL4NLP-part1.pdf?dl=0
    readings:
      - name: Word2Vec
        url: https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    references:
      - name: GloVe
        url: https://aclanthology.org/D14-1162.pdf
      - name: SGNS
        url: https://papers.nips.cc/paper/2014/file/feab05aa91085b7a8012516bc3533958-Paper.pdf
  - date: Oct.<br>7, 4
    topics:
      - name: Attention, Transformers
    #        url: https://www.dropbox.com/s/bjxvre1d3w30iqj/Week5-DL4NLP-part1.pdf?dl=0
    readings:
      - name: The annotated Transformer (blog)
        url: https://nlp.seas.harvard.edu/annotated-transformer/
      - name: An introduction of Positional Encoding
        url: https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    references:
      - name: Transformer
        url: https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
      - name: ViT
        url: https://arxiv.org/abs/2010.11929
      - name: Rotary Position Embedding
        url: https://arxiv.org/abs/2104.09864
      - name: FlashAttention
        url: https://arxiv.org/abs/2205.14135
  - date: Oct.<br>16, 11
    topics:
      - name: Introduction to Huggingface & Kaggle Challenge (Notebook)
    readings:
      - name: Training a causal language model from scratch (practical guides)
        url: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    references:
      - name: transfer learning (T5)
        url: https://arxiv.org/abs/1910.10683
      - name: auto-regressive or causal language model (GPT-2)
        url: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
      - name: Transformers library
        url: https://huggingface.co/docs/transformers/index
      - name: Accelerate library
        url: https://huggingface.co/docs/accelerate/index
    homeworks:
      - name: HW2 (to be announced)
  #        url: https://www.dropbox.com/s/j2w4cpq14jypkbe/HW2.pdf?dl=0
  #      - name: Instruction
  #        url: https://www.dropbox.com/s/j2w4cpq14jypkbe/HW2.pdf?dl=0
  #      - name: Kaggle
  #        url: https://www.kaggle.com/c/math60630aw21
  - date: Oct.<br>28, 25
    topics:
      - name: Large Language Models I - Pre-training and Fine-tuning
    #        url: https://www.dropbox.com/s/366364m5gmu6gkd/Week7-DL4NLP-part2.pdf?dl=0
    readings:
      - name: Survey of Pre-trained LMs
        url: https://arxiv.org/pdf/2302.09419
      - name: LLM Course
        url: https://github.com/mlabonne/llm-course
      - name: GPT in 60 Lines of NumPy
        url: https://jaykmody.com/blog/gpt-from-scratch/
      - name: Fine-Tune Your Own Llama 2 Model in a Colab Notebook
        url: https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html
    references:
      - name: BERT
        url: https://arxiv.org/pdf/1810.04805
      - name: GPT-3
        url: https://arxiv.org/abs/2005.14165
      - name: T5
        url: https://arxiv.org/abs/1910.10683
      - name: Instruction Tuning
        url: https://arxiv.org/abs/2109.01652
      - name: LoRA
        url: https://arxiv.org/abs/2104.09864
      - name: Scaling Law
        url: https://arxiv.org/abs/2001.08361

  - date: Nov.<br>04, 01
    topics:
      - name: Large Language Models II - Prompt Tuning
    #        url: https://www.dropbox.com/s/gcd1bu7bxd5gigm/Week8-DL4NLP-part3.pptx?dl=0
    readings:
      - name: Prompt Engineering (Blog by Lilian Weng)
        url: https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
    references:
      - name: Chain-of-Thought
        url: https://arxiv.org/pdf/2201.11903
      - name: Self Consistency
        url: https://arxiv.org/pdf/2203.11171
      - name: ReAct
        url: https://arxiv.org/pdf/2210.03629
      - name: Tree of Thoughts
        url: https://arxiv.org/abs/2305.10601
  - date: Nov.<br>11, 08
    topics:
      - name: Generative Models I
    #        url: https://www.dropbox.com/s/nf4ohrqjqg7rb66/Week10-Graph-part2.pdf?dl=0
    readings:
      - name: Evidence Lower Bound ELBO — What & Why
        url: https://yunfanj.com/blog/2021/01/11/ELBO.html
      - name: From Autoencoder to Beta-VAE
        url: https://lilianweng.github.io/posts/2018-08-12-vae/
      - name: From GAN to WGAN
        url: https://lilianweng.github.io/posts/2017-08-20-gan/
      - name: Understanding Failure Modes of GAN Training
        url: https://medium.com/game-of-bits/understanding-failure-modes-of-gan-training-eae62dbcf1dd
    references:
      - name: VAE
        url: https://arxiv.org/abs/1312.6114
      - name: GAN
        url: https://arxiv.org/abs/1406.2661
  - date: Nov.<br>18, 15
    topics:
      - name: Diffusion, text-to-image generation
    #        url: https://www.dropbox.com/s/nf4ohrqjqg7rb66/Week10-Graph-part2.pdf?dl=0
    readings:
      - name: Hands-on Notebooks "Diffusion Models from Scratch"
        url: https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb
      - name: How to Use DALL-E 3 Tips, Examples, and Features
        url: https://www.datacamp.com/tutorial/an-introduction-to-dalle3
      - name: Sora
        url: https://openai.com/index/sora/
      - name: Kling
        url: https://kling.kuaishou.com/en
    references:
      - name: Diffusion Probabilistic Model
        url: https://arxiv.org/abs/2006.11239
      - name: DALL·E3
        url: https://cdn.openai.com/papers/dall-e-3.pdf
      - name: Sora
        url: https://arxiv.org/pdf/2402.17177
  - date: Nov.<br>25, 22
    topics:
      - name: Graph Representation Learning
    #        url: https://www.dropbox.com/s/3e09x5i9wyn8q3c/Week9-Graph-part1.pdf?dl=0
    readings:
      - name: Design Space for Graph Neural Networks
        url: https://arxiv.org/pdf/2011.08843
      - name: A Comprehensive Survey on Graph Neural Networks
        url: https://arxiv.org/abs/1901.00596
      - name: Graph Neural Networks Implementation Tutorial 
        url: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html
    references:
      - name: DeepWalk
        url: https://arxiv.org/pdf/1403.6652
      - name: LINE
        url: https://arxiv.org/pdf/1503.03578
      - name: GCN
        url: https://arxiv.org/pdf/1609.02907
#      - name: Bayesian Personalized Ranking
#        url: https://arxiv.org/abs/1205.2618
#      - name: Factorization Machines
#        url: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
---

# Course Schedule
<table>
  <tr>
    <th>Date</th>
    <th>Topic</th>
    <th style="width: 45%;">Suggested Readings</th>
    <th style="width: 35%;">Reference</th>
    <th>Homework</th>
  </tr>
  {% for week in page.schedule %}
    <tr>
      <td>{{ week.date }}</td>
      <td>
      {% for topic in week.topics %}
        {% include href item=topic %}<br>
      {% endfor %}
      </td>
      <td style="width: 45%;">
      {% for reading in week.readings %}
        {% include href item=reading %}<br>
      {% endfor %}
      </td>
      <td style="width: 35%;">
      {% for presentation in week.references %}
        {% include href item=presentation %}<br>
      {% endfor %}
      </td>
      <td>
      {% for homework in week.homeworks %}
        {% include href item=homework %}<br>
      {% endfor %}
      </td>
    </tr>
  {% endfor %}
</table>
