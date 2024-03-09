# NLP Explainability Techniques

This repository is the code source of the paper: NLP Explainability Techniques in 2024. 

In 2020, Danilevsky et al introduced [A Survey of the State of Explainable AI for Natural Language Processing](https://arxiv.org/pdf/2010.00711.pdf) where they explained, among others, the explainability techniques used in NLP. 4 years later, we propose we evaluate some of the tools and methods they introduce, as well as propose some additional approaches.

The five explainability techniques we review are:
1. Feature Importance
2. Surrogate Model
3. Example Driven
4. Provenance Based
5. Declarative Induction

We introduce 5 notebooks where the user can play around and test the techniques and use the code to understand their models.

### Data

The data used in the experiments is the [poem_sentiment](https://huggingface.co/datasets/poem_sentiment) dataset from Hugging Face. It is a dataset that contains verses of poems and classifies them according to their sentiment nature:
* 0: for negative.
* 1: for positive.
* 2: for neutral.
* 3: for mixed feelings.

Since the dataset is quite limited, we have merged the train+test data to make the training set bigger, preserved the validation set and created our own test dataset which consists of a Taylor Swift song called ["All too well (10 Minute Verion)(Taylor's Version)(From the Vault)](https://www.youtube.com/watch?v=sRxrwjOtIag)" and labelled it manually.
Since our experiments do not require a big amount of data, these has been enough to test our techniques.

### Models

The models used differ depending on the explainability technique we are evaluating, since not all techniques are suitable for neural networks (this is the last technique: declarative induction). For this reason, we have used:

* A finetuned `google-bert/bert-base-cased` model with out `poem_sentiment data` (finetuning code in the file `src/finetune.py`).
* A Random Forest from sklearn.

## 🚀 Getting started

To be able to reproduce all the results and visualizations shown in this repository, you need to install the requirements:

```console
pip install -r requirements.txt
```

## To DO:

- [ ] Acabar techniques
  - [ ] Feature Importance
  - [ ] Surrogate Model
  - [ ] Example Driven
  - [ ] Provenance Based
  - [ ] Declarative Induction (falten conclusions)
  - [ ] Estructura techniques: Intro, experiments, conclusions
- [ ] Cleaning
  - [ ] Linting
  - [ ] Requirements
  - [ ] ReadMe (Getting started)
- [ ] Add paper later :D
- [ ] Add poster
