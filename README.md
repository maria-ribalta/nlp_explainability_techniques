# NLP Explainability Techniques

This repository is the code source of the paper: NLP Explainability Techniques. 

In 2020, Danilevsky et al introduced [A Survey of the State of Explainable AI for Natural Language Processing](https://arxiv.org/pdf/2010.00711.pdf) where they explained, among others, the explainability techniques used in NLP. 

4 years later, we evaluate some of the tools and methods they introduce, as well as propose some additional approaches that have appeared since the publication of the paper in 2020.

The five explainability techniques we review are:
1. [Feature Importance](nlp_explainability_techniques/1_Feature_importance.ipynb)
2. [Surrogate Model](nlp_explainability_techniques/2_Surrogate_model.ipynb)
3. [Example Driven](nlp_explainability_techniques/3_Example_driven.ipynb)
4. [Provenance Based](nlp_explainability_techniques/4_Provenance_based.ipynb)
5. [Declarative Induction](nlp_explainability_techniques/5_Declarative_induction.ipynb)

We introduce 5 notebooks where the user can play around and test the techniques and use the code to understand their models.

### Data

The data used in the experiments is the [poem_sentiment](https://huggingface.co/datasets/poem_sentiment) dataset from Hugging Face. It is a dataset that contains verses of poems and classifies them according to their sentiment nature:
* 0: for negative.
* 1: for positive.
* 2: for neutral.
* 3: for mixed feelings.

Since the dataset is quite limited, we have merged the train+test data to make the training set bigger, preserved the validation set and created our own test dataset which consists of a Taylor Swift song called ["All too well (10 Minute Version)(Taylor's Version)(From the Vault)](https://www.youtube.com/watch?v=sRxrwjOtIag)" and labelled it manually.
Since our experiments do not require a big amount of data, these has been enough to test our techniques.

To load the data, it is enough with doing:

```python
from src.preprocess import get_train_dev_test_data

train, dev, test = get_train_dev_test_data()
```

### Models

The models used differ depending on the explainability technique we are evaluating, since not all techniques are suitable for neural networks (this is the last technique: declarative induction). For this reason, we have used:

* A finetuned `google-bert/bert-base-cased` model with `poem_sentiment` data (finetuning code in the file `src/finetune.py`).
* The `meta-llama/Llama-2-7b-chat-hf` model.
* A Random Forest from sklearn.

To finetune the model with our training data, one must simply run in the console:

```console
python src/finetune.py
```
The finetuned model will be saved in the folder `pretrained_model` and to use it, one simply loads the Bert tokenizer and the model like this:

```python
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "pretrained_model/", num_labels=4
)
```

## 🚀 Getting started

To be able to reproduce all the results and visualizations shown in this repository, you need to install the requirements:

```console
pip install -r requirements.txt
```

