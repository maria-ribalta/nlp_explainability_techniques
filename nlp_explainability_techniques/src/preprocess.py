from datasets import load_dataset
from datasets import concatenate_datasets


def get_train_dev_test_data():
    poem_dataset = load_dataset("poem_sentiment").rename_column("label", "labels")

    train_dataset = concatenate_datasets(
        [poem_dataset["train"], poem_dataset["test"]]
    )  # join train and test from poem_sentiment because we use all_too_well.csv as test
    validation_dataset = poem_dataset["validation"]  # keep the same validation
    test_dataset = load_dataset(
        "csv", data_files="data/all_too_well.csv"  # load our own test dataset annotated
    )["train"]

    return train_dataset, validation_dataset, test_dataset


def tokenize(data, tokenizer):
    def tokenize_function(sample):
        return tokenizer(sample["verse_text"], padding="max_length", truncation=True)

    return data.map(tokenize_function, batched=True)
