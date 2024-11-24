import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import trange
from collections.abc import Sequence

import langcodes
from sklearn.metrics import confusion_matrix
import fasttext
from lingua import LanguageDetectorBuilder
from huggingface_hub import hf_hub_download


def replace_lang_code(model_predictions: list[str], mapping: dict[str, str]) -> None:
    """
    Replace predicted language codes in the `model_predictions` list
    using `mapping`, where:
      - Keys represent the original language codes (predicted by the model)
      - Values represent the target language codes to replace them with.

    The purpose of this function is to standardize language codes
    by combining multiple variants of the same language into a unified code
    in order to match supported languages.
    """
    for i in trange(len(model_predictions)):
        if model_predictions[i] in mapping:
            model_predictions[i] = mapping[model_predictions[i]]


def get_fasttext_predictions(texts: list[str]) -> list[str]:
    """
    Detect the languages of the given texts using FastText model.

    Args:
        texts: A sequence of texts for which the language detection
               needs to be performed.

    Returns:
        A list of predicted ISO 639-1 language codes.
    """
    # texts preprocessing
    table = str.maketrans({"\n": " ", "\r": " "})
    texts = [text.translate(table).lower() for text in texts]

    # load model
    fasttext_model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    fasttext_model = fasttext.load_model(fasttext_model_path)

    # get predictions
    predictions, _ = fasttext_model.predict(texts)
    predicted_lang = [predictions[i][0].removeprefix("__label__") for i in range(len(texts))]
    # get iso 639-3
    fasttext_preds = [predicted_lang[i].split("_")[0] for i in range(len(predicted_lang))]
    # get iso 639-1
    fasttext_preds = [langcodes.standardize_tag(lang_code, macro=True) for lang_code in fasttext_preds]
    return fasttext_preds


def get_lingua_predictions(texts: Sequence[str]) -> list[str]:
    """
    Detect the languages of the given texts using the Lingua language detector.

    Args:
        texts: A sequence of texts for which the language detection
               needs to be performed.

    Returns:
        A list of predicted ISO 639-1 language codes.
        If the model is uncertain about the language of a text, "xx" is returned
        as a placeholder.
    """
    texts = [text.lower() for text in texts]
    lingua_detector = LanguageDetectorBuilder.from_all_languages().build()

    predictions = lingua_detector.detect_languages_in_parallel_of(texts)

    lingua_preds = [prediction.iso_code_639_1.name.lower() if prediction is not None else "xx" 
                    for prediction in predictions]
    return lingua_preds


def calculate_metrics(cm: np.ndarray, labels: Sequence, model_name: str) -> pd.DataFrame:
    """
    Calculate precision, recall and f1-score.
    Args:
        cm: confusion matrix
        labels: languages, for which the metrics need to be calculated
        model_name: model name (needed for column names in DataFrame)

    Returns: pandas.DataFrame with computed metrics for each language.
    """
    tp_and_fn = cm.sum(axis=1)
    tp_and_fp = cm.sum(axis=0)
    tp = cm.diagonal()

    precision = np.divide(tp, tp_and_fp, out=np.zeros_like(tp, dtype=float), where=tp_and_fp > 0)
    recall = np.divide(tp, tp_and_fn, out=np.zeros_like(tp, dtype=float), where=tp_and_fn > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision, dtype=float),
                   where=(precision + recall) > 0)

    df = pd.DataFrame({
        "lang": labels,
        f"{model_name}_precision": precision,
        f"{model_name}_recall": recall,
        f"{model_name}_f1": f1,
    })

    return df


def main():
    texts_under_10_words = pd.read_csv(os.path.join(Path(__file__).parent, "texts_under_10_words.csv"))
    texts = texts_under_10_words.ingredients_text.tolist()
    true_labels = texts_under_10_words.lang.tolist()
    possible_class_labels = texts_under_10_words["lang"].value_counts().index.tolist()  # use value_counts in order to get sorted by frequency list

    fasttext_preds = get_fasttext_predictions(texts)
    lingua_preds = get_lingua_predictions(texts)

    mapping = {"yue": "zh"}  # yue is a type of Chinese
    replace_lang_code(fasttext_preds, mapping)
    replace_lang_code(lingua_preds, mapping)

    predictions = [fasttext_preds, lingua_preds]
    model_names = ["fasttext", "lingua"]
    metrics = []
    for preds, model_name in zip(predictions, model_names):
        cm = confusion_matrix(true_labels, preds, labels=possible_class_labels)
        cm_df = pd.DataFrame(cm, index=possible_class_labels, columns=possible_class_labels)
        cm_df.to_csv(os.path.join(Path(__file__).parent, f"{model_name}_confusion_matrix.csv"))

        metrics_df = calculate_metrics(cm, possible_class_labels, model_name)
        metrics_df.set_index("lang", inplace=True)
        metrics.append(metrics_df)

    # combine results
    metrics_df = pd.DataFrame(texts_under_10_words.lang.value_counts())
    metrics_df = pd.concat((metrics_df, *metrics), axis=1)

    # change columns order
    metrics_df = metrics_df[
        ["count"] + [f"{model}_{metric}" for metric in ["precision", "recall", "f1"] for model in model_names]
    ]

    metrics_df.to_csv(os.path.join(Path(__file__).parent, "10_words_metrics.csv"))

if __name__ == "__main__":
    main()
