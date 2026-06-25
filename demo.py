import os
import gc
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from IPython.display import display

from dataset_utils import load_hateval
from iterative_pipeline import train_and_evaluate, combine_datasets
from xai_core import (
    BaselineClassifier,
    compute_token_attributions,
    align_subwords_to_words,
    select_least_important_words,
)
from augmentation_engines import (
    XaiSynonymBackTranslator,
    XaiParaphraseBackTranslator,
)


#configs
MODEL_NAME = "xlm-roberta-base" #or bert-base-multilingual-cased
NUM_LABELS = 2
SOURCE_LANG = "es"
DEMO_TRAIN_SIZE = 30 #for demo
DEMO_VAL_SIZE = 12

N_NOT_HATE_AUG = 1 #1 not-hate + 2 hate for augmentation
N_HATE_AUG = 2
N_NOT_HATE_EVAL = 3 #3 not-hate + 3 hate for evaluation
N_HATE_EVAL = 3

EPOCHS = 1
BATCH_SIZE = 1
SEED = 45

RUN_PR_BT = True #sr+pr

DEMO_DIR = os.environ.get(
    "DEMO_DIR",
    "outputs/live_xai_pipeline_demo"
)
os.makedirs(DEMO_DIR, exist_ok=True)

label_names = {
    0: "not hate",
    1: "hate",
}


#functions

#same subset creating with balanced based on labels
def make_subset(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

    if len(df) <= n:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    try:
        _, subset = train_test_split(
            df,
            test_size=n,
            stratify=df["label"],
            random_state=seed,
        )
    except ValueError:
        subset = df.sample(n=n, random_state=seed)

    return subset.reset_index(drop=True)


def balanced_sample_binary( #choose non-hate and hate samples
    df: pd.DataFrame,
    n_not_hate: int,
    n_hate: int,
    seed: int = 45
) -> pd.DataFrame:

    not_hate_df = df[df["label"] == 0] #not hate samples
    hate_df = df[df["label"] == 1] #hate samples

    available_not_hate = len(not_hate_df)
    available_hate = len(hate_df)

    n_not_hate_final = min(n_not_hate, available_not_hate) #numb of samples taken
    n_hate_final = min(n_hate, available_hate)



    samples = []

    #randomly select the samples and add them to the list
    if n_not_hate_final > 0:
        samples.append(
            not_hate_df.sample(n=n_not_hate_final, random_state=seed)
        )

    if n_hate_final > 0:
        samples.append(
            hate_df.sample(n=n_hate_final, random_state=seed + 1)
        )

    if not samples:
        raise ValueError("No samples available")

    return pd.concat(samples, ignore_index=True) #combine the samples


def label_to_name(label):
    return label_names.get(int(label), str(label))
    #convert labels to names 0-not hate and 1-hate


def cleanup(): #clean
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_with_checkpoint(model_path: str, texts: list[str], num_labels: int):
    #load a saved model checkpoint and predict laber

    clf = BaselineClassifier(
        model_name=model_path,
        num_labels=num_labels,
    )

    preds = []

    for text in texts:
        pred_class, _ = clf.predict(text)
        preds.append(pred_class)

    del clf
    cleanup()

    return preds


def explain_and_augment_sentence(
    text: str,
    true_label: int,
    classifier,
    sr_engine,
    pr_engine=None,
    max_fraction: float = 0.30,
    n_steps_ig: int = 10,
):
    #prediction -> importance scoring -> least important k selection -> sr-bt (and/or pr-bt)
    pred_before, _ = classifier.predict(text)

    tokens, scores, _ = compute_token_attributions(
        classifier,
        text,
        n_steps=n_steps_ig,
        internal_batch_size=2,
    )

    word_attrs = align_subwords_to_words(tokens, scores)

    target_words = select_least_important_words(
        word_attrs,
        max_fraction=max_fraction,
    )

    selected_words = [w.word for w in target_words]

    selected_words_with_scores = [
        f"{w.word} ({w.score:.4f})" for w in target_words
    ]

    #SR-BT augmentation
    sr_text, sr_results = sr_engine.augment(text, target_words)

    sr_changes = [
        f"{r.original_word} → {r.replacement_word}"
        for r in sr_results
        if r.succeeded
    ]

    #PR-BT augmentation
    if pr_engine is not None:
        pr_text, pr_results = pr_engine.augment(text, target_words)

        pr_changes = [
            f"{r.original_word} → {r.replacement_word}"
            for r in pr_results
            if r.succeeded
        ]
    else:
        pr_changes = []

    table_row = {
        "true_label": label_to_name(true_label),
        "baseline_prediction": label_to_name(pred_before),
        "original_sentence": text,
        "least_important_words_selected": ", ".join(selected_words),
        "selected_words_with_IG_scores": ", ".join(selected_words_with_scores),
        "SR-BT_changed_words": ", ".join(sr_changes) if sr_changes else "No successful replacement",
        "SR-BT_augmented_sentence": sr_text,
        "PR-BT_changed_words": ", ".join(pr_changes) if pr_changes else "No successful replacement / not run",
        "PR-BT_augmented_sentence": pr_text,
    }

    return table_row, sr_text, pr_text



#load and prep data
print("HatEval Spanish dataset")
train_df_full, val_df_full = load_hateval()

train_subset = make_subset(train_df_full, DEMO_TRAIN_SIZE, SEED)
val_subset = make_subset(val_df_full, DEMO_VAL_SIZE, SEED)


#train baseline model
print("\n" + "=" * 80)
print("TRAIN BASELINE MODEL")
print("=" * 80)

BEFORE_MODEL_PATH = os.path.join(DEMO_DIR, "before_model")

before_metrics = train_and_evaluate(
    train_df=train_subset,
    val_df=val_subset,
    model_name=MODEL_NAME,
    num_labels=NUM_LABELS,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    save_dir=BEFORE_MODEL_PATH,
)

print("\nBaseline metrics:")
print(before_metrics)

cleanup()


#demo with couple sentences

print("\n" + "=" * 80)
print("SENTENCES THROUGH XAI + AUGMENTATION")
print("=" * 80)

#1 not-hate and 2 hate samples.
pipeline_df = balanced_sample_binary(
    train_subset,
    n_not_hate=N_NOT_HATE_AUG,
    n_hate=N_HATE_AUG,
    seed=7,
)

print("\nSentences chosen:")
display(pipeline_df[["text", "label"]])


guiding_classifier = BaselineClassifier( #baseline model load
    model_name=BEFORE_MODEL_PATH,
    num_labels=NUM_LABELS,
)

#sr-bt
sr_engine = XaiSynonymBackTranslator(source_lang=SOURCE_LANG)

if RUN_PR_BT:
    #pr-bt
    pr_engine = XaiParaphraseBackTranslator(source_lang=SOURCE_LANG)
else:
    pr_engine = None

augmentation_rows = []
sr_augmented_rows = []
pr_augmented_rows = []

for _, row in pipeline_df.iterrows():
    original_text = row["text"]
    true_label = int(row["label"])

    table_row, sr_text, pr_text = explain_and_augment_sentence(
        text=original_text,
        true_label=true_label,
        classifier=guiding_classifier,
        sr_engine=sr_engine,
        pr_engine=pr_engine,
        max_fraction=0.30,
        n_steps_ig=10,
    )

    augmentation_rows.append(table_row)

    sr_augmented_rows.append({
        "text": sr_text,
        "label": true_label,
    })

    if RUN_PR_BT:
        pr_augmented_rows.append({
            "text": pr_text,
            "label": true_label,
        })

augmentation_demo_df = pd.DataFrame(augmentation_rows)

pd.set_option("display.max_colwidth", 300)

print("\nAUGMENTATION DEMO TABLE:")
display(augmentation_demo_df)

augmentation_demo_path = os.path.join(DEMO_DIR, "live_demo_sentence_augmentation_balanced.csv")
augmentation_demo_df.to_csv(augmentation_demo_path, index=False)
print("Saved augmentation demo table to:", augmentation_demo_path)

#clean up
del guiding_classifier
del sr_engine

if pr_engine is not None:
    del pr_engine

cleanup()


#train after sr-bt augmentation

print("\n" + "=" * 80)
print("TRAIN MODEL AFTER SR-BT AUGMENTATION")
print("=" * 80)

sr_augmented_df = pd.DataFrame(sr_augmented_rows)
sr_combined_train_df = combine_datasets(train_subset, sr_augmented_df)

SR_BT_MODEL_PATH = os.path.join(DEMO_DIR, "after_sr_bt_model")

sr_metrics = train_and_evaluate(
    train_df=sr_combined_train_df,
    val_df=val_subset,
    model_name=MODEL_NAME,
    num_labels=NUM_LABELS,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    save_dir=SR_BT_MODEL_PATH,
)

print("\nSR-BT metrics:")
print(sr_metrics)

cleanup()


#train after pr-bt

if RUN_PR_BT:
    print("\n" + "=" * 80)
    print("TRAIN MODEL AFTER PR-BT AUGMENTATION")
    print("=" * 80)

    pr_augmented_df = pd.DataFrame(pr_augmented_rows)
    pr_combined_train_df = combine_datasets(train_subset, pr_augmented_df)

    PR_BT_MODEL_PATH = os.path.join(DEMO_DIR, "after_pr_bt_model")

    pr_metrics = train_and_evaluate(
        train_df=pr_combined_train_df,
        val_df=val_subset,
        model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        save_dir=PR_BT_MODEL_PATH,
    )

    print("\nPR-BT metrics:")
    print(pr_metrics)

    cleanup()
else:
    PR_BT_MODEL_PATH = None
    pr_metrics = None


#sample comparisons

print("\n" + "=" * 80)
print("SAMPLES BEFORE AND AFTER TRAINING")
print("=" * 80)

#3 not-hate and 3 hate samples.
eval_demo_df = balanced_sample_binary(
    val_subset,
    n_not_hate=N_NOT_HATE_EVAL,
    n_hate=N_HATE_EVAL,
    seed=16,
)

eval_texts = eval_demo_df["text"].tolist()

print("\nEvaluation samples chosen:")
display(eval_demo_df[["text", "label"]])

#prediction
before_preds = predict_with_checkpoint(
    BEFORE_MODEL_PATH,
    eval_texts,
    NUM_LABELS,
)

#sr-bt
sr_preds = predict_with_checkpoint(
    SR_BT_MODEL_PATH,
    eval_texts,
    NUM_LABELS,
)

if RUN_PR_BT:
    #pr-bt
    pr_preds = predict_with_checkpoint(
        PR_BT_MODEL_PATH,
        eval_texts,
        NUM_LABELS,
    )
else:
    pr_preds = [None] * len(eval_texts)

prediction_demo_df = pd.DataFrame({
    "evaluation_sentence": eval_texts,
    "true_label": [label_to_name(x) for x in eval_demo_df["label"]],
    "prediction_before_augmentation_training": [label_to_name(x) for x in before_preds],
    "prediction_after_SR_BT_training": [label_to_name(x) for x in sr_preds],
    "prediction_after_PR_BT_training": [
        label_to_name(x) if x is not None else "PR-BT not run"
        for x in pr_preds
    ],
})

print("\nEVALUATION PREDICTION DEMO TABLE:")
display(prediction_demo_df)

prediction_demo_path = os.path.join(DEMO_DIR, "live_demo_evaluation_predictions_balanced.csv")
prediction_demo_df.to_csv(prediction_demo_path, index=False)
print("Saved evaluation prediction demo table to:", prediction_demo_path)


#results

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

summary_rows = []

summary_rows.append({
    "stage": "Before augmentation",
    "accuracy": before_metrics["accuracy"],
    "f1": before_metrics["f1"],
    "delta_accuracy": 0.0,
    "delta_f1": 0.0,
})

summary_rows.append({
    "stage": "After SR-BT augmentation",
    "accuracy": sr_metrics["accuracy"],
    "f1": sr_metrics["f1"],
    "delta_accuracy": sr_metrics["accuracy"] - before_metrics["accuracy"],
    "delta_f1": sr_metrics["f1"] - before_metrics["f1"],
})

if RUN_PR_BT and pr_metrics is not None:
    summary_rows.append({
        "stage": "After PR-BT augmentation",
        "accuracy": pr_metrics["accuracy"],
        "f1": pr_metrics["f1"],
        "delta_accuracy": pr_metrics["accuracy"] - before_metrics["accuracy"],
        "delta_f1": pr_metrics["f1"] - before_metrics["f1"],
    })

mini_results_df = pd.DataFrame(summary_rows)

print("\nRESULTS TABLE:")
display(mini_results_df)

mini_results_path = os.path.join(DEMO_DIR, "live_demo_mini_results_balanced.csv")
mini_results_df.to_csv(mini_results_path, index=False)
