from __future__ import annotations
import gc
from dataclasses import dataclass, field
from typing import List
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from augmentation_engines import XaiParaphraseBackTranslator, XaiSynonymBackTranslator

from xai_core import (
    BaselineClassifier,
    align_subwords_to_words,
    compute_token_attributions,
    select_least_important_words,
)


#augmentation
@dataclass
class AugmentationRunConfig:
    method: str = "xai_sr_bt"          #xai_sr_bt or xai_pr_bt
    source_lang: str = "am"
    start_fraction: float = 0.20        #initial threshold for least important
    max_fraction: float = 0.30          #max from the paper
    fraction_step: float = 0.05
    min_success_rate: float = 0.80      #threlhold to try again
    max_threshold_retries: int = 3
    n_steps_ig: int = 50
    ig_internal_batch_size: int = 4


@dataclass
class AugmentationStats:
    threshold_used: List[float] = field(default_factory=list)
    success_rate_per_pass: List[float] = field(default_factory=list)
    final_success_rate: float = 0.0


def _build_engine(config: AugmentationRunConfig): #choose the baseline model
    if config.method == "xai_sr_bt":
        return XaiSynonymBackTranslator(source_lang=config.source_lang)
    if config.method == "xai_pr_bt":
        return XaiParaphraseBackTranslator(source_lang=config.source_lang)
    raise ValueError(f"Unknown augmentation method")

#start with the initial threshold and the original text to augment a sentence
def augment_single_example(text: str, classifier: BaselineClassifier, engine,
                            config: AugmentationRunConfig) -> tuple[str, float]:
    #IG attribution -> word alignment -> bottom-k selection ->  engine -> success-rate check -> threshold widening retry loop
    fraction = config.start_fraction
    augmented_text = text
    success_rate = 0.0

    #find the tokens and the scores for the words in the sentence, and align split words
    tokens, scores, _ = compute_token_attributions(
        classifier, text, n_steps=config.n_steps_ig,
        internal_batch_size=config.ig_internal_batch_size,
    )
    word_attrs = align_subwords_to_words(tokens, scores)

    #if replacement is not succesfull enough (%80), try again until max retries is reached
    for attempt in range(config.max_threshold_retries + 1):
        target_words = select_least_important_words(word_attrs, max_fraction=fraction) #select bottom %20
        if not target_words: #if there are no words to replace, stop
            break

        augmented_text, results = engine.augment(text, target_words) #augment/replace texts with sr or pr
        n_success = sum(1 for r in results if r.succeeded) #find the succesfull replacements
        success_rate = n_success / len(results) if results else 0.0

        if success_rate >= config.min_success_rate:
            break

        # increase the threshold and try again to create a bigger candidate pool
        fraction = min(config.max_fraction, fraction + config.fraction_step)

    return augmented_text, success_rate #retun the new sentence and the success rate

#apply augmentation to the dataset
def build_augmented_dataset(df: pd.DataFrame, classifier: BaselineClassifier,
                             config: AugmentationRunConfig) -> tuple[pd.DataFrame, AugmentationStats]:
    engine = _build_engine(config)
    stats = AugmentationStats()
    augmented_rows = []

    #augment a text and save success rate, then add to the augmented list
    for _, row in df.iterrows(): #loop every row of dataset
        augmented_text, success_rate = augment_single_example(row["text"], classifier, engine, config)
        stats.success_rate_per_pass.append(success_rate)
        augmented_rows.append({"text": augmented_text, "label": row["label"]})
    #calculate the avg rate for all text
    stats.final_success_rate = (
        sum(stats.success_rate_per_pass) / len(stats.success_rate_per_pass)
        if stats.success_rate_per_pass else 0.0
    )

    #XAI-PR-BT loads a second model (PEGASUS) onto the GPU
    del engine #to mae sure we have space for pegasus
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(augmented_rows), stats


def combine_datasets(original_df: pd.DataFrame, augmented_df: pd.DataFrame) -> pd.DataFrame:
    #combine the original and augmented dataset
    return pd.concat([original_df, augmented_df], ignore_index=True)


#prep for training
class TextClassificationDataset(Dataset): #convert the data for pytorch
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): #number of samples in dataset
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer( #tokenize a sample text and pad so all are same size
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()} #fix syntax
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long) #add the label
        return item #return the training ready example

#training
def train_and_evaluate(train_df: pd.DataFrame, val_df: pd.DataFrame, model_name: str,
                        num_labels: int, epochs: int = 3, batch_size: int = 8,
                        learning_rate: float = 2e-5, device: str | None = None,
                        use_amp: bool = True, save_dir: str | None = None
                        ) -> dict:

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = use_amp and device == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_name) #load tokenizer for the models
    model = AutoModelForSequenceClassification.from_pretrained( #load model
        model_name, num_labels=num_labels
    ).to(device)

    train_loader = DataLoader( #load the training samples
        TextClassificationDataset(train_df, tokenizer), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader( #load the validation samples
        TextClassificationDataset(val_df, tokenizer), batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #adam optimizer for model weights
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    #TRAIN
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(**batch) #loss+predictions
            scaler.scale(outputs.loss).backward() #compute gradients from loss
            scaler.step(optimizer) #update weights
            scaler.update()

    #EVALUATION
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad(): #no gradients for evaluation
        for batch in val_loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(**batch).logits #get logits from the model(raw scores)
            preds = torch.argmax(logits, dim=-1).cpu() #choose class with highest score
            all_preds.extend(preds.tolist()) #add predics and labels to the list
            all_labels.extend(labels.tolist())
    #calculate accuracy and f1
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    if save_dir is not None:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    del model, optimizer, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"accuracy": acc, "f1": f1, "save_dir": save_dir}



def run_full_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame, model_name: str,
                         augmentation_method: str, source_lang: str,
                         num_labels: int = 3, min_improvement: float = 0.01,
                         max_outer_iterations: int = 3, epochs: int = 3,
                         batch_size: int = 8, checkpoint_dir: str = "checkpoints") -> dict:

    #train and evaluate baseline model without augmentation
    before = train_and_evaluate(train_df, val_df, model_name, num_labels, epochs=epochs,
                                 batch_size=batch_size, save_dir=f"{checkpoint_dir}/before")

    #classify which words are least important
    guiding_classifier = BaselineClassifier(model_name=model_name, num_labels=num_labels)
    method = augmentation_method #choose aug method
    best_after = None
    history = []

    for outer_iter in range(max_outer_iterations):
        config = AugmentationRunConfig(method=method, source_lang=source_lang)
        augmented_df, aug_stats = build_augmented_dataset(train_df, guiding_classifier, config)
        #create augmented configs and training data
        combined_df = combine_datasets(train_df, augmented_df) #join original + augmented dataset

        #create checkpoints to save
        iter_save_dir = f"{checkpoint_dir}/iter_{outer_iter + 1}_{method}"

        #train a new model on the combined dataset and evaluate
        after = train_and_evaluate(combined_df, val_df, model_name, num_labels, epochs=epochs,
                                    batch_size=batch_size, save_dir=iter_save_dir)
        delta_acc = after["accuracy"] - before["accuracy"]


        history.append({ #save results
            "iteration": outer_iter + 1,
            "method": method,
            "accuracy": after["accuracy"],
            "f1": after["f1"],
            "delta_acc": delta_acc,
            "substitution_success_rate": aug_stats.final_success_rate,
            "checkpoint": iter_save_dir,
        })
        #save best result
        if best_after is None or after["accuracy"] > best_after["accuracy"]:
            best_after = after

        if delta_acc >= min_improvement:
            break

        method = "xai_pr_bt" if method == "xai_sr_bt" else "xai_sr_bt"

    del guiding_classifier #clean memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return { #results
        "before": {"accuracy": before["accuracy"], "f1": before["f1"]},
        "after_best": {"accuracy": best_after["accuracy"], "f1": best_after["f1"]},
        "history": history,
    }


#comparison calcualtions
def run_table_style_comparison(train_df: pd.DataFrame, val_df: pd.DataFrame,
                                 model_name: str, source_lang: str, dataset_name: str,
                                 num_labels: int = 2, epochs: int = 3, batch_size: int = 8,
                                 checkpoint_dir: str = "checkpoints") -> dict:

    safe_name = f"{dataset_name}_{model_name}".replace("/", "_")



    before_save_dir = f"{checkpoint_dir}/{safe_name}_before"

    before = train_and_evaluate( #train and evaluate before aug
        train_df,
        val_df,
        model_name,
        num_labels,
        epochs=epochs,
        batch_size=batch_size,
        save_dir=before_save_dir,
    )


    guiding_classifier = BaselineClassifier( #use the finetuned baseline
        model_name=before_save_dir,
        num_labels=num_labels,
    )

    #XAI-SR-BT (synonym + back-translation)
    #create config, augmented data, combined dataset, train and evaluate for syn
    sr_config = AugmentationRunConfig(method="xai_sr_bt", source_lang=source_lang)
    sr_augmented, sr_stats = build_augmented_dataset(train_df, guiding_classifier, sr_config)
    sr_combined = combine_datasets(train_df, sr_augmented)
    after_sr = train_and_evaluate(sr_combined, val_df, model_name, num_labels, epochs=epochs,
                                   batch_size=batch_size,
                                   save_dir=f"{checkpoint_dir}/{safe_name}_xai_sr_bt")

    #XAI-PR-BT (paraphrase + back-translation)
    #create config, augmented data, combined dataset, train and evaluate for paraph
    pr_config = AugmentationRunConfig(method="xai_pr_bt", source_lang=source_lang)
    pr_augmented, pr_stats = build_augmented_dataset(train_df, guiding_classifier, pr_config)
    pr_combined = combine_datasets(train_df, pr_augmented)
    after_pr = train_and_evaluate(pr_combined, val_df, model_name, num_labels, epochs=epochs,
                                   batch_size=batch_size,
                                   save_dir=f"{checkpoint_dir}/{safe_name}_xai_pr_bt")

    del guiding_classifier #clean memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return { #give all table values
        "dataset": dataset_name,
        "model": model_name,
        "acc_before": before["accuracy"], "f1_before": before["f1"],
        "acc_sr_bt": after_sr["accuracy"], "f1_sr_bt": after_sr["f1"],
        "delta_acc_sr_bt": after_sr["accuracy"] - before["accuracy"],
        "delta_f1_sr_bt": after_sr["f1"] - before["f1"],
        "sr_bt_substitution_success": sr_stats.final_success_rate,
        "acc_pr_bt": after_pr["accuracy"], "f1_pr_bt": after_pr["f1"],
        "delta_acc_pr_bt": after_pr["accuracy"] - before["accuracy"],
        "delta_f1_pr_bt": after_pr["f1"] - before["f1"],
        "pr_bt_substitution_success": pr_stats.final_success_rate,
    }


#run all combinations of model, language to create a table
def run_comparison_matrix(dataset_specs: list[dict], model_names: list[str],
                           num_labels: int = 2, epochs: int = 3, batch_size: int = 8,
                           checkpoint_dir: str = "checkpoints") -> pd.DataFrame:
    rows = []
    for spec in dataset_specs:
        for model_name in model_names:
            row = run_table_style_comparison(
                train_df=spec["train_df"], val_df=spec["val_df"],
                model_name=model_name, source_lang=spec["source_lang"],
                dataset_name=spec["name"], num_labels=num_labels,
                epochs=epochs, batch_size=batch_size, checkpoint_dir=checkpoint_dir,
            )
            rows.append(row)
    return pd.DataFrame(rows)
