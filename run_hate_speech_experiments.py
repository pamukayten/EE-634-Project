import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from sklearn.model_selection import train_test_split

from dataset_utils import (
    load_hateval,
    #load_haspeede_italian,
)
from iterative_pipeline import run_comparison_matrix


#configs
MODEL_NAMES = ["xlm-roberta-base", "bert-base-multilingual-cased"]

EPOCHS = 3
BATCH_SIZE = 1
NUM_LABELS = 2

TRAIN_SUBSET_SIZE = 1000
VAL_SUBSET_SIZE = 250
SEED = 45

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "outputs/hate_speech_subset1000_250"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS_TO_RUN = [
    "hateval_spanish",
   # "haspeede_italian",
]


#since computation is very costly, I need subsets
def make_subset(df: pd.DataFrame, n: int, seed: int = 45) -> pd.DataFrame:
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True) #remove rows without text or labels

    if len(df) <= n:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True) #use dataset size if it is smaller than chosen size

    try: #create the subsets based on the label distribution
        _, subset = train_test_split(
            df,
            test_size=n,
            stratify=df["label"],
            random_state=seed, #fşxed selection
        )
    except ValueError:
        subset = df.sample(n=n, random_state=seed) #if it cant be seperated based on dist. seperate randomly

    return subset.reset_index(drop=True)

#prepare the dataset so it can use the func from iterative pipeline
def add_dataset_spec(specs, name, train_df, val_df, source_lang):
    train_subset = make_subset(train_df, TRAIN_SUBSET_SIZE, SEED)
    val_subset = make_subset(val_df, VAL_SUBSET_SIZE, SEED)


    specs.append({ #dictionary for values
        "name": name,
        "train_df": train_subset,
        "val_df": val_subset,
        "source_lang": source_lang,
    })


def build_dataset_specs() -> list[dict]:
    specs = []
    #when running spanish data set; load it, create the training and validation subsets, create the dictionary
    if "hateval_spanish" in DATASETS_TO_RUN:
        print("HatEval (Spanish)")
        train_df, val_df = load_hateval()
        add_dataset_spec(specs, "HatEval_Spanish_subset1000_250", train_df, val_df, "es")
    return specs


def main():
    dataset_specs = build_dataset_specs() #build dataset

    if not dataset_specs:
        return

    all_results = []
    out_path = os.path.join(OUTPUT_DIR, "hate_speech_results_subset1000_250_incremental.csv")

    for dataset_spec in dataset_specs: #in the dataset
        for model_name in MODEL_NAMES: #over two models

            partial_results = run_comparison_matrix( #run experiment
                dataset_specs=[dataset_spec],
                model_names=[model_name],
                num_labels=NUM_LABELS,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                checkpoint_dir=os.path.join(
                    OUTPUT_DIR,
                    "checkpoints",
                    dataset_spec["name"],
                    model_name.replace("/", "_")
                ),
            )

            all_results.append(partial_results)#save results

            current_results_df = pd.concat(all_results, ignore_index=True)
            current_results_df.to_csv(out_path, index=False)



    results_df = pd.concat(all_results, ignore_index=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("\n" + "=" * 70)
    print("FINAL HATE SPEECH RESULTS TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    final_out_path = os.path.join(OUTPUT_DIR, "hate_speech_results_subset1000_250_final.csv")
    results_df.to_csv(final_out_path, index=False)


if __name__ == "__main__":
    main()