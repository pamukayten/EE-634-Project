import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from sklearn.model_selection import train_test_split

from dataset_utils import load_afrisenti
from iterative_pipeline import run_comparison_matrix


#configurations
MODEL_NAMES = ["xlm-roberta-base", "bert-base-multilingual-cased"]

EPOCHS = 3
BATCH_SIZE = 1
NUM_LABELS = 3     #neg/neutral/pos

TRAIN_SUBSET_SIZE = 1000
VAL_SUBSET_SIZE = 250
SEED = 45

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "outputs/sentiment_subset1000_250"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


LANGUAGES_TO_RUN = {
    "amh": ("Amharic", "am"),
    "hau": ("Hausa", "ha"),
    "kin": ("Kinyarwanda", "rw"),
    "swa": ("Swahili", "sw"),
}


def make_subset(df: pd.DataFrame, n: int, seed: int = 45) -> pd.DataFrame:
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


def add_dataset_spec(specs, name, train_df, val_df, source_lang):
    train_subset = make_subset(train_df, TRAIN_SUBSET_SIZE, SEED)
    val_subset = make_subset(val_df, VAL_SUBSET_SIZE, SEED)

    specs.append({
        "name": name,
        "train_df": train_subset,
        "val_df": val_subset,
        "source_lang": source_lang,
    })


def build_dataset_specs() -> list[dict]:
    specs = []

    for afrisenti_code, (display_name, translator_code) in LANGUAGES_TO_RUN.items():
        print(f"AfriSenti ({display_name})")
        train_df, val_df = load_afrisenti(language=afrisenti_code)

        add_dataset_spec(
            specs=specs,
            name=f"AfriSenti_{display_name}_subset1000_250",
            train_df=train_df,
            val_df=val_df,
            source_lang=translator_code,
        )

    return specs


def main():
    dataset_specs = build_dataset_specs()

    if not dataset_specs:
        return

    all_results = []

    incremental_out_path = os.path.join(
        OUTPUT_DIR,
        "sentiment_results_subset1000_250_incremental.csv"
    )

    for dataset_spec in dataset_specs:
        for model_name in MODEL_NAMES:

            safe_model_name = model_name.replace("/", "_")

            partial_results = run_comparison_matrix(
                dataset_specs=[dataset_spec],
                model_names=[model_name],
                num_labels=NUM_LABELS,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                checkpoint_dir=os.path.join(
                    OUTPUT_DIR,
                    "checkpoints",
                    dataset_spec["name"],
                    safe_model_name,
                ),
            )

            all_results.append(partial_results)

            current_results_df = pd.concat(all_results, ignore_index=True)
            current_results_df.to_csv(incremental_out_path, index=False)


    results_df = pd.concat(all_results, ignore_index=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("\n" + "=" * 70)
    print("FINAL SENTIMENT ANALYSIS RESULTS TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    final_out_path = os.path.join(
        OUTPUT_DIR,
        "sentiment_results_subset1000_250_final.csv"
    )

    results_df.to_csv(final_out_path, index=False)


if __name__ == "__main__":
    main()