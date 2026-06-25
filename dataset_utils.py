from __future__ import annotations

import random
import pandas as pd

def load_afrisenti(language: str = "amh", val_fraction: float = 0.1,
                    seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:

    from datasets import load_dataset

    valid_langs = {"amh", "hau", "kin", "swa"}
    if language not in valid_langs:
        raise ValueError(f"language must be one of {valid_langs}, got {language!r}")

    try:
        ds = load_dataset("shmuhammad/AfriSenti-twitter-sentiment", language)
    except Exception:
        ds = load_dataset("masakhane/afrisenti", language)

    label_map = {"negative": 0, "neutral": 1, "positive": 2}

    def _to_df(split):
        rows = ds[split]
        texts = rows["tweet"] if "tweet" in rows.column_names else rows["text"]
        raw_labels = rows["label"]

        if isinstance(raw_labels[0], str):
            labels = [label_map[l] for l in raw_labels]
        else:
            labels = list(raw_labels)
        return pd.DataFrame({"text": texts, "label": labels})

    if "validation" in ds:
        train_df = _to_df("train")
        val_df = _to_df("validation")
    else:
        full_df = _to_df("train")
        val_df = full_df.sample(frac=val_fraction, random_state=seed)
        train_df = full_df.drop(val_df.index).reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    return train_df, val_df


def load_hateval(val_fraction: float = 0.15, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset

    ds = load_dataset("valeriobasile/HatEval")

    def _to_df(split):
        rows = ds[split]
        df = pd.DataFrame(rows)
        if "language" in df.columns:
            df = df[df["language"] == "es"]
        text_col = "text" if "text" in df.columns else "tweet"
        label_col = "HS" if "HS" in df.columns else "label"
        out = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
        out["label"] = out["label"].astype(int)
        return out.reset_index(drop=True)

    if "test" in ds:
        train_df = _to_df("train")
        val_df = _to_df("test") if "test" in ds else _to_df("validation")
    else:
        full_df = _to_df("train")
        val_df = full_df.sample(frac=val_fraction, random_state=seed)
        train_df = full_df.drop(val_df.index).reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    return train_df, val_df



def load_haspeede_italian() -> tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset

    ds = load_dataset("evalitahf/hatespeech_detection")

    def _to_df(split):
        df = pd.DataFrame(ds[split])
        out = df[["full_text", "hs"]].rename(columns={"full_text": "text", "hs": "label"})
        out["label"] = out["label"].astype(int)
        return out.dropna(subset=["text", "label"]).reset_index(drop=True)

    train_df = _to_df("dev")
    val_df = _to_df("test_all")
    return train_df, val_df


def load_india_hate_speech_superset(
    val_fraction: float = 0.2,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset

    ds = load_dataset("manueltonneau/india-hate-speech-superset")

    def _normalize_df(split) -> pd.DataFrame:
        df = pd.DataFrame(split)

        if "text" in df.columns:
            text_col = "text"
        elif "tweet" in df.columns:
            text_col = "tweet"
        elif "content" in df.columns:
            text_col = "content"
        else:
            raise ValueError(
                f"Could not find text column in India dataset. Columns: {df.columns.tolist()}"
            )

        if "labels" in df.columns:
            label_col = "labels"
        elif "label" in df.columns:
            label_col = "label"
        elif "hate" in df.columns:
            label_col = "hate"
        else:
            raise ValueError(
                f"Could not find label column in India dataset. Columns: {df.columns.tolist()}"
            )

        out = df[[text_col, label_col]].rename(
            columns={text_col: "text", label_col: "label"}
        )

        if out["label"].dtype == object:
            label_map = {
                "not_hateful": 0,
                "not hateful": 0,
                "non-hate": 0,
                "non hate": 0,
                "normal": 0,
                "none": 0,
                "hateful": 1,
                "hate": 1,
                "hate speech": 1,
            }

            lowered = out["label"].astype(str).str.lower().str.strip()

            if lowered.isin(label_map.keys()).all():
                out["label"] = lowered.map(label_map)
            else:
                out["label"] = out["label"].astype(int)
        else:
            out["label"] = out["label"].astype(int)

        return out.dropna(subset=["text", "label"]).reset_index(drop=True)

    if "train" in ds and "test" in ds:
        train_df = _normalize_df(ds["train"])
        val_df = _normalize_df(ds["test"])

    elif "train" in ds and "validation" in ds:
        train_df = _normalize_df(ds["train"])
        val_df = _normalize_df(ds["validation"])

    elif "train" in ds:
        full_df = _normalize_df(ds["train"])
        val_df = full_df.sample(frac=val_fraction, random_state=seed)
        train_df = full_df.drop(val_df.index).reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    else:
        first_split = list(ds.keys())[0]
        full_df = _normalize_df(ds[first_split])
        val_df = full_df.sample(frac=val_fraction, random_state=seed)
        train_df = full_df.drop(val_df.index).reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    return train_df, val_df



def load_tupy_portuguese(
    val_fraction: float = 0.2,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset

    try:
        ds = load_dataset("Silly-Machine/TuPy-Dataset", "binary")
    except Exception:
        ds = load_dataset("Silly-Machine/TuPy-Dataset")

    def _to_df(split_name: str) -> pd.DataFrame:
        df = pd.DataFrame(ds[split_name])

        if "text" not in df.columns:
            raise ValueError(
                f"Could not find text column in TuPy dataset. Columns: {df.columns.tolist()}"
            )

        if "hate" in df.columns:
            out = df[["text", "hate"]].rename(columns={"hate": "label"})
        elif "label" in df.columns:
            out = df[["text", "label"]]
        elif "labels" in df.columns:
            out = df[["text", "labels"]].rename(columns={"labels": "label"})
        else:
            raise ValueError(
                f"Could not find label column in TuPy dataset. Columns: {df.columns.tolist()}"
            )

        out["label"] = out["label"].astype(int)

        return out.dropna(subset=["text", "label"]).reset_index(drop=True)

    if "train" in ds and "test" in ds:
        train_df = _to_df("train")
        val_df = _to_df("test")

    elif "train" in ds and "validation" in ds:
        train_df = _to_df("train")
        val_df = _to_df("validation")

    elif "train" in ds:
        full_df = _to_df("train")
        val_df = full_df.sample(frac=val_fraction, random_state=seed)
        train_df = full_df.drop(val_df.index).reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    else:
        first_split = list(ds.keys())[0]
        full_df = _to_df(first_split)
        val_df = full_df.sample(frac=val_fraction, random_state=seed)
        train_df = full_df.drop(val_df.index).reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    return train_df, val_df


