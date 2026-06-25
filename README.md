# XAI-Guided Context-Aware Data Augmentation Experiments

Upload these files to the same Colab working directory:
```text
augmentation_engines.py
dataset_utils.py
demo.py
iterative_pipeline.py
xai_core.py
run_hate_speech_experiments.py
run_hate_speech_experiments_second.py
run_sentiment_experiments.py
requirements.txt
```

## Main pipeline
- `xai_core.py`: models, Integrated Gradients, subword-to-word alignment, and least-important word selection
- `augmentation_engines.py`: XAI-SR-BT and XAI-PR-BT augmentation engines
- `iterative_pipeline.py`: dataset augmentation, training, evaluation, checkpoint saving, and comparison table generation
- `dataset_utils.py`: dataset loaders for AfriSenti, HatEval, HaSpeeDe, etc.

## Google Colab setup
### 1. Enable GPU

In Colab:
```
Runtime → Change runtime type → Hardware accelerator → GPU
```

Then check GPU availability:
```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

### 2. Install requirements
Upload `requirements.txt`, then run:
```bash
!pip install -r requirements.txt
```

Download NLTK WordNet resources:
```python
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
```

### 3. Mount Google Drive
Mount Drive before running long experiments:
```python
from google.colab import drive
drive.mount("/content/drive")
```

### 4. Hugging Face login for gated datasets

Some datasets require Hugging Face login and dataset access approval. Run:
```python
from huggingface_hub import login
from getpass import getpass

hf_token = getpass("Paste your Hugging Face token here: ")
login(token=hf_token)
```
token = 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 
token can be unusable due to time-out. I can re-enable it with a new one if you let me know or you can create and acces token yourself.

## Hate Speech experiments
For the Spanish HatEval experiment:
```bash
!python run_hate_speech_experiments.py
```

For the second hate-speech experiment with Italian, India Hindi-English, and Portuguese TuPy:
```bash
!python run_hate_speech_experiments_second.py
```

### Sentiment experiment
For the AfriSenti sentiment experiment:
```bash
!python run_sentiment_experiments.py
```



## Notes and common warnings
### `UNEXPECTED` and `MISSING` when loading XLM-R or mBERT
These messages are usually normal. The base pretrained model is being loaded into a sequence-classification architecture, so the classification head is newly initialized and trained during your run.
### `torch.cuda.amp` FutureWarning
This is not fatal. It only means PyTorch recommends a newer AMP syntax.
### SentencePiece / PEGASUS tokenizer errors
Make sure `sentencepiece` and `protobuf` are installed:
```bash
!pip install sentencepiece protobuf
```
for PEGASUS to avoid tokenizer conversion issues.
### "module dill._dill has no attributes" error
latest version of dill is not functional with the python environment for this experiment, so instead use the version dill 0.3.6 or earlier.

