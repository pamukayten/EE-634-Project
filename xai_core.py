from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch
from captum.attr import LayerIntegratedGradients #library for XAI in pytorch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#Baseline classifier + tokenization
class BaselineClassifier:
    def __init__(self, model_name: str = "xlm-roberta-base", num_labels: int = 3,
                 device: str | None = None):
        #gpu usage
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) #load tokenizers that match model for hugging face/ text to number

        #load a pretrained transformer (xlm-roberta and mbert)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        self.model.eval()

    def tokenize(self, text: str, max_length: int = 128): #to turn text to numbers/token ids
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

    @torch.no_grad() #look at the string and return the class label and the token id
    def predict(self, text: str) -> Tuple[int, torch.Tensor]:
        encoded = self.tokenize(text) #text to numbers
        logits = self.model(**encoded).logits #largest logit is the predicted class
        pred_class = int(torch.argmax(logits, dim=-1).item())
        return pred_class, logits.squeeze(0).cpu()

    def get_embedding_layer(self): #find the model to use
        base_model = getattr(self.model, "roberta", None) or \
                     getattr(self.model, "bert", None) or \
                     self.model.base_model
        return base_model.embeddings.word_embeddings #embedding layer turns tokens to vectors so  integrated gradients can find the least imp words


#integrated gradieng (ig) gives a score to words based on the model, the token id, class label and the attention mask
def _forward_for_ig(model, input_ids, attention_mask, target_class):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits[:, target_class] #get logits based on the target label/class


def compute_token_attributions(classifier: BaselineClassifier, text: str,
                                n_steps: int = 50, max_length: int = 128,
                                internal_batch_size: int = 4
                                ) -> Tuple[List[str], List[float], int]:
    model, tokenizer, device = classifier.model, classifier.tokenizer, classifier.device
#run several ig steps on the input to assign scores to tokens

    encoded = classifier.tokenize(text, max_length=max_length) #tokenize the input. text to number
    input_ids = encoded["input_ids"] #get token id
    attention_mask = encoded["attention_mask"] #get attention mask (1-real, 0-padding)

    with torch.no_grad():
        #predict the label of input by using logits(raw scores) and find the highest
        pred_class = int(torch.argmax(
            model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=-1
        ).item())

    #baseline is all pad sequence with sentence structure
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0 #get the id for padding token
    baseline_ids = torch.full_like(input_ids, pad_id) #create a tensor the size of input, fill with pad
    if tokenizer.bos_token_id is not None: #if beginning of sequence exist keep it
        baseline_ids[:, 0] = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None: #if end of sequence exists keep it
        baseline_ids[:, -1] = tokenizer.eos_token_id

    #integrated gradient
    embedding_layer = classifier.get_embedding_layer()
    #based on the tokens, attention mask, and pred class, return the logit for predicted class
    lig = LayerIntegratedGradients(
        lambda ids, mask: _forward_for_ig(model, ids, mask, pred_class),
        embedding_layer,
    )

    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )
    #convert attr to token scores
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().tolist()
    #convert token scores to string/num to text
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    return tokens, token_scores, pred_class


#Subword -> word alignment, ranking, and top-k least-important selection
SPECIAL_TOKEN_MARKERS = {"<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]", "<unk>", "[UNK]"}
#"<s>" begin and end of a sentence in roberta
#"[CLS]" classification for mbert
#"[SEP]" seperator token for mbert
#"[PAD]" padding token

@dataclass
class WordAttribution:
    word: str
    score: float
    token_span: Tuple[int, int]


def align_subwords_to_words(tokens: List[str], scores: List[float]
                             ) -> List[WordAttribution]:
#take subword token, importance score for tokens and returns list of words with the scores
    words: List[WordAttribution] = [] #storage for full words
    current_word_pieces: List[str] = [] #store for pieces of words
    current_score = 0.0
    span_start = None



    def _flush(end_idx): # finish current word and save
        nonlocal current_word_pieces, current_score, span_start
        if current_word_pieces:
            word_text = "".join(current_word_pieces).replace("▁", "") #join the pieces into a word
            words.append(WordAttribution(word_text, current_score, (span_start, end_idx))) #save the word, score, and token span
        current_word_pieces = [] #reset
        current_score = 0.0
        span_start = None

    for i, (tok, score) in enumerate(zip(tokens, scores)):
        if tok in SPECIAL_TOKEN_MARKERS: #check if special token
            _flush(i - 1) #if a special token appears finish the prior word
            continue

        # roberta uses _ to start a word
        # mbert uses ## to continue a word

        #does the current token begin a new word
        is_new_word = tok.startswith("▁") or (not tok.startswith("##") and i > 0
                                               and tokens[i - 1] in SPECIAL_TOKEN_MARKERS) \
                      or current_word_pieces == []

        if tok.startswith("##"):
            is_new_word = False

        if is_new_word and current_word_pieces: #if already building a word finish
            _flush(i - 1)

        if span_start is None:
            span_start = i
        current_word_pieces.append(tok.lstrip("▁").replace("##", "")) #clean token from model markers and add to current word
        current_score += score

    _flush(len(tokens) - 1) #save the final word at the index of the last token
    return [w for w in words if w.word.strip() != ""] #return list of words without empty words

#at most 30% and at least 1 of the words are selected as the least important
def compute_k(num_words: int, max_fraction: float = 0.30, min_k: int = 1) -> int:
    return max(min_k, int(round(num_words * max_fraction)))
#calculate the 30% of the number of words, then round and convert them to int to find k

#rank the list with the words and the scores from high to low, then choose the bottom k as the least important
def select_least_important_words(word_attrs: List[WordAttribution],
                                  max_fraction: float = 0.30
                                  ) -> List[WordAttribution]:
    k = compute_k(len(word_attrs), max_fraction=max_fraction)
    ranked = sorted(word_attrs, key=lambda w: abs(w.score))
    return ranked[:k]
