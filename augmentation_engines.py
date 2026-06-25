from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
import nltk
from deep_translator import GoogleTranslator #for translation
from nltk.corpus import wordnet as wn #for syn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer #for hugging face
from xai_core import WordAttribution

#check if worldnet is there, if not download resources
try:
    wn.synsets("test")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


@dataclass #data container
class ReplacementResult:
    original_word: str
    replacement_word: Optional[str] #syn or paraphrased new word
    succeeded: bool


#translation bw source lang and eng
class TranslationBridge:
    def __init__(self, source_lang: str = "am"):
        self.source_lang = source_lang
        self._to_en = GoogleTranslator(source=source_lang, target="en")
        self._from_en = GoogleTranslator(source="en", target=source_lang)

    def to_english(self, text: str) -> str:
        try:
            return self._to_en.translate(text) #from source to eng translation
        except Exception:
            return text 

    def from_english(self, text: str) -> str:
        try:
            return self._from_en.translate(text) #from eng to source language
        except Exception:
            return text


#XAI-SR-BT: Synonym Replacement + Back Translation
#least important word,
    #translate to eng - find syn - translate syn - replace word
class XaiSynonymBackTranslator:
    def __init__(self, source_lang: str = "am"):
        self.bridge = TranslationBridge(source_lang=source_lang)

    def _english_synonym(self, english_word: str) -> Optional[str]:
        synsets = wn.synsets(english_word) #check wordnet for the meaning
        if not synsets:
            return None
        for syn in synsets:
            for lemma in syn.lemmas(): #select candidate for each meaning
                candidate = lemma.name().replace("_", " ") #if it is two words, fix the syntax
                if candidate.lower() != english_word.lower(): #dont choose the same word
                    return candidate
        return None

    def replace_word(self, word: str) -> ReplacementResult:
        english = self.bridge.to_english(word) #from source to english translation
        if not english or english.strip() == "":
            return ReplacementResult(word, None, False)

        synonym = self._english_synonym(english.strip().lower()) #find syn
        if synonym is None:
            return ReplacementResult(word, None, False)

        back_translated = self.bridge.from_english(synonym) #translate back to source lang
        if not back_translated or back_translated.strip().lower() == word.strip().lower():
            return ReplacementResult(word, None, False)

        return ReplacementResult(word, back_translated, True)

    def augment(self, text: str, target_words: List[WordAttribution]
                ) -> tuple[str, List[ReplacementResult]]: #modify the word/augment
        augmented = text #original sentence
        results: List[ReplacementResult] = [] #storage for results
        for w in target_words: #for each target word, replace the word and save the result
            result = self.replace_word(w.word)
            results.append(result)
            if result.succeeded:
                augmented = _safe_replace(augmented, w.word, result.replacement_word) #replace the original with the new word
        return augmented, results


# XAI-PR-BT: Paraphrasing Replacement + Back Translation
#least important word,
    #translate to en - paraphrase with pegasus - tranlate to source lang - replace in sentence
class XaiParaphraseBackTranslator:
    def __init__(self, source_lang: str = "am",
                 paraphrase_model: str = "tuner007/pegasus_paraphrase"):
        self.bridge = TranslationBridge(source_lang=source_lang)
        
        #if available use gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(paraphrase_model, use_fast=False) #converts tex to number for the pegassus

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            paraphrase_model,
            use_safetensors=False
        ).to(self.device)
        self.model.eval()

    def _paraphrase(self, english_text: str) -> Optional[str]: #paraphraser
        if not english_text or english_text.strip() == "":
            return None
        

        inputs = self.tokenizer(english_text, return_tensors="pt", truncation=True).to(self.device) #text to numbers
        try:
            with torch.no_grad(): #we dont need gradients in evaluation, we need it in training
                outputs = self.model.generate(
                    **inputs,
                    max_length=60,
                    num_beams=5,
                    num_return_sequences=1,
                    temperature=1.0,
                ) #generate the paraphrase
            candidate = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip() #numbers to text
        except Exception:
            return None
            
        if candidate.lower() == english_text.strip().lower(): # make sure it is not the same word
            return None
        return candidate

    def replace_word(self, word: str) -> ReplacementResult:
        english = self.bridge.to_english(word)#source to eng
        paraphrase = self._paraphrase(english) #find a paraphrase
        if paraphrase is None:
            return ReplacementResult(word, None, False)

        back_translated = self.bridge.from_english(paraphrase) #eng to source
        if not back_translated or back_translated.strip().lower() == word.strip().lower():
            return ReplacementResult(word, None, False)

        return ReplacementResult(word, back_translated, True)

    def augment(self, text: str, target_words: List[WordAttribution]
                ) -> tuple[str, List[ReplacementResult]]:
        augmented = text
        results: List[ReplacementResult] = []
        for w in target_words: #for each target word find the paraphrased word and replace the original
            result = self.replace_word(w.word)
            results.append(result)
            if result.succeeded:
                augmented = _safe_replace(augmented, w.word, result.replacement_word)
        return augmented, results


#only replace one word in the sentence
def _safe_replace(text: str, old_word: str, new_word: str) -> str:
    import re

    pattern = re.compile(rf"\b{re.escape(old_word)}\b", flags=re.IGNORECASE)
    #dont pay attention to uppercase/lowercase and find full worlds(not pieces of words)
    if pattern.search(text):
        return pattern.sub(new_word, text, count=1) #after finding full word, replace once
    if old_word in text:
        return text.replace(old_word, new_word, 1) #if full word does not work, replace the old word with the new one once
    return text