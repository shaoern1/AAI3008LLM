# evaluation.py
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sacrebleu
from bert_score import score
import torch

# Ensure you have downloaded the necessary NLTK data
nltk.download('punkt_tab')

def compute_bleu(reference, candidate):
    """
    Compute BLEU score between reference and candidate texts using NLTK.
    """
    # Tokenize the reference and candidate sentences
    reference_tokens = [nltk.word_tokenize(reference)]
    candidate_tokens = nltk.word_tokenize(candidate)

    # Smoothing method to avoid zero scores for short sentences
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)

    return bleu_score

def compute_sacrebleu(reference, candidate):
    """
    Compute BLEU score between reference and candidate texts using sacrebleu.
    """
    bleu = sacrebleu.metrics.BLEU(effective_order=True)
    return bleu.sentence_score(candidate, [reference]).score # sacrebleu expects a list of references

def compute_bert_score(reference, candidate):
    # from bert_score import score # Import library when compute_bert_score is called
    """
    Compute BERTScore between reference and candidate texts.
    """
    # return None
    P, R, F1 = score([candidate], [reference], lang="en", device = "cpu") # load the score here as this loads a model
    return {
       "precision": P.item(),
       "recall": R.item(),
       "f1": F1.item()
    }