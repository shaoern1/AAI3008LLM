import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score

# Ensure you have downloaded the necessary NLTK data
nltk.download('punkt_tab')

def compute_bleu(reference, candidate):
    """
    Compute BLEU score between reference and candidate texts.
    """
    # Tokenize the reference and candidate sentences
    reference_tokens = [nltk.word_tokenize(reference)]
    candidate_tokens = nltk.word_tokenize(candidate)

    # Smoothing method to avoid zero scores for short sentences
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)

    return bleu_score

def compute_bert_score(reference, candidate):
    """
    Compute BERTScore between reference and candidate texts.
    """
    # Using BERTScore's score function to calculate precision, recall, and F1 score
    P, R, F1 = score([candidate], [reference], lang="en")
    
    return {
        "precision": P.item(),
        "recall": R.item(),
        "f1": F1.item()
    }
