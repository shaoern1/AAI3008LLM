# evaluation_debug.py
"""
This module serves as a replacement for your evaluation module to help debug issues.
It provides simplified versions of the evaluation functions that log their operations.
"""

import logging
import traceback

logger = logging.getLogger(__name__)

def safe_compute_bleu(reference, candidate):
    """A simplified and safer version of BLEU computation for debugging."""
    try:
        logger.info(f"Computing BLEU with ref: '{reference[:30]}...' and cand: '{candidate[:30]}...'")
        
        # Simple word-based BLEU calculation
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        # Count matching words
        matches = sum(1 for w in cand_words if w in ref_words)
        precision = matches / max(len(cand_words), 1)
        
        logger.info(f"Computed simplified BLEU: {precision}")
        return precision
    except Exception as e:
        logger.error(f"Error in safe_compute_bleu: {e}")
        logger.error(traceback.format_exc())
        return 0.0

def safe_compute_sacrebleu(reference, candidate):
    """A simplified and safer version of SacreBLEU computation for debugging."""
    try:
        logger.info(f"Computing SacreBLEU with ref: '{reference[:30]}...' and cand: '{candidate[:30]}...'")
        
        # For debugging, return a slightly different value from BLEU
        result = safe_compute_bleu(reference, candidate) * 0.95
        
        logger.info(f"Computed simplified SacreBLEU: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in safe_compute_sacrebleu: {e}")
        logger.error(traceback.format_exc())
        return 0.0

def safe_compute_bert_score(reference, candidate):
    """A simplified and safer version of BERTScore computation for debugging."""
    try:
        logger.info(f"Computing BERTScore with ref: '{reference[:30]}...' and cand: '{candidate[:30]}...'")
        
        # For debugging, return dummy BERTScore values
        result = {
            'precision': 0.75, 
            'recall': 0.72,
            'f1': 0.735
        }
        
        logger.info(f"Computed simplified BERTScore: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in safe_compute_bert_score: {e}")
        logger.error(traceback.format_exc())
        return None

# Export the safe versions with the original names
compute_bleu = safe_compute_bleu
compute_sacrebleu = safe_compute_sacrebleu
compute_bert_score = safe_compute_bert_score