"""
Evaluation metrics for clinical RAG validation.

This module provides lightweight, lexical-based metrics that serve as 
proxies for retrieval and generation quality without requiring LLM-as-judge 
invocations. These are useful for rapid iteration and offline validation.
"""
import re
from difflib import SequenceMatcher


def token_overlap(text_a: str, text_b: str) -> float:
    """
    Computes the Jaccard similarity between two strings based on token overlap.
    
    In clinical contexts, this measures the lexical consistency between 
    generated answers and ground truth, capturing the presence of specific 
    medical entities.
    """
    a_tokens = set(re.findall(r"\w+", text_a.lower()))
    b_tokens = set(re.findall(r"\w+", text_b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def fuzzy_similarity(text_a: str, text_b: str) -> float:
    """
    Computes fuzzy string similarity using SequenceMatcher.
    
    Useful for capturing similarities where medical terms might have slight 
    variations or typos.
    """
    return SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()


def context_hit_rate(question: str, contexts: list[str]) -> float:
    """
    Measures the fraction of retrieved contexts containing question keywords.
    
    Acts as a proxy for retrieval relevance. In clinical RAG, high hit rates 
    suggest that the retriever is successfully identifying document chunks 
    aligned with the query's clinical entities.
    """
    keywords = set(re.findall(r"\w{4,}", question.lower()))
    if not keywords:
        return 0.0
    hits = sum(
        1 for ctx in contexts
        if any(kw in ctx.lower() for kw in keywords)
    )
    return hits / len(contexts)


def answer_length_score(answer: str) -> float:
    """
    Penalizes overly concise or empty clinical responses.
    
    Clinical answers often require specific detail; this metric ensures the 
    model isn't producing critically underspecified output.
    """
    words = len(answer.split())
    return min(words / 50.0, 1.0)


def no_hallucination_proxy(answer: str, contexts: list[str]) -> float:
    """
    Faithfulness proxy based on token grounding in retrieved contexts.
    
    Calculates the fraction of significant tokens in the answer that are 
    present in the source contexts. High scores indicate the answer is likely 
    grounded in the provided evidence, reducing the probability of hallucinations.
    """
    answer_tokens = set(re.findall(r"\w{4,}", answer.lower()))
    if not answer_tokens:
        return 0.0

    all_context_text = " ".join(contexts).lower()
    grounded = sum(1 for tok in answer_tokens if tok in all_context_text)
    return grounded / len(answer_tokens)


def compute_all_custom_metrics(item: dict) -> dict:
    """
    Aggregates all custom metrics for a single evaluation item.
    """
    return {
        "token_overlap_vs_gt"   : token_overlap(item["answer"], item["ground_truth"]),
        "fuzzy_similarity_vs_gt": fuzzy_similarity(item["answer"], item["ground_truth"]),
        "context_hit_rate"      : context_hit_rate(item["question"], item["contexts"]),
        "answer_length_score"   : answer_length_score(item["answer"]),
        "grounding_proxy"       : no_hallucination_proxy(item["answer"], item["contexts"])
    }
