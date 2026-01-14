"""
Semantic Contradiction Detector
Assignment - Part 2
"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json


@dataclass
class ContradictionResult:
    has_contradiction: bool
    confidence: float
    contradicting_pairs: List[Tuple[str, str]]
    explanation: str


class SemanticContradictionDetector:

    def __init__(self, model_name: str = "roberta-large-mnli"):
        
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.labels = ['contradiction', 'neutral', 'entailment']

        # use gpu if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    # ----------------------------------------------------
    # STEP 1 — Preprocessing
    # ----------------------------------------------------
    def preprocess(self, text: str) -> List[str]:
        
        # remove extra whitespace
        text = text.strip()

        # split on punctuation boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # return non-empty sentences
        return [s for s in sentences if s.strip()]

    # ----------------------------------------------------
    # STEP 2 — Claim Extraction
    # ----------------------------------------------------
    def extract_claims(self, sentences: List[str]) -> List[Dict[str, Any]]:
        
        claims = []
        for s in sentences:
            claims.append({
                "text": s,
                "tokens": s.split(),
                "length": len(s.split())
            })
        return claims

    def check_contradiction(self, claim_a: Dict, claim_b: Dict) -> Tuple[bool, float]:

        encoded = self.tokenizer(
            claim_a["text"],
            claim_b["text"],
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = F.softmax(logits, dim=1)

        label_index = int(torch.argmax(probs))
        label = self.labels[label_index]
        confidence = float(torch.max(probs))

        return (label == "contradiction", confidence)

    # ----------------------------------------------------
    # STEP 4 — Full Pipeline
    # ----------------------------------------------------
    def analyze(self, text: str) -> ContradictionResult:
        
        
        sentences = self.preprocess(text)

        
        claims = self.extract_claims(sentences)

        contradicting_pairs = []
        confidences = []

        
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                is_contra, conf = self.check_contradiction(claims[i], claims[j])
                if is_contra:
                    contradicting_pairs.append((claims[i]["text"], claims[j]["text"]))
                    confidences.append(conf)

        if contradicting_pairs:
            return ContradictionResult(
                has_contradiction=True,
                confidence=float(np.mean(confidences)),
                contradicting_pairs=contradicting_pairs,
                explanation="At least one sentence pair shows contradiction."
            )

        # otherwise no contradiction
        return ContradictionResult(
            has_contradiction=False,
            confidence=0.0,
            contradicting_pairs=[],
            explanation="No contradictory claims detected."
        )



def evaluate(detector: SemanticContradictionDetector,
             test_data: List[Dict]) -> Dict[str, float]:
    """
    Evaluate detector performance.
    """

    y_true = []
    y_pred = []

    for sample in test_data:
        result = detector.analyze(sample["text"])

        # ground-truth labels expected in sample["label"]
        y_true.append(sample.get("has_contradiction", False))
        y_pred.append(result.has_contradiction)

    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)

    accuracy = float(np.mean(y_true == y_pred))
    tp = np.sum(y_true & y_pred)
    precision = float(tp / (np.sum(y_pred) + 1e-8))
    recall = float(tp / (np.sum(y_true) + 1e-8))
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))

    return dict(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1
    )


if __name__ == "__main__":
    detector = SemanticContradictionDetector()

    with open(r'data\dataset.txt', 'r', encoding='utf-8') as f:
        raw_json = f.read()

    corrected_json = re.sub(r'\((\d+,\s*\d+)\)', r'[\1]', raw_json)
    corrected_json = corrected_json.replace('True', 'true').replace('False', 'false')

    data = json.loads(corrected_json)
    

    for review in data:
        result = detector.analyze(review["text"])
        print(f"\nReview {review.get('id', 'N/A')}")
        print(f"Text: {review['text']}")
        print(f"Has contradiction: {result.has_contradiction}")
        print(f"Confidence: {result.confidence}")
        print(f"Contradicting pairs: {result.contradicting_pairs}")
        print(f"Explanation: {result.explanation}")

    metrics = evaluate(detector, data)
    print("\nMetrics:", metrics)
