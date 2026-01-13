
import re
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from typing import List, Dict, Any


class ContradictionDetector:
    
    def __init__(self):
        pass

    def find_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 3]

    def detect(self, text):
        sentences = self.find_sentences(text)
        contradictions = []

        opposites = [('fast','slow'), ('good','bad'), ('long','short'), ('great','terrible')]
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                s1 = sentences[i].lower()
                s2 = sentences[j].lower()
                for a,b in opposites:
                    if (a in s1 and b in s2) or (b in s1 and a in s2):
                        contradictions.append((sentences[i], sentences[j]))
        confidence = min(1.0, 0.3*len(contradictions))
        return contradictions, confidence

class ManipulationDetector:
    URGENCY_KEYWORDS = ["hurry", "limited time", "only", "last chance", "buy now", "act fast"]
    SCARCITY_KEYWORDS = ["limited stock", "almost gone", "only a few left", "last items"]
    EMOTIONAL_KEYWORDS = ["you deserve", "must have", "best gift", "love it"]

    def detect(self, text):
        text_lower = text.lower()
        patterns = []
        for kw in self.URGENCY_KEYWORDS:
            if kw in text_lower:
                patterns.append(f"Urgency: {kw}")
        for kw in self.SCARCITY_KEYWORDS:
            if kw in text_lower:
                patterns.append(f"Scarcity: {kw}")
        for kw in self.EMOTIONAL_KEYWORDS:
            if kw in text_lower:
                patterns.append(f"Emotional: {kw}")
        confidence = min(1.0, len(patterns)/5)
        return patterns, confidence

class StylometryDetector:
    def extract_features(self, text):
        words = text.split()
        sentences = re.split(r'[.!?]', text)
        sentences = [s for s in sentences if s.strip()]
        avg_sent_len = len(words)/max(len(sentences),1)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        exclamations = text.count('!')
        questions = text.count('?')
        upper_ratio = sum(1 for c in text if c.isupper())/max(len(text),1)
        vocab_richness = len(set(words))/max(len(words),1)
        return [avg_sent_len, avg_word_len, exclamations, questions, upper_ratio, vocab_richness]

    def cluster_reviews(self, texts, review_ids, n_clusters=3):
        features = np.array([self.extract_features(t) for t in texts])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        clusters = defaultdict(list)
        for idx,label in enumerate(labels):
            clusters[label].append({"id": review_ids[idx], "text": texts[idx]})
        return clusters


def decide_review(contradictions, manip_patterns, cluster_size):
    evidence = []
    score = 0
    if contradictions:
        evidence.append("Contradiction")
        score += 1
    if manip_patterns:
        evidence.append("Manipulation")
        score += 1
    if cluster_size >= 3:
        evidence.append("Fingerprint")
        score += 1
    confidence = score/3
    if confidence >= 0.66:
        label = "Fake"
    elif confidence >= 0.33:
        label = "Suspicious"
    else:
        label = "Real"
    return label, confidence, evidence

# ----------------- Full Pipeline -----------------
def analyze_all_reviews(reviews: List[Dict[str,Any]]):
    contra = ContradictionDetector()
    manip = ManipulationDetector()
    stylo = StylometryDetector()

    texts = [r["text"] for r in reviews]
    review_ids = [r["id"] for r in reviews]
    clusters = stylo.cluster_reviews(texts, review_ids)

    results = []
    for idx, review in enumerate(reviews):
        contradictions, contra_conf = contra.detect(review["text"])
        manip_patterns, manip_conf = manip.detect(review["text"])
        # Find cluster
        cluster_size = max([len(c) for c in clusters.values() if any(r['id']==review['id'] for r in c)])
        label, confidence, evidence = decide_review(contradictions, manip_patterns, cluster_size)

        results.append({
            "id": review["id"],
            "label": label,
            "confidence": confidence,
            "evidence": evidence,
            "contradictions": contradictions,
            "manipulation_patterns": manip_patterns,
            "same_author_flag": cluster_size>=3
        })
    return results

# ----------------- Evaluation -----------------
def evaluate_reviews(reviews, results):
    
    y_true = np.array([r.get("deceptive",0) for r in reviews])
    y_pred = np.array([1 if r["label"]=="Fake" else 0 for r in results])
    accuracy = np.mean(y_true==y_pred)
    precision = np.sum((y_true==1)&(y_pred==1))/max(np.sum(y_pred==1),1)
    recall = np.sum((y_true==1)&(y_pred==1))/max(np.sum(y_true==1),1)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# ----------------- Main -----------------
if __name__ == "__main__":
    print("Module ready. Use analyze_all_reviews() and evaluate_reviews() with a list of reviews.")
