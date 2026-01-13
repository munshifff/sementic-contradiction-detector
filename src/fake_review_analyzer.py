
import re
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

labels = ['contradiction', 'neutral', 'entailment']

tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def sent_tokenize(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

def find_sentences_for_span(span, text):
    start, end = span
    sentences = sent_tokenize(text)
    selected = []
    pos = 0
    for s in sentences:
        s_start = pos
        s_end = pos + len(s)
        if s_end >= start and s_start <= end:
            selected.append(s)
        pos += len(s) + 1
    return " ".join(selected).strip()

def detect_contradiction(review_item):
    if review_item.get('has_contradiction') and len(review_item.get('contradiction_spans', [])) == 2:
        span1, span2 = review_item['contradiction_spans']
        sent1 = find_sentences_for_span(span1, review_item['text'])
        sent2 = find_sentences_for_span(span2, review_item['text'])
        encoded = tokenizer(sent1, sent2, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = F.softmax(logits, dim=1)
            predicted_label = labels[torch.argmax(probs)]
            confidence = float(torch.max(probs))
        return {'contradiction': predicted_label == 'contradiction', 'confidence': confidence, 
                'evidence': (sent1, sent2)}
    return {'contradiction': False, 'confidence': 0.0, 'evidence': None}


urgency_keywords = ["only", "hurry", "limited time", "buy now", "act fast"]
scarcity_keywords = ["limited stock", "almost gone", "last chance"]
emotional_keywords = ["everyone is buying", "you deserve", "must-have"]

def detect_manipulation(text):
    text_lower = text.lower()
    urgency_found = any(k in text_lower for k in urgency_keywords)
    scarcity_found = any(k in text_lower for k in scarcity_keywords)
    emotional_found = any(k in text_lower for k in emotional_keywords)
    sentiment = TextBlob(text).sentiment.polarity
    sentiment_score = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"

    score = 0
    if urgency_found or scarcity_found:
        score += 1
    if emotional_found:
        score += 1
    if sentiment_score == "positive":
        score += 0.5
    is_manipulative = score > 1.5
    evidence = []
    if urgency_found: evidence.append("urgency")
    if scarcity_found: evidence.append("scarcity")
    if emotional_found: evidence.append("emotional")
    return {'manipulative': is_manipulative, 'score': score, 'evidence': evidence}


def extract_style_features(text):
    words = text.split()
    sentences = re.split(r'[.!?]', text)
    sentences = [s for s in sentences if s.strip()]
    avg_sentence_len = len(words)/max(len(sentences),1)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    exclamations = text.count('!')
    questions = text.count('?')
    uppercase_ratio = sum(1 for c in text if c.isupper())/max(len(text),1)
    vocab_richness = len(set(words))/max(len(words),1)
    return [avg_sentence_len, avg_word_len, exclamations, questions, uppercase_ratio, vocab_richness]

class StylometryDetector:
    def cluster_reviews(self, texts, review_ids, n_clusters=3):
        features = np.array([extract_style_features(t) for t in texts])
        n_clusters = min(n_clusters, len(texts))  # Fix: cannot have more clusters than reviews
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        clusters = defaultdict(list)
        for idx,label in enumerate(labels):
            clusters[label].append({'id': review_ids[idx], 'text': texts[idx]})
        return clusters


def decide_review(contradiction, manipulation, stylometry_match):
    signals = 0
    evidence = []

    if contradiction['contradiction']:
        signals += 1
        evidence.append("Contradiction")
    if manipulation['manipulative']:
        signals += 1
        evidence.append("Manipulation")
    if stylometry_match:
        signals += 1
        evidence.append("Fingerprint")

    if signals == 0:
        label = "Real"
        confidence = 0.0
    else:
        label = "Fake" if signals > 1 else "Suspicious"
        confidence = signals / 3
    return label, confidence, evidence if evidence else None


def analyze_all_reviews(reviews):
    results = []
    stylo = StylometryDetector()
    texts = [r['text'] for r in reviews]
    review_ids = [r['id'] for r in reviews]
    clusters = stylo.cluster_reviews(texts, review_ids)

    for idx, r in enumerate(reviews):
        contradiction = detect_contradiction(r)
        manipulation = detect_manipulation(r['text'])
        # If review shares cluster with others (>1 review in cluster)
        cluster_label = None
        for label,revs in clusters.items():
            if any(rv['id']==r['id'] for rv in revs):
                cluster_label = label
                break
        stylometry_match = len(clusters[cluster_label]) > 1 if cluster_label is not None else False

        label, confidence, evidence = decide_review(contradiction, manipulation, stylometry_match)
        results.append({'id': r['id'], 'deceptive': label=="Fake", 'confidence': confidence, 'evidence': evidence})
    return results

# -------------------- Evaluation --------------------
def evaluate_reviews(results, ground_truth):
    y_true = [r.get('deceptive',False) for r in ground_truth]
    y_pred = [r['deceptive'] for r in results]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = float(np.mean(y_true==y_pred))
    precision = float(np.sum((y_true & y_pred))/(np.sum(y_pred)+1e-8))
    recall = float(np.sum((y_true & y_pred))/(np.sum(y_true)+1e-8))
    f1 = float(2*precision*recall/(precision+recall+1e-8))
    return {'accuracy': accuracy,'precision': precision,'recall': recall,'f1': f1}


if __name__=="__main__":
    print("This module is ready to run on any list of reviews with keys 'id' and 'text'.")
    print("Use 'analyze_all_reviews(reviews)' to process reviews and 'evaluate_reviews(results, ground_truth)' for metrics.")
