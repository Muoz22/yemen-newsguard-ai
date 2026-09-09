"""News analysis utilities for Yemen NewsGuard AI.

The app works offline with a transparent heuristic model and can optionally
use an OpenAI-compatible model when OPENAI_API_KEY is configured.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ARABIC_STOPWORDS = {
    "من", "في", "على", "إلى", "عن", "هذا", "هذه", "ذلك", "تلك", "الذي",
    "التي", "و", "أو", "ثم", "أن", "إن", "كان", "يكون", "تم", "قد", "ما",
    "لا", "لم", "لن", "هو", "هي", "هم", "مع", "بعد", "قبل", "لدى", "بشأن",
}

SENSATIONAL_TERMS = {
    "عاجل", "خطير", "صادم", "كارثة", "فضيحة", "لن تصدق", "هام جداً", "حصري",
    "انهيار", "مؤامرة", "مفاجأة", "عاجلا", "هام جدا",
}

@dataclass
class AnalysisResult:
    verdict: str
    score: float
    confidence: float
    reasons: list[str]
    recommendations: list[str]
    similar_examples: list[dict[str, Any]]
    source: str
    model: str


def _clean(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\sء-ي]", " ", text, flags=re.UNICODE)
    words = [w for w in text.split() if w not in ARABIC_STOPWORDS and len(w) > 1]
    return " ".join(words)


def _similar_examples(text: str, dataset: pd.DataFrame, limit: int = 3) -> list[dict[str, Any]]:
    if dataset.empty or "text" not in dataset.columns:
        return []
    corpus = [_clean(text)] + [_clean(x) for x in dataset["text"].fillna("").tolist()]
    if not any(corpus[1:]) or not corpus[0]:
        return []
    matrix = TfidfVectorizer(ngram_range=(1, 2), max_features=3000).fit_transform(corpus)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    indexes = scores.argsort()[::-1][:limit]
    examples = []
    for idx in indexes:
        if scores[idx] <= 0:
            continue
        row = dataset.iloc[idx]
        examples.append({
            "text": str(row.get("text", ""))[:180],
            "label": str(row.get("label", "UNVERIFIED")),
            "source": str(row.get("source", "")),
            "similarity": round(float(scores[idx]), 2),
        })
    return examples


def local_analysis(text: str, dataset: pd.DataFrame) -> AnalysisResult:
    """Provide an explainable triage result, not a claim of factual proof."""
    normalized = _clean(text)
    reasons: list[str] = []
    score = 0.5
    links = re.findall(r"https?://\S+|www\.\S+", text or "")
    if len(text.split()) < 12:
        score += 0.12
        reasons.append("النص قصير ولا يقدم تفاصيل كافية للتحقق المستقل.")
    if not links:
        score += 0.12
        reasons.append("لا يوجد رابط للمصدر الأصلي أو وثيقة يمكن مراجعتها.")
    if any(term in (text or "").lower() for term in SENSATIONAL_TERMS):
        score += 0.12
        reasons.append("توجد عبارات مثيرة أو عاطفية قد تدفع إلى المشاركة قبل التحقق.")
    if re.search(r"قالت مصادر|مصدر خاص|بحسب مصادر مطلعة", text or ""):
        score += 0.1
        reasons.append("الخبر يعتمد على مصادر غير مسماة؛ يلزم طلب اسم الجهة أو التصريح الأصلي.")
    if re.search(r"\d", text or ""):
        score -= 0.03
    if any(word in normalized for word in ["وزارة", "الحكومة", "البنك", "الأمم المتحدة", "محافظة"]):
        reasons.append("يتضمن ادعاءً يمكن مقارنته ببيانات الجهة الرسمية أو تقارير موثوقة.")
    score = max(0.05, min(0.95, score))
    if score >= 0.72:
        verdict = "يحتاج إلى تحقق عاجل"
    elif score >= 0.55:
        verdict = "غير موثق مبدئياً"
    else:
        verdict = "مؤشرات أولية أفضل"
    recommendations = [
        "تحقق من تاريخ النشر والسياق، وليس من العنوان فقط.",
        "ابحث عن التصريح أو الوثيقة في الموقع الرسمي للجهة المعنية.",
        "قارن الخبر مع مصدرين مستقلين موثوقين قبل إعادة النشر.",
    ]
    return AnalysisResult(
        verdict=verdict,
        score=round(score, 2),
        confidence=round(min(0.9, 0.52 + abs(score - 0.5) * 0.7), 2),
        reasons=reasons or ["لم تظهر إشارات كافية؛ لا تعتبر النتيجة إثباتاً لصحة الخبر."],
        recommendations=recommendations,
        similar_examples=_similar_examples(text, dataset),
        source="محرك محلي قابل للتفسير",
        model="heuristic-v1",
    )


def _openai_result(text: str, dataset: pd.DataFrame, api_key: str, model: str) -> AnalysisResult | None:
    prompt = f"""أنت محلل تحقق من الأخبار في اليمن. حلل النص التالي دون اختلاق أدلة.
أعد JSON فقط بالمفاتيح: verdict (واحد من: مؤشرات تضليل, غير موثق, يحتاج تحقق, مؤشرات موثوقية),
score (رقم بين 0 و1 يمثل احتمال التضليل), confidence (0 إلى 1), reasons (قائمة عربية قصيرة),
recommendations (قائمة عربية قصيرة). أكد أن التحليل أولي وليس إثباتاً.
النص: {text}"""
    base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
    try:
        response = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "temperature": 0.1, "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": "أجب بالعربية وبـ JSON صالح فقط."}, {"role": "user", "content": prompt}]},
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
        data = json.loads(payload["choices"][0]["message"]["content"])
        local = local_analysis(text, dataset)
        return AnalysisResult(
            verdict=str(data.get("verdict", "غير موثق")),
            score=float(data.get("score", local.score)),
            confidence=float(data.get("confidence", local.confidence)),
            reasons=list(data.get("reasons", local.reasons)),
            recommendations=list(data.get("recommendations", local.recommendations)),
            similar_examples=local.similar_examples,
            source="نموذج ذكاء اصطناعي اختياري",
            model=model,
        )
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return None


def analyze_news(text: str, dataset: pd.DataFrame, use_ai: bool = True, model: str = "gpt-4o-mini") -> AnalysisResult:
    if use_ai and os.getenv("OPENAI_API_KEY"):
        result = _openai_result(text, dataset, os.environ["OPENAI_API_KEY"], model)
        if result:
            return result
    return local_analysis(text, dataset)
