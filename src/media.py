"""Multimedia helpers for public, user-provided content."""
from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from PIL import Image


def image_to_data_url(raw: bytes, mime: str = "image/jpeg", max_side: int = 1600) -> str:
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_video_frames(raw: bytes, suffix: str = ".mp4", count: int = 3) -> list[bytes]:
    """Extract representative JPEG frames using ffmpeg when available."""
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / f"source{suffix}"
        pattern = str(Path(temp) / "frame-%02d.jpg")
        source.write_bytes(raw)
        command = ["ffmpeg", "-y", "-i", str(source), "-vf", "fps=1/5,scale=1280:-2", "-frames:v", str(count), pattern]
        try:
            subprocess.run(command, capture_output=True, timeout=45, check=True)
        except (FileNotFoundError, subprocess.SubprocessError):
            return []
        return [p.read_bytes() for p in sorted(Path(temp).glob("frame-*.jpg"))[:count]]


def fetch_public_facebook(url: str) -> dict[str, Any]:
    """Read public OpenGraph metadata only; private/login-protected posts are unsupported."""
    result: dict[str, Any] = {"url": url, "title": "", "description": "", "image": "", "status": ""}
    if not re.match(r"^https?://(www\.)?facebook\.com/", url.strip(), re.I):
        result["status"] = "الرابط ليس رابط Facebook عاماً صالحاً."
        return result
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for key, field in [("og:title", "title"), ("og:description", "description"), ("og:image", "image")]:
            tag = soup.find("meta", property=key)
            result[field] = tag.get("content", "") if tag else ""
        result["status"] = "تم استخراج البيانات العامة المتاحة."
    except requests.RequestException:
        result["status"] = "تعذر قراءة المنشور؛ قد يكون خاصاً أو يتطلب تسجيل الدخول. الصق نص المنشور يدوياً."
    return result


def vision_analysis(data_urls: list[str], context: str, api_key: str, model: str = "gpt-4o-mini") -> dict[str, Any] | None:
    """Ask a vision-capable OpenAI-compatible model to inspect images/frames."""
    if not data_urls:
        return None
    prompt = f"""أنت محلل تحقق رقمي عربي. افحص الصورة/الإطارات المرفقة بحثاً عن مؤشرات التلاعب أو إخراج الصورة من سياقها.
استخرج النص الظاهر في الصورة إن أمكن، صف العناصر والتاريخ/المكان، وحدد ما الذي يمكن وما الذي لا يمكن إثباته بصرياً.
لا تجزم بصحة الادعاء من الصورة وحدها. أعد JSON فقط بالمفاتيح:
extracted_text, visual_summary, risk_signals (قائمة), verification_steps (قائمة), score (0 إلى 1).
سياق المستخدم: {context or 'لا يوجد'}"""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": u, "detail": "high"}} for u in data_urls[:4])
    base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
    try:
        response = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "temperature": 0.1, "response_format": {"type": "json_object"},
                  "messages": [{"role": "user", "content": content}]},
            timeout=90,
        )
        response.raise_for_status()
        return __import__("json").loads(response.json()["choices"][0]["message"]["content"])
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return None
