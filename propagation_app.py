from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.intelligence import key_phrases, persist_report
from src.monitoring import monitor_claim

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "data" / "sources.csv"
DB = ROOT / "data" / "newsguard_history.db"

st.set_page_config(page_title="Yemen NewsGuard — Propagation Intelligence", page_icon="🛰️", layout="wide")
st.title("🛰️ Yemen NewsGuard — Propagation Intelligence")
st.caption("رصد انتشار الخبر عبر المصادر العامة المفهرسة، مع بصمة خبر وسجل تاريخي")

for key in ("OPENAI_API_KEY", "OPENAI_API_BASE", "NEWSGUARD_SEARCH_MODEL", "TAVILY_API_KEY"):
    if key in st.secrets:
        os.environ[key] = str(st.secrets[key])

claim = st.text_area("ألصق نص الخبر أو الادعاء", height=180, placeholder="الصق هنا الخبر كاملاً أو أهم فقراته...")
max_results = st.slider("عدد النتائج القصوى", 10, 100, 50, 10)

if st.button("🛰️ تتبع انتشار الخبر", type="primary", use_container_width=True):
    if not claim.strip():
        st.warning("أدخل نص الخبر أولاً.")
    else:
        with st.spinner("يتم البحث والمقارنة وبناء سجل الانتشار..."):
            report = monitor_claim(claim.strip(), SOURCES, max_results=max_results)
            summary = persist_report(report, DB)
        st.session_state["prop_report"] = report
        st.session_state["prop_summary"] = summary

report = st.session_state.get("prop_report")
summary = st.session_state.get("prop_summary")
if report:
    evidence = report.get("evidence", [])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("النتائج", len(evidence))
    c2.metric("تطابق قوي", sum(1 for x in evidence if x.get("match", 0) >= .75))
    c3.metric("نطاقات", len(summary.get("domain_counts", {})))
    c4.metric("منصات", len(summary.get("platform_counts", {})))

    st.subheader("🧬 بصمة الخبر")
    st.write("العبارات المفتاحية المستخدمة لاكتشاف النسخ المعاد صياغتها:")
    st.code("\n".join(key_phrases(claim)), language="text")

    st.subheader("📍 المرشح الأول للظهور")
    first = summary.get("first_observed_candidate")
    if first:
        st.write(f"**{first.get('domain','')}** — {first.get('title','')}")
        st.write(f"التاريخ المفهرس: {first.get('published','') or 'غير متاح'} | التطابق: {float(first.get('match',0)):.0%}")
        st.link_button("فتح المصدر", first.get("url", ""))
    else:
        st.info("لم تظهر نتيجة قابلة للرصد.")

    st.subheader("📡 خريطة الانتشار حسب المنصة")
    pc = pd.DataFrame([{"المنصة": k, "النتائج": v} for k, v in summary.get("platform_counts", {}).items()])
    if not pc.empty:
        st.bar_chart(pc.set_index("المنصة"))

    st.subheader("🌐 خريطة الانتشار حسب النطاق")
    dc = pd.DataFrame([{"النطاق": k, "النتائج": v} for k, v in summary.get("domain_counts", {}).items()])
    if not dc.empty:
        st.dataframe(dc, use_container_width=True, hide_index=True)

    st.subheader("🧾 سلسلة الأدلة")
    table = pd.DataFrame(evidence)
    cols = [c for c in ["platform", "domain", "account_or_page", "published", "match", "match_type", "title", "url"] if c in table.columns]
    table = table[cols].copy()
    if "match" in table:
        table["match"] = table["match"].map(lambda x: f"{float(x):.0%}")
    st.dataframe(table, use_container_width=True, hide_index=True, column_config={"url": st.column_config.LinkColumn("الرابط")})

    st.download_button("تنزيل سجل النتائج CSV", report.get("claim", "report").encode("utf-8"), file_name="claim.txt")

st.divider()
st.warning("هذا النظام يتتبع المحتوى العام المفهرس والمتاح فقط. لا يمكنه الوصول إلى المنشورات الخاصة أو المحتوى خلف تسجيل الدخول، ولا يمكنه الجزم بأن أقدم نتيجة مفهرسة هي أول نشر فعلي.")
st.caption("NewsGuard — التطابق دليل على التشابه/النشر المرصود، وليس حكماً على صحة الخبر.")
