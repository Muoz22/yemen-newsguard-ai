from pathlib import Path
import os
import pandas as pd
import streamlit as st

from src.monitoring import monitor_claim, report_csv, report_html, report_json

st.set_page_config(page_title="Yemen NewsGuard — Intelligence Monitor", page_icon="🛰️", layout="wide")

try:
    for name in ("OPENAI_API_KEY", "OPENAI_API_BASE", "NEWSGUARD_SEARCH_MODEL", "TAVILY_API_KEY"):
        if name in st.secrets and st.secrets[name]:
            os.environ[name] = str(st.secrets[name])
except Exception:
    pass

BASE = Path(__file__).parent
SOURCES = BASE / "data" / "sources.csv"
QUERIES = BASE / "data" / "monitor_queries.csv"

st.markdown("<h1 dir='rtl'>🛰️ Yemen NewsGuard — رصد انتشار الأخبار</h1><p dir='rtl'>محرك منظم للعثور على النسخ المنشورة والنتائج العامة ذات الصلة عبر المواقع والمنصات المفهرسة.</p>", unsafe_allow_html=True)

if not os.getenv("TAVILY_API_KEY"):
    st.warning("لأوسع تغطية للويب العام، أضف TAVILY_API_KEY إلى Secrets. بدونها سيعمل البحث عبر Google News/Bing RSS والصفحات العامة المهيأة فقط.")
if not os.getenv("OPENAI_API_KEY"):
    st.info("OPENAI_API_KEY اختياري لتحسين توليد عبارات البحث الدقيقة؛ البحث الأساسي لا يتوقف عليه.")

left, right = st.columns([3, 1])
with left:
    claim = st.text_area("الخبر أو الادعاء المراد رصد انتشاره", height=180, placeholder="الصق الخبر كاملاً أو عنوانه، ويمكن إضافة رابط المنشور الأصلي.")
with right:
    limit = st.number_input("أقصى عدد نتائج", min_value=10, max_value=100, value=50, step=10)
    st.caption("التغطية تعتمد على الفهرسة العامة وواجهات البحث المتاحة، وليست مسحاً حرفياً لكل الإنترنت.")

if st.button("🔎 ابدأ الرصد والتحليل", type="primary", use_container_width=True):
    if not claim.strip():
        st.error("أدخل الخبر أو الادعاء أولاً.")
    else:
        with st.spinner("يبحث عن النسخ والتغطيات ويزيل التكرارات ويصنف المنصات..."):
            st.session_state["monitor_report"] = monitor_claim(claim.strip(), SOURCES, int(limit))

report = st.session_state.get("monitor_report")
if report:
    st.divider()
    a,b,c,d = st.columns(4)
    a.metric("إجمالي النتائج", report["total_results"])
    b.metric("تطابق قوي", report["strong_matches"])
    c.metric("نطاقات مختلفة", len(report["domains"]))
    d.metric("حسابات/صفحات مرصودة", len(report["accounts_or_pages"]))

    st.markdown("### 📊 خريطة الانتشار")
    p1,p2 = st.columns(2)
    with p1:
        st.markdown("**حسب المنصة**")
        st.dataframe(pd.DataFrame(list(report["platforms"].items()), columns=["المنصة","العدد"]), hide_index=True, use_container_width=True)
    with p2:
        st.markdown("**حسب النطاق**")
        st.dataframe(pd.DataFrame(list(report["domains"].items()), columns=["النطاق","العدد"]), hide_index=True, use_container_width=True)

    evidence = pd.DataFrame(report["evidence"])
    if not evidence.empty:
        evidence = evidence.rename(columns={"title":"العنوان","url":"الرابط","domain":"النطاق","platform":"المنصة","account_or_page":"الحساب/الصفحة","source":"المصدر","published":"النشر","match":"التطابق","match_type":"نوع التطابق","query":"عبارة البحث","snippet":"مقتطف"})
        st.markdown("### 🧾 الأدلة والنتائج")
        st.dataframe(evidence[["المنصة","النطاق","الحساب/الصفحة","التطابق","نوع التطابق","النشر","العنوان","الرابط"]], hide_index=True, use_container_width=True, column_config={"الرابط": st.column_config.LinkColumn("فتح")})
        st.caption("التطابق = تشابه نصي/دلالي مع الخبر المدخل. لا يعني صحة الخبر ولا يثبت أن الحساب أعاد نشر النص حرفياً.")
    else:
        st.warning("لم تظهر نتائج عامة قابلة للإثبات من الفهارس المتاحة. قد يكون المحتوى غير مفهرس أو خاصاً أو محمياً بتسجيل الدخول.")

    st.markdown("### 📥 تصدير التقرير")
    x,y,z = st.columns(3)
    x.download_button("CSV", report_csv(report), "news_monitor_report.csv", "text/csv")
    y.download_button("JSON", report_json(report), "news_monitor_report.json", "application/json")
    z.download_button("HTML", report_html(report), "news_monitor_report.html", "text/html")

st.divider()
with st.expander("⚙️ قوائم الرصد الدورية"):
    if QUERIES.exists():
        qdf = pd.read_csv(QUERIES).fillna("")
        st.dataframe(qdf, hide_index=True, use_container_width=True)
    st.caption("يمكن إضافة عبارات رصد يومية في data/monitor_queries.csv، وسيستخدمها مجدول GitHub Actions في التقارير الدورية.")
