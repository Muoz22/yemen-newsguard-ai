from pathlib import Path
import pandas as pd
import streamlit as st

from src.analyzer import analyze_news

st.set_page_config(page_title="Yemen NewsGuard AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
.block-container { max-width: 1200px; padding-top: 2rem; }
.hero { background: linear-gradient(135deg,#102a43 0%,#176b87 100%); color:white; padding:2.2rem 2.5rem; border-radius:22px; margin-bottom:1.5rem; }
.hero h1 { font-size:2.35rem; margin:0 0 .4rem; }
.hero p { color:#d8f3f0; margin:0; font-size:1.05rem; }
.badge { display:inline-block; background:#d8f3f0; color:#102a43; border-radius:999px; padding:.25rem .8rem; font-size:.82rem; font-weight:700; margin-bottom:.8rem; }
.notice { background:#fff8e6; border-right:4px solid #e0a11a; padding:1rem; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "data" / "news_dataset.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH).fillna("")

try:
    df = load_data()
except Exception as exc:
    st.error(f"تعذر تحميل البيانات: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("## 🛡️ NewsGuard")
    st.caption("نسخة أولية للتحقق المساند من الأخبار في السياق اليمني")
    st.divider()
    use_ai = st.toggle("استخدام نموذج AI عند توفر المفتاح", value=True)
    st.info("النتيجة مؤشر أولي للمساعدة في الفرز، وليست حكماً نهائياً على صحة الخبر.")
    st.divider()
    st.markdown("**خطوات التحقق**")
    st.markdown("1. اقرأ الادعاء كاملاً\n2. افحص المصدر والتاريخ\n3. قارن بمصادر مستقلة\n4. لا تعِد النشر قبل التثبت")

st.markdown('<div class="hero" dir="rtl"><div class="badge">أداة تحقق مساندة</div><h1>Yemen NewsGuard AI</h1><p>حلّل الخبر، افهم إشارات الخطورة، واتخذ خطوة تحقق أفضل قبل المشاركة.</p></div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("سجلات مرجعية", len(df))
m2.metric("مصادر معروفة", df["source"].nunique() if "source" in df else 0)
m3.metric("حقول متاحة", len(df.columns))
m4.metric("حالة النظام", "جاهز")

st.markdown("## 🔍 تحليل خبر أو ادعاء")
left, right = st.columns([1.35, .65], gap="large")
with left:
    news_text = st.text_area("ألصق نص الخبر هنا", height=230, placeholder="مثال: أعلنت جهة رسمية اليوم... أضف رابط المصدر إن وجد.", key="news")
    analyze = st.button("تحليل الخبر الآن", type="primary", use_container_width=True)
with right:
    st.markdown("### كيف يعمل؟")
    st.markdown("يبحث النظام عن مؤشرات مثل غياب المصدر، اللغة العاطفية، قِصر النص، والاعتماد على مصادر مجهولة، ثم يقارن النص بسجلات المشروع.")
    st.markdown('<div class="notice">لا تعني عبارة «مؤشرات أفضل» أن الخبر صحيح؛ التحقق من الأدلة والمصادر مسؤولية المستخدم.</div>', unsafe_allow_html=True)

if analyze:
    if not news_text.strip():
        st.warning("يرجى إدخال نص الخبر أولاً.")
    else:
        with st.spinner("يجري تحليل النص..."):
            result = analyze_news(news_text, df, use_ai=use_ai)
        st.divider()
        st.markdown("## 📋 نتيجة التحليل")
        r1, r2, r3 = st.columns(3)
        r1.metric("التقييم المبدئي", result.verdict)
        r2.metric("مؤشر التضليل", f"{result.score:.0%}")
        r3.metric("درجة الثقة", f"{result.confidence:.0%}")
        st.caption(f"المحرك: {result.model} — {result.source}")
        st.progress(result.score, text="مؤشر الحاجة إلى التحقق (ليس احتمالاً إحصائياً مؤكداً)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### لماذا؟")
            for reason in result.reasons:
                st.markdown(f"- {reason}")
        with c2:
            st.markdown("### الخطوة التالية")
            for item in result.recommendations:
                st.markdown(f"- {item}")
        if result.similar_examples:
            st.markdown("### سجلات مشابهة في قاعدة المشروع")
            st.dataframe(pd.DataFrame(result.similar_examples), use_container_width=True, hide_index=True)

with st.expander("📚 استعراض قاعدة الأخبار المرجعية"):
    st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.caption("Yemen NewsGuard AI · مشروع بحثي مفتوح المصدر · استخدم النتائج كمساعدة لا كبديل عن التحقق الصحفي")
