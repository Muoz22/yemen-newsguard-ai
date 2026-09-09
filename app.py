from pathlib import Path
import os
import pandas as pd
import streamlit as st

from src.analyzer import analyze_news
from src.media import extract_video_frames, fetch_public_facebook, image_to_data_url, vision_analysis
from src.web_search import search_public_web, track_propagation

# Streamlit Cloud stores secrets in st.secrets rather than os.environ.
# Mirror only the relevant values so the shared analysis helpers can use them.
try:
    for _secret_name in ("OPENAI_API_KEY", "OPENAI_API_BASE", "NEWSGUARD_VISION_MODEL", "TAVILY_API_KEY"):
        if _secret_name in st.secrets and st.secrets[_secret_name]:
            os.environ[_secret_name] = str(st.secrets[_secret_name])
except Exception:
    pass

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
SOURCES_PATH = DATA_PATH.parent / "sources.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH).fillna("")

df = load_data()
sources_df = pd.read_csv(SOURCES_PATH).fillna("") if SOURCES_PATH.exists() else pd.DataFrame()

with st.sidebar:
    st.markdown("## 🛡️ NewsGuard")
    st.caption("منصة تحقق متعددة الوسائط للسياق اليمني")
    st.divider()
    use_ai = st.toggle("تفعيل تحليل AI", value=True)
    ai_key_loaded = bool(os.getenv("OPENAI_API_KEY"))
    if ai_key_loaded:
        st.success("اتصال AI مفعّل")
    else:
        st.warning("لم تتم إضافة OPENAI_API_KEY؛ التحليل النصي المحلي متاح، وتحليل الصور/الفيديو يحتاج مفتاحاً.")
    if os.getenv("TAVILY_API_KEY") and not os.getenv("TAVILY_API_KEY").startswith("مفتاح"):
        st.success("بحث الويب الموسع مفعّل")
    else:
        st.warning("بحث Facebook وX الموسع يحتاج TAVILY_API_KEY حقيقياً")
    st.info("النتيجة مساعدة أولية وليست حكماً نهائياً أو دليلاً على صحة المحتوى.")

st.markdown('<div class="hero" dir="rtl"><div class="badge">نص · صورة · فيديو · Facebook</div><h1>Yemen NewsGuard AI</h1><p>افحص الادعاء قبل المشاركة: حلّل النص، اقرأ ما في الصورة، راجع إطارات الفيديو، واستخرج البيانات العامة من منشورات Facebook.</p></div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("سجلات مرجعية", len(df))
m2.metric("مصادر البحث المضافة", len(sources_df))
m3.metric("أنواع المحتوى", "4")
m4.metric("حالة النظام", "جاهز")

tab_text, tab_image, tab_video, tab_fb = st.tabs(["📝 نص / خبر", "🖼️ صورة", "🎬 فيديو", "📘 Facebook"])

with tab_text:
    st.markdown("### بحث ومقارنة خبر أو ادعاء")
    text = st.text_area("ألصق نص الخبر هنا", height=220, placeholder="أدخل نص الخبر أو الادعاء مع الرابط الأصلي إن وجد.", key="news_text")
    search_web = st.checkbox("البحث في الويب والأخبار العامة والمصادر اليمنية", value=True, key="search_web")
    if st.button("ابحث وقارن الخبر", type="primary", key="analyze_text"):
        if not text.strip():
            st.warning("أدخل نصاً أولاً.")
        else:
            with st.spinner("يجري تحليل النص..."):
                result = analyze_news(text, df, use_ai=use_ai)
            st.session_state["text_result"] = result
            if search_web:
                with st.spinner("يبحث في مصادر الأخبار العامة والويب..."):
                    st.session_state["web_results"] = search_public_web(text, DATA_PATH.parent / "sources.csv")
                evidence = st.session_state["web_results"]
                result.model = "evidence-search-v2"
                result.source = "بحث الويب العام + صفحات المصادر"
                if evidence:
                    best = max(float(item.get("match", 0)) for item in evidence)
                    domains = {str(item.get("source", "")) for item in evidence if item.get("source")}
                    result.score = round(1 - best, 2)
                    result.confidence = round(min(0.95, 0.45 + best * 0.45 + min(len(domains), 3) * 0.04), 2)
                    result.verdict = "تطابق قوي مع خبر منشور" if best >= 0.75 else ("تطابق جزئي يحتاج مراجعة" if best >= 0.45 else "لم يثبت تطابق الخبر")
                    result.reasons = [f"أفضل تطابق خارجي: {best:.0%}.", f"عدد النطاقات الظاهرة: {len(domains)}.", "التقييم مبني على نتائج البحث العامة، وليس على قاعدة الأخبار التجريبية فقط."]
                else:
                    result.verdict = "لم تظهر أدلة منشورة متاحة"
                    result.score = 0.5
                    result.confidence = 0.35
                    result.reasons = ["لم يعثر البحث العام على نتيجة مطابقة بالحد الأدنى.", "عدم العثور على الخبر لا يثبت أنه لم يُنشر."]
    result = st.session_state.get("text_result")
    if result:
        st.divider()
        a, b, c = st.columns(3)
        a.metric("التقييم المبدئي", result.verdict)
        b.metric("مؤشر غياب التطابق", f"{result.score:.0%}")
        c.metric("الثقة", f"{result.confidence:.0%}")
        st.caption(f"المحرك: {result.model} — {result.source}")
        st.warning("هذه النسبة هي مؤشر لقوة الدليل الخارجي: كلما ارتفعت، كان التطابق مع الخبر المنشور أضعف. لا تمثل احتمال صحة الخبر.")
        st.progress(result.score)
        x, y = st.columns(2)
        with x:
            st.markdown("#### الإشارات المرصودة")
            for item in result.reasons: st.markdown(f"- {item}")
        with y:
            st.markdown("#### خطوات التحقق")
            for item in result.recommendations: st.markdown(f"- {item}")
        if result.similar_examples:
            st.markdown("#### سجلات مشابهة")
            st.dataframe(pd.DataFrame(result.similar_examples), use_container_width=True, hide_index=True)
        web_results = st.session_state.get("web_results", [])
        st.markdown("#### أين ظهر هذا الخبر؟")
        if web_results:
            web_df = pd.DataFrame(web_results)
            web_df = web_df.rename(columns={"title": "العنوان", "url": "الرابط", "source": "الموقع", "published": "التاريخ", "snippet": "ملخص", "match": "التشابه النصي", "channel": "القناة", "searched_query": "عبارة البحث", "match_type": "نوع التطابق", "relation": "علاقة الحدث", "attributed_source": "المصدر المنسوب", "publisher_type": "نوع الناشر"})
            social_df = web_df[web_df["القناة"].astype(str).str.contains("Facebook|Twitter|X /", case=False, regex=True)]
            news_df = web_df[~web_df.index.isin(social_df.index)]
            if not news_df.empty:
                st.markdown("##### مواقع الأخبار والويب")
                st.dataframe(news_df[["العنوان", "الموقع", "نوع الناشر", "المصدر المنسوب", "علاقة الحدث", "التاريخ", "التشابه النصي", "نوع التطابق", "الرابط"]], use_container_width=True, hide_index=True, column_config={"الرابط": st.column_config.LinkColumn("الرابط")})
            if not social_df.empty:
                st.markdown("##### منشورات وصفحات Facebook وX العامة")
                st.dataframe(social_df[["العنوان", "الموقع", "القناة", "علاقة الحدث", "التاريخ", "التشابه النصي", "نوع التطابق", "الرابط"]], use_container_width=True, hide_index=True, column_config={"الرابط": st.column_config.LinkColumn("الرابط")})
            elif os.getenv("TAVILY_API_KEY"):
                st.info("لم يجد فهرس الويب منشوراً عاماً مطابقاً من Facebook أو X لهذا النص. قد يكون المنشور غير مفهرس أو يتطلب تسجيل الدخول.")
            max_match = max(float(value) for value in web_df["التشابه النصي"])
            if max_match < 0.45:
                st.warning("لم يظهر تطابق قريب بما يكفي مع نص الخبر. النتائج أعلاه مجرد مواد مرتبطة للمتابعة وليست دليلاً على أن نفس الخبر نُشر فيها.")
            else:
                st.success("ظهرت نتائج ذات تشابه نصي مرتفع نسبياً؛ افتح الروابط وراجع التاريخ والمصدر قبل اعتبارها إعادة نشر لنفس الخبر.")
            st.caption("النتائج مأخوذة من مصادر عامة وفهارس أخبار متاحة؛ عدم ظهور الخبر لا يعني أنه لم يُنشر في أي مكان.")
        else:
            st.info("لم تظهر نتائج عامة متاحة. جرّب عبارة أقصر أو أضف كلمات مثل المكان والجهة والتاريخ.")

with tab_image:
    st.markdown("### تحليل صورة أو لقطة شاشة")
    image = st.file_uploader("ارفع صورة (PNG/JPG/WebP)", type=["png", "jpg", "jpeg", "webp"], key="image_upload")
    image_context = st.text_area("ما الادعاء المرتبط بالصورة؟", placeholder="مثال: يقال إن هذه الصورة من حدث وقع اليوم في صنعاء.", key="image_context")
    if image:
        st.image(image, caption="المحتوى المرفوع", use_container_width=True)
    if st.button("فحص الصورة واستخراج النص", type="primary", key="analyze_image"):
        if not image:
            st.warning("ارفع صورة أولاً.")
        elif not os.getenv("OPENAI_API_KEY"):
            st.error("تحليل الصور يحتاج إضافة OPENAI_API_KEY في Streamlit Secrets.")
        else:
            with st.spinner("يفحص النموذج الصورة ويستخرج النص الظاهر..."):
                data = vision_analysis([image_to_data_url(image.getvalue(), image.type)], image_context, os.environ["OPENAI_API_KEY"], os.getenv("NEWSGUARD_VISION_MODEL", "gpt-4o-mini"))
            if data:
                st.session_state["image_result"] = data
            else: st.error("تعذر تحليل الصورة. تحقق من المفتاح وحجم الصورة.")
    if st.session_state.get("image_result"):
        data = st.session_state["image_result"]
        st.markdown("#### نتيجة الفحص البصري")
        st.write("**النص المستخرج:**", data.get("extracted_text", "لم يتم التعرف على نص"))
        st.write("**الوصف:**", data.get("visual_summary", ""))
        st.write("**مؤشر المخاطر:**", f"{float(data.get('score', 0.5)):.0%}")
        for item in data.get("risk_signals", []): st.markdown(f"- {item}")
        st.markdown("#### ماذا تتحقق منه؟")
        for item in data.get("verification_steps", []): st.markdown(f"- {item}")

with tab_video:
    st.markdown("### تحليل فيديو عبر لقطات ممثلة")
    video = st.file_uploader("ارفع فيديو (MP4/MOV/WEBM)", type=["mp4", "mov", "webm"], key="video_upload")
    video_context = st.text_area("ما الادعاء المرتبط بالفيديو؟", key="video_context")
    if video: st.video(video)
    if st.button("استخراج اللقطات وفحص الفيديو", type="primary", key="analyze_video"):
        if not video: st.warning("ارفع فيديو أولاً.")
        elif not os.getenv("OPENAI_API_KEY"): st.error("تحليل الفيديو يحتاج إضافة OPENAI_API_KEY في Streamlit Secrets.")
        else:
            with st.spinner("يستخرج النظام لقطات من الفيديو..."):
                frames = extract_video_frames(video.getvalue(), Path(video.name).suffix.lower(), 3)
                urls = [image_to_data_url(frame) for frame in frames]
                data = vision_analysis(urls, video_context, os.environ["OPENAI_API_KEY"], os.getenv("NEWSGUARD_VISION_MODEL", "gpt-4o-mini"))
            if not frames: st.error("تعذر استخراج اللقطات. قد يحتاج خادم Streamlit إلى دعم ffmpeg.")
            elif data:
                st.session_state["video_result"] = data
                st.image(frames, caption=[f"لقطة {i+1}" for i in range(len(frames))], width=250)
            else: st.error("تعذر تحليل لقطات الفيديو.")
    if st.session_state.get("video_result"):
        data = st.session_state["video_result"]
        st.markdown("#### نتيجة فحص اللقطات")
        st.write("**النص المستخرج:**", data.get("extracted_text", ""))
        st.write("**الوصف:**", data.get("visual_summary", ""))
        for item in data.get("risk_signals", []): st.markdown(f"- {item}")
        st.caption("الفحص يعتمد على لقطات ممثلة، ولا يثبت صحة الصوت أو كل ثانية من الفيديو. راجع الفيديو الأصلي ومصدره.")

with tab_fb:
    st.markdown("### فحص منشور Facebook عام")
    fb_url = st.text_input("رابط المنشور العام", placeholder="https://www.facebook.com/...", key="fb_url")
    fb_text = st.text_area("الصق نص المنشور أو الادعاء يدوياً (موصى به)", height=150, key="fb_text")
    if st.button("جلب البيانات العامة وتحليل المنشور", type="primary", key="analyze_fb"):
        if not fb_url and not fb_text.strip(): st.warning("أدخل رابطاً أو الصق نص المنشور.")
        else:
            metadata = fetch_public_facebook(fb_url) if fb_url else {}
            combined = "\n".join(x for x in [metadata.get("title", ""), metadata.get("description", ""), fb_text] if x)
            if combined:
                st.session_state["fb_metadata"] = metadata
                st.session_state["fb_result"] = analyze_news(combined, df, use_ai=use_ai)
            else: st.warning("لم يمكن قراءة البيانات العامة؛ الصق نص المنشور يدوياً.")
    if st.session_state.get("fb_metadata", {}).get("status"): st.info(st.session_state["fb_metadata"]["status"])
    if st.session_state.get("fb_result"):
        result = st.session_state["fb_result"]
        a, b, c = st.columns(3)
        a.metric("التقييم", result.verdict); b.metric("المؤشر", f"{result.score:.0%}"); c.metric("الثقة", f"{result.confidence:.0%}")
        for item in result.reasons: st.markdown(f"- {item}")
    st.markdown("#### تتبع إعادة النشر والصيغ المشابهة")
    st.caption("أدخل نص المنشور أو رابطاً عاماً ثم شغّل التتبع للبحث عن الحسابات والصفحات التي أعادت نشره أو ناقشت الحدث نفسه.")
    if st.button("تتبع الانتشار في الويب وFacebook وX", key="track_fb", type="secondary"):
        metadata = st.session_state.get("fb_metadata", {})
        if fb_url and not metadata:
            metadata = fetch_public_facebook(fb_url)
        combined = "\n".join(x for x in [metadata.get("title", ""), metadata.get("description", ""), fb_text] if x)
        if not combined.strip():
            st.warning("أدخل نص المنشور أولاً ثم اضغط تتبع الانتشار.")
        else:
            with st.spinner("يبحث عن إعادة النشر والصيغ المشابهة..."):
                st.session_state["fb_propagation"] = track_propagation(combined, SOURCES_PATH, max_results=40)
            st.session_state["fb_propagation_query"] = combined
    propagation = st.session_state.get("fb_propagation", [])
    if st.session_state.get("fb_propagation_query"):
        st.markdown("#### تتبع انتشار المنشور")
        st.caption("النتائج مصنفة إلى إعادة نشر محتملة، أو نفس الحدث، أو موضوع قريب. لا تعني كثرة النتائج صحة الادعاء.")
        st.info(f"عدد الصفحات والنتائج المطابقة أو المرتبطة: {len(propagation)}")
        if not propagation:
            st.warning("لم يعثر البحث العام على صفحات مفهرسة تحمل النص أو سياقاً قريباً. هذا لا يعني عدم وجود إعادة نشر؛ قد تكون المنشورات خاصة أو غير مفهرسة أو محجوبة عن محركات البحث.")
    if propagation:
        propagation_df = pd.DataFrame(propagation).rename(columns={"title": "العنوان", "url": "الرابط", "source": "الموقع", "published": "التاريخ", "match": "التشابه", "channel": "القناة", "relation": "العلاقة", "publisher_type": "نوع الناشر", "attributed_source": "المصدر المنسوب", "searched_query": "عبارة البحث"})
        columns = ["العنوان", "الموقع", "القناة", "العلاقة", "نوع الناشر", "المصدر المنسوب", "التشابه", "التاريخ", "الرابط"]
        st.dataframe(propagation_df[[column for column in columns if column in propagation_df.columns]], use_container_width=True, hide_index=True, column_config={"الرابط": st.column_config.LinkColumn("الرابط")})

with st.expander("📚 قاعدة الأخبار المرجعية"):
    st.dataframe(df, use_container_width=True, hide_index=True)
st.divider()
st.markdown('<div class="notice">تنبيه: لا يستطيع أي نظام وحده إثبات أن صورة أو فيديو أو منشور صحيح. المحتوى الخاص في Facebook غير متاح دون صلاحية المستخدم، والفيديو يفحص لقطات ممثلة فقط. تحقق دائماً من المصدر والتاريخ والسياق.</div>', unsafe_allow_html=True)
st.caption("Yemen NewsGuard AI · مشروع بحثي مفتوح المصدر · لا تعِد النشر قبل التحقق")
