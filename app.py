from pathlib import Path
import os
from datetime import datetime
import pandas as pd
import streamlit as st
from src.analyzer import analyze_news
from src.media import extract_video_frames, fetch_public_facebook, image_to_data_url, vision_analysis
from src.monitoring import monitor_claim, report_csv, report_json, report_html
from src.intelligence import fingerprint, key_phrases, persist_report

try:
    for name in ("OPENAI_API_KEY", "OPENAI_API_BASE", "NEWSGUARD_VISION_MODEL", "TAVILY_API_KEY"):
        if name in st.secrets and st.secrets[name]: os.environ[name] = str(st.secrets[name])
except Exception: pass

st.set_page_config(page_title="Yemen NewsGuard AI", page_icon="🛡️", layout="wide")
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "news_dataset.csv"
SOURCES_PATH = ROOT / "data" / "sources.csv"
HISTORY_DB = ROOT / "data" / "newsguard_history.db"

df = pd.read_csv(DATA_PATH).fillna("")
sources_df = pd.read_csv(SOURCES_PATH).fillna("") if SOURCES_PATH.exists() else pd.DataFrame()


def pretty_time(value: str) -> str:
    """Convert ISO/UTC timestamps to a compact, readable local-style display."""
    value = str(value or "").strip()
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value[:16].replace("T", " ")


def display_source(item: dict) -> tuple[str, str]:
    """Return original publisher and aggregator separately."""
    source = str(item.get("source", "") or "").strip()
    domain = str(item.get("domain", "") or "").strip().lower()
    if domain == "sahaafa.net":
        return (source if source and source != "صحافة نت" else "غير مستخرج", "صحافة نت")
    return (source or domain or "غير معروف", "—")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Cairo',sans-serif}.block-container{max-width:1250px;padding-top:2rem}.hero{background:linear-gradient(135deg,#102a43,#176b87);color:white;padding:2.2rem 2.5rem;border-radius:22px;margin-bottom:1.5rem}.hero h1{font-size:2.35rem;margin:0 0 .4rem}.hero p{color:#d8f3f0;margin:0;font-size:1.05rem}.badge{display:inline-block;background:#d8f3f0;color:#102a43;border-radius:999px;padding:.25rem .8rem;font-size:.82rem;font-weight:700;margin-bottom:.8rem}.notice{background:#fff8e6;border-right:4px solid #e0a11a;padding:1rem;border-radius:10px}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🛡️ NewsGuard")
    st.caption("تحقق متعدد الوسائط + استخبارات انتشار الأخبار")
    use_ai = st.toggle("تفعيل تحليل AI", value=True)
    if os.getenv("OPENAI_API_KEY"):
        st.success("اتصال AI مفعّل")
    else:
        st.warning("OPENAI_API_KEY غير مضاف")
    if os.getenv("TAVILY_API_KEY"):
        st.success("بحث الويب الموسع مفعّل")
    else:
        st.warning("TAVILY_API_KEY غير مضاف")
    st.info("التطابق والانتشار لا يثبتان صحة الخبر.")

st.markdown('<div class="hero" dir="rtl"><div class="badge">NewsGuard · Propagation Intelligence</div><h1>Yemen NewsGuard AI</h1><p>ابحث عن الخبر في المصادر العامة، وابنِ بصمته، وسجّل أين ظهر ومتى رُصد وأي منصات أو حسابات عامة تشير إليه.</p></div>', unsafe_allow_html=True)

m1,m2,m3,m4=st.columns(4)
m1.metric("سجلات مرجعية",len(df)); m2.metric("مصادر البحث",len(sources_df)); m3.metric("أنواع المحتوى","4"); m4.metric("الحالة","جاهز")

tab_text,tab_image,tab_video,tab_fb=st.tabs(["📝 نص / خبر","🖼️ صورة","🎬 فيديو","📘 Facebook"])

with tab_text:
    st.markdown("### بحث ومقارنة وتتبع انتشار خبر")
    text=st.text_area("ألصق نص الخبر هنا",height=220,placeholder="أدخل نص الخبر أو الادعاء مع الرابط الأصلي إن وجد.",key="news_text")
    search_web=st.checkbox("البحث في الويب والأخبار العامة والمصادر اليمنية",value=True,key="search_web")
    max_results=st.slider("عدد النتائج القصوى",20,100,50,10,key="max_results")
    if st.button("ابحث وقارن وتتبع الانتشار",type="primary",key="analyze_text"):
        if not text.strip(): st.warning("أدخل نصاً أولاً.")
        else:
            with st.spinner("تحليل الخبر وبناء سجل الانتشار..."):
                result=analyze_news(text,df,use_ai=use_ai)
                report=monitor_claim(text,SOURCES_PATH,max_results=max_results) if search_web else {"claim":text,"evidence":[],"total_results":0,"strong_matches":0,"domains":{},"platforms":{},"accounts_or_pages":[]}
                summary=persist_report(report,HISTORY_DB) if search_web else None
            st.session_state["text_result"]=result
            st.session_state["propagation_report"]={"report":report,"summary":summary}

    result=st.session_state.get("text_result")
    prop=st.session_state.get("propagation_report")
    if result:
        a,b,c=st.columns(3); a.metric("التقييم",result.verdict); b.metric("مؤشر غياب التطابق",f"{result.score:.0%}"); c.metric("الثقة",f"{result.confidence:.0%}")
        st.caption(f"المحرك: {result.model} — {result.source}")
        for item in result.reasons: st.markdown(f"- {item}")
        if result.similar_examples: st.dataframe(pd.DataFrame(result.similar_examples),use_container_width=True,hide_index=True)

    if prop and prop.get("summary"):
        report=prop["report"]; summary=prop["summary"]
        st.divider(); st.markdown("## 📡 أين ظهر هذا الخبر؟")
        st.caption("أول ظهور هنا يعني أول ظهور مرصود/مفهرس في البيانات المتاحة، وليس بالضرورة أول نشر في العالم.")
        p1,p2,p3,p4,p5=st.columns(5)
        p1.metric("النتائج",summary["observations"]); p2.metric("تطابقات قوية",summary["strong_matches"]); p3.metric("المصادر المتنوعة",len(summary["domain_counts"])); p4.metric("المنصات",len(summary["platform_counts"])); p5.metric("Story ID",summary["story_id"][:10])
        st.markdown("### 🔎 بصمة الخبر"); st.code(fingerprint(text)); st.write("**عبارات مفتاحية:**"," · ".join(key_phrases(text)))
        history=summary.get("history",[])
        if history:
            rows=[]
            for x in history:
                match=float(x.get("match",0) or 0); domain=str(x.get("domain","")); platform=str(x.get("platform",""))
                original_source, aggregator = display_source(x)
                if "sahaafa.net" in domain: status="مجمع/فهرسة"
                elif platform in {"Facebook","X / Twitter","Telegram","TikTok","Instagram","YouTube"}: status="إشارة/إعادة نشر محتملة"
                elif match>=.85: status="تطابق قوي"
                elif match>=.70: status="تطابق محتمل"
                else: status="مادة مرتبطة"
                rows.append({
                    "عنوان الخبر":x.get("title") or "—",
                    "المصدر الأصلي":original_source,
                    "المجمّع":aggregator,
                    "النطاق":domain or "—",
                    "المنصة":platform or "—",
                    "الحساب/الصفحة":x.get("account_or_page","") or "—",
                    "التاريخ المنشور":pretty_time(x.get("published")),
                    "أول رصد":pretty_time(x.get("first_seen")),
                    "آخر رصد":pretty_time(x.get("last_seen")),
                    "التطابق":f"{match:.0%}",
                    "الحالة":status,
                    "الرابط":x.get("url","")
                })
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,column_config={"الرابط":st.column_config.LinkColumn("الرابط")})
            candidate=summary.get("first_observed_candidate")
            if candidate:
                candidate_source, candidate_aggregator = display_source(candidate)
                candidate_time = pretty_time(candidate.get("published") or candidate.get("first_seen"))
                st.info(f"أقدم مرشح مرصود وفق التاريخ المتاح: {candidate_source} — {candidate_time}. هذا ليس إثباتاً للمصدر الأصلي.")
            d1,d2=st.columns(2)
            with d1: st.markdown("**حسب المنصة**"); st.bar_chart(pd.Series(summary["platform_counts"],name="عدد النتائج"))
            with d2: st.markdown("**حسب المصدر/النطاق**"); st.bar_chart(pd.Series(summary["domain_counts"],name="عدد النتائج"))
            timeline=[{"الوقت/التاريخ المتاح":pretty_time(x.get("published") or x.get("first_seen")),"عنوان الخبر":x.get("title") or "—","المصدر":display_source(x)[0],"المنصة":x.get("platform"),"التطابق":float(x.get("match",0) or 0)} for x in history if x.get("published") or x.get("first_seen")]
            st.markdown("### 🧭 خط زمني للرصد"); st.dataframe(pd.DataFrame(timeline),use_container_width=True,hide_index=True)
        st.markdown("### 📥 التقرير")
        d1,d2,d3=st.columns(3)
        d1.download_button("تنزيل CSV",report_csv(report),"newsguard_propagation.csv","text/csv")
        d2.download_button("تنزيل JSON",report_json(report),"newsguard_propagation.json","application/json")
        d3.download_button("تنزيل HTML",report_html(report),"newsguard_propagation.html","text/html")

with tab_image:
    st.markdown("### تحليل صورة أو لقطة شاشة")
    image=st.file_uploader("ارفع صورة",type=["png","jpg","jpeg","webp"],key="image_upload")
    ctx=st.text_area("ما الادعاء المرتبط بالصورة؟",key="image_context")
    if image: st.image(image,use_container_width=True)
    if st.button("فحص الصورة",type="primary",key="analyze_image"):
        if not image: st.warning("ارفع صورة أولاً.")
        elif not os.getenv("OPENAI_API_KEY"): st.error("تحليل الصور يحتاج OPENAI_API_KEY.")
        else:
            data=vision_analysis([image_to_data_url(image.getvalue(),image.type)],ctx,os.environ["OPENAI_API_KEY"],os.getenv("NEWSGUARD_VISION_MODEL","gpt-4o-mini")); st.session_state["image_result"]=data
    if st.session_state.get("image_result"):
        data=st.session_state["image_result"]; st.write("**النص:**",data.get("extracted_text","")); st.write("**الوصف:**",data.get("visual_summary",""))
        for x in data.get("risk_signals",[]): st.markdown(f"- {x}")

with tab_video:
    st.markdown("### تحليل فيديو عبر لقطات ممثلة")
    video=st.file_uploader("ارفع فيديو",type=["mp4","mov","webm"],key="video_upload")
    ctx=st.text_area("ما الادعاء المرتبط بالفيديو؟",key="video_context")
    if video: st.video(video)
    if st.button("فحص الفيديو",type="primary",key="analyze_video"):
        if not video: st.warning("ارفع فيديو أولاً.")
        elif not os.getenv("OPENAI_API_KEY"): st.error("تحليل الفيديو يحتاج OPENAI_API_KEY.")
        else:
            frames=extract_video_frames(video.getvalue(),Path(video.name).suffix.lower(),3); urls=[image_to_data_url(f) for f in frames]
            data=vision_analysis(urls,ctx,os.environ["OPENAI_API_KEY"],os.getenv("NEWSGUARD_VISION_MODEL","gpt-4o-mini")) if urls else None
            if data: st.session_state["video_result"]=data
            if frames: st.image(frames,caption=[f"لقطة {i+1}" for i in range(len(frames))],width=250)
    if st.session_state.get("video_result"):
        data=st.session_state["video_result"]; st.write("**النص:**",data.get("extracted_text","")); st.write("**الوصف:**",data.get("visual_summary",""))
        st.caption("الفحص يعتمد على لقطات ممثلة فقط.")

with tab_fb:
    st.markdown("### فحص منشور Facebook عام")
    fb_url=st.text_input("رابط المنشور العام",key="fb_url"); fb_text=st.text_area("الصق نص المنشور",key="fb_text")
    if st.button("جلب وتحليل",type="primary",key="analyze_fb"):
        metadata=fetch_public_facebook(fb_url) if fb_url else {}; combined="\n".join(x for x in [metadata.get("title",""),metadata.get("description",""),fb_text] if x)
        if combined: st.session_state["fb_result"]=analyze_news(combined,df,use_ai=use_ai)
        else: st.warning("لم يمكن قراءة البيانات؛ الصق النص يدوياً.")
    if st.session_state.get("fb_result"):
        r=st.session_state["fb_result"]; st.write(r.verdict); [st.markdown(f"- {x}") for x in r.reasons]

with st.expander("📚 قاعدة الأخبار المرجعية"):
    st.dataframe(df,use_container_width=True,hide_index=True)
st.markdown('<div class="notice">الطبقة الجديدة ترصد المحتوى العام المفهرس فقط. المحتوى الخاص أو غير المفهرس لا يظهر. أول رصد ليس بالضرورة أول نشر، والتطابق لا يثبت صحة الخبر.</div>',unsafe_allow_html=True)
