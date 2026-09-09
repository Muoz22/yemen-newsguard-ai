import streamlit as st

st.set_page_config(
    page_title="Yemen NewsGuard AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Yemen NewsGuard AI")
st.subheader("نظام ذكي لرصد الأخبار المضللة والتحقق من الادعاءات")

st.markdown("""
هذا الإصدار الأول من النظام.
سيتم تطويره لاحقًا ليشمل تحليل الأخبار، استخراج الادعاءات،
والبحث عن الأدلة والمصادر.
""")

st.divider()

news_text = st.text_area(
    "ألصق الخبر هنا",
    height=200,
    placeholder="اكتب أو ألصق نص الخبر الذي تريد تحليله..."
)

if st.button("🔍 تحليل الخبر"):
    if news_text.strip():
        st.info("النسخة الأولى جاهزة لاستقبال الخبر. سيتم إضافة نموذج التحليل في المرحلة التالية.")
    else:
        st.warning("يرجى إدخال نص الخبر أولًا.")
