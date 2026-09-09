import streamlit as st
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Yemen NewsGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🛡️ Yemen NewsGuard AI")
st.subheader("نظام ذكي لرصد وتحليل الأخبار المضللة في السياق اليمني")

st.markdown(
    """
    هذا هو الإصدار الأول من **Yemen NewsGuard AI**.
    
    يقوم النظام حاليًا بتحميل قاعدة الأخبار المستخدمة في المشروع،
    تمهيدًا لإضافة نماذج الذكاء الاصطناعي والتحقق من الادعاءات.
    """
)

st.divider()

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

DATA_PATH = Path("data/news_dataset.csv")

try:
    df = pd.read_csv(DATA_PATH)

    st.success(f"تم تحميل قاعدة البيانات بنجاح — {len(df)} خبر")

    # --------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("عدد الأخبار", len(df))

    with col2:
        st.metric("عدد المصادر", df["source"].nunique())

    with col3:
        st.metric("التصنيفات", df["label"].nunique())

    st.divider()

    # --------------------------------------------------
    # Dataset preview
    # --------------------------------------------------

    st.subheader("📊 قاعدة البيانات")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # News analysis interface
    # --------------------------------------------------

    st.subheader("🔍 تحليل خبر")

    news_text = st.text_area(
        "ألصق نص الخبر هنا:",
        height=220,
        placeholder="اكتب أو ألصق نص الخبر الذي تريد تحليله..."
    )

    if st.button("تحليل الخبر", type="primary"):

        if not news_text.strip():

            st.warning("يرجى إدخال نص الخبر أولًا.")

        else:

            st.info(
                "تم استلام الخبر. "
                "سيتم ربط نموذج الذكاء الاصطناعي في المرحلة التالية."
            )

except FileNotFoundError:

    st.error(
        "لم يتم العثور على ملف البيانات: "
        "`data/news_dataset.csv`"
    )

except Exception as e:

    st.error(f"حدث خطأ أثناء تحميل البيانات: {e}")
