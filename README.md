# Yemen NewsGuard AI

نظام مفتوح المصدر للتحقق المساند من الأخبار والادعاءات في السياق اليمني. يوفر التطبيق واجهة عربية في Streamlit، ومحركاً محلياً قابلاً للتفسير يعمل دون مفتاح API، مع دعم اختياري لنموذج متوافق مع OpenAI عند إضافة `OPENAI_API_KEY`.

## التشغيل محلياً

```bash
git clone https://github.com/Muoz22/yemen-newsguard-ai.git
cd yemen-newsguard-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

ثم افتح العنوان الذي يظهر في الطرفية. يعمل التحليل المحلي دون اتصال بخدمة خارجية.

## تفعيل نموذج الذكاء الاصطناعي اختيارياً

أضف المفتاح في بيئة التشغيل أو في Streamlit Cloud من خلال **Settings → Secrets**:

```toml
OPENAI_API_KEY = "ضع-المفتاح-هنا"
# اختياري لمزود متوافق مع OpenAI
# OPENAI_API_BASE = "https://api.openai.com/v1"
```

إذا لم يوجد المفتاح أو تعذر الاتصال، يعود التطبيق تلقائياً إلى المحرك المحلي بدلاً من التوقف.

## النشر على Streamlit Community Cloud

1. ادفع هذه الملفات إلى مستودع GitHub.
2. افتح [share.streamlit.io](https://share.streamlit.io) وسجّل الدخول بحساب GitHub.
3. اختر المستودع `Muoz22/yemen-newsguard-ai` والفرع `main` والملف `app.py`.
4. اضغط **Deploy**.
5. من إعدادات التطبيق أضف `OPENAI_API_KEY` فقط إذا أردت تفعيل النموذج الخارجي.

## تنبيه منهجي

النتيجة ليست إثباتاً آلياً بأن الخبر صحيح أو مضلل. هي أداة فرز وتلخيص لإشارات تستحق المراجعة. يجب الرجوع إلى المصدر الأصلي، والبيانات الرسمية، ومصادر مستقلة قبل نشر أي حكم.

## بنية المشروع

- `app.py`: واجهة Streamlit العربية.
- `src/analyzer.py`: التحليل المحلي والاتصال الاختياري بالنموذج.
- `data/news_dataset.csv`: السجلات المرجعية.
- `data/schema.csv`: مخطط الحقول المقترح لتوسيع البيانات.
