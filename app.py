import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load('diabetes_model.joblib')

model = load_model()

st.title("🩺 أداة فحص مخاطر مرض السكري")
st.write("أداة فحص تعتمد على استبيان صحي قصير لتقدير احتمالية الإصابة بمرض السكري أو ما قبل السكري.")

st.warning("⚠️ **تنبيه هام:** هذه الأداة هي أداة فحص مبدئي فقط وليست تشخيصاً طبياً رسمياً، ويُرجى دائماً مراجعة الطبيب المختص.")

st.subheader("املأ البيانات التالية لتقييم حالتك:")

with st.form("diabetes_form"):
    high_bp = st.selectbox("هل تعاني من ارتفاع ضغط الدم؟", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    high_chol = st.selectbox("هل تعاني من ارتفاع الكوليسترول؟", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    bmi = st.number_input("مؤشر كتلة الجسم (BMI)", min_value=10.0, max_value=60.0, value=25.0)
    gen_hlth = st.selectbox("كيف تقيم صحتك العامة؟ (1 ممتاز إلى 5 سيء)", [1, 2, 3, 4, 5])
    age = st.slider("اختر الفئة العمرية (حسب تصنيف الاستطلاع من 1 إلى 13)", 1, 13, 5)
    
    submit_button = st.form_submit_button(label="فحص المخاطر")

if submit_button:
    input_data = pd.DataFrame(columns=model.named_steps['classifier'].feature_names_in_)
    input_data.loc[0] = 0  
    if 'HighBP' in input_data.columns: input_data['HighBP'] = high_bp
    if 'HighChol' in input_data.columns: input_data['HighChol'] = high_chol
    if 'BMI' in input_data.columns: input_data['BMI'] = bmi
    if 'GenHlth' in input_data.columns: input_data['GenHlth'] = gen_hlth
    if 'Age' in input_data.columns: input_data['Age'] = age

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("نتيجة الفحص:")
    if prediction == 1:
        st.error(f"⚠️ **مخاطر مرتفعة** (احتمالية الإصابة: {probability * 100:.1f}%). يُنصح بجدولة موعد مع الطبيب لإجراء الفحوصات الطبية اللازمة.")
    else:
        st.success(f"✅ **مخاطر منخفضة** (احتمالية الإصابة: {probability * 100:.1f}%). استمر في الحفاظ على نمط حياة صحي ونشط.")
