# main.py
import streamlit as st
import pandas as pd

# ==================== إعداد صفحة التطبيق ====================
st.set_page_config(
    page_title="Data App",
    layout="wide"
)

st.title("👋 Welcome to the Data Analysis & Machine Learning App")

st.markdown("""
This application helps you to:
- Upload your dataset (CSV or Excel).
- Explore and clean the data.
- Train machine learning models.
---
""")

# ==================== رفع الملف ====================
uploaded_file = st.file_uploader(
    "📂 Upload your data file (CSV or Excel)",
    type=["csv", "xls", "xlsx"]
)

# ==================== معالجة الملف بعد الرفع ====================
if uploaded_file:
    try:
        # قراءة الملف حسب الامتداد
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # حفظ الملف في session_state للوصول إليه من الصفحات الأخرى
        st.session_state["df"] = df

        st.success("✅ File uploaded successfully!")

        # ==================== معاينة البيانات ====================
        st.subheader("👀 Preview of the first 5 rows")
        st.write(df.head())

        # ==================== معلومات عن البيانات ====================
        st.subheader("📌 Dataset Information")
        st.write(f"Shape: {df.shape}")
        st.write("Data Types:")
        st.write(df.dtypes)

    except Exception as e:
        st.error(f"Error while reading the file: {e}")
