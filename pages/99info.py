# pages/99_info.py
import streamlit as st

st.set_page_config(page_title="ℹ️ Info", layout="centered")

# ==================== عنوان الصفحة ====================
st.markdown("<h1 style='text-align: center;'>ℹ️ About the Application & Developer</h1>", unsafe_allow_html=True)

st.markdown("---")

# ==================== شرح التطبيق ====================
st.markdown("""
<div style="font-size:16px; line-height:1.8;">
<strong>About the Application:</strong><br>
This Data Analysis & Machine Learning App allows users to:
<ul>
<li>Upload datasets in CSV or Excel format.</li>
<li>Explore and analyze data with interactive visualizations.</li>
<li>Train and compare machine learning models.</li>
</ul>
It is designed to be user-friendly and suitable for both beginners and advanced users in data analytics.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================== معلومات التواصل ====================
st.markdown("""
<div style="font-size:16px; line-height:1.8;">
📧 <b>Email:</b> <a href="mailto:alroby5200@gmail.com">alroby5200@gmail.com</a><br>
📞 <b>Phone:</b> +20 106 389 4392<br>
🔗 <b>LinkedIn:</b> <a href="https://www.linkedin.com/in/mohamed-ruby-8a3582253?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank">Mohamed Ruby</a><br>
🏠 <b>Address:</b> Egypt - Cairo
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================== نسخة التطبيق ====================
st.markdown("""
<div style="text-align:center; font-size:14px; color:gray;">
Version 1.0.0
</div>
""", unsafe_allow_html=True)
