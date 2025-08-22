import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

st.set_page_config(page_title="Data Cleaning", layout="wide")

st.title("🛠️ Data Cleaning & Preprocessing")

# Load dataset from session state
if "df" not in st.session_state:
    st.warning("⚠️ No dataset found. Please upload a file from the home page.")
else:
    # always work on the current version
    df = st.session_state["df"].copy()

    st.subheader("📂 Preview")
    st.write(df.head())

    st.markdown("### 🔎 Data Checking")
    col1, col2, col3 = st.columns(3)

    # --- Check buttons ---
    with col1:
        if st.button("🔍 Check Missing Values"):
            missing_counts = df.isnull().sum()
            st.session_state["missing_info"] = missing_counts[missing_counts > 0]

        if "missing_info" in st.session_state:
            if not st.session_state["missing_info"].empty:
                st.write(st.session_state["missing_info"])
            else:
                st.success("✅ No missing values!")

    with col2:
        if st.button("🔍 Check Duplicates"):
            st.session_state["dup_count"] = df.duplicated().sum()

        if "dup_count" in st.session_state:
            if st.session_state["dup_count"] > 0:
                st.info(f"Duplicates: {st.session_state['dup_count']}")
            else:
                st.success("✅ No duplicates!")

    with col3:
        if st.button("🔍 Check Categorical"):
            st.session_state["cat_cols"] = list(df.select_dtypes(include=["object"]).columns)

        if "cat_cols" in st.session_state:
            if st.session_state["cat_cols"]:
                st.write(st.session_state["cat_cols"])
            else:
                st.success("✅ No categorical columns!")

    # --- Missing values handling ---
    st.markdown("#### 🧹 Handle Missing Values")
    missing_action = st.radio(
        "Choose action:",
        ["Do Nothing", "Fill with Mean", "Drop Rows"]
    )

    # --- Categorical handling ---
    st.markdown("#### 🔤 Handle Categorical Values")
    if "cat_cols" in st.session_state and st.session_state["cat_cols"]:
        encode_col = st.selectbox("Select column:", st.session_state["cat_cols"])
        text_action = st.radio("Encoding method:", ["Do Nothing", "Label Encoding", "One Hot Encoding"])
    else:
        encode_col, text_action = None, "Do Nothing"

    # --- Duplicates handling ---
    st.markdown("#### 📑 Handle Duplicates")
    dup_action = st.checkbox("Remove duplicates?")

    # --- Apply ---
    if st.button("💾 Apply Changes"):
        # Missing values
        if missing_action == "Fill with Mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif missing_action == "Drop Rows":
            df = df.dropna()

        # Categorical
        if encode_col and text_action == "Label Encoding":
            le = LabelEncoder()
            df[encode_col] = le.fit_transform(df[encode_col])
        elif encode_col and text_action == "One Hot Encoding":
            df = pd.get_dummies(df, columns=[encode_col], drop_first=True)

        # Duplicates
        if dup_action:
            df = df.drop_duplicates()

        # 🔴 overwrite main dataset
        st.session_state["df"] = df.copy()

        st.success("✅ Changes applied!")
        st.write(st.session_state["df"].head())

    # --- Download ---
    st.subheader("⬇️ Download Cleaned Dataset")
    csv = st.session_state["df"].to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="cleaned.csv", mime="text/csv")

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        st.session_state["df"].to_excel(writer, index=False, sheet_name="Cleaned")
    st.download_button(
        "Download Excel",
        data=buf.getvalue(),
        file_name="cleaned.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
