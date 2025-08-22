# advanced_data_analysis_app_fixed_v2.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import io

st.set_page_config(page_title="Advanced Data Analysis", layout="wide")
st.title("📊 Advanced Data Analysis & Visualization (Fixed v2)")

# --- Dataset Upload & Session State ---
st.markdown("### 📂 Dataset Source")
source = st.radio("Select dataset source:", ["Cleaned Dataset", "Upload New Dataset"])

df = None

if source == "Cleaned Dataset":
    if "df" in st.session_state:
        df = st.session_state["df"].copy()
        st.success("✅ Using cleaned dataset from session.")
    else:
        st.error("⚠️ No cleaned dataset found! Please upload a new dataset.")
elif source == "Upload New Dataset":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df.copy()
        st.success("✅ Dataset uploaded and saved to session.")

# --- Function to download Plotly figure ---
def download_plotly_fig(fig, filename="chart.html"):
    buf = io.StringIO()
    fig.write_html(buf, include_plotlyjs='cdn')
    st.download_button(
        label="📥 Download Chart (HTML)",
        data=buf.getvalue().encode(),  # تحويل النص إلى bytes
        file_name=filename,
        mime="text/html"
    )

# --- Main Analysis Section ---
if df is not None:
    st.subheader("🔍 Dataset Preview")
    st.write(df.head())

    st.markdown("### ℹ️ Dataset Info")
    st.write({
        "Number of rows": df.shape[0],
        "Number of columns": df.shape[1],
        "Columns with missing values": df.isna().sum().to_dict()
    })

    # Descriptive Statistics
    if st.checkbox("Show Descriptive Statistics"):
        st.write(df.describe(include="all"))

    # --- Tabs for Visualizations ---
    st.markdown("---")
    st.subheader("📈 Visualizations")
    tabs = st.tabs(["Bar Chart", "Line Chart", "Pie Chart", "Histogram", "Scatter Plot", "Box Plot", "Heatmap"])

    # --- Bar Chart ---
    with tabs[0]:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            col = st.selectbox("Select categorical column", cat_cols, key="bar_col")
            bar_data = df[col].value_counts().reset_index()
            bar_data.columns = [col, "Count"]
            fig = px.bar(bar_data, x=col, y="Count", title=f"Bar Chart of {col}")
            st.plotly_chart(fig, use_container_width=True)
            download_plotly_fig(fig, f"bar_chart_{col}.html")
        else:
            st.warning("⚠️ No categorical columns available.")

    # --- Line Chart ---
    with tabs[1]:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            col = st.selectbox("Select numeric column", num_cols, key="line_col")
            fig = px.line(df, y=col, title=f"Line Chart of {col}")
            st.plotly_chart(fig, use_container_width=True)
            download_plotly_fig(fig, f"line_chart_{col}.html")
        else:
            st.warning("⚠️ No numeric columns available.")

    # --- Pie Chart ---
    with tabs[2]:
        if cat_cols:
            col = st.selectbox("Select categorical column for Pie Chart", cat_cols, key="pie_col")
            pie_data = df[col].value_counts().reset_index()
            pie_data.columns = [col, "Count"]
            fig = px.pie(pie_data, names=col, values="Count", title=f"Pie Chart of {col}")
            st.plotly_chart(fig, use_container_width=True)
            download_plotly_fig(fig, f"pie_chart_{col}.html")
        else:
            st.warning("⚠️ No categorical columns available.")

    # --- Histogram ---
    with tabs[3]:
        if num_cols:
            col = st.selectbox("Select numeric column", num_cols, key="hist_col")
            bins = st.slider("Number of bins", min_value=5, max_value=100, value=20)
            fig = px.histogram(df, x=col, nbins=bins, title=f"Histogram of {col}")
            st.plotly_chart(fig, use_container_width=True)
            download_plotly_fig(fig, f"histogram_{col}.html")
        else:
            st.warning("⚠️ No numeric columns available.")

    # --- Scatter Plot ---
    with tabs[4]:
        if len(num_cols) >= 2:
            x_axis = st.selectbox("Select X-axis", num_cols, key="scatter_x")
            y_axis = st.selectbox("Select Y-axis", num_cols, key="scatter_y")
            hue_col = st.selectbox("Optional categorical hue", [None]+cat_cols, key="scatter_hue")
            fig = px.scatter(df, x=x_axis, y=y_axis, color=hue_col, title=f"Scatter Plot: {x_axis} vs {y_axis}")
            st.plotly_chart(fig, use_container_width=True)
            download_plotly_fig(fig, f"scatter_{x_axis}_vs_{y_axis}.html")
        else:
            st.warning("⚠️ Not enough numeric columns for scatter plot.")

    # --- Box Plot ---
    with tabs[5]:
        if num_cols:
            col = st.selectbox("Select numeric column", num_cols, key="box_col")
            group_col = st.selectbox("Optional categorical grouping", [None]+cat_cols, key="box_group")
            fig = px.box(df, y=col, color=group_col, title=f"Box Plot of {col}")
            st.plotly_chart(fig, use_container_width=True)
            download_plotly_fig(fig, f"box_{col}.html")
        else:
            st.warning("⚠️ No numeric columns available.")

    # --- Heatmap ---
    with tabs[6]:
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig = ff.create_annotated_heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.columns),
                annotation_text=np.round(corr.values, 2),
                colorscale="Viridis"
            )
            fig.update_layout(title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            download_plotly_fig(fig, "heatmap.html")
        else:
            st.warning("⚠️ Not enough numeric columns for heatmap.")

    # --- Download Cleaned Dataset ---
    st.markdown("---")
    st.subheader("📥 Download Data")
    st.download_button(
        "Download Dataset as CSV",
        df.to_csv(index=False),
        "dataset.csv",
        "text/csv"
    )
