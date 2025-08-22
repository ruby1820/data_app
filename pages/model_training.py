# pages/model_training_advanced.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, confusion_matrix

st.set_page_config(page_title="Advanced ML Training", layout="wide")
st.title("🤖 Advanced Machine Learning Dashboard")

# --- Dataset Selection ---
st.markdown("### 📂 Dataset")
use_session_data = st.checkbox("Use cleaned dataset from session (if available)")

df = None
if use_session_data and "df" in st.session_state:
    df = st.session_state["df"].copy()
    st.success("✅ Using cleaned dataset from session.")
else:
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded successfully.")

# --- Dataset Info ---
if df is not None:
    st.subheader("🔍 Dataset Preview")
    st.write(df.head())
    
    st.markdown("### ℹ️ Dataset Info")
    st.write({
        "Number of rows": df.shape[0],
        "Number of columns": df.shape[1],
        "Columns with missing values": df.isna().sum().to_dict(),
        "Data types": df.dtypes.to_dict()
    })

    # --- Target & Features ---
    st.markdown("---")
    st.subheader("🎯 Select Target and Features")
    all_cols = df.columns.tolist()
    target_col = st.selectbox("Select Target Column", all_cols)
    feature_cols = st.multiselect("Select Feature Columns", [col for col in all_cols if col != target_col])

    if target_col and feature_cols:
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Encode categorical features
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        if y.dtype == "object" or str(y.dtype) == "category":
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        else:
            le_target = None

        # --- Model Type ---
        st.subheader("🧰 Model Type Selection")
        model_type = st.radio("Select model type:", ["Regression", "Classification"])

        # --- Train/Test Split ---
        st.subheader("⚖️ Train/Test Split")
        test_size = st.slider("Test set size (%)", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        # Scaling optional
        scale_features = st.checkbox("Scale features (StandardScaler)")
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # --- Model Selection ---
        st.subheader("🔹 Choose Models & Hyperparameters")
        selected_models = []

        if model_type == "Regression":
            if st.checkbox("Linear Regression"):
                selected_models.append(("Linear Regression", LinearRegression(), {}))
            if st.checkbox("Random Forest Regressor"):
                n_estimators = st.number_input("RF Regressor n_estimators", 10, 500, 100)
                max_depth = st.number_input("RF Regressor max_depth (0=None)", 0, 50, 10)
                max_depth_val = None if max_depth==0 else max_depth
                selected_models.append(("Random Forest Regressor", RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth_val, random_state=42), {}))
            if st.checkbox("Gradient Boosting Regressor"):
                n_estimators_gb = st.number_input("GB Regressor n_estimators", 10, 500, 100)
                learning_rate = st.number_input("GB Regressor learning_rate", 0.01, 1.0, 0.1)
                selected_models.append(("Gradient Boosting Regressor", GradientBoostingRegressor(n_estimators=n_estimators_gb, learning_rate=learning_rate, random_state=42), {}))
        else:
            if st.checkbox("Logistic Regression"):
                selected_models.append(("Logistic Regression", LogisticRegression(max_iter=500), {}))
            if st.checkbox("Random Forest Classifier"):
                n_estimators = st.number_input("RF Classifier n_estimators", 10, 500, 100)
                max_depth = st.number_input("RF Classifier max_depth (0=None)", 0, 50, 10)
                max_depth_val = None if max_depth==0 else max_depth
                selected_models.append(("Random Forest Classifier", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth_val, random_state=42), {}))
            if st.checkbox("Gradient Boosting Classifier"):
                n_estimators_gb = st.number_input("GB Classifier n_estimators", 10, 500, 100)
                learning_rate = st.number_input("GB Classifier learning_rate", 0.01, 1.0, 0.1)
                selected_models.append(("Gradient Boosting Classifier", GradientBoostingClassifier(n_estimators=n_estimators_gb, learning_rate=learning_rate, random_state=42), {}))

        # --- Train Models ---
        if st.button("Train Selected Models") and selected_models:
            st.subheader("📊 Training & Evaluation")
            results = []
            best_model = None
            best_score = None

            for name, model, params in selected_models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if model_type == "Regression":
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    results.append({"Model": name, "MSE": mse, "MAE": mae, "R2": r2})
                    # Determine best by R2
                    if best_score is None or r2 > best_score:
                        best_score = r2
                        best_model = (name, model)

                    # Plot Actual vs Predicted
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title(f"{name}: Actual vs Predicted")
                    st.pyplot(fig)

                else:  # Classification
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    results.append({"Model": name, "Accuracy": acc, "F1-score": f1})
                    # Determine best by Accuracy
                    if best_score is None or acc > best_score:
                        best_score = acc
                        best_model = (name, model)

                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"{name}: Confusion Matrix")
                    st.pyplot(fig)

            # Show results table
            st.subheader("📈 Models Comparison")
            st.table(pd.DataFrame(results))

            # Download best model
            st.subheader("📥 Download Best Model")
            buffer = io.BytesIO()
            pickle.dump(best_model[1], buffer)
            st.download_button(
                label=f"Download {best_model[0]} (Pickle)",
                data=buffer.getvalue(),
                file_name=f"{best_model[0].replace(' ','_')}.pkl"
            )
