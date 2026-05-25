import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# -----------------------------
# App Configuration and Header
# -----------------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("Business Intelligence Predictive Modeling Application")
st.write(
    "This app predicts customer churn for a telecom company using the Telco Customer Churn dataset. "
    "It demonstrates data preprocessing, model training, evaluation, and comparison in Streamlit."
)


# -----------------------------
# Load Dataset (No Upload)
# -----------------------------
DATA_PATH = "assets/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from the local path."""
    return pd.read_csv(path)


df = load_data(DATA_PATH)

st.subheader("Dataset Preview")
st.dataframe(df, use_container_width=True)

st.subheader("Basic Dataset Information")
col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", f"{df.shape[0]:,}")
col2.metric("Total Columns", f"{df.shape[1]:,}")
col3.metric("Target Column", "Churn")

st.write("Statistical Summary (Numeric Columns)")
st.dataframe(df.describe(), use_container_width=True)


# -----------------------------
# Data Preprocessing
# -----------------------------
@st.cache_data
def preprocess_data(raw_df: pd.DataFrame):
    """Clean, encode, scale, and split the dataset."""
    data = raw_df.copy()

    # Fix hidden missing values in TotalCharges and convert to numeric
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].replace(" ", pd.NA), errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(0)

    # Drop non-predictive column
    data = data.drop(columns=["customerID"])

    # Encode target variable (Churn: Yes/No)
    target_encoder = LabelEncoder()
    data["Churn"] = target_encoder.fit_transform(data["Churn"])  # Yes=1, No=0

    # Encode binary categorical columns with LabelEncoder
    binary_cols = [
        col for col in data.columns
        if data[col].dtype == "object" and data[col].nunique() == 2
    ]
    for col in binary_cols:
        data[col] = LabelEncoder().fit_transform(data[col])

    # One-hot encode multi-class categorical columns
    multi_cols = [
        col for col in data.columns
        if data[col].dtype == "object" and data[col].nunique() > 2
    ]
    data = pd.get_dummies(data, columns=multi_cols, drop_first=True)

    # Separate features and target
    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    # Scale numeric columns
    scaler = StandardScaler()
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = preprocess_data(df)

st.success("Data preprocessing completed successfully.")


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Model Training Controls")
model_choice = st.sidebar.selectbox(
    "Select a model:",
    ["K-Nearest Neighbor (KNN)", "Support Vector Machine (SVM)", "Artificial Neural Network (ANN)"]
)
st.sidebar.subheader("Hyperparameters")

with st.sidebar.expander("KNN Settings", expanded=False):
    knn_k = st.slider("Number of neighbors (k)", min_value=1, max_value=20, value=5, step=1)

with st.sidebar.expander("SVM Settings", expanded=False):
    svm_c = st.slider("Regularization (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

with st.sidebar.expander("ANN Settings", expanded=False):
    ann_layer1 = st.number_input("Hidden layer 1 size", min_value=10, max_value=256, value=64, step=1)
    ann_layer2 = st.number_input("Hidden layer 2 size", min_value=10, max_value=256, value=32, step=1)
    ann_max_iter = st.number_input("Max iterations", min_value=100, max_value=1000, value=500, step=50)

train_button = st.sidebar.button("Train and Evaluate Model")


# -----------------------------
# Model Training and Evaluation
# -----------------------------
if train_button:
    if model_choice == "K-Nearest Neighbor (KNN)":
        model = KNeighborsClassifier(n_neighbors=knn_k)
    elif model_choice == "Support Vector Machine (SVM)":
        model = SVC(kernel="rbf", C=svm_c, probability=True)
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(ann_layer1, ann_layer2),
            max_iter=ann_max_iter,
            random_state=42
        )

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.subheader("Model Evaluation Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1-score", f"{f1:.3f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


# -----------------------------
# Compare All Models
# -----------------------------
st.subheader("Compare All Models")
if st.button("Run Model Comparison"):
    results = []

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=knn_k),
        "SVM": SVC(kernel="rbf", C=svm_c),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(ann_layer1, ann_layer2),
            max_iter=ann_max_iter,
            random_state=42
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results.append({"Model": name, "Accuracy": acc})

    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

    st.write("Model Accuracy Comparison")
    st.dataframe(results_df, use_container_width=True)

    # Bar chart for comparison
    st.bar_chart(results_df.set_index("Model")["Accuracy"])