# -*- coding: utf-8 -*-
"""
Brainy Data Explorer
=====================
A dataset-agnostic Streamlit app for exploratory data analysis and
supervised modeling (regression or classification, auto-detected from
the target column you choose). Upload any CSV or Excel file — no code
edits required.

Run with:
    streamlit run app.py
"""

import io
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================
st.set_page_config(
    page_title="Brainy Data Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6C63FF 0%, #FF6584 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #8B8B9E;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #f7f7fb;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e8e8f0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        color: #6C63FF;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">🧠 Brainy Data Explorer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload any dataset, explore it, and train a model — '
    "no code required. 🧠✨</p>",
    unsafe_allow_html=True,
)

sns.set_theme(style="whitegrid")


# =============================================================================
# HELPERS
# =============================================================================
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes each categorical column by its relative frequency in the
    training data. Used for high-cardinality categoricals, where one-hot
    encoding would blow up dimensionality and hurt distance-based models
    like KNN."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.freq_maps_ = {col: X[col].value_counts(normalize=True) for col in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            fmap = self.freq_maps_.get(col, pd.Series(dtype=float))
            X[col] = X[col].map(fmap).fillna(0.0)
        return X.values.astype(float)


@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a .csv or .xlsx/.xls file.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def infer_column_types(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    return numeric_cols, categorical_cols, datetime_cols


def build_preprocessor(numeric_cols, low_card_cols, high_card_cols):
    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))
    if low_card_cols:
        transformers.append((
            "cat_low",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]),
            low_card_cols,
        ))
    if high_card_cols:
        transformers.append((
            "cat_high",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("freq", FrequencyEncoder()),
            ]),
            high_card_cols,
        ))
    return ColumnTransformer(transformers)


def detect_problem_type(y: pd.Series, override: str):
    if override != "Auto-detect":
        return "Regression" if override == "Regression" else "Classification"
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        return "Regression"
    return "Classification"


def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return buf


# =============================================================================
# SIDEBAR — FILE UPLOAD
# =============================================================================
with st.sidebar:
    st.markdown("### 🧠 Get started")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    st.caption("Your data stays in this session and is never hard-coded into the app.")

if uploaded_file is None:
    st.info("👈 Upload a dataset from the sidebar to begin exploring.")
    st.stop()

try:
    df_raw = load_data(uploaded_file)
except Exception as e:
    st.error(f"Couldn't read that file: {e}")
    st.stop()

if df_raw.empty:
    st.error("The uploaded file has no rows.")
    st.stop()

# =============================================================================
# TABS
# =============================================================================
tab_preview, tab_eda, tab_model = st.tabs(["🔍 Data Preview", "📊 Exploratory Analysis", "🧠 Modeling"])

# -----------------------------------------------------------------------------
# TAB 1 — PREVIEW
# -----------------------------------------------------------------------------
with tab_preview:
    st.subheader("Dataset preview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df_raw.shape[0]:,}")
    c2.metric("Columns", f"{df_raw.shape[1]:,}")
    c3.metric("Missing cells", f"{int(df_raw.isnull().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df_raw.duplicated().sum()):,}")

    st.dataframe(df_raw.head(20), use_container_width=True)

    with st.expander("Column info & data types"):
        info_df = pd.DataFrame({
            "Column": df_raw.columns,
            "Dtype": df_raw.dtypes.astype(str).values,
            "Missing": df_raw.isnull().sum().values,
            "Missing %": (df_raw.isnull().mean() * 100).round(2).values,
            "Unique values": [df_raw[c].nunique() for c in df_raw.columns],
        })
        st.dataframe(info_df, use_container_width=True)

    with st.expander("Summary statistics"):
        st.dataframe(df_raw.describe(include="all").transpose(), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2 — EDA
# -----------------------------------------------------------------------------
with tab_eda:
    numeric_cols_all, categorical_cols_all, _ = infer_column_types(df_raw)
    st.subheader("Exploratory Data Analysis")

    if numeric_cols_all:
        st.markdown("#### Distribution of a numeric column")
        col_choice = st.selectbox("Choose a numeric column", numeric_cols_all, key="hist_col")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df_raw[col_choice].dropna(), kde=True, ax=ax, color="#6C63FF")
        ax.set_title(f"Distribution of {col_choice}")
        st.pyplot(fig)
        plt.close(fig)

    if len(numeric_cols_all) >= 2:
        st.markdown("#### Correlation heatmap")
        fig, ax = plt.subplots(figsize=(min(1 + 0.6 * len(numeric_cols_all), 12), 6))
        sns.heatmap(df_raw[numeric_cols_all].corr(), annot=len(numeric_cols_all) <= 15,
                    fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    if numeric_cols_all:
        st.markdown("#### Boxplots (outlier check)")
        box_cols = st.multiselect("Columns to plot", numeric_cols_all,
                                   default=numeric_cols_all[:min(6, len(numeric_cols_all))])
        if box_cols:
            fig, ax = plt.subplots(figsize=(min(1.5 * len(box_cols), 12), 4.5))
            df_raw[box_cols].boxplot(ax=ax, rot=45)
            st.pyplot(fig)
            plt.close(fig)

    if categorical_cols_all:
        st.markdown("#### Categorical column breakdown")
        cat_choice = st.selectbox("Choose a categorical column", categorical_cols_all, key="cat_col")
        top_n = df_raw[cat_choice].value_counts().head(20)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=top_n.values, y=top_n.index.astype(str), ax=ax, color="#FF6584")
        ax.set_xlabel("Count")
        ax.set_title(f"Top categories in {cat_choice}")
        st.pyplot(fig)
        plt.close(fig)

    if len(numeric_cols_all) >= 2:
        st.markdown("#### Pairwise relationships")
        pair_cols = st.multiselect(
            "Columns for pairplot (2–5 recommended)", numeric_cols_all,
            default=numeric_cols_all[:min(3, len(numeric_cols_all))],
        )
        if 2 <= len(pair_cols) <= 5:
            fig = sns.pairplot(df_raw[pair_cols].dropna())
            st.pyplot(fig)
            plt.close("all")
        elif len(pair_cols) > 5:
            st.caption("Pick 5 or fewer columns to keep the pairplot readable.")

# -----------------------------------------------------------------------------
# TAB 3 — MODELING
# -----------------------------------------------------------------------------
with tab_model:
    st.subheader("Train a model")

    all_cols = df_raw.columns.tolist()
    target_col = st.selectbox(
        "🎯 Choose the target variable (the column you want to predict)",
        all_cols, index=len(all_cols) - 1,
    )

    problem_override = st.radio(
        "Problem type", ["Auto-detect", "Regression", "Classification"],
        horizontal=True,
    )

    df_model = df_raw.dropna(subset=[target_col]).copy()
    y_full = df_model[target_col]
    problem_type = detect_problem_type(y_full, problem_override)
    st.caption(f"Detected problem type: **{problem_type}** "
               f"(target has {y_full.nunique()} unique values, dtype `{y_full.dtype}`).")

    default_features = [c for c in all_cols if c != target_col]
    feature_cols = st.multiselect(
        "Feature columns to use (deselect IDs or leakage-prone columns)",
        default_features, default=default_features,
    )
    if not feature_cols:
        st.warning("Select at least one feature column to continue.")
        st.stop()

    X_full = df_model[feature_cols].copy()

    numeric_cols, categorical_cols, datetime_cols = infer_column_types(X_full)
    if datetime_cols:
        st.caption(f"Dropping datetime columns not usable as-is: {datetime_cols}")
        X_full = X_full.drop(columns=datetime_cols)

    HIGH_CARD_THRESHOLD = 30
    high_card_cols = [c for c in categorical_cols if X_full[c].nunique() > HIGH_CARD_THRESHOLD]
    low_card_cols = [c for c in categorical_cols if c not in high_card_cols]

    with st.expander("How features will be encoded"):
        st.write(f"**Numeric** (scaled): {numeric_cols or 'none'}")
        st.write(f"**Low-cardinality categorical** (one-hot encoded, ≤{HIGH_CARD_THRESHOLD} categories): {low_card_cols or 'none'}")
        st.write(f"**High-cardinality categorical** (frequency-encoded, >{HIGH_CARD_THRESHOLD} categories): {high_card_cols or 'none'}")

    st.markdown("#### Training options")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    with col_b:
        tune_k = st.checkbox("Tune K with GridSearchCV", value=True)
    with col_c:
        k_max = st.slider("Max K to try", 5, 30, 15)

    compare_models = st.multiselect(
        "Also train these models for comparison",
        ["Linear/Logistic Regression", "Random Forest"] + (["XGBoost"] if XGBOOST_AVAILABLE else []),
        default=["Random Forest"],
    )
    if not XGBOOST_AVAILABLE:
        st.caption("XGBoost isn't installed in this environment, so it's excluded from comparisons.")

    train_clicked = st.button("🧠 Train model", type="primary")

    if train_clicked:
        y = y_full.loc[X_full.index]

        label_encoder = None
        if problem_type == "Classification" and not pd.api.types.is_numeric_dtype(y):
            y = y.astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y, test_size=test_size, random_state=RANDOM_STATE,
            stratify=y if problem_type == "Classification" and y.nunique() > 1 else None,
        )

        preprocessor = build_preprocessor(numeric_cols, low_card_cols, high_card_cols)

        if problem_type == "Regression":
            knn_model = KNeighborsRegressor()
            scoring = "r2"
        else:
            knn_model = KNeighborsClassifier()
            scoring = "f1_macro"

        knn_pipeline = Pipeline([("preprocessor", preprocessor), ("model", knn_model)])

        with st.spinner("Training KNN..."):
            if tune_k:
                param_grid = {"model__n_neighbors": list(range(2, k_max + 1))}
                cv_folds = min(5, y_train.value_counts().min()) if problem_type == "Classification" else 5
                cv_folds = max(cv_folds, 2)
                grid = GridSearchCV(knn_pipeline, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1)
                grid.fit(X_train, y_train)
                best_knn = grid.best_estimator_
                st.success(f"Best K found: {grid.best_params_['model__n_neighbors']}")
            else:
                best_knn = knn_pipeline
                best_knn.set_params(model__n_neighbors=5)
                best_knn.fit(X_train, y_train)

        pred = best_knn.predict(X_test)

        st.markdown("### 📈 KNN Results")
        if problem_type == "Regression":
            mae = mean_absolute_error(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, pred)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE", f"{mae:,.3f}")
            m2.metric("RMSE", f"{rmse:,.3f}")
            m3.metric("MSE", f"{mse:,.3f}")
            m4.metric("R²", f"{r2:.4f}")

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_test, pred, alpha=0.4, color="#6C63FF")
            lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
            ax.plot(lims, lims, "r--")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)
            plt.close(fig)
        else:
            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.4f}")
            m2.metric("Precision", f"{prec:.4f}")
            m3.metric("Recall", f"{rec:.4f}")
            m4.metric("F1-score", f"{f1:.4f}")

            with st.expander("Full classification report"):
                st.text(classification_report(y_test, pred, zero_division=0))

            labels = sorted(y.unique().tolist())
            cm = confusion_matrix(y_test, pred, labels=labels)
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues", ax=ax, colorbar=False)
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("### 📉 Learning curve")
        with st.spinner("Computing learning curve..."):
            try:
                cv_lc = min(5, y_train.value_counts().min()) if problem_type == "Classification" else 5
                cv_lc = max(cv_lc, 2)
                train_sizes, train_scores, val_scores = learning_curve(
                    best_knn, X_train, y_train, cv=cv_lc, scoring=scoring,
                    train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1,
                )
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train score", color="#6C63FF")
                ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation score", color="#FF6584")
                ax.set_xlabel("Training examples")
                ax.set_ylabel(scoring)
                ax.set_title("Learning Curve — KNN")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.caption(f"Learning curve unavailable: {e}")

        # ---------------------------------------------------------------
        # Model comparison
        # ---------------------------------------------------------------
        models_to_run = {"KNN": best_knn}

        if "Linear/Logistic Regression" in compare_models:
            base = LinearRegression() if problem_type == "Regression" else LogisticRegression(max_iter=1000)
            pipe = Pipeline([("preprocessor", preprocessor), ("model", base)])
            pipe.fit(X_train, y_train)
            models_to_run["Linear/Logistic"] = pipe

        if "Random Forest" in compare_models:
            base = (RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
                    if problem_type == "Regression"
                    else RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
            pipe = Pipeline([("preprocessor", preprocessor), ("model", base)])
            pipe.fit(X_train, y_train)
            models_to_run["Random Forest"] = pipe

        if "XGBoost" in compare_models and XGBOOST_AVAILABLE:
            if problem_type == "Regression":
                base = XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=RANDOM_STATE)
                pipe = Pipeline([("preprocessor", preprocessor), ("model", base)])
                pipe.fit(X_train, y_train)
            else:
                le = LabelEncoder()
                y_train_enc = le.fit_transform(y_train)
                base = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=6,
                                      random_state=RANDOM_STATE, eval_metric="mlogloss")
                pipe = Pipeline([("preprocessor", preprocessor), ("model", base)])
                pipe.fit(X_train, y_train_enc)
                pipe._label_encoder = le
            models_to_run["XGBoost"] = pipe

        if len(models_to_run) > 1:
            st.markdown("### 🏆 Model comparison")
            rows = {}
            for name, m in models_to_run.items():
                if name == "XGBoost" and problem_type == "Classification":
                    p = m._label_encoder.inverse_transform(m.predict(X_test))
                else:
                    p = m.predict(X_test)
                if problem_type == "Regression":
                    rows[name] = {
                        "MAE": mean_absolute_error(y_test, p),
                        "RMSE": np.sqrt(mean_squared_error(y_test, p)),
                        "R2": r2_score(y_test, p),
                    }
                else:
                    rows[name] = {
                        "Accuracy": accuracy_score(y_test, p),
                        "Precision": precision_score(y_test, p, average="weighted", zero_division=0),
                        "Recall": recall_score(y_test, p, average="weighted", zero_division=0),
                        "F1": f1_score(y_test, p, average="weighted", zero_division=0),
                    }
            comp_df = pd.DataFrame(rows).T
            st.dataframe(comp_df.style.highlight_max(axis=0, color="#d4f7dc"), use_container_width=True)

            fig, axes = plt.subplots(1, len(comp_df.columns), figsize=(4.5 * len(comp_df.columns), 4))
            if len(comp_df.columns) == 1:
                axes = [axes]
            for ax, metric in zip(axes, comp_df.columns):
                comp_df[metric].plot(kind="bar", ax=ax, color=["#6C63FF", "#FF6584", "#55A868", "#DD8452"])
                ax.set_title(metric)
                ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ---------------------------------------------------------------
        # Feature importance (tree-based models only)
        # ---------------------------------------------------------------
        importance_source = models_to_run.get("Random Forest") or models_to_run.get("XGBoost")
        if importance_source is not None:
            st.markdown("### 🌲 Feature importance")
            try:
                fitted_pre = importance_source.named_steps["preprocessor"]
                names = []
                if numeric_cols:
                    names += numeric_cols
                if low_card_cols:
                    names += fitted_pre.named_transformers_["cat_low"].named_steps["ohe"].get_feature_names_out(low_card_cols).tolist()
                if high_card_cols:
                    names += high_card_cols
                importances = importance_source.named_steps["model"].feature_importances_
                imp_df = pd.DataFrame({"Feature": names, "Importance": importances}).sort_values(
                    "Importance", ascending=False).head(20)
                fig, ax = plt.subplots(figsize=(8, min(0.4 * len(imp_df) + 1, 10)))
                sns.barplot(x="Importance", y="Feature", data=imp_df, ax=ax, color="#6C63FF")
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.caption(f"Feature importance unavailable: {e}")

        # ---------------------------------------------------------------
        # Download predictions
        # ---------------------------------------------------------------
        results_df = X_test.copy()
        results_df["Actual"] = y_test.values
        results_df["Predicted"] = pred
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download test-set predictions (CSV)",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption("🧠 Brainy Data Explorer — dataset-agnostic EDA & modeling, built with Streamlit + scikit-learn.")
