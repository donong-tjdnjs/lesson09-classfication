import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 1. Page Configuration
st.set_page_config(
    page_title="채점표 - 뇌졸중 예측 실습실",
    page_icon="📝",
    layout="wide",
)


# 2. Load Data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df


df_raw = load_data()

st.title("📝 모델 채점표")
st.markdown("---")

# Feature Mapping (Korean labels <-> Column names)
FEATURE_MAP = {
    "나이": "age",
    "평균 혈당": "avg_glucose_level",
    "체질량지수": "bmi",
    "고혈압": "hypertension",
    "심장병": "heart_disease",
}
REVERSE_FEATURE_MAP = {v: k for k, v in FEATURE_MAP.items()}

# ---------------------------------------------------------
# 1. Feature Selection & Train/Test Split (Identical to Page 2)
# ---------------------------------------------------------
default_selected_labels = ["나이", "평균 혈당", "고혈압", "심장병"]
selected_cols = [FEATURE_MAP[label] for label in default_selected_labels]

# Sort by ID/Index and split (group of 10 -> first 3 test, remaining 7 train)
if "id" in df_raw.columns:
    df_sorted = df_raw.sort_values("id").reset_index(drop=True)
else:
    df_sorted = df_raw.reset_index(drop=True)

test_mask = (df_sorted.index % 10) < 3
df_test = df_sorted[test_mask].copy()
df_train = df_sorted[~test_mask].copy()

# Imputation (Median from train)
train_medians = df_train[selected_cols].median()

X_train = df_train[selected_cols].copy()
X_test = df_test[selected_cols].copy()

if "bmi" in selected_cols:
    bmi_median = train_medians["bmi"]
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

y_train = df_train["stroke"].values
y_test = df_test["stroke"].values

# ---------------------------------------------------------
# 2. Model Training
# ---------------------------------------------------------
# (1) Baseline Model (Majority class from train)
majority_class = int(pd.Series(y_train).mode()[0])
baseline_preds = np.full_like(y_test, fill_value=majority_class)

# (2) Logistic Regression (Standardized)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
log_reg_preds = log_reg.predict(X_test_scaled)

# (3) Decision Tree
dt_cls = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42
)
dt_cls.fit(X_train, y_train)
dt_preds = dt_cls.predict(X_test)


# Helper function to compute Confusion Matrix & Metrics
def get_cm_and_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    acc = (tp + tn) / len(y_true) * 100 if len(y_true) > 0 else 0.0

    # Recall (Sensitivity): TP / (TP + FN)
    actual_positives = tp + fn
    recall = (tp / actual_positives * 100) if actual_positives > 0 else 0.0

    # Precision: TP / (TP + FP)
    predicted_positives = tp + fp
    if predicted_positives == 0:
        precision_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"
    else:
        precision = tp / predicted_positives * 100
        precision_str = f"{precision:.2f}%"

        # F1 score: 2 * (Precision * Recall) / (Precision + Recall)
        p_val = precision / 100
        r_val = recall / 100
        if (p_val + r_val) > 0:
            f1 = 2 * (p_val * r_val) / (p_val + r_val)
            f1_str = f"{f1:.4f}"
        else:
            f1_str = "0.0000"

    recall_str = f"{recall:.2f}%"
    acc_str = f"{acc:.2f}%"

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Accuracy": acc_str,
        "Recall": recall_str,
        "Precision": precision_str,
        "F1": f1_str,
    }


baseline_res = get_cm_and_metrics(y_test, baseline_preds)
log_reg_res = get_cm_and_metrics(y_test, log_reg_preds)
dt_res = get_cm_and_metrics(y_test, dt_preds)

# ---------------------------------------------------------
# 3. Test Set Overview
# ---------------------------------------------------------
test_total_count = len(df_test)
test_stroke_count = int(np.sum(y_test == 1))

st.markdown(
    f"📊 테스트용 사람은 총 **{test_total_count:,}명**이며, 그중 실제 뇌졸중인 사람은 **{test_stroke_count:,}명**입니다."
)

st.markdown("---")

# ---------------------------------------------------------
# 4. Confusion Matrix Displays (2x2 Tables)
# ---------------------------------------------------------
st.subheader("1️⃣ 모델별 혼동행렬 (2×2 표)")

col_cm1, col_cm2 = st.columns(2)


def render_confusion_matrix(title, res):
    tp, fp, fn, tn = res["TP"], res["FP"], res["FN"], res["TN"]
    row_act_1_sum = tp + fn
    row_act_0_sum = fp + tn
    col_pred_1_sum = tp + fp
    col_pred_0_sum = fn + tn
    total_sum = row_act_1_sum + row_act_0_sum

    cm_df = pd.DataFrame(
        [
            {
                "실제 \\ 예측": "실제 뇌졸중 (1)",
                "예측 뇌졸중 (1)": f"{tp:,} 명 (TP)",
                "예측 아님 (0)": f"{fn:,} 명 (FN)",
                "합계": f"{row_act_1_sum:,} 명",
            },
            {
                "실제 \\ 예측": "실제 아님 (0)",
                "예측 뇌졸중 (1)": f"{fp:,} 명 (FP)",
                "예측 아님 (0)": f"{tn:,} 명 (TN)",
                "합계": f"{row_act_0_sum:,} 명",
            },
            {
                "실제 \\ 예측": "합계",
                "예측 뇌졸중 (1)": f"{col_pred_1_sum:,} 명",
                "예측 아님 (0)": f"{col_pred_0_sum:,} 명",
                "합계": f"{total_sum:,} 명",
            },
        ]
    ).set_index("실제 \\ 예측")

    st.markdown(f"##### {title}")
    st.table(cm_df)


with col_cm1:
    render_confusion_matrix(
        "📈 로지스틱 회귀 (확률로 답하는 모델)", log_reg_res
    )

with col_cm2:
    render_confusion_matrix("🌳 의사결정트리 (질문으로 답하는 모델)", dt_res)

st.markdown("---")

# ---------------------------------------------------------
# 5. Performance Metrics Table
# ---------------------------------------------------------
st.subheader("2️⃣ 모델 성능 평가지표 비교 표")

metrics_summary_df = pd.DataFrame(
    [
        {
            "모델": "훈련용에서 많은 쪽으로만 답하는 모델",
            "정확도(Accuracy)": baseline_res["Accuracy"],
            "재현율(Recall)": baseline_res["Recall"],
            "정밀도(Precision)": baseline_res["Precision"],
            "F1 점수": baseline_res["F1"],
        },
        {
            "모델": "로지스틱 회귀 (확률로 답하는 모델)",
            "정확도(Accuracy)": log_reg_res["Accuracy"],
            "재현율(Recall)": log_reg_res["Recall"],
            "정밀도(Precision)": log_reg_res["Precision"],
            "F1 점수": log_reg_res["F1"],
        },
        {
            "모델": "의사결정트리 (질문으로 답하는 모델)",
            "정확도(Accuracy)": dt_res["Accuracy"],
            "재현율(Recall)": dt_res["Recall"],
            "정밀도(Precision)": dt_res["Precision"],
            "F1 점수": dt_res["F1"],
        },
    ]
)

st.dataframe(metrics_summary_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------
# 6. Detailed Error Analysis Lists (FN & FP)
# ---------------------------------------------------------
st.subheader("3️⃣ 오답 명단 상세보기 (놓친 환자 및 헛짚은 사람)")

model_option = st.selectbox(
    "상세 명단을 확인할 모델을 선택하세요:",
    ["로지스틱 회귀 (확률로 답하는 모델)", "의사결정트리 (질문으로 답하는 모델)"],
)

if model_option == "로지스틱 회귀 (확률로 답하는 모델)":
    selected_preds = log_reg_preds
    selected_res = log_reg_res
else:
    selected_preds = dt_preds
    selected_res = dt_res

df_test_analysis = df_test.copy()
df_test_analysis["pred_stroke"] = selected_preds

# 1. FN: Missed Stroke Patients (Actual=1, Pred=0)
fn_df = df_test_analysis[
    (df_test_analysis["stroke"] == 1) & (df_test_analysis["pred_stroke"] == 0)
].copy()
fn_df["고혈압 여부"] = fn_df["hypertension"].map(
    {0: "고혈압 없음", 1: "고혈압 있음"}
)
fn_df["심장병 여부"] = fn_df["heart_disease"].map(
    {0: "심장병 없음", 1: "심장병 있음"}
)
fn_df = fn_df.sort_values("age", ascending=False)

# 2. FP: False Positive Cases (Actual=0, Pred=1)
fp_df = df_test_analysis[
    (df_test_analysis["stroke"] == 0) & (df_test_analysis["pred_stroke"] == 1)
].copy()
fp_df["고혈압 여부"] = fp_df["hypertension"].map(
    {0: "고혈압 없음", 1: "고혈압 있음"}
)
fp_df["심장병 여부"] = fp_df["heart_disease"].map(
    {0: "심장병 없음", 1: "심장병 있음"}
)
fp_df = fp_df.sort_values("age", ascending=False)

display_cols = ["id", "age", "avg_glucose_level", "고혈압 여부", "심장병 여부"]
rename_dict = {
    "id": "번호",
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
}

col_list1, col_list2 = st.columns(2)

with col_list1:
    fn_total_len = len(fn_df)
    st.markdown(
        f"##### ⚠️ 놓친 환자 (실제 뇌졸중이나 아니라고 예측) [총 {fn_total_len:,}명]"
    )
    if fn_total_len > 20:
        st.caption("※ 나이가 많은 순서로 상위 20명만 표시합니다.")
        fn_show_df = fn_df.head(20)[display_cols].rename(columns=rename_dict)
    else:
        fn_show_df = fn_df[display_cols].rename(columns=rename_dict)

    st.dataframe(fn_show_df, use_container_width=True, hide_index=True)

with col_list2:
    fp_total_len = len(fp_df)
    st.markdown(
        f"##### 🔍 헛짚은 사람 (실제 아님이나 뇌졸중이라고 예측) [총 {fp_total_len:,}명]"
    )
    if fp_total_len > 20:
        st.caption("※ 나이가 많은 순서로 상위 20명만 표시합니다.")
        fp_show_df = fp_df.head(20)[display_cols].rename(columns=rename_dict)
    else:
        fp_show_df = fp_df[display_cols].rename(columns=rename_dict)

    st.dataframe(fp_show_df, use_container_width=True, hide_index=True)

st.markdown(
    f"💡 **확인:** 위 명단의 놓친 환자 수(**{fn_total_len:,}명**)와 헛짚은 사람 수(**{fp_total_len:,}명**)는 "
    f"선택한 모델 혼동행렬의 **FN({selected_res['FN']:,}명)** 및 **FP({selected_res['FP']:,}명)** 값과 정확히 일치합니다."
)
