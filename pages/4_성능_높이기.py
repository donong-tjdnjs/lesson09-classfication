import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 1. Page Configuration
st.set_page_config(
    page_title="성능 높이기 - 뇌졸중 예측 실습실",
    page_icon="⚡",
    layout="wide",
)

# 2. Load Data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df_raw = load_data()

st.title("⚡ 클래스 가중치 조절로 성능 높이기")
st.markdown("---")

# Feature Mapping (Korean labels <-> Column names)
FEATURE_MAP = {
    "나이": "age",
    "평균 혈당": "avg_glucose_level",
    "체질량지수": "bmi",
    "고혈압": "hypertension",
    "심장병": "heart_disease"
}
REVERSE_FEATURE_MAP = {v: k for k, v in FEATURE_MAP.items()}

# ---------------------------------------------------------
# 1. Feature Selection UI
# ---------------------------------------------------------
st.subheader("1️⃣ 학습에 사용할 속성 선택")

default_selected_labels = ["나이", "평균 혈당", "고혈압", "심장병"]
selected_labels = st.multiselect(
    "모델 학습 및 예측에 사용할 속성을 선택하세요 (최소 2개 이상):",
    options=list(FEATURE_MAP.keys()),
    default=default_selected_labels
)

if len(selected_labels) < 2:
    st.warning("⚠️ 속성을 최소 2개 이상 선택해 주세요!")
    st.stop()

selected_cols = [FEATURE_MAP[label] for label in selected_labels]

# ---------------------------------------------------------
# 2. Train / Test Split & Imputation (Identical to Page 2 & 3)
# ---------------------------------------------------------
if "id" in df_raw.columns:
    df_sorted = df_raw.sort_values("id").reset_index(drop=True)
else:
    df_sorted = df_raw.reset_index(drop=True)

test_mask = (df_sorted.index % 10) < 3
df_test = df_sorted[test_mask].copy()
df_train = df_sorted[~test_mask].copy()

# Median imputation from train
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
# 3. Model Training (Before vs After Weight Balancing)
# ---------------------------------------------------------
# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# (1) Logistic Regression: Before & After
log_reg_before = LogisticRegression(random_state=42)
log_reg_before.fit(X_train_scaled, y_train)

log_reg_after = LogisticRegression(class_weight="balanced", random_state=42)
log_reg_after.fit(X_train_scaled, y_train)

# (2) Decision Tree: Before & After
dt_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_before.fit(X_train, y_train)

dt_after = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=42)
dt_after.fit(X_train, y_train)

# Metrics calculation helper
def get_metrics_dict(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    acc = (tp + tn) / len(y_true) * 100 if len(y_true) > 0 else 0.0
    actual_positives = tp + fn
    recall = (tp / actual_positives * 100) if actual_positives > 0 else 0.0
    predicted_positives = tp + fp

    if predicted_positives == 0:
        precision_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"
    else:
        precision = tp / predicted_positives * 100
        precision_str = f"{precision:.2f}%"
        p_val = precision / 100
        r_val = recall / 100
        if (p_val + r_val) > 0:
            f1 = 2 * (p_val * r_val) / (p_val + r_val)
            f1_str = f"{f1:.4f}"
        else:
            f1_str = "0.0000"

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy": f"{acc:.2f}%",
        "Recall": f"{recall:.2f}%",
        "Precision": precision_str,
        "F1": f1_str
    }

lr_b_res = get_metrics_dict(y_test, log_reg_before.predict(X_test_scaled))
lr_a_res = get_metrics_dict(y_test, log_reg_after.predict(X_test_scaled))

dt_b_res = get_metrics_dict(y_test, dt_before.predict(X_test))
dt_a_res = get_metrics_dict(y_test, dt_after.predict(X_test))

# ---------------------------------------------------------
# 4. Results Comparison Tables
# ---------------------------------------------------------
st.subheader("2️⃣ 가중치 조정 전/후 비교 표")

def render_model_comparison(title, b_res, a_res):
    st.markdown(f"#### 📌 {title}")
    
    comp_df = pd.DataFrame([
        {
            "구분": "가중치 조정 전 (기본)",
            "TP (찾아낸 환자)": f"{b_res['TP']:,} 명",
            "FP (헛짚은 사람)": f"{b_res['FP']:,} 명",
            "FN (놓친 환자)": f"{b_res['FN']:,} 명",
            "TN (정상 예측)": f"{b_res['TN']:,} 명",
            "정확도(Accuracy)": b_res["Accuracy"],
            "재현율(Recall)": b_res["Recall"],
            "정밀도(Precision)": b_res["Precision"],
            "F1 점수": b_res["F1"],
        },
        {
            "구분": "가중치 조정 후 (Balanced)",
            "TP (찾아낸 환자)": f"{a_res['TP']:,} 명",
            "FP (헛짚은 사람)": f"{a_res['FP']:,} 명",
            "FN (놓친 환자)": f"{a_res['FN']:,} 명",
            "TN (정상 예측)": f"{a_res['TN']:,} 명",
            "정확도(Accuracy)": a_res["Accuracy"],
            "재현율(Recall)": a_res["Recall"],
            "정밀도(Precision)": a_res["Precision"],
            "F1 점수": a_res["F1"],
        }
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    summary_text = (
        f"💡 **가중치 조정 결과:** "
        f"**찾아낸 환자 수(TP)**는 **{b_res['TP']}명 → {a_res['TP']}명**, "
        f"**놓친 환자 수(FN)**는 **{b_res['FN']}명 → {a_res['FN']}명**, "
        f"**헛짚은 사람 수(FP)**는 **{b_res['FP']}명 → {a_res['FP']}명**으로 바뀌었습니다."
    )
    st.markdown(summary_text)

render_model_comparison("로지스틱 회귀 (확률로 답하는 모델)", lr_b_res, lr_a_res)
st.markdown("<br>", unsafe_allow_html=True)
render_model_comparison("의사결정트리 (질문으로 답하는 모델)", dt_b_res, dt_a_res)

st.markdown("---")

# ---------------------------------------------------------
# 5. 2D Decision Boundary Comparison
# ---------------------------------------------------------
st.subheader("3️⃣ 2차원 결정 경계선 비교 (전/후 겹쳐 그리기)")

c_x, c_y = st.columns(2)
with c_x:
    x_axis_label = st.selectbox("가로축(X축) 속성 선택:", selected_labels, index=0, key="2d_x")
with c_y:
    y_axis_label = st.selectbox(
        "세로축(Y축) 속성 선택:", 
        selected_labels, 
        index=1 if len(selected_labels) > 1 else 0,
        key="2d_y"
    )

if x_axis_label == y_axis_label:
    st.info("💡 서로 다른 두 속성을 선택하면 2차원 결정 경계선을 확인할 수 있습니다.")
else:
    x_col = FEATURE_MAP[x_axis_label]
    y_col = FEATURE_MAP[y_axis_label]

    test_medians = X_test.median()
    unselected_cols = [c for c in selected_cols if c not in [x_col, y_col]]
    fixed_texts = [f"**{REVERSE_FEATURE_MAP[c]}**: {test_medians[c]:.2f}" for c in unselected_cols]
    if fixed_texts:
        st.markdown(f"📌 **고정된 속성의 테스트 데이터 중앙값**: {', '.join(fixed_texts)}")

    x_min, x_max = X_test[x_col].min() - 1, X_test[x_col].max() + 1
    y_min, y_max = X_test[y_col].min() - 1, X_test[y_col].max() + 1
    if X_test[x_col].nunique() <= 2:
        x_min, x_max = -0.5, 1.5
    if X_test[y_col].nunique() <= 2:
        y_min, y_max = -0.5, 1.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    grid_df = pd.DataFrame(index=range(xx.size))
    for c in selected_cols:
        if c == x_col:
            grid_df[c] = xx.ravel()
        elif c == y_col:
            grid_df[c] = yy.ravel()
        else:
            grid_df[c] = test_medians[c]

    grid_df_scaled = scaler.transform(grid_df)
    probs_lr_b = log_reg_before.predict_proba(grid_df_scaled)[:, 1].reshape(xx.shape)
    probs_lr_a = log_reg_after.predict_proba(grid_df_scaled)[:, 1].reshape(xx.shape)

    fig_2d = go.Figure()

    # Scatter points
    plot_test_df = df_test.copy()
    for val, name, color in zip([0, 1], ["뇌졸중 없음(0)", "뇌졸중(1)"], ["#1f77b4", "#d62728"]):
        sub_df = plot_test_df[plot_test_df["stroke"] == val]
        fig_2d.add_trace(
            go.Scatter(
                x=sub_df[x_col], y=sub_df[y_col],
                mode="markers", name=name,
                marker=dict(color=color, size=6, opacity=0.7)
            )
        )

    # Before boundary line (Blue)
    lr_b_in = (probs_lr_b.min() < 0.5) and (probs_lr_b.max() > 0.5)
    if lr_b_in:
        fig_2d.add_trace(
            go.Contour(
                x=np.linspace(x_min, x_max, 150), y=np.linspace(y_min, y_max, 150),
                z=probs_lr_b, contours_start=0.5, contours_end=0.5,
                contours_coloring="lines", line_width=3, line_color="blue",
                name="가중치 조정 전 0.5 경계선", showscale=False
            )
        )
    else:
        st.info("ℹ️ 가중치 조정 전 로지스틱 회귀 모델의 0.5 경계선이 현재 그림 범위 밖에 있습니다.")

    # After boundary line (Red)
    lr_a_in = (probs_lr_a.min() < 0.5) and (probs_lr_a.max() > 0.5)
    if lr_a_in:
        fig_2d.add_trace(
            go.Contour(
                x=np.linspace(x_min, x_max, 150), y=np.linspace(y_min, y_max, 150),
                z=probs_lr_a, contours_start=0.5, contours_end=0.5,
                contours_coloring="lines", line_width=3, line_color="crimson",
                name="가중치 조정 후 0.5 경계선", showscale=False
            )
        )
    else:
        st.info("ℹ️ 가중치 조정 후 로지스틱 회귀 모델의 0.5 경계선이 현재 그림 범위 밖에 있습니다.")

    fig_2d.update_layout(
        title=f"로지스틱 회귀 2D 결정 경계 비교 ({x_axis_label} vs {y_axis_label})",
        xaxis_title=x_axis_label, yaxis_title=y_axis_label,
        legend_title="구분", hovermode="closest"
    )
    st.plotly_chart(fig_2d, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 6. 3D Decision Plane Comparison (If >= 3 Features Selected)
# ---------------------------------------------------------
if len(selected_labels) >= 3:
    st.subheader("4️⃣ 3차원 결정 경계 평면 비교 (속성 3개 선택 시)")

    col_3d_x, col_3d_y, col_3d_z = st.columns(3)
    with col_3d_x:
        x_3d_label = st.selectbox("3D X축 속성:", selected_labels, index=0, key="3d_x")
    with col_3d_y:
        y_3d_label = st.selectbox("3D Y축 속성:", selected_labels, index=1, key="3d_y")
    with col_3d_z:
        z_3d_label = st.selectbox("3D Z축 속성:", selected_labels, index=2, key="3d_z")

    if len({x_3d_label, y_3d_label, z_3d_label}) < 3:
        st.info("💡 3D 그래프를 확인하려면 X, Y, Z축에 서로 다른 속성을 지정하세요.")
    else:
        x_3d_col = FEATURE_MAP[x_3d_label]
        y_3d_col = FEATURE_MAP[y_3d_label]
        z_3d_col = FEATURE_MAP[z_3d_label]

        test_medians_3d = X_test.median()
        unselected_3d_cols = [c for c in selected_cols if c not in [x_3d_col, y_3d_col, z_3d_col]]
        fixed_texts_3d = [f"**{REVERSE_FEATURE_MAP[c]}**: {test_medians_3d[c]:.2f}" for c in unselected_3d_cols]
        if fixed_texts_3d:
            st.markdown(f"📌 **고정된 속성의 테스트 데이터 중앙값**: {', '.join(fixed_texts_3d)}")

        x3_min, x3_max = X_test[x_3d_col].min(), X_test[x_3d_col].max()
        y3_min, y3_max = X_test[y_3d_col].min(), X_test[y_3d_col].max()
        z3_min, z3_max = X_test[z_3d_col].min(), X_test[z_3d_col].max()

        # Grid generation for 3D surface
        x3_mesh, y3_mesh = np.meshgrid(
            np.linspace(x3_min, x3_max, 30),
            np.linspace(y3_min, y3_max, 30)
        )

        # Calculate z-plane values for 0.5 probability (w_x*x + w_y*y + w_z*z + b = 0)
        def compute_05_z_plane(model, x_col, y_col, z_col, x_mesh, y_mesh):
            coefs = model.coef_[0]
            intercept = model.intercept_[0]

            idx_x = selected_cols.index(x_col)
            idx_y = selected_cols.index(y_col)
            idx_z = selected_cols.index(z_col)

            # Get unselected column values contribution to intercept
            unsel_contrib = 0.0
            for i, col_name in enumerate(selected_cols):
                if i not in [idx_x, idx_y, idx_z]:
                    unsel_contrib += coefs[i] * ((test_medians_3d[col_name] - scaler.mean_[i]) / scaler.scale_[i])

            w_x = coefs[idx_x] / scaler.scale_[idx_x]
            w_y = coefs[idx_y] / scaler.scale_[idx_y]
            w_z = coefs[idx_z] / scaler.scale_[idx_z]

            mean_x, scale_x = scaler.mean_[idx_x], scaler.scale_[idx_x]
            mean_y, scale_y = scaler.mean_[idx_y], scaler.scale_[idx_y]
            mean_z, scale_z = scaler.mean_[idx_z], scaler.scale_[idx_z]

            # Linear formula: sum(w_i * (val_i - mean_i)/scale_i) + intercept = 0
            # Solve for z_mesh (raw value)
            if abs(w_z) < 1e-6:
                return None
            
            z_mesh = mean_z - scale_z * (
                w_x * (x_mesh - mean_x) / scale_x +
                w_y * (y_mesh - mean_y) / scale_y +
                unsel_contrib + intercept
            ) / coefs[idx_z]
            return z_mesh

        z_plane_b = compute_05_z_plane(log_reg_before, x_3d_col, y_3d_col, z_3d_col, x3_mesh, y3_mesh)
        z_plane_a = compute_05_z_plane(log_reg_after, x_3d_col, y_3d_col, z_3d_col, x3_mesh, y3_mesh)

        fig_3d = go.Figure()

        # 3D Scatter points
        plot_test_3d = df_test.copy()
        for val, name, color in zip([0, 1], ["뇌졸중 없음(0)", "뇌졸중(1)"], ["#1f77b4", "#d62728"]):
            sub_df = plot_test_3d[plot_test_3d["stroke"] == val]
            fig_3d.add_trace(
                go.Scatter3d(
                    x=sub_df[x_3d_col], y=sub_df[y_3d_col], z=sub_df[z_3d_col],
                    mode="markers", name=name,
                    marker=dict(color=color, size=4, opacity=0.7)
                )
            )

        # Before 3D plane (Blue)
        if z_plane_b is not None:
            fig_3d.add_trace(
                go.Surface(
                    x=x3_mesh, y=y3_mesh, z=z_plane_b,
                    colorscale=[[0, "blue"], [1, "blue"]], opacity=0.4,
                    showscale=False, name="가중치 조정 전 0.5 평면"
                )
            )

        # After 3D plane (Red)
        if z_plane_a is not None:
            fig_3d.add_trace(
                go.Surface(
                    x=x3_mesh, y=y3_mesh, z=z_plane_a,
                    colorscale=[[0, "crimson"], [1, "crimson"]], opacity=0.5,
                    showscale=False, name="가중치 조정 후 0.5 평면"
                )
            )

        fig_3d.update_layout(
            title="3D 결정 경계 평면 비교 (마우스로 드래그하여 회전할 수 있습니다)",
            scene=dict(
                xaxis_title=x_3d_label,
                yaxis_title=y_3d_label,
                zaxis_title=z_3d_label,
                zaxis=dict(range=[z3_min - 1, z3_max + 1])
            ),
            legend_title="구분"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
