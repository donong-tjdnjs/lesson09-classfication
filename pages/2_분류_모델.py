import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, _tree

# 1. Page Configuration
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실",
    page_icon="🤖",
    layout="wide",
)

# 2. Load Data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df_raw = load_data()

st.title("🤖 뇌졸중 분류 모델 실습")
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

default_selected = ["나이", "평균 혈당", "고혈압", "심장병"]
selected_labels = st.multiselect(
    "모델 학습 및 예측에 사용할 속성을 선택하세요 (최소 2개 이상):",
    options=list(FEATURE_MAP.keys()),
    default=default_selected
)

if len(selected_labels) < 2:
    st.warning("⚠️ 속성을 최소 2개 이상 선택해 주세요!")
    st.stop()

selected_cols = [FEATURE_MAP[label] for label in selected_labels]

# ---------------------------------------------------------
# 2. Train / Test Split (Group of 10 -> 3 Test, 7 Train)
# ---------------------------------------------------------
# Sort by id or index
if "id" in df_raw.columns:
    df_sorted = df_raw.sort_values("id").reset_index(drop=True)
else:
    df_sorted = df_raw.reset_index(drop=True)

# Select test mask: first 3 of every 10 items
test_mask = (df_sorted.index % 10) < 3
df_test = df_sorted[test_mask].copy()
df_train = df_sorted[~test_mask].copy()

# ---------------------------------------------------------
# 3. Imputation (BMI median calculated strictly from train set)
# ---------------------------------------------------------
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
# 4. Model Training
# ---------------------------------------------------------
# (1) Baseline Model (Majority Class from Train)
majority_class = int(pd.Series(y_train).mode()[0])
baseline_train_acc = (y_train == majority_class).mean() * 100
baseline_test_acc = (y_test == majority_class).mean() * 100

# (2) Logistic Regression (with Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

log_reg_train_acc = log_reg.score(X_train_scaled, y_train) * 100
log_reg_test_acc = log_reg.score(X_test_scaled, y_test) * 100

# (3) Decision Tree
dt_cls = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)
dt_cls.fit(X_train, y_train)

dt_train_acc = dt_cls.score(X_train, y_train) * 100
dt_test_acc = dt_cls.score(X_test, y_test) * 100

# ---------------------------------------------------------
# 5. Model Metrics Display Cards
# ---------------------------------------------------------
st.subheader("2️⃣ 모델 평가 (정확도)")

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown("##### 📌 기준 모델")
    st.caption("입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델")
    st.metric("테스트 정확도", f"{baseline_test_acc:.2f}%")
    st.caption(f"훈련: {baseline_train_acc:.2f}% | 테스트: {baseline_test_acc:.2f}%")

with col_m2:
    st.markdown("##### 📈 로지스틱 회귀")
    st.caption("확률로 답하는 모델")
    st.metric("테스트 정확도", f"{log_reg_test_acc:.2f}%")
    st.caption(f"훈련: {log_reg_train_acc:.2f}% | 테스트: {log_reg_test_acc:.2f}%")

with col_m3:
    st.markdown("##### 🌳 의사결정트리")
    st.caption("질문으로 답하는 모델")
    st.metric("테스트 정확도", f"{dt_test_acc:.2f}%")
    st.caption(f"훈련: {dt_train_acc:.2f}% | 테스트: {dt_test_acc:.2f}%")

st.markdown("---")

# ---------------------------------------------------------
# 6. 2D Boundary & Decision Regions Visualization (Plotly)
# ---------------------------------------------------------
st.subheader("3️⃣ 속성 2개 기준 경계 및 칸 분할 시각화")

c_x, c_y = st.columns(2)
with c_x:
    x_axis_label = st.selectbox("가로축(X축) 속성 선택:", selected_labels, index=0)
with c_y:
    y_axis_label = st.selectbox(
        "세로축(Y축) 속성 선택:", 
        selected_labels, 
        index=1 if len(selected_labels) > 1 else 0
    )

if x_axis_label == y_axis_label:
    st.info("💡 서로 다른 두 속성을 선택하면 2차원 경계 그래프를 확인할 수 있습니다.")
else:
    x_col = FEATURE_MAP[x_axis_label]
    y_col = FEATURE_MAP[y_axis_label]

    # Calculate medians of test data for unselected dimensions
    test_medians = X_test.median()
    unselected_cols = [c for c in selected_cols if c not in [x_col, y_col]]

    fixed_info_texts = []
    for c in unselected_cols:
        lbl = REVERSE_FEATURE_MAP[c]
        val = test_medians[c]
        fixed_info_texts.append(f"**{lbl}**: {val:.2f}")

    if fixed_info_texts:
        st.markdown(f"📌 **고정된 속성의 테스트 데이터 중앙값**: {', '.join(fixed_info_texts)}")

    # Meshgrid setup
    x_min, x_max = X_test[x_col].min() - 1, X_test[x_col].max() + 1
    y_min, y_max = X_test[y_col].min() - 1, X_test[y_col].max() + 1

    # Adjust for binary/discrete features
    if X_test[x_col].nunique() <= 2:
        x_min, x_max = -0.5, 1.5
    if X_test[y_col].nunique() <= 2:
        y_min, y_max = -0.5, 1.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 150),
        np.linspace(y_min, y_max, 150)
    )

    # Build full input grid for prediction
    grid_df = pd.DataFrame(index=range(xx.size))
    for c in selected_cols:
        if c == x_col:
            grid_df[c] = xx.ravel()
        elif c == y_col:
            grid_df[c] = yy.ravel()
        else:
            grid_df[c] = test_medians[c]

    # Decision Tree Predictions for Contour (Regions)
    dt_grid_preds = dt_cls.predict(grid_df).reshape(xx.shape)

    # Logistic Regression Predictions for 0.5 Boundary Line
    grid_df_scaled = scaler.transform(grid_df)
    log_reg_grid_probs = log_reg.predict_proba(grid_df_scaled)[:, 1].reshape(xx.shape)

    fig_boundary = go.Figure()

    # 1. Decision Tree Regions (Shaded background)
    fig_boundary.add_trace(
        go.Heatmap(
            x=np.linspace(x_min, x_max, 150),
            y=np.linspace(y_min, y_max, 150),
            z=dt_grid_preds,
            colorscale=[[0, "rgba(173, 216, 230, 0.3)"], [1, "rgba(255, 182, 193, 0.4)"]],
            showscale=False,
            hoverinfo="skip",
            name="의사결정트리 분할 구역"
        )
    )

    # 2. Logistic Regression Boundary Line (Prob = 0.5)
    # Check if boundary line falls within plot range
    has_boundary_line = (log_reg_grid_probs.min() < 0.5) and (log_reg_grid_probs.max() > 0.5)

    if has_boundary_line:
        fig_boundary.add_trace(
            go.Contour(
                x=np.linspace(x_min, x_max, 150),
                y=np.linspace(y_min, y_max, 150),
                z=log_reg_grid_probs,
                contours_start=0.5,
                contours_end=0.5,
                contours_coloring="lines",
                line_width=3,
                line_color="crimson",
                name="로지스틱 회귀 경계선 (0.5)",
                showscale=False
            )
        )
    else:
        st.info("ℹ️ 로지스틱 회귀 모델의 0.5 기준 결정 경계선이 현재 선택한 속성 축 및 범위 밖 영역에 존재합니다.")

    # 3. Test Scatter Points
    plot_test_df = df_test.copy()
    plot_test_df["stroke_label"] = plot_test_df["stroke"].map({0: "뇌졸중 없음(0)", 1: "뇌졸중(1)"})

    for val, name, color in zip([0, 1], ["뇌졸중 없음(0)", "뇌졸중(1)"], ["#1f77b4", "#d62728"]):
        sub_df = plot_test_df[plot_test_df["stroke"] == val]
        fig_boundary.add_trace(
            go.Scatter(
                x=sub_df[x_col],
                y=sub_df[y_col],
                mode="markers",
                name=name,
                marker=dict(color=color, size=6, opacity=0.8),
                hovertemplate=f"<b>{x_axis_label}</b>: %{{x}}<br><b>{y_axis_label}</b>: %{{y}}<extra></extra>"
            )
        )

    fig_boundary.update_layout(
        title=f"테스트 데이터 산점도 및 결정 경계 ({x_axis_label} vs {y_axis_label})",
        xaxis_title=x_axis_label,
        yaxis_title=y_axis_label,
        legend_title="실제 뇌졸중 여부",
        hovermode="closest"
    )

    st.plotly_chart(fig_boundary, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 7. Decision Tree Graphviz Diagram & Rule Summary
# ---------------------------------------------------------
st.subheader("4️⃣ 의사결정트리 나무 구조 시각화")

def generate_dot_tree(tree, feature_names, class_names=["뇌졸중 없음(0)", "뇌졸중(1)"]):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    dot_lines = [
        'digraph Tree {',
        'node [shape=box, style="filled, rounded", color="black", fontname="helvetica"] ;',
        'edge [fontname="helvetica"] ;'
    ]

    def recurse(node, parent_id=None, edge_label=""):
        samples = tree_.n_node_samples[node]
        value = tree_.value[node][0]
        pos_count = int(value[1]) if len(value) > 1 else 0
        pos_ratio = (pos_count / samples * 100) if samples > 0 else 0.0

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            kor_name = REVERSE_FEATURE_MAP.get(name, name)
            
            label = (
                f"{kor_name} <= {threshold:.2f}\\n"
                f"사람 수: {samples}명\\n"
                f"뇌졸중: {pos_count}명 ({pos_ratio:.1f}%)"
            )
            fillcolor = "#ffffff"
            dot_lines.append(f'  {node} [label="{label}", fillcolor="{fillcolor}"] ;')

            if parent_id is not None:
                dot_lines.append(f'  {parent_id} -> {node} [label="{edge_label}"] ;')

            left_child = tree_.children_left[node]
            right_child = tree_.children_right[node]
            recurse(left_child, node, "예")
            recurse(right_child, node, "아니요")
        else:
            pred_class = np.argmax(value)
            # Distinct colors for leaf nodes based on final decision
            fillcolor = "#ffcccc" if pred_class == 1 else "#cce5ff"
            class_str = class_names[pred_class]

            label = (
                f"예측: {class_str}\\n"
                f"사람 수: {samples}명\\n"
                f"뇌졸중: {pos_count}명 ({pos_ratio:.1f}%)"
            )
            dot_lines.append(f'  {node} [label="{label}", fillcolor="{fillcolor}"] ;')

            if parent_id is not None:
                dot_lines.append(f'  {parent_id} -> {node} [label="{edge_label}"] ;')

    recurse(0)
    dot_lines.append('}')
    return "\n".join(dot_lines)

dot_graph = generate_dot_tree(dt_cls, selected_cols)
st.graphviz_chart(dot_graph)

# Decision Tree Summary Statistics
tree_ = dt_cls.tree_
leaf_nodes = [i for i in range(tree_.node_count) if tree_.children_left[i] == _tree.TREE_UNDEFINED]
total_leaves = len(leaf_nodes)

no_stroke_leaves = sum(
    1 for i in leaf_nodes if np.argmax(tree_.value[i][0]) == 0
)

# Identify features actually used by the tree
used_feature_indices = set(tree_.feature[tree_.feature != _tree.TREE_UNDEFINED])
used_feature_names = [selected_cols[i] for i in used_feature_indices]
used_kor_names = [REVERSE_FEATURE_MAP[col] for col in used_feature_names]

st.markdown("##### 📝 의사결정트리 구조 요약")
st.write(f"- 최종 답을 내는 마디(잎 노드)는 총 **{total_leaves}개**이며, 그중 **{no_stroke_leaves}개**가 '뇌졸중 없음(0)'으로 답합니다.")
st.write(f"- 고른 속성 중 이 나무가 실제로 질문에 활용한 속성은 **{', '.join(used_kor_names) if used_kor_names else '없음'}** 입니다.")
