import pandas as pd
import plotly.express as px
import streamlit as st

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="탐색 - 뇌졸중 예측 실습실",
    page_icon="🔍",
    layout="wide",
)


# 2. 데이터 로드 (캐싱 적용, UTF-8 인코딩)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df


df = load_data()

st.title("🔍 데이터 탐색하기")
st.markdown("---")

# ---------------------------------------------------------
# 1. 나이와 평균 혈당의 분포 (히스토그램)
# ---------------------------------------------------------
st.subheader("1️⃣ 나이 및 평균 혈당 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age_hist = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이(Age) 분포",
        labels={"age": "나이", "count": "인원 수"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig_age_hist.update_layout(bargap=0.1)
    st.plotly_chart(fig_age_hist, use_container_width=True)

with col2:
    fig_glucose_hist = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당(Avg Glucose Level) 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "인원 수"},
        color_discrete_sequence=["#ff7f0e"],
    )
    fig_glucose_hist.update_layout(bargap=0.1)
    st.plotly_chart(fig_glucose_hist, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 2. 뇌졸중 여부에 따른 나이 및 평균 혈당 비교 (상자그림 & 평균 표)
# ---------------------------------------------------------
st.subheader("2️⃣ 뇌졸중 여부별 나이 및 평균 혈당 비교")

# 라벨링 추가 (0: 미발병, 1: 발병)
df["stroke_label"] = df["stroke"].map({0: "뇌졸중 미발병(0)", 1: "뇌졸중 발병(1)"})

b_col1, b_col2 = st.columns(2)

with b_col1:
    fig_age_box = px.box(
        df,
        x="stroke_label",
        y="age",
        color="stroke_label",
        title="뇌졸중 여부별 나이 분포 (Boxplot)",
        labels={"stroke_label": "구분", "age": "나이"},
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with b_col2:
    fig_glucose_box = px.box(
        df,
        x="stroke_label",
        y="avg_glucose_level",
        color="stroke_label",
        title="뇌졸중 여부별 평균 혈당 분포 (Boxplot)",
        labels={"stroke_label": "구분", "avg_glucose_level": "평균 혈당"},
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)

# 그룹별 평균값 표 계산
mean_table = (
    df.groupby("stroke_label")[["age", "avg_glucose_level"]]
    .mean()
    .reset_index()
    .rename(
        columns={
            "stroke_label": "뇌졸중 여부",
            "age": "평균 나이(세)",
            "avg_glucose_level": "평균 혈당 수치",
        }
    )
)
st.markdown("##### 📊 뇌졸중 그룹별 평균값")
st.dataframe(mean_table, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 3. 고혈압/심장병 유무에 따른 뇌졸중 비율 (막대그래프)
# ---------------------------------------------------------
st.subheader("3️⃣ 고혈압 및 심장병 유무별 뇌졸중 비율")

df["hypertension_label"] = df["hypertension"].map({0: "고혈압 없음", 1: "고혈압 있음"})
df["heart_disease_label"] = df["heart_disease"].map(
    {0: "심장병 없음", 1: "심장병 있음"}
)

# 고혈압 비율 계산
ht_ratio = (
    df.groupby("hypertension_label")["stroke"]
    .mean()
    .reset_index()
    .rename(columns={"stroke": "stroke_ratio"})
)
ht_ratio["stroke_ratio_pct"] = ht_ratio["stroke_ratio"] * 100

# 심장병 비율 계산
hd_ratio = (
    df.groupby("heart_disease_label")["stroke"]
    .mean()
    .reset_index()
    .rename(columns={"stroke": "stroke_ratio"})
)
hd_ratio["stroke_ratio_pct"] = hd_ratio["stroke_ratio"] * 100

m_col1, m_col2 = st.columns(2)

with m_col1:
    fig_ht_bar = px.bar(
        ht_ratio,
        x="hypertension_label",
        y="stroke_ratio_pct",
        text="stroke_ratio_pct",
        title="고혈압 유무별 뇌졸중 비율 (%)",
        labels={"hypertension_label": "고혈압 여부", "stroke_ratio_pct": "뇌졸중 비율 (%)"},
        color="hypertension_label",
    )
    fig_ht_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig_ht_bar, use_container_width=True)

with m_col2:
    fig_hd_bar = px.bar(
        hd_ratio,
        x="heart_disease_label",
        y="stroke_ratio_pct",
        text="stroke_ratio_pct",
        title="심장병 유무별 뇌졸중 비율 (%)",
        labels={"heart_disease_label": "심장병 여부", "stroke_ratio_pct": "뇌졸중 비율 (%)"},
        color="heart_disease_label",
    )
    fig_hd_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig_hd_bar, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 4. BMI 결측치 집단 vs 전체 뇌졸중 비율 비교
# ---------------------------------------------------------
st.subheader("4️⃣ BMI 결측치 집단과 전체 뇌졸중 비율 비교")

bmi_null_df = df[df["bmi"].isnull()]

total_stroke_ratio = (df["stroke"].mean()) * 100
null_bmi_stroke_ratio = (bmi_null_df["stroke"].mean()) * 100

bmi_compare_df = pd.DataFrame(
    [
        {
            "구분": "전체 데이터",
            "총 인원 수": len(df),
            "뇌졸중 환자 수": int(df["stroke"].sum()),
            "뇌졸중 비율 (%)": f"{total_stroke_ratio:.2f}%",
        },
        {
            "구분": "BMI 결측치 집단",
            "총 인원 수": len(bmi_null_df),
            "뇌졸중 환자 수": int(bmi_null_df["stroke"].sum()),
            "뇌졸중 비율 (%)": f"{null_bmi_stroke_ratio:.2f}%",
        },
    ]
)

st.dataframe(bmi_compare_df, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 5. 흡연 상태(smoking_status) 값별 사람 수
# ---------------------------------------------------------
st.subheader("5️⃣ 흡연 상태별 인원 수")

smoke_count_df = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .reset_index()
    .rename(columns={"smoking_status": "흡연 상태", "count": "인원 수"})
)

# 컬럼 명칭 하위 버전 지원 처리
if "index" in smoke_count_df.columns:
    smoke_count_df = smoke_count_df.rename(
        columns={"index": "흡연 상태", "smoking_status": "인원 수"}
    )

st.dataframe(smoke_count_df, use_container_width=True)
