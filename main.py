import json
import os
import pandas as pd
import streamlit as st

# 1. 페이지 설정 및 브라우저 탭 아이콘 설정
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

# 파일 저장 폴더 및 파일 경로 설정
SAVE_DIR = "내_기록"
METADATA_FILE = os.path.join(SAVE_DIR, "metadata.json")
SOURCE_FILE = os.path.join(SAVE_DIR, "source.json")

# 폴더 생성 함수
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# 저장된 기록 로드/저장 함수
def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 2. 데이터 로드 (캐싱)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df


df = load_data()

# 3. 화면 헤더
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("---")

# 4. 숫자 카드 4개 (전체 사람 수, 열 개수, stroke가 1인 사람 수, 그 비율)
total_count = len(df)
col_count = len(df.columns)
stroke_count = int(df["stroke"].sum()) if "stroke" in df.columns else 0
stroke_ratio = (stroke_count / total_count * 100) if total_count > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("👥 전체 사람 수", f"{total_count:,} 명")
c2.metric("📊 열 개수", f"{col_count} 개")
c3.metric("🚨 stroke=1 환자 수", f"{stroke_count:,} 명")
c4.metric("📈 stroke=1 비율", f"{stroke_ratio:.2f} %")

st.markdown("---")

# 5. 열 정보 및 교재용 우리말 뜻 직접 입력 표
st.subheader("📋 데이터 열 메타데이터 (우리말 뜻 입력)")

# 기존 유저 입력 내용 로드
saved_meanings = load_json(METADATA_FILE, {})

cols_info = []
for col in df.columns:
    null_count = int(df[col].isnull().sum())
    dtype_str = str(df[col].dtype)
    current_meaning = saved_meanings.get(col, "")
    cols_info.append(
        {
            "열 이름": col,
            "우리말 뜻": current_meaning,
            "값의 종류": dtype_str,
            "빈 값 개수": null_count,
        }
    )

meta_df = pd.DataFrame(cols_info)

# Streamlit data_editor를 통해 직접 편집 가능한 테이블 생성
edited_meta_df = st.data_editor(
    meta_df,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    use_container_width=True,
    num_rows="fixed",
    key="meta_editor",
)

# 편집된 '우리말 뜻' 저장
if st.button("💾 우리말 뜻 저장하기"):
    new_meanings = dict(
        zip(edited_meta_df["열 이름"], edited_meta_df["우리말 뜻"])
    )
    save_json(METADATA_FILE, new_meanings)
    st.success("우리말 뜻이 '내_기록/metadata.json'에 정상적으로 저장되었습니다!")

st.markdown("---")

# 6. 상위 5개 데이터 미리보기
st.subheader("🔍 데이터 미리보기 (상위 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.markdown("---")

# 7. 데이터 출처 입력 구역 (교재 내용 저장)
st.subheader("📚 데이터 출처")

saved_source = load_json(SOURCE_FILE, {"source": ""}).get("source", "")

source_input = st.text_area(
    "교재에 적힌 데이터 출처를 아래에 직접 적어주세요:",
    value=saved_source,
    height=100,
    placeholder="예: Kaggle Stroke Prediction Dataset...",
)

if st.button("💾 출처 저장하기"):
    save_json(SOURCE_FILE, {"source": source_input})
    st.success("데이터 출처가 '내_기록/source.json'에 정상적으로 저장되었습니다!")
