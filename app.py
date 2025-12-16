import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# 기본 설정
# =========================
DATA_PATH = "diet_log.csv"

st.set_page_config(page_title="식단 취향 기반 메뉴 추천 AI", layout="centered")
st.title("🍽 식단 기록 기반 맞춤 메뉴 추천 웹")

st.markdown("""
이 웹은 개인 식단 데이터를 기록하고,  
그 데이터를 이용해 '오늘의 메뉴'를 추천해줍니다.
""")

# 우리가 사용할 기분 옵션 (기록 + 추천 공통)
MOOD_OPTIONS = ["happy", "stressed", "tired", "rushed", "neutral", "hungry"]
MOOD_LABEL_KO = {
    "happy": "기분 좋음 😊",
    "stressed": "스트레스 😵",
    "tired": "피곤함 😪",
    "rushed": "바쁨/시간 없음 ⏱",
    "neutral": "그냥 저냥 😐",
    "hungry": "배고픔 🤤"
}
MOOD_KO_TO_EN = {v: k for k, v in MOOD_LABEL_KO.items()}


# =========================
# 데이터 로드 / 저장
# =========================
def load_data(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["date", "meal", "food", "category", "reason", "mood", "satisfaction"])
    # 타입 정리
    if "satisfaction" in df.columns:
        df["satisfaction"] = pd.to_numeric(df["satisfaction"], errors="coerce").fillna(3)
    return df


def save_data(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


df = load_data(DATA_PATH)


# =========================
# 딥러닝 모델 학습 함수
# =========================
@st.cache_resource(show_spinner=True)
def train_model(df_input: pd.DataFrame):
    """
    df_input: date, meal, food, category, reason, mood, satisfaction 컬럼을 포함한 DataFrame
    반환: (model, mood_le, food_le)
    """
    df_train = df_input.copy()

    # fast / need_soup feature 생성 (간편식 여부, 국물 여부)
    if "fast" not in df_train.columns:
        df_train["fast"] = df_train["category"].fillna("").str.contains(
            "간편|패스트푸드|분식|샌드위치|김밥|햄버거"
        ).astype(int)

    if "need_soup" not in df_train.columns:
        df_train["need_soup"] = df_train["category"].fillna("").str.contains(
            "국물|찌개|국|라멘|라면|순두부|우동|짬뽕|부대찌개"
        ).astype(int)

    # mood / food 라벨 인코딩
    mood_le = LabelEncoder()
    food_le = LabelEncoder()

    # 결측값 대비
    df_train["mood"] = df_train["mood"].fillna("neutral")
    df_train["food"] = df_train["food"].fillna("알수없음")

    df_train["mood_idx"] = mood_le.fit_transform(df_train["mood"])
    df_train["food_idx"] = food_le.fit_transform(df_train["food"])

    num_mood_classes = df_train["mood_idx"].nunique()
    num_food_classes = df_train["food_idx"].nunique()

    # 입력/출력 정의
    X = df_train[["mood_idx", "fast", "need_soup", "satisfaction"]]
    y = df_train["food_idx"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Keras 모델 정의
    mood_input = layers.Input(shape=(1,), name="mood_input")
    fast_input = layers.Input(shape=(1,), name="fast_input")
    soup_input = layers.Input(shape=(1,), name="soup_input")
    sat_input = layers.Input(shape=(1,), name="sat_input")

    # mood embedding
    mood_embed_dim = min(4, num_mood_classes)  # 너무 크지 않게
    mood_embed = layers.Embedding(
        input_dim=num_mood_classes,
        output_dim=mood_embed_dim,
        name="mood_embedding"
    )(mood_input)
    mood_embed = layers.Flatten()(mood_embed)

    # 다른 입력과 결합
    x = layers.Concatenate()([mood_embed, fast_input, soup_input, sat_input])

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(num_food_classes, activation="softmax", name="food_output")(x)

    model = models.Model(
        inputs=[mood_input, fast_input, soup_input, sat_input],
        outputs=output
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 학습
    model.fit(
        {
            "mood_input": X_train["mood_idx"],
            "fast_input": X_train["fast"],
            "soup_input": X_train["need_soup"],
            "sat_input": X_train["satisfaction"],
        },
        y_train,
        validation_data=(
            {
                "mood_input": X_test["mood_idx"],
                "fast_input": X_test["fast"],
                "soup_input": X_test["need_soup"],
                "sat_input": X_test["satisfaction"],
            },
            y_test,
        ),
        epochs=40,
        batch_size=8,
        verbose=0
    )

    return model, mood_le, food_le


def recommend_food(model, mood_le, food_le, mood_str, fast, need_soup, satisfaction=3.0, top_k=3):
    # mood_str이 학습에 없으면 가장 가까운 걸로 대체
    if mood_str not in mood_le.classes_:
        mood_idx = 0
    else:
        mood_idx = mood_le.transform([mood_str])[0]

    inp = {
        "mood_input": np.array([mood_idx]),
        "fast_input": np.array([fast]),
        "soup_input": np.array([need_soup]),
        "sat_input": np.array([satisfaction]),
    }

    probs = model.predict(inp, verbose=0)[0]
    top_indices = probs.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        food_name = food_le.inverse_transform([idx])[0]
        results.append((food_name, float(probs[idx])))
    return results


# =========================
# UI: 탭 구성
# =========================
tab1, tab2 = st.tabs(["🥗 식단 기록하기", "🤖 메뉴 추천"])

# -------------------------
# 탭 1: 식단 기록
# -------------------------
with tab1:
    st.subheader("1️⃣ 오늘 먹은 메뉴 기록하기")

    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("날짜", datetime.now())
    with col2:
        meal = st.selectbox("끼니", ["lunch", "dinner"])

    food = st.text_input("음식 이름 (예: 부대찌개, 김밥 등)")
    category = st.text_input("카테고리 (예: 한식/국물, 간편식, 분식, 일식 등)")
    reason = st.text_area("선택 이유 (기분/상황 등 자유롭게 적기)")

    mood_choice_ko = st.selectbox(
        "당시 기분",
        [MOOD_LABEL_KO[m] for m in MOOD_OPTIONS]
    )
    mood_en = MOOD_KO_TO_EN[mood_choice_ko]

    satisfaction = st.slider("만족도 (1~5)", 1, 5, 4)

    if st.button("기록 저장", key="save_record"):
        if food.strip() == "":
            st.warning("음식 이름은 반드시 입력해야 합니다.")
        else:
            new_row = {
                "date": date.strftime("%Y-%m-%d"),
                "meal": meal,
                "food": food,
                "category": category,
                "reason": reason,
                "mood": mood_en,
                "satisfaction": satisfaction
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_data(df, DATA_PATH)
            st.success("식단 기록이 저장되었습니다!")

    st.markdown("---")
    st.markdown("### 최근 기록된 식단 (최신 10개)")
    if len(df) > 0:
        st.dataframe(df.tail(10))
    else:
        st.info("아직 기록된 데이터가 없습니다.")


# -------------------------
# 탭 2: 메뉴 추천
# -------------------------
with tab2:
    st.subheader("2️⃣ 맞춤 메뉴 추천 받기")

    if len(df) < 20:
        st.warning("데이터가 너무 적습니다. 최소 20개 이상 기록해 주세요.")
    else:
        st.markdown("현재 학습에 사용 가능한 식단 데이터 개수: **{}개**".format(len(df)))

        mood_choice_ko_rec = st.selectbox(
            "오늘 현재 기분",
            [MOOD_LABEL_KO[m] for m in MOOD_OPTIONS],
            key="rec_mood"
        )
        mood_en_rec = MOOD_KO_TO_EN[mood_choice_ko_rec]

        situation = st.selectbox(
            "지금 상황은 어떤가요?",
            ["여유 있게 먹고 싶음", "시간이 없어서 빨리 먹고 싶음"]
        )

        want_soup = st.checkbox("따뜻한 국물 있는 메뉴가 좋다 🔥", value=False)

        # fast, need_soup 플래그로 변환
        fast_flag = 1 if situation == "시간이 없어서 빨리 먹고 싶음" else 0
        soup_flag = 1 if want_soup else 0

        sat_for_pred = st.slider(
            "예상 만족도 가중치 (적당히 1~5, 3이면 중간)",
            1, 5, 3
        )

        if st.button("메뉴 추천 받기 👇"):
            with st.spinner("로딩 중입니다..."):
                model, mood_le, food_le = train_model(df)

            results = recommend_food(
                model, mood_le, food_le,
                mood_en_rec, fast_flag, soup_flag,
                satisfaction=float(sat_for_pred),
                top_k=3
            )

            if not results:
                st.warning("추천 결과가 없습니다. 데이터를 조금 더 모아보세요.")
            else:
                st.success("오늘 이런 메뉴는 어떠세요? 😋")

                for food_name, prob in results:
                    st.markdown(f"""
**🍽 {food_name}**  
- 예측 확률: `{prob:.2f}`  
---
""")


st.markdown("---")
st.caption("개인 식단 데이터를 활용한 맞춤 메뉴 추천 앱")
