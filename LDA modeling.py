import pandas as pd
from konlpy.tag import Okt
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt
import wordcloud
import numpy as np

# --- 설정 변수 ---
# ⚠️ 주의: 토픽 모델링을 실행할 최종 데이터 파일 경로로 변경하세요.
# (예: 'soop_data_labeled_final_for_model.csv' 또는 'soop_community_data_raw.csv')
FILE_PATH = 'soop_community_data_raw.csv'
NUM_TOPICS = 5  # 추출할 토픽 개수 (5~10개 사이 권장)
MIN_WORD_COUNT = 5  # 최소 5회 이상 등장한 단어만 사용

# --- 전처리 및 토큰화 함수 ---
def preprocess_text(text):
    """KoNLPy를 사용하여 텍스트를 명사 토큰화하고 불용어를 제거합니다."""
    # KoNLPy Okt 초기화 (시간이 걸릴 수 있습니다)
    okt = Okt()

    # 1. 도메인 특화 불용어 리스트
    stop_words = set([
        '이다', '하다', '있다', '없다', '되다', '이다', '것', '수', '이', '그', '저', '저희',
        '우리', '같다', '말', '안', '좀', '정말', '진짜', '너무', '요청', '문의', '개선',
        '숲', 'SOOP', 'BJ', '방송', '유저', '운영자', '문제', '내용', '부분', '점',
        '하나', '이렇', '또한', '어요', '저렇', '보이', '아니', '그렇' # 일반적인 연결어 및 동사 추가
    ])

    tokens = []
    if pd.isna(text):
        return []

    for word in okt.nouns(str(text)):
        # 2글자 이상 단어만 사용 및 불용어 제거
        if len(word) > 1 and word not in stop_words:
            tokens.append(word)
    return tokens


# --- 메인 LDA 함수 ---
def perform_lda_analysis():
    print("### 1. 데이터 로드 및 전처리 시작 ###")
    try:
        # 지정된 파일 경로에서 데이터 로드
        df = pd.read_csv(FILE_PATH)
        df = df.dropna(subset=['content']).reset_index(drop=True)
        print(f"-> 총 분석 대상 문서 수: {len(df)}건")
        print(f"-> 파일 경로: {FILE_PATH}")

    except FileNotFoundError:
        print(f"오류: 지정된 파일 '{FILE_PATH}'을(를) 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # 텍스트 전처리 및 토큰화
    df['tokens'] = df['content'].apply(preprocess_text)

    # 2. 딕셔너리 및 코퍼스 생성
    dictionary = corpora.Dictionary(df['tokens'])

    # 최소 빈도수 미만 단어 제거 (노이즈 감소)
    dictionary.filter_extremes(no_below=MIN_WORD_COUNT)

    corpus = [dictionary.doc2bow(doc) for doc in df['tokens']]

    # 3. LDA 모델 학습
    print(f"### 2. LDA 모델 학습 시작 (토픽 {NUM_TOPICS}개) ###")
    # LDA 모델 학습 실행
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        random_state=42,
        passes=20,  # 학습 횟수 증가 (정확도 향상)
        alpha='auto',
        per_word_topics=True
    )

    # 4. 결과 출력 및 보고서 활용
    print("### 3. 추출된 토픽 및 핵심 키워드 ###")
    topics = lda_model.show_topics(num_words=10, formatted=False)

    for i, topic in enumerate(topics):
        keywords = [word[0] for word in topic[1]]
        print(f"토픽 {i + 1}: {', '.join(keywords)}")

    # 5. 워드 클라우드 생성 (보고서 시각 자료)
    generate_word_cloud(lda_model, dictionary, NUM_TOPICS)


def generate_word_cloud(lda_model, dictionary, num_topics):
    """토픽별 핵심 키워드를 워드 클라우드로 시각화합니다."""
    # ⚠️ 폰트 경로 설정: 시스템 환경에 맞게 한글 폰트 경로를 설정해야 합니다.
    try:
        font_path = 'c:/Windows/Fonts/malgun.ttf'
        wc = wordcloud.WordCloud(
            font_path=font_path,
            background_color='white',
            width=800,
            height=400,
            max_words=50
        )
    except Exception:
        # 폰트 로드 실패 시 기본 폰트로 대체 (한글 깨짐 발생 가능)
        wc = wordcloud.WordCloud(
            background_color='white',
            width=800,
            height=400,
            max_words=50
        )

    fig, axes = plt.subplots(1, num_topics, figsize=(20, 10))

    print("\n### 4. 워드 클라우드 생성 (시각 자료) ###")
    for i, topic in enumerate(lda_model.show_topics(num_topics=num_topics, num_words=20, formatted=False)):
        # 토픽 키워드 빈도 사전 생성
        topic_words = {word: weight for word, weight in topic[1]}

        wc.generate_from_frequencies(topic_words)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].axis("off")
        axes[i].set_title(f"Topic {i + 1}", fontsize=14)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 이 코드는 토픽 모델링을 수행하는 분석용 코드입니다.
    perform_lda_analysis()
