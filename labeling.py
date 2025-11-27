import pandas as pd
import numpy as np
import re
import os

# --- 설정 변수 ---
# 크롤링된 전체 Raw 데이터 파일 (24,905건)
RAW_FILE_PATH = 'soop_community_data_raw.csv'
# 수동 라벨링을 시작해야 할 파일 (2,000건 샘플)
SAMPLE_FILE_PATH = 'soop_data_labeling_sample.csv'
# 모델 학습에 사용될 최종 라벨링 파일 (2,000건)
FINAL_LABELED_FILE = 'soop_data_labeled_final_for_model.csv'

# ⚠️ 5개 클래스 라벨링 기준을 키워드 기반으로 시뮬레이션
# 주의: 이 시뮬레이션은 정확하지 않으며, 보고서의 '90% 자동 라벨링' 부분의 계획으로 활용된다.
URGENCY_KEYWORDS = {
    # 3: 운영자 공지 (가장 높은 우선순위로 먼저 확인)
    3: ['안녕하세요', '아프리카tv입니다', '소통센터장', '안내드립니다', '공지', '클린아티'],
    # 1: 기술/서비스 장애 (서버, 버퍼링, 결제 오류 등 즉각적인 기술 결함)
    # 키워드 보강: '안모아지네요'와 같이 지급/결제 관련 오류 키워드 추가
    1: ['오류', '버그', '접속', '렉', '짤려', '끊김', '멈춤', '무한로딩', '서버', '충전', '결제', '환불', '팅기는', '다운', '인증', '복구', '밀림', '안됨',
        '튕김', '먹통', '딜레이', '버벅', '로그인', '느려요', '느림', '안돼', '설치', '안모아지네요'],
    # 2: 운영/정책 문제 (블랙, 제재, 정책, 이미지 관련 비판)
    # 키워드 보강: 보기 싫은 방송/BJ 차단/숨기기 관련 정책적 요구 키워드 추가
    2: ['블랙', '차단', '정지', '제재', '정책', '영구정지', '관리', '운영', '베비', '규정', '이미지', '불공평', '갑질', '차별', '논란', '월권', '보기싫은',
        '안보이게', '숨기기', '차단하기','제도'],
    # 0: 기타/기능 건의 (UI/UX 개선 요청, 기능 추가 등 일반적인 건의 사항)
    # 💡 포인트/혜택/선물 관련 키워드를 추가하여 건의사항 분류 정확도 향상
    0: ['개선', '요청', '기능', '추가', '수정', '업데이트', '변경', 'UI', 'UX', '화면', '줄여주세요', '없앨수', '취소', '확인화면', '얼리기', '불편', '옵션',
        '개편', '자동', '설정', '컨텐츠', '화질', '선물', '혜택', '구독', '아이템', '포인트', '광고'],
    # 4: 단순 비난/잡담 (키워드가 없거나 비난성 텍스트)
}


def simulate_labeling():
    """크롤링된 Raw 파일을 로드하여 5개 클래스 라벨링을 시뮬레이션한다."""
    try:
        # Raw 데이터 로드 (전체 24,905건)
        df_raw = pd.read_csv(RAW_FILE_PATH)
        # 병합 키로 사용할 post_id와 content에 결측치 없는 행만 사용
        df_raw = df_raw.dropna(subset=['content', 'post_id']).reset_index(drop=True)

        print(f"### 1. 전체 {len(df_raw)}건에 대해 5-Tier 라벨링 시뮬레이션 시작 ###")

    except FileNotFoundError:
        print(f"오류: 원본 파일 '{RAW_FILE_PATH}'을 찾을 수 없다. 크롤링을 먼저 실행해야 한다.")
        return

    def assign_temp_label(text):
        text = str(text).lower()

        # 1. Tier 3 (운영자 공지): 가장 높은 우선순위로 분류
        for keyword in URGENCY_KEYWORDS[3]:
            if keyword in text and len(text) > 50:
                return 3

        # 2. Tier 1 (기술/서비스 장애)
        for keyword in URGENCY_KEYWORDS[1]:
            if keyword in text:
                return 1

        # 3. Tier 2 (운영/정책 문제)
        for keyword in URGENCY_KEYWORDS[2]:
            if keyword in text:
                return 2

        # 4. Tier 0 (기타/기능 건의)
        for keyword in URGENCY_KEYWORDS[0]:
            if keyword in text:
                return 0

        # 5. 키워드가 없으면 단순 비난/잡담(4)로 분류
        return 4

    df_raw['simulated_label'] = df_raw['content'].apply(assign_temp_label)

    # --------------------------------------------------------------------------------------------------
    # ⚠️ [핵심 단계]: 2,000건 샘플에만 이 시뮬레이션 라벨을 적용하여 최종 학습 파일 생성
    # --------------------------------------------------------------------------------------------------
    try:
        df_sample = pd.read_csv(SAMPLE_FILE_PATH)
        df_sample = df_sample.dropna(subset=['content', 'post_id']).reset_index(drop=True)

        # post_id와 content를 기준으로 df_raw의 시뮬레이션 라벨을 df_sample에 병합
        df_merged = pd.merge(df_sample[['post_id', 'content']],
                             df_raw[['post_id', 'content', 'simulated_label']],
                             on=['post_id', 'content'], how='left')

        # 모델 학습 파일 준비 (content, label 만 남김)
        df_final = df_merged.rename(columns={'simulated_label': 'label'})
        # 매칭 안된 행이 있을 경우, 기본값 4로 채우고 정수형으로 변환
        df_final['label'] = df_final['label'].fillna(4).astype(int)
        df_final = df_final[['content', 'label']]

        df_final.to_csv(FINAL_LABELED_FILE, index=False, encoding='utf-8-sig')

        print(f"\n-> 최종 {len(df_final)}건 모델 학습용 파일 준비 완료.")
        print("-> 임시 라벨 분포:")
        print(df_final['label'].value_counts())

        print("\n\n################################################################")
        print("⚠️ 경고: 이 파일은 임시 라벨링 결과이며, 보고서 제출 전 반드시 수동 검토해야 한다.")
        print("################################################################")

    except FileNotFoundError:
        print(f"오류: 샘플 파일 '{SAMPLE_FILE_PATH}'을 찾을 수 없다. 크롤링을 먼저 실행해야 한다.")
        return


if __name__ == '__main__':
    # ⚠️ 이 코드는 수동 라벨링을 대체하는 임시 파일 생성용이다.
    simulate_labeling()
