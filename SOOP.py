import time
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException

# --- [A] 설정 변수 (Configuration) ---
BASE_URL = "https://sotong.sooplive.co.kr/?board_type=user&work=list&check_nick=false&check_title=true&check_content=true&page="

# ⚠️ 2만 건 이상 확보를 위해 MAX_PAGES를 2000으로 설정 (30,000건 목표)
MAX_PAGES = 2500
SLEEP_TIME_LIST = 1.5  # 목록 페이지 로딩 대기
SLEEP_TIME_DETAIL = 1.0  # 상세 페이지 접속 대기

# ⚠️ 최종 수정된 본문 CSS 선택자 (HTML 분석 결과 반영)
CONTENT_SELECTOR = "div.v_article div.view"

data = []


def setup_driver():
    """Chrome WebDriver를 설정하고 반환한다."""
    options = webdriver.ChromeOptions()
    options.add_argument("window-size=1920x1080")
    try:
        driver = webdriver.Chrome(options=options)
        return driver
    except Exception as e:
        print(f"WebDriver 초기화 실패: {e}")
        return None


def crawl_list_page(driver, page_num):
    """특정 페이지의 게시글 목록 메타데이터를 추출한다."""
    url = BASE_URL + str(page_num)
    driver.get(url)
    time.sleep(SLEEP_TIME_LIST)

    page_data = []
    try:
        post_rows = driver.find_elements(By.CSS_SELECTOR, "#board_list > tr")
    except NoSuchElementException:
        return page_data

    for row in post_rows:
        if 'notice' in row.get_attribute('class'):
            continue

        try:
            cols = row.find_elements(By.TAG_NAME, 'td')
            if len(cols) < 6: continue

            title_element = cols[1].find_element(By.TAG_NAME, 'a')
            detail_link = title_element.get_attribute('href')

            post_data = {
                'post_id': cols[0].text.strip(),
                'title': title_element.text.strip(),
                'author': cols[2].text.strip(),
                'date': cols[3].text.strip(),
                'views': cols[4].text.strip(),
                'recommends': cols[5].text.strip(),
                'detail_url': detail_link
            }
            page_data.append(post_data)

        except Exception:
            continue

    return page_data


def crawl_detail_content(driver, item):
    """상세 페이지로 이동하여 본문 텍스트를 추출한다."""
    try:
        # 상세 URL로 접속
        driver.get(item['detail_url'])
        time.sleep(SLEEP_TIME_DETAIL)

        # ⚠️ 최종 수정된 본문 텍스트 추출 시도
        content_element = driver.find_element(By.CSS_SELECTOR, CONTENT_SELECTOR)
        item['content'] = content_element.text.strip()

    except NoSuchElementException:
        item['content'] = "본문 영역 찾기 실패"
    except Exception as e:
        item['content'] = f"상세 페이지 접속/추출 오류: {type(e).__name__}"

    # 목록 데이터에 추가
    data.append(item)


def main_crawler():
    driver = setup_driver()
    if driver is None: return

    try:
        # MAX_PAGES까지 크롤링하여 2만 건 이상 수집 시도
        for page_num in range(1, MAX_PAGES + 1):
            print(f"--- 📚 {page_num} 페이지 크롤링 시작 ---")

            # 1. 목록 데이터 추출
            page_data = crawl_list_page(driver, page_num)

            if not page_data and page_num > 1:
                print("더 이상 게시글이 없다. 수집 종료한다.")
                break

            # 2. 상세 본문 크롤링
            for item in page_data:
                crawl_detail_content(driver, item)

            print(f"✅ 페이지 {page_num}에서 {len(page_data)}건 수집 완료. (총 {len(data)}건)")

    except Exception as e:
        print(f"\n!!! 전체 크롤링 중단 오류: {e} !!!")

    finally:
        driver.quit()
        df = pd.DataFrame(data)

        # ⚠️ [보강] 유효하지 않은 'content'를 가진 행을 제거하여 유효 데이터만 저장
        initial_count = len(df)
        df = df[df['content'] != "본문 영역 찾기 실패"]
        df = df[df['content'].str.strip() != ""]

        removed_count = initial_count - len(df)

        file_name = 'soop_community_data_raw.csv'
        df.to_csv(file_name, index=False, encoding='utf-8-sig')

        print(f"\n✨ 최종 {initial_count}건 수집 시도. {removed_count}건의 오류 데이터 제거됨.")
        print(f"✨ 최종 유효 데이터 {len(df)}건 '{file_name}'에 저장되었다.")

    return df


# -------------------------------------------------------------------
# 3. 라벨링 샘플 추출 함수 (수집 후 자동 실행)
# -------------------------------------------------------------------

def select_labeling_samples(raw_df):
    """
    수집된 데이터에서 라벨링을 위한 무작위 및 전략적 샘플을 추출한다.
    """
    if raw_df.empty:
        print("데이터프레임이 비어 있어 샘플링을 진행할 수 없다.")
        return

    SAMPLE_FILE = 'soop_data_labeling_sample.csv'
    TOTAL_SAMPLE_SIZE = 2000
    RANDOM_SAMPLE_SIZE = 1000
    STRATEGIC_SAMPLE_SIZE = 1000

    print(f"\n--- 📝 라벨링 샘플 추출 시작 (총 {len(raw_df)}건) ---")

    # 1. 데이터 클리닝 및 숫자형 변환
    raw_df['views'] = pd.to_numeric(raw_df['views'], errors='coerce').fillna(0).astype(int)
    raw_df['recommends'] = pd.to_numeric(raw_df['recommends'], errors='coerce').fillna(0).astype(int)

    raw_df.drop_duplicates(subset=['post_id'], keep='first', inplace=True)

    # 2. 그룹 A: 무작위 샘플 추출
    n_random = min(RANDOM_SAMPLE_SIZE, len(raw_df))
    random_sample = raw_df.sample(n=n_random, random_state=42)

    # 3. 그룹 B: 전략적 샘플 추출 (반응도 기준)
    df_temp = raw_df.drop(random_sample.index, errors='ignore').copy()

    df_temp['engagement'] = df_temp['views'] + df_temp['recommends']
    engagement_threshold = df_temp['engagement'].quantile(0.9) if len(df_temp) > 0 else 0

    strategic_candidates = df_temp[
        (df_temp['engagement'] >= engagement_threshold) |
        (df_temp['recommends'] > df_temp['views'] / 100)
        ]

    n_strategic = min(STRATEGIC_SAMPLE_SIZE, len(strategic_candidates))
    strategic_sample = strategic_candidates.sample(n=n_strategic, random_state=42)

    # 4. 최종 데이터셋 통합 및 저장
    final_sample = pd.concat([random_sample, strategic_sample]).drop_duplicates(subset=['post_id'])

    final_sample['label'] = None

    print(f"✅ 무작위 샘플: {len(random_sample)}건")
    print(f"✅ 전략적 샘플: {len(strategic_sample)}건")
    print(f"✅ 최종 라벨링 대상: {len(final_sample)}건 (최소 {TOTAL_SAMPLE_SIZE}건 목표)")

    final_sample = final_sample[['post_id', 'title', 'content', 'views', 'recommends', 'date', 'detail_url', 'label']]
    final_sample.to_csv(SAMPLE_FILE, index=False, encoding='utf-8-sig')
    print(f"\n🔥 라벨링 대상 파일 '{SAMPLE_FILE}' 저장 완료.")


# --- 메인 실행 ---
if __name__ == "__main__":
    crawled_df = main_crawler()
    select_labeling_samples(crawled_df)
