# 연습실 예약 마감 위험도 예측 PoC (2026)

## Overview
연습실 예약 플랫폼의 "시간대별 예약 마감 위험도 예측" 기능을 시각적으로 보여주는 PoC(Proof of Concept) 웹 앱입니다.
**머신러닝(RandomForest) 기반** 예측 모델을 사용합니다.

## Tech Stack
- Python 3.11
- Streamlit (UI)
- pandas, numpy (데이터 처리)
- scikit-learn (RandomForest ML 모델)
- plotly (시각화)

## Features
1. **ML 모델 학습**: RandomForest 모델로 예약 패턴 학습
2. **2026년 일정 반영**: 휴일, 시험기간, 공연시즌 자동 판단
3. **날짜 선택**: 달력 위젯으로 2026년 예약 희망 날짜 선택
4. **위험도 예측**: ML 기반 시간대별 예약 마감 확률을 4단계로 분류
   - 여유 (0~25%) - 🟢 초록색: 지금 예약하지 않아도 충분히 여유 있음
   - 주의 (25~50%) - 🟡 노랑색: 조금씩 찰 가능성 있음
   - 임박 (50~75%) - 🟠 주황색: 예약 지연 시 확보 어려움
   - 위험 (75~100%) - 🔴 빨간색: 빠른 예약 필요
5. **과거 날짜/시간 차단**: 이미 지난 날짜와 시간대는 예약 불가
6. **기간 특성 표시**: 휴일/시험기간/공연시즌 배지 표시
7. **시각화**: Plotly 막대그래프로 시간대별 위험도 표시
8. **예약하기**: PoC 결제 화면 안내

## 2026년 주요 일정
- **휴일**: 1/1, 3/1, 3/2, 5/5, 5/24, 6/6, 8/15, 10/3, 10/9, 12/25
- **시험기간**: 4/13~4/24, 6/8~6/19, 10/12~10/23, 12/7~12/18
- **공연시즌**: 5/11~6/5, 11/2~11/27

## Project Structure
```
├── app.py                    # 메인 Streamlit 앱 (Python 백엔드)
├── index.html                # 정적 HTML 앱 (Netlify 배포용)
├── netlify.toml              # Netlify 배포 설정
├── attached_assets/          # 첨부 파일 (CSV 데이터, 이미지)
├── .streamlit/config.toml    # Streamlit 설정
└── pyproject.toml            # Python 패키지 의존성
```

## Running the App

### Streamlit 버전 (Replit)
```bash
streamlit run app.py --server.port 5000
```

### 정적 HTML 버전 (Netlify)
index.html 파일을 Netlify에 배포하면 됩니다.
- Netlify에서 GitHub 연동 또는 폴더 드래그 드롭으로 배포
- netlify.toml 설정 파일 포함됨

## Data Format
CSV 데이터 컬럼:
- 연도, 월, 일, 요일 (0=일요일~6=토요일)
- 시간 (9~22시)
- 휴일 여부, 시험기간 여부, 공연시즌 여부
- 예약 여부 (0 또는 1)
- 리드타임_시간, 취소 여부

## Notes
- PoC 데모이므로 실제 결제 연동 없음
- RandomForest ML 모델 기반 예측 (scikit-learn 사용)
- SpaceCloud (https://www.spacecloud.kr) 참고 디자인
