# 연습실 예약 마감 위험도 예측 PoC

## Overview
연습실 예약 플랫폼의 "시간대별 예약 마감 위험도 예측" 기능을 시각적으로 보여주는 PoC(Proof of Concept) 웹 앱입니다.

## Tech Stack
- Python 3.11
- Streamlit (UI)
- pandas, numpy (데이터 처리)
- plotly (시각화)

## Features
1. **더미 데이터 생성**: 버튼 클릭 시 가상의 예약 데이터 생성 (실제 CSV 데이터 사용 가능)
2. **날짜 선택**: 달력 위젯으로 예약 희망 날짜 선택
3. **위험도 예측**: 시간대별 예약 마감 확률을 3단계로 분류
   - 여유 (40% 미만) - 초록색
   - 보통 (40~70%) - 주황색
   - 위험 (70% 이상) - 빨간색
4. **시각화**: Plotly 막대그래프로 시간대별 위험도 표시
5. **예약하기**: PoC 결제 화면 안내

## Project Structure
```
├── app.py                    # 메인 Streamlit 앱
├── attached_assets/          # 첨부 파일 (CSV 데이터, 이미지)
├── .streamlit/config.toml    # Streamlit 설정
└── pyproject.toml            # Python 패키지 의존성
```

## Running the App
```bash
streamlit run app.py --server.port 5000
```

## Data Format
CSV 데이터 컬럼:
- 연도, 월, 일, 요일 (0=일요일~6=토요일)
- 시간 (9~22시)
- 휴일 여부, 시험기간 여부, 공연시즌 여부
- 예약 여부 (0 또는 1)
- 리드타임_시간, 취소 여부

## Notes
- PoC 데모이므로 실제 결제 연동 없음
- 단순 통계 기반 확률 계산 (머신러닝 모델 미사용)
- SpaceCloud (https://www.spacecloud.kr) 참고 디자인
