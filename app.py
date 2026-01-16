import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import os

st.set_page_config(
    page_title="연습실 예약 마감 위험도 예측 (2026)",
    page_icon="🎵",
    layout="wide"
)

WEEKDAY_NAMES = {
    0: '월요일',
    1: '화요일',
    2: '수요일',
    3: '목요일',
    4: '금요일',
    5: '토요일',
    6: '일요일'
}

TIME_SLOTS = list(range(9, 23))

HOLIDAYS_2026 = [
    '2026-01-01', '2026-03-01', '2026-03-02', '2026-05-05',
    '2026-05-24', '2026-06-06', '2026-08-15', '2026-10-03', 
    '2026-10-09', '2026-12-25'
]

EXAM_PERIODS_2026 = [
    ('2026-04-13', '2026-04-24'),
    ('2026-06-08', '2026-06-19'),
    ('2026-10-12', '2026-10-23'),
    ('2026-12-07', '2026-12-18')
]

PERFORMANCE_SEASONS_2026 = [
    ('2026-05-11', '2026-06-05'),
    ('2026-11-02', '2026-11-27')
]

class PracticeRoomPredictor:
    def __init__(self, df):
        self.df = df
        self.features = ['월', '일', '요일', '시간', '휴일 여부', '시험기간 여부', '공연시즌 여부']
        self.model = self._train_model()
    
    def _train_model(self):
        if '시험기간 여부' not in self.df.columns:
            self.df['시험기간 여부'] = 0
        if '공연시즌 여부' not in self.df.columns:
            self.df['공연시즌 여부'] = 0
        
        X = self.df[self.features]
        y = self.df['예약 여부']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    
    def _is_in_period(self, target_date, periods):
        for start_str, end_str in periods:
            start = datetime.strptime(start_str, '%Y-%m-%d').date()
            end = datetime.strptime(end_str, '%Y-%m-%d').date()
            if start <= target_date <= end:
                return 1
        return 0
    
    def predict(self, date_obj, hour):
        if isinstance(date_obj, str):
            target_dt = datetime.strptime(date_obj, '%Y-%m-%d')
            target_date = target_dt.date()
        else:
            target_date = date_obj
            target_dt = datetime.combine(date_obj, datetime.min.time())
        
        month = target_date.month
        day = target_date.day
        weekday = target_date.weekday()
        
        date_str = target_date.strftime('%Y-%m-%d')
        is_holiday = 1 if (weekday >= 5 or date_str in HOLIDAYS_2026) else 0
        is_exam = self._is_in_period(target_date, EXAM_PERIODS_2026)
        is_perf = self._is_in_period(target_date, PERFORMANCE_SEASONS_2026)
        
        input_data = pd.DataFrame([[
            month, day, weekday, hour, is_holiday, is_exam, is_perf
        ]], columns=self.features)
        
        prob = self.model.predict_proba(input_data)[0][1]
        
        return prob * 100

def load_real_data():
    csv_path = "attached_assets/practice_room_ML_data_2025_1768532371118.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        return df
    return None

def generate_training_data():
    np.random.seed(42)
    
    data = []
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    current_date = start_date
    
    exam_periods_2025 = [
        ('2025-04-14', '2025-04-25'),
        ('2025-06-09', '2025-06-20'),
        ('2025-10-13', '2025-10-24'),
        ('2025-12-08', '2025-12-19')
    ]
    
    perf_seasons_2025 = [
        ('2025-05-12', '2025-06-06'),
        ('2025-11-03', '2025-11-28')
    ]
    
    holidays_2025 = [
        '2025-01-01', '2025-01-28', '2025-01-29', '2025-01-30',
        '2025-03-01', '2025-05-05', '2025-05-06', '2025-06-06',
        '2025-08-15', '2025-10-03', '2025-10-06', '2025-10-07',
        '2025-10-08', '2025-10-09', '2025-12-25'
    ]
    
    def is_in_period(date_obj, periods):
        for start_str, end_str in periods:
            start = datetime.strptime(start_str, '%Y-%m-%d').date()
            end = datetime.strptime(end_str, '%Y-%m-%d').date()
            if start <= date_obj.date() <= end:
                return 1
        return 0
    
    while current_date <= end_date:
        weekday = current_date.weekday()
        date_str = current_date.strftime('%Y-%m-%d')
        
        is_weekend = weekday >= 5
        is_holiday = 1 if (is_weekend or date_str in holidays_2025) else 0
        is_exam = is_in_period(current_date, exam_periods_2025)
        is_perf = is_in_period(current_date, perf_seasons_2025)
        
        for hour in TIME_SLOTS:
            base_prob = 0.25
            
            if is_holiday:
                base_prob += 0.20
            
            if is_exam:
                base_prob += 0.25
            
            if is_perf:
                base_prob += 0.15
            
            if 18 <= hour <= 21:
                base_prob += 0.30
            elif 14 <= hour <= 17:
                base_prob += 0.15
            elif 9 <= hour <= 11:
                base_prob += 0.05
            
            is_booked = 1 if np.random.random() < base_prob else 0
            
            lead_time = np.random.uniform(2, 300) if is_booked else 0.0
            is_cancelled = 1 if is_booked and np.random.random() < 0.05 else 0
            
            data.append({
                '연도': current_date.year,
                '월': current_date.month,
                '일': current_date.day,
                '요일': weekday,
                '시간': hour,
                '휴일 여부': is_holiday,
                '시험기간 여부': is_exam,
                '공연시즌 여부': is_perf,
                '예약 여부': is_booked,
                '리드타임_시간': round(lead_time, 1),
                '취소 여부': is_cancelled
            })
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(data)

def get_risk_level(probability):
    if probability >= 75:
        return "위험", "#FF4B4B", "🔴"
    elif probability >= 50:
        return "임박", "#FFA500", "🟠"
    elif probability >= 25:
        return "주의", "#FFD700", "🟡"
    else:
        return "여유", "#00CC66", "🟢"

def get_period_info(date_obj):
    if isinstance(date_obj, str):
        date_obj = datetime.strptime(date_obj, '%Y-%m-%d').date()
    
    date_str = date_obj.strftime('%Y-%m-%d')
    weekday = date_obj.weekday()
    
    is_holiday = weekday >= 5 or date_str in HOLIDAYS_2026
    
    is_exam = False
    for start_str, end_str in EXAM_PERIODS_2026:
        start = datetime.strptime(start_str, '%Y-%m-%d').date()
        end = datetime.strptime(end_str, '%Y-%m-%d').date()
        if start <= date_obj <= end:
            is_exam = True
            break
    
    is_perf = False
    for start_str, end_str in PERFORMANCE_SEASONS_2026:
        start = datetime.strptime(start_str, '%Y-%m-%d').date()
        end = datetime.strptime(end_str, '%Y-%m-%d').date()
        if start <= date_obj <= end:
            is_perf = True
            break
    
    return {
        'is_holiday': is_holiday,
        'is_exam': is_exam,
        'is_perf': is_perf
    }

def create_time_slot_chart(time_data, selected_slot=None):
    hours = [f"{h}:00~{h+1}:00" for h in TIME_SLOTS]
    probabilities = [time_data[h]['probability'] for h in TIME_SLOTS]
    colors = [time_data[h]['color'] for h in TIME_SLOTS]
    
    if selected_slot is not None:
        colors = [
            c if h != selected_slot else '#7B68EE' 
            for h, c in zip(TIME_SLOTS, colors)
        ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=hours,
            y=probabilities,
            marker_color=colors,
            text=[f"{p:.0f}%" for p in probabilities],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>마감 확률: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="시간대별 예약 마감 위험도 (ML 예측)",
            font=dict(size=18)
        ),
        xaxis_title="시간대",
        yaxis_title="마감 확률 (%)",
        yaxis=dict(range=[0, 110]),
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.add_shape(
        type="line",
        x0=-0.5, x1=len(TIME_SLOTS)-0.5,
        y0=75, y1=75,
        line=dict(color="#FF4B4B", dash="dash", width=1)
    )
    fig.add_shape(
        type="line",
        x0=-0.5, x1=len(TIME_SLOTS)-0.5,
        y0=50, y1=50,
        line=dict(color="#FFA500", dash="dash", width=1)
    )
    fig.add_shape(
        type="line",
        x0=-0.5, x1=len(TIME_SLOTS)-0.5,
        y0=25, y1=25,
        line=dict(color="#FFD700", dash="dash", width=1)
    )
    
    return fig

st.title("🎵 연습실 예약 마감 위험도 예측 (2026)")
st.markdown("""
이 서비스는 **머신러닝(RandomForest) 기반 예약 마감 확률**을 제공하여 사용자가 더 합리적으로 예약 결정을 내릴 수 있도록 돕습니다.

| 단계 | 위험도 | 마감 확률 | 의미 |
|:---:|:---:|:---:|:---|
| 🟢 | **여유** | 0~25% | 지금 예약하지 않아도 충분히 여유 있음 |
| 🟡 | **주의** | 25~50% | 조금씩 찰 가능성 있음 |
| 🟠 | **임박** | 50~75% | 예약 지연 시 확보 어려움 |
| 🔴 | **위험** | 75~100% | 빠른 예약 필요 |
""")

st.divider()

st.subheader("1. ML 모델 준비")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("📊 학습 데이터 생성", type="primary", use_container_width=True):
        with st.spinner("학습 데이터 생성 및 모델 학습 중..."):
            training_data = generate_training_data()
            st.session_state['training_data'] = training_data
            st.session_state['predictor'] = PracticeRoomPredictor(training_data)
            st.session_state['data_source'] = "생성된 학습 데이터"
        st.success("ML 모델 학습 완료!")
        st.rerun()

with col2:
    real_data_available = os.path.exists("attached_assets/practice_room_ML_data_2025_1768532371118.csv")
    if real_data_available:
        if st.button("📁 실제 데이터로 학습", use_container_width=True):
            with st.spinner("실제 데이터 로드 및 모델 학습 중..."):
                real_data = load_real_data()
                st.session_state['training_data'] = real_data
                st.session_state['predictor'] = PracticeRoomPredictor(real_data)
                st.session_state['data_source'] = "실제 CSV 데이터"
            st.success("ML 모델 학습 완료!")
            st.rerun()

with col3:
    if 'predictor' in st.session_state:
        st.info(f"✅ {st.session_state['data_source']}로 학습 완료 ({len(st.session_state['training_data'])}개 레코드)")

if 'training_data' in st.session_state:
    with st.expander("📋 학습 데이터 미리보기 (처음 20개 행)"):
        display_df = st.session_state['training_data'].head(20).copy()
        display_df['요일명'] = display_df['요일'].map(WEEKDAY_NAMES)
        display_df['시간대'] = display_df['시간'].apply(lambda x: f"{x}:00~{x+1}:00")
        display_df['예약 상태'] = display_df['예약 여부'].map({0: '미예약', 1: '예약됨'})
        
        cols_to_show = ['연도', '월', '일', '요일명', '시간대', '휴일 여부', '시험기간 여부', '공연시즌 여부', '예약 상태']
        cols_available = [c for c in cols_to_show if c in display_df.columns]
        
        st.dataframe(
            display_df[cols_available],
            use_container_width=True,
            hide_index=True
        )

st.divider()

st.subheader("2. 2026년 날짜 선택")

if 'predictor' not in st.session_state:
    st.warning("⚠️ 먼저 '학습 데이터 생성' 버튼을 클릭해주세요.")
else:
    col_date1, col_date2 = st.columns([1, 2])
    
    today = datetime.now().date()
    min_date = max(today, datetime(2026, 1, 1).date())
    
    with col_date1:
        default_date = max(min_date, datetime(2026, 3, 15).date())
        selected_date = st.date_input(
            "예약 희망 날짜를 선택하세요",
            value=default_date,
            min_value=min_date,
            max_value=datetime(2026, 12, 31),
            format="YYYY-MM-DD"
        )
    
    with col_date2:
        weekday_num = selected_date.weekday()
        weekday_name = WEEKDAY_NAMES[weekday_num]
        period_info = get_period_info(selected_date)
        
        period_badges = []
        if period_info['is_holiday']:
            period_badges.append("🎉 휴일")
        if period_info['is_exam']:
            period_badges.append("📚 시험기간")
        if period_info['is_perf']:
            period_badges.append("🎭 공연시즌")
        
        period_text = " | ".join(period_badges) if period_badges else "📆 평일"
        
        st.markdown(f"""
        **선택된 날짜 정보:**
        - 📅 날짜: {selected_date.strftime('%Y년 %m월 %d일')} ({weekday_name})
        - {period_text}
        """)
        
        if period_info['is_exam']:
            st.warning("📚 시험기간에는 예약 수요가 높습니다!")
        if period_info['is_perf']:
            st.info("🎭 공연시즌에는 연습실 수요가 증가합니다.")
    
    st.divider()
    
    st.subheader("3. 시간대별 예약 마감 위험도 (ML 예측)")
    
    predictor = st.session_state['predictor']
    time_data = {}
    
    for hour in TIME_SLOTS:
        prob = predictor.predict(selected_date, hour)
        risk_level, color, emoji = get_risk_level(prob)
        time_data[hour] = {
            'probability': prob,
            'risk_level': risk_level,
            'color': color,
            'emoji': emoji
        }
    
    selected_time = st.session_state.get('selected_time', None)
    
    chart = create_time_slot_chart(time_data, selected_time)
    st.plotly_chart(chart, use_container_width=True)
    
    st.markdown("##### 시간대 선택")
    st.caption("⏰ 이미 지난 시간대는 선택할 수 없습니다.")
    
    now = datetime.now()
    current_hour = now.hour
    is_today = selected_date == now.date()
    
    cols = st.columns(7)
    for idx, hour in enumerate(TIME_SLOTS):
        col_idx = idx % 7
        with cols[col_idx]:
            risk_info = time_data[hour]
            is_past_time = is_today and hour <= current_hour
            
            if is_past_time:
                st.button(
                    f"{hour}:00\n⛔",
                    key=f"time_{hour}",
                    use_container_width=True,
                    disabled=True
                )
            else:
                if st.button(
                    f"{hour}:00\n{risk_info['emoji']}",
                    key=f"time_{hour}",
                    use_container_width=True
                ):
                    st.session_state['selected_time'] = hour
                    st.rerun()
    
    st.markdown("---")
    col_legend1, col_legend2, col_legend3, col_legend4 = st.columns(4)
    with col_legend1:
        st.markdown("🟢 **여유** (0~25%)")
    with col_legend2:
        st.markdown("🟡 **주의** (25~50%)")
    with col_legend3:
        st.markdown("🟠 **임박** (50~75%)")
    with col_legend4:
        st.markdown("🔴 **위험** (75~100%)")
    
    if 'selected_time' in st.session_state and st.session_state['selected_time'] is not None:
        st.divider()
        
        st.subheader("4. 선택한 시간대 상세 정보")
        
        sel_hour = st.session_state['selected_time']
        sel_info = time_data[sel_hour]
        
        info_col1, info_col2 = st.columns([2, 1])
        
        with info_col1:
            risk_color = sel_info['color']
            period_text_short = []
            if period_info['is_holiday']:
                period_text_short.append("휴일")
            if period_info['is_exam']:
                period_text_short.append("시험기간")
            if period_info['is_perf']:
                period_text_short.append("공연시즌")
            period_str = ", ".join(period_text_short) if period_text_short else "평일"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {risk_color}22, {risk_color}44);
                border-left: 5px solid {risk_color};
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            ">
                <h3 style="margin: 0; color: #333;">📍 예약 정보 (ML 예측)</h3>
                <p style="font-size: 16px; margin-top: 10px;">
                    <strong>날짜:</strong> {selected_date.strftime('%Y년 %m월 %d일')} ({weekday_name})<br>
                    <strong>시간:</strong> {sel_hour}:00 ~ {sel_hour+1}:00 (1시간)<br>
                    <strong>기간 특성:</strong> {period_str}<br>
                    <strong>마감 확률:</strong> <span style="font-size: 24px; font-weight: bold; color: {risk_color};">{sel_info['probability']:.1f}%</span><br>
                    <strong>위험도:</strong> {sel_info['emoji']} {sel_info['risk_level']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if sel_info['risk_level'] == "위험":
                st.error("🔴 이 시간대는 마감 가능성이 매우 높습니다. 빠른 예약이 필요합니다!")
            elif sel_info['risk_level'] == "임박":
                st.warning("🟠 예약 지연 시 확보가 어려울 수 있습니다. 빠른 결정을 권장합니다.")
            elif sel_info['risk_level'] == "주의":
                st.info("🟡 조금씩 찰 가능성이 있습니다. 여유를 두고 예약하세요.")
            else:
                st.success("🟢 이 시간대는 충분히 여유가 있습니다. 천천히 예약해도 괜찮습니다.")
        
        with info_col2:
            st.markdown("""
            **예상 이용료**
            """)
            hourly_rate = 110000
            st.markdown(f"""
            <div style="
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            ">
                <p style="margin: 0; color: #666;">1시간 기준</p>
                <h2 style="margin: 10px 0; color: #7B68EE;">₩{hourly_rate:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader("5. 예약하기")
        
        reserve_col1, reserve_col2, reserve_col3 = st.columns([1, 2, 1])
        
        with reserve_col2:
            if st.button(
                "🎯 예약하기",
                type="primary",
                use_container_width=True
            ):
                st.session_state['show_payment'] = True
        
        if st.session_state.get('show_payment', False):
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #7B68EE, #9370DB);
                color: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin: 20px 0;
            ">
                <h2 style="margin: 0;">💳 결제 화면으로 이동 (PoC)</h2>
                <p style="margin-top: 15px; font-size: 16px;">
                    이것은 PoC 데모입니다.<br>
                    실제 서비스에서는 이 화면에서 결제가 진행됩니다.
                </p>
                <hr style="border-color: rgba(255,255,255,0.3); margin: 20px 0;">
                <p style="margin: 0;">
                    <strong>예약 정보:</strong> {date} ({weekday}) {time}:00~{time_end}:00<br>
                    <strong>결제 금액:</strong> ₩{price:,}
                </p>
            </div>
            """.format(
                date=selected_date.strftime('%Y.%m.%d'),
                weekday=weekday_name,
                time=sel_hour,
                time_end=sel_hour+1,
                price=hourly_rate
            ), unsafe_allow_html=True)
            
            if st.button("닫기", use_container_width=True):
                st.session_state['show_payment'] = False
                st.rerun()

st.divider()
st.caption("🎵 연습실 예약 마감 위험도 예측 PoC (2026) | ML 기반 예측 | SpaceCloud 참고")
