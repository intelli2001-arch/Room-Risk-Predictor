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
        self.features = ['월', '일', '요일', '시간', '휴일 여부', '시험기간 여부', '공연시즌 여부', '리드타임_시간']
        self.model = self._train_model()
    
    def _train_model(self):
        if '시험기간 여부' not in self.df.columns:
            self.df['시험기간 여부'] = 0
        if '공연시즌 여부' not in self.df.columns:
            self.df['공연시즌 여부'] = 0
        if '리드타임_시간' not in self.df.columns:
            self.df['리드타임_시간'] = 72
        
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
    
    def calculate_lead_time(self, date_obj, hour):
        """현재 시점부터 예약 시간까지의 리드타임(시간) 계산"""
        now = datetime.now()
        if isinstance(date_obj, str):
            target_date = datetime.strptime(date_obj, '%Y-%m-%d').date()
        else:
            target_date = date_obj
        
        target_datetime = datetime.combine(target_date, datetime.min.time().replace(hour=hour))
        lead_time_hours = (target_datetime - now).total_seconds() / 3600
        return max(0, lead_time_hours)
    
    def predict(self, date_obj, hour, lead_time_hours=None):
        if isinstance(date_obj, str):
            target_dt = datetime.strptime(date_obj, '%Y-%m-%d')
            target_date = target_dt.date()
        else:
            target_date = date_obj
            target_dt = datetime.combine(date_obj, datetime.min.time())
        
        if lead_time_hours is None:
            lead_time_hours = self.calculate_lead_time(target_date, hour)
        
        month = target_date.month
        day = target_date.day
        weekday = target_date.weekday()
        
        date_str = target_date.strftime('%Y-%m-%d')
        is_holiday = 1 if (weekday >= 5 or date_str in HOLIDAYS_2026) else 0
        is_exam = self._is_in_period(target_date, EXAM_PERIODS_2026)
        is_perf = self._is_in_period(target_date, PERFORMANCE_SEASONS_2026)
        
        input_data = pd.DataFrame([[
            month, day, weekday, hour, is_holiday, is_exam, is_perf, lead_time_hours
        ]], columns=self.features)
        
        prob = self.model.predict_proba(input_data)[0][1]
        
        return prob * 100, lead_time_hours

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
                base_prob -= 0.15  # 시험기간에는 연습 수요 감소
            
            if is_perf:
                base_prob += 0.30  # 공연시즌에는 연습 수요 크게 증가
            
            if 18 <= hour <= 21:
                base_prob += 0.30
            elif 14 <= hour <= 17:
                base_prob += 0.15
            elif 9 <= hour <= 11:
                base_prob += 0.05
            
            lead_time_ranges = [
                (0, 6),      # 당일 예약 (0~6시간 전)
                (6, 24),     # 하루 전 예약
                (24, 72),    # 1~3일 전 예약
                (72, 168),   # 3~7일 전 예약
                (168, 336),  # 1~2주 전 예약
                (336, 720),  # 2주~1달 전 예약
            ]
            
            for lead_min, lead_max in lead_time_ranges:
                lead_time = np.random.uniform(lead_min, lead_max)
                
                time_factor = 1.0
                if lead_time < 6:
                    time_factor = 0.85
                elif lead_time < 24:
                    time_factor = 0.70
                elif lead_time < 72:
                    time_factor = 0.55
                elif lead_time < 168:
                    time_factor = 0.40
                elif lead_time < 336:
                    time_factor = 0.25
                else:
                    time_factor = 0.15
                
                adjusted_prob = base_prob * time_factor
                is_booked = 1 if np.random.random() < adjusted_prob else 0
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

def render_model_training(key_prefix=""):
    st.subheader("ML 모델 준비")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("📊 학습 데이터 생성", type="primary", use_container_width=True, key=f"{key_prefix}train_btn"):
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
            if st.button("📁 실제 데이터로 학습", use_container_width=True, key=f"{key_prefix}real_btn"):
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

def analyze_utilization(predictor):
    st.subheader("📊 공간 활용률 분석")
    
    analysis_type = st.selectbox(
        "분석 유형 선택",
        ["월별 평균 수요", "요일별 평균 수요", "시간대별 평균 수요", "기간 특성별 수요"]
    )
    
    results = []
    
    if analysis_type == "월별 평균 수요":
        for month in range(1, 13):
            probs = []
            for day in [10, 15, 20]:
                for hour in TIME_SLOTS:
                    try:
                        test_date = datetime(2026, month, day).date()
                        prob, _ = predictor.predict(test_date, hour, lead_time_hours=72)
                        probs.append(prob)
                    except:
                        pass
            if probs:
                results.append({'기간': f"{month}월", '평균 수요(%)': np.mean(probs)})
        
        df = pd.DataFrame(results)
        fig = go.Figure(data=[
            go.Bar(x=df['기간'], y=df['평균 수요(%)'], 
                   marker_color=['#FF4B4B' if v > 60 else '#FFA500' if v > 40 else '#FFD700' if v > 25 else '#00CC66' 
                                 for v in df['평균 수요(%)']])
        ])
        fig.update_layout(title="월별 평균 예약 수요", yaxis_title="수요 (%)", xaxis_title="월")
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "요일별 평균 수요":
        weekday_probs = {i: [] for i in range(7)}
        for month in [3, 6, 9, 11]:
            for day in range(1, 29):
                try:
                    test_date = datetime(2026, month, day).date()
                    for hour in TIME_SLOTS:
                        prob, _ = predictor.predict(test_date, hour, lead_time_hours=72)
                        weekday_probs[test_date.weekday()].append(prob)
                except:
                    pass
        
        for wd in range(7):
            if weekday_probs[wd]:
                results.append({'요일': WEEKDAY_NAMES[wd], '평균 수요(%)': np.mean(weekday_probs[wd])})
        
        df = pd.DataFrame(results)
        colors = ['#FF4B4B' if v > 60 else '#FFA500' if v > 40 else '#FFD700' if v > 25 else '#00CC66' 
                  for v in df['평균 수요(%)']]
        fig = go.Figure(data=[go.Bar(x=df['요일'], y=df['평균 수요(%)'], marker_color=colors)])
        fig.update_layout(title="요일별 평균 예약 수요", yaxis_title="수요 (%)", xaxis_title="요일")
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "시간대별 평균 수요":
        hour_probs = {h: [] for h in TIME_SLOTS}
        for month in [3, 6, 9, 11]:
            for day in [10, 15, 20]:
                try:
                    test_date = datetime(2026, month, day).date()
                    for hour in TIME_SLOTS:
                        prob, _ = predictor.predict(test_date, hour, lead_time_hours=72)
                        hour_probs[hour].append(prob)
                except:
                    pass
        
        for hour in TIME_SLOTS:
            if hour_probs[hour]:
                results.append({'시간': f"{hour}:00", '평균 수요(%)': np.mean(hour_probs[hour])})
        
        df = pd.DataFrame(results)
        colors = ['#FF4B4B' if v > 60 else '#FFA500' if v > 40 else '#FFD700' if v > 25 else '#00CC66' 
                  for v in df['평균 수요(%)']]
        fig = go.Figure(data=[go.Bar(x=df['시간'], y=df['평균 수요(%)'], marker_color=colors)])
        fig.update_layout(title="시간대별 평균 예약 수요", yaxis_title="수요 (%)", xaxis_title="시간")
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "기간 특성별 수요":
        period_probs = {'평일': [], '휴일/주말': [], '시험기간': [], '공연시즌': []}
        for month in range(1, 13):
            for day in range(1, 29):
                try:
                    test_date = datetime(2026, month, day).date()
                    period = get_period_info(test_date)
                    for hour in TIME_SLOTS:
                        prob, _ = predictor.predict(test_date, hour, lead_time_hours=72)
                        if period['is_exam']:
                            period_probs['시험기간'].append(prob)
                        elif period['is_perf']:
                            period_probs['공연시즌'].append(prob)
                        elif period['is_holiday']:
                            period_probs['휴일/주말'].append(prob)
                        else:
                            period_probs['평일'].append(prob)
                except:
                    pass
        
        for period_name, probs in period_probs.items():
            if probs:
                results.append({'기간': period_name, '평균 수요(%)': np.mean(probs)})
        
        df = pd.DataFrame(results)
        colors = ['#00CC66', '#FFA500', '#3498db', '#FF4B4B']
        fig = go.Figure(data=[go.Bar(x=df['기간'], y=df['평균 수요(%)'], marker_color=colors)])
        fig.update_layout(title="기간 특성별 평균 예약 수요", yaxis_title="수요 (%)", xaxis_title="기간")
        st.plotly_chart(fig, use_container_width=True)
    
    st.caption("💡 수요가 낮은 시간대/기간을 타임세일이나 오픈연습실로 전환하면 수익을 높일 수 있습니다.")

def render_promotion_management(predictor):
    st.subheader("🏷️ 저수요 시간대 프로모션 관리")
    
    col_date, col_threshold = st.columns(2)
    
    with col_date:
        today = datetime.now().date()
        min_date = max(today, datetime(2026, 1, 1).date())
        default_date = max(min_date, today)
        promo_date = st.date_input(
            "분석할 날짜 선택",
            value=default_date,
            min_value=min_date,
            max_value=datetime(2026, 12, 31),
            format="YYYY-MM-DD",
            key="promo_date"
        )
    
    with col_threshold:
        threshold = st.slider("저수요 기준 (% 이하)", 10, 50, 30)
    
    if 'promo_slots' not in st.session_state:
        st.session_state['promo_slots'] = {}
    
    promo_key = promo_date.strftime('%Y-%m-%d')
    if promo_key not in st.session_state['promo_slots']:
        st.session_state['promo_slots'][promo_key] = {}
    
    if 'booked_slots_cache' not in st.session_state:
        st.session_state['booked_slots_cache'] = {}
    
    if promo_key not in st.session_state['booked_slots_cache']:
        period_info = get_period_info(promo_date)
        date_seed = promo_date.toordinal()
        np.random.seed(date_seed)
        booked_slots = set()
        for hour in TIME_SLOTS:
            prob, _ = predictor.predict(promo_date, hour, lead_time_hours=72)
            close_chance = prob / 100 * 0.4
            if period_info['is_perf']:
                close_chance *= 1.5
            if period_info['is_holiday']:
                close_chance *= 1.3
            if period_info['is_exam']:
                close_chance *= 0.3
            if 18 <= hour <= 20:
                close_chance *= 1.4
            if np.random.random() < close_chance:
                booked_slots.add(hour)
        st.session_state['booked_slots_cache'][promo_key] = booked_slots
    
    booked_slots = st.session_state['booked_slots_cache'].get(promo_key, set())
    
    low_demand_slots = []
    booked_excluded_count = 0
    
    for hour in TIME_SLOTS:
        if hour in booked_slots:
            booked_excluded_count += 1
            continue
        prob, _ = predictor.predict(promo_date, hour, lead_time_hours=72)
        if prob < threshold:
            low_demand_slots.append({'hour': hour, 'prob': prob})
    
    if booked_excluded_count > 0:
        st.warning(f"⚠️ {booked_excluded_count}개 시간대가 이미 예약 완료되어 프로모션 전환이 불가합니다.")
    
    if low_demand_slots:
        st.info(f"📉 {len(low_demand_slots)}개 저수요 시간대 발견 (수요 {threshold}% 미만, 예약 가능한 시간대만)")
        
        cols = st.columns(min(len(low_demand_slots), 7))
        
        for idx, slot in enumerate(low_demand_slots):
            hour = slot['hour']
            prob = slot['prob']
            col_idx = idx % min(len(low_demand_slots), 7)
            
            with cols[col_idx]:
                current_promo = st.session_state['promo_slots'][promo_key].get(hour, '일반')
                
                st.markdown(f"""
                <div style="
                    background: {'#e3f2fd' if current_promo == '오픈연습실' else '#fff3e0' if current_promo == '타임세일' else '#f5f5f5'};
                    padding: 10px;
                    border-radius: 8px;
                    text-align: center;
                    margin-bottom: 10px;
                    border: 2px solid {'#2196F3' if current_promo == '오픈연습실' else '#FF9800' if current_promo == '타임세일' else '#ddd'};
                ">
                    <strong>{hour}:00</strong><br>
                    <span style="color: #666;">수요: {prob:.0f}%</span><br>
                    <span style="font-size: 0.8em; color: {'#2196F3' if current_promo == '오픈연습실' else '#FF9800' if current_promo == '타임세일' else '#999'};">
                        {current_promo}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                new_promo = st.selectbox(
                    f"{hour}시",
                    ['일반', '오픈연습실', '타임세일'],
                    index=['일반', '오픈연습실', '타임세일'].index(current_promo),
                    key=f"promo_{promo_key}_{hour}",
                    label_visibility="collapsed"
                )
                
                if new_promo != current_promo:
                    st.session_state['promo_slots'][promo_key][hour] = new_promo
                    st.rerun()
        
        st.divider()
        
        st.markdown("##### 프로모션 요약")
        open_practice = [h for h, p in st.session_state['promo_slots'].get(promo_key, {}).items() if p == '오픈연습실']
        time_sale = [h for h, p in st.session_state['promo_slots'].get(promo_key, {}).items() if p == '타임세일']
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="background: #e3f2fd; padding: 15px; border-radius: 10px;">
                <h4 style="margin: 0; color: #2196F3;">🎸 오픈연습실</h4>
                <p style="margin: 5px 0;">누구나 자유롭게 이용 가능한 개방 시간</p>
                <strong>{len(open_practice)}개 시간대</strong>
                {('<br>' + ', '.join([f"{h}:00" for h in sorted(open_practice)])) if open_practice else ''}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: #fff3e0; padding: 15px; border-radius: 10px;">
                <h4 style="margin: 0; color: #FF9800;">💰 타임세일</h4>
                <p style="margin: 5px 0;">할인된 가격으로 예약 유도</p>
                <strong>{len(time_sale)}개 시간대</strong>
                {('<br>' + ', '.join([f"{h}:00" for h in sorted(time_sale)])) if time_sale else ''}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success(f"✅ 모든 시간대의 수요가 {threshold}% 이상입니다. 프로모션이 필요하지 않습니다.")

tab_customer, tab_business = st.tabs(["👤 고객", "🏢 사업자"])

with tab_customer:
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
    
    render_model_training("customer_")
    
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
            today = datetime.now().date()
            default_date = max(min_date, today)
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
                st.info("📚 시험기간에는 예약 수요가 낮습니다. 여유롭게 예약 가능합니다.")
            if period_info['is_perf']:
                st.info("🎭 공연시즌에는 연습실 수요가 증가합니다.")
        
        st.divider()
        
        st.subheader("3. 시간대별 예약 마감 위험도 (ML 예측)")
        
        st.caption("📊 **리드타임 기반 예측**: 현재 시점 기준으로 예약이 마감될 확률을 예측합니다.")
        
        predictor = st.session_state['predictor']
        time_data = {}
        
        date_seed = selected_date.toordinal()
        np.random.seed(date_seed)
        
        if 'booked_slots_cache' not in st.session_state:
            st.session_state['booked_slots_cache'] = {}
        
        booked_cache_key = selected_date.strftime('%Y-%m-%d')
        
        if booked_cache_key not in st.session_state['booked_slots_cache']:
            booked_slots = set()
            for hour in TIME_SLOTS:
                prob, lead_time = predictor.predict(selected_date, hour)
                
                close_chance = prob / 100 * 0.4
                if period_info['is_perf']:
                    close_chance *= 1.5
                if period_info['is_holiday']:
                    close_chance *= 1.3
                if period_info['is_exam']:
                    close_chance *= 0.3
                
                if 18 <= hour <= 20:
                    close_chance *= 1.4
                
                if np.random.random() < close_chance:
                    booked_slots.add(hour)
            
            st.session_state['booked_slots_cache'][booked_cache_key] = booked_slots
        else:
            booked_slots = st.session_state['booked_slots_cache'][booked_cache_key]
        
        for hour in TIME_SLOTS:
            prob, lead_time = predictor.predict(selected_date, hour)
            risk_level, color, emoji = get_risk_level(prob)
            time_data[hour] = {
                'probability': prob,
                'lead_time': lead_time,
                'risk_level': risk_level,
                'color': color,
                'emoji': emoji,
                'is_booked': hour in booked_slots
            }
        
        if 'selected_times' not in st.session_state:
            st.session_state['selected_times'] = []
        
        st.session_state['selected_times'] = [h for h in st.session_state['selected_times'] if h not in booked_slots]
        selected_times = st.session_state.get('selected_times', [])
        
        chart = create_time_slot_chart(time_data, selected_times[0] if selected_times else None)
        st.plotly_chart(chart, use_container_width=True)
        
        booked_count = len(booked_slots)
        if booked_count > 0:
            st.warning(f"⚠️ {booked_count}개 시간대가 이미 마감되었습니다.")
        
        st.markdown("##### 시간대 선택 (복수 선택 가능)")
        st.caption("⏰ 이미 지난 시간대와 마감된 시간대는 선택할 수 없습니다. 클릭하여 선택/해제하세요.")
        
        now = datetime.now()
        current_hour = now.hour
        is_today = selected_date == now.date()
        
        date_key = selected_date.strftime('%Y-%m-%d')
        promo_for_date = st.session_state.get('promo_slots', {}).get(date_key, {})
        
        cols = st.columns(7)
        for idx, hour in enumerate(TIME_SLOTS):
            col_idx = idx % 7
            with cols[col_idx]:
                risk_info = time_data[hour]
                is_past_time = is_today and hour <= current_hour
                is_booked = risk_info['is_booked']
                is_selected = hour in selected_times
                promo_status = promo_for_date.get(hour, '일반')
                
                if is_past_time:
                    st.button(
                        f"{hour}:00\n⛔ 지남",
                        key=f"time_{hour}",
                        use_container_width=True,
                        disabled=True
                    )
                elif is_booked:
                    st.button(
                        f"{hour}:00\n🚫 마감",
                        key=f"time_{hour}",
                        use_container_width=True,
                        disabled=True
                    )
                else:
                    promo_emoji = ""
                    if promo_status == '오픈연습실':
                        promo_emoji = "🎸"
                    elif promo_status == '타임세일':
                        promo_emoji = "💰"
                    
                    button_label = f"{'✅ ' if is_selected else ''}{hour}:00\n{promo_emoji if promo_emoji else risk_info['emoji']}"
                    if st.button(
                        button_label,
                        key=f"time_{hour}",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        if hour in st.session_state['selected_times']:
                            st.session_state['selected_times'].remove(hour)
                        else:
                            st.session_state['selected_times'].append(hour)
                            st.session_state['selected_times'].sort()
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
        
        if selected_times:
            st.divider()
            
            st.subheader(f"4. 선택한 시간대 상세 정보 ({len(selected_times)}개 선택)")
            
            period_text_short = []
            if period_info['is_holiday']:
                period_text_short.append("휴일")
            if period_info['is_exam']:
                period_text_short.append("시험기간")
            if period_info['is_perf']:
                period_text_short.append("공연시즌")
            period_str = ", ".join(period_text_short) if period_text_short else "평일"
            
            hourly_rate = 110000
            total_hours = len(selected_times)
            total_price = hourly_rate * total_hours
            
            time_ranges = []
            for h in selected_times:
                time_ranges.append(f"{h}:00~{h+1}:00")
            time_str = ", ".join(time_ranges)
            
            avg_prob = sum(time_data[h]['probability'] for h in selected_times) / len(selected_times)
            max_risk_hour = max(selected_times, key=lambda h: time_data[h]['probability'])
            max_risk_info = time_data[max_risk_hour]
            
            first_hour = selected_times[0]
            lead_time = time_data[first_hour]['lead_time']
            if lead_time < 24:
                lead_time_str = f"{lead_time:.1f}시간 전"
            elif lead_time < 168:
                lead_time_str = f"{lead_time/24:.1f}일 전"
            else:
                lead_time_str = f"{lead_time/168:.1f}주 전"
            
            info_col1, info_col2 = st.columns([2, 1])
            
            with info_col1:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {max_risk_info['color']}22, {max_risk_info['color']}44);
                    border-left: 5px solid {max_risk_info['color']};
                    padding: 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                ">
                    <h3 style="margin: 0; color: #333;">📍 예약 정보 (ML 예측)</h3>
                    <p style="font-size: 16px; margin-top: 10px;">
                        <strong>날짜:</strong> {selected_date.strftime('%Y년 %m월 %d일')} ({weekday_name})<br>
                        <strong>시간:</strong> {time_str} ({total_hours}시간)<br>
                        <strong>기간 특성:</strong> {period_str}<br>
                        <strong>예약 시점:</strong> 🕐 {lead_time_str} (리드타임: {lead_time:.0f}시간)<br>
                        <strong>평균 마감 확률:</strong> <span style="font-size: 24px; font-weight: bold; color: {max_risk_info['color']};">{avg_prob:.1f}%</span><br>
                        <strong>가장 높은 위험:</strong> {max_risk_info['emoji']} {max_risk_hour}:00 ({max_risk_info['probability']:.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                for sel_hour in selected_times:
                    sel_info = time_data[sel_hour]
                    st.markdown(f"- **{sel_hour}:00~{sel_hour+1}:00**: {sel_info['emoji']} {sel_info['risk_level']} ({sel_info['probability']:.1f}%)")
            
            with info_col2:
                st.markdown("""
                **예상 이용료**
                """)
                st.markdown(f"""
                <div style="
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                ">
                    <p style="margin: 0; color: #666;">{total_hours}시간 기준</p>
                    <h2 style="margin: 10px 0; color: #7B68EE;">₩{total_price:,}</h2>
                    <p style="margin: 0; font-size: 12px; color: #999;">시간당 ₩{hourly_rate:,}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("선택 초기화", use_container_width=True):
                    st.session_state['selected_times'] = []
                    st.rerun()
            
            st.divider()
            
            st.subheader("5. 예약하기")
            
            reserve_col1, reserve_col2, reserve_col3 = st.columns([1, 2, 1])
            
            with reserve_col2:
                if st.button(
                    f"🎯 {total_hours}시간 예약하기",
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
                        <strong>예약 정보:</strong> {date} ({weekday})<br>
                        <strong>시간:</strong> {time_str}<br>
                        <strong>결제 금액:</strong> ₩{price:,}
                    </p>
                </div>
                """.format(
                    date=selected_date.strftime('%Y.%m.%d'),
                    weekday=weekday_name,
                    time_str=time_str,
                    price=total_price
                ), unsafe_allow_html=True)
                
                if st.button("닫기", use_container_width=True):
                    st.session_state['show_payment'] = False
                    st.rerun()

with tab_business:
    st.markdown("""
    **사업자용 대시보드**입니다. ML 분석 결과를 활용하여 공간 운영을 최적화하세요.
    
    - 📊 **공간 활용률 분석**: 시기별, 시간대별 수요 패턴 파악
    - 🏷️ **프로모션 관리**: 저수요 시간대를 오픈연습실이나 타임세일로 전환
    """)
    
    st.divider()
    
    if 'predictor' not in st.session_state:
        render_model_training("business_")
        st.warning("⚠️ 먼저 '학습 데이터 생성' 버튼을 클릭해주세요.")
    else:
        st.info(f"✅ ML 모델 준비 완료 ({st.session_state['data_source']})")
        
        predictor = st.session_state['predictor']
        
        biz_tab1, biz_tab2 = st.tabs(["📊 공간 활용률 분석", "🏷️ 프로모션 관리"])
        
        with biz_tab1:
            analyze_utilization(predictor)
        
        with biz_tab2:
            render_promotion_management(predictor)

st.divider()
st.caption("🎵 연습실 예약 마감 위험도 예측 PoC (2026) | ML 기반 예측 | SpaceCloud 참고")
