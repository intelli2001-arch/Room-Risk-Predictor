import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

st.set_page_config(
    page_title="연습실 예약 마감 위험도 예측",
    page_icon="🎵",
    layout="wide"
)

WEEKDAY_NAMES = {
    0: '일요일',
    1: '월요일',
    2: '화요일',
    3: '수요일',
    4: '목요일',
    5: '금요일',
    6: '토요일'
}

TIME_SLOTS = list(range(9, 23))

def load_real_data():
    csv_path = "attached_assets/practice_room_ML_data_2025_1768532371118.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        return df
    return None

def generate_dummy_data():
    np.random.seed(42)
    
    data = []
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 6, 30)
    current_date = start_date
    
    while current_date <= end_date:
        weekday = current_date.weekday()
        python_weekday = (weekday + 1) % 7
        
        is_weekend = weekday >= 5
        
        for hour in TIME_SLOTS:
            base_prob = 0.3
            
            if is_weekend:
                base_prob += 0.25
            
            if 18 <= hour <= 21:
                base_prob += 0.3
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
                '요일': python_weekday,
                '시간': hour,
                '휴일 여부': 1 if is_weekend else 0,
                '시험기간 여부': 0,
                '공연시즌 여부': 0,
                '예약 여부': is_booked,
                '리드타임_시간': round(lead_time, 1),
                '취소 여부': is_cancelled
            })
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(data)

def calculate_booking_probability(df, target_weekday, target_hour, is_holiday=0):
    filtered = df[(df['요일'] == target_weekday) & (df['시간'] == target_hour)]
    
    if is_holiday:
        filtered = filtered[filtered['휴일 여부'] == 1]
    
    if len(filtered) == 0:
        return np.random.uniform(0.2, 0.8)
    
    booking_rate = filtered['예약 여부'].mean()
    
    booking_rate = booking_rate * 100
    
    noise = np.random.uniform(-5, 5)
    booking_rate = max(5, min(95, booking_rate + noise))
    
    return booking_rate

def get_risk_level(probability):
    if probability >= 70:
        return "위험", "#FF4B4B", "🔴"
    elif probability >= 40:
        return "보통", "#FFA500", "🟠"
    else:
        return "여유", "#00CC66", "🟢"

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
            text="시간대별 예약 마감 위험도",
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
        y0=70, y1=70,
        line=dict(color="#FF4B4B", dash="dash", width=1)
    )
    fig.add_shape(
        type="line",
        x0=-0.5, x1=len(TIME_SLOTS)-0.5,
        y0=40, y1=40,
        line=dict(color="#FFA500", dash="dash", width=1)
    )
    
    return fig

st.title("🎵 연습실 예약 마감 위험도 예측")
st.markdown("""
이 서비스는 **예약 마감 확률**을 제공하여 사용자가 더 합리적으로 예약 결정을 내릴 수 있도록 돕습니다.

- **위험 (70% 이상)**: 마감 가능성이 높아 빠른 예약을 권장합니다.
- **보통 (40~70%)**: 적당한 시간 내 예약을 권장합니다.
- **여유 (40% 미만)**: 여유롭게 예약해도 괜찮습니다.
""")

st.divider()

st.subheader("1. 데이터 준비")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("📊 더미 데이터 생성", type="primary", use_container_width=True):
        with st.spinner("데이터 생성 중..."):
            st.session_state['booking_data'] = generate_dummy_data()
            st.session_state['data_source'] = "더미 데이터"
        st.success("더미 데이터가 생성되었습니다!")
        st.rerun()

with col2:
    real_data_available = os.path.exists("attached_assets/practice_room_ML_data_2025_1768532371118.csv")
    if real_data_available:
        if st.button("📁 실제 데이터 로드", use_container_width=True):
            with st.spinner("데이터 로드 중..."):
                st.session_state['booking_data'] = load_real_data()
                st.session_state['data_source'] = "실제 데이터"
            st.success("실제 데이터가 로드되었습니다!")
            st.rerun()

with col3:
    if 'booking_data' in st.session_state:
        st.info(f"✅ {st.session_state['data_source']}가 로드되었습니다. ({len(st.session_state['booking_data'])}개 레코드)")

if 'booking_data' in st.session_state:
    with st.expander("📋 데이터 미리보기 (처음 20개 행)"):
        display_df = st.session_state['booking_data'].head(20).copy()
        display_df['요일명'] = display_df['요일'].map(WEEKDAY_NAMES)
        display_df['시간대'] = display_df['시간'].apply(lambda x: f"{x}:00~{x+1}:00")
        display_df['예약 상태'] = display_df['예약 여부'].map({0: '미예약', 1: '예약됨'})
        
        st.dataframe(
            display_df[['연도', '월', '일', '요일명', '시간대', '휴일 여부', '예약 상태']],
            use_container_width=True,
            hide_index=True
        )

st.divider()

st.subheader("2. 날짜 선택")

if 'booking_data' not in st.session_state:
    st.warning("⚠️ 먼저 '더미 데이터 생성' 버튼을 클릭해주세요.")
else:
    col_date1, col_date2 = st.columns([1, 2])
    
    with col_date1:
        selected_date = st.date_input(
            "예약 희망 날짜를 선택하세요",
            value=datetime(2025, 3, 15),
            min_value=datetime(2025, 1, 1),
            max_value=datetime(2025, 12, 31),
            format="YYYY-MM-DD"
        )
    
    with col_date2:
        weekday_num = (selected_date.weekday() + 1) % 7
        weekday_name = WEEKDAY_NAMES[weekday_num]
        is_weekend = selected_date.weekday() >= 5
        
        st.markdown(f"""
        **선택된 날짜 정보:**
        - 📅 날짜: {selected_date.strftime('%Y년 %m월 %d일')} ({weekday_name})
        - {'🎉 주말/휴일' if is_weekend else '📆 평일'}
        """)
    
    st.divider()
    
    st.subheader("3. 시간대별 예약 마감 위험도")
    
    df = st.session_state['booking_data']
    time_data = {}
    
    for hour in TIME_SLOTS:
        prob = calculate_booking_probability(
            df, 
            weekday_num, 
            hour, 
            is_holiday=1 if is_weekend else 0
        )
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
    
    cols = st.columns(7)
    for idx, hour in enumerate(TIME_SLOTS):
        col_idx = idx % 7
        with cols[col_idx]:
            risk_info = time_data[hour]
            button_label = f"{hour}:00\n{risk_info['emoji']}\n{risk_info['probability']:.0f}%"
            
            if st.button(
                f"{hour}:00\n{risk_info['emoji']}",
                key=f"time_{hour}",
                use_container_width=True
            ):
                st.session_state['selected_time'] = hour
                st.rerun()
    
    st.markdown("---")
    col_legend1, col_legend2, col_legend3 = st.columns(3)
    with col_legend1:
        st.markdown("🟢 **여유** (40% 미만)")
    with col_legend2:
        st.markdown("🟠 **보통** (40~70%)")
    with col_legend3:
        st.markdown("🔴 **위험** (70% 이상)")
    
    if 'selected_time' in st.session_state and st.session_state['selected_time'] is not None:
        st.divider()
        
        st.subheader("4. 선택한 시간대 상세 정보")
        
        sel_hour = st.session_state['selected_time']
        sel_info = time_data[sel_hour]
        
        info_col1, info_col2 = st.columns([2, 1])
        
        with info_col1:
            risk_color = sel_info['color']
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {risk_color}22, {risk_color}44);
                border-left: 5px solid {risk_color};
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            ">
                <h3 style="margin: 0; color: #333;">📍 예약 정보</h3>
                <p style="font-size: 16px; margin-top: 10px;">
                    <strong>날짜:</strong> {selected_date.strftime('%Y년 %m월 %d일')} ({weekday_name})<br>
                    <strong>시간:</strong> {sel_hour}:00 ~ {sel_hour+1}:00 (1시간)<br>
                    <strong>마감 확률:</strong> <span style="font-size: 24px; font-weight: bold; color: {risk_color};">{sel_info['probability']:.1f}%</span><br>
                    <strong>위험도:</strong> {sel_info['emoji']} {sel_info['risk_level']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if sel_info['risk_level'] == "위험":
                st.warning("⚠️ 이 시간대는 마감 가능성이 높습니다. 빠른 예약을 권장합니다!")
            elif sel_info['risk_level'] == "보통":
                st.info("ℹ️ 이 시간대는 보통 수준의 수요가 있습니다. 적당한 시간 내 예약을 권장합니다.")
            else:
                st.success("✅ 이 시간대는 여유가 있습니다. 천천히 예약해도 괜찮습니다.")
        
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
st.caption("🎵 연습실 예약 마감 위험도 예측 PoC | SpaceCloud 참고")
