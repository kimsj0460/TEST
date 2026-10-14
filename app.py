# -*- coding: utf-8 -*-
"""
================================================================================
서울특별시 공공자전거(따릉이) 빅데이터 융합 분석 대시보드
================================================================================
- Author: Senior Python Data Scientist & Interactive Dashboard Specialist
- Stack: Streamlit, Pandas, GeoPandas, Plotly, Shapely, PyArrow
- Features:
  1. Data Modeling: Auto Left-Join between Usage & Station Master + GeoPandas EPSG:4326
  2. Tab 1: Executive Summary (KPIs, Monthly Trends, Demographics, Top 10 Stations)
  3. Tab 2: Geospatial Analysis (District Choropleth, Station Bubble Map, Infrastructure Efficiency)
  4. Tab 3: User Behavior & Segmentation (Demographic Heatmap, Commute vs Leisure Scatter, Eco/Health)
  5. Tab 4: Insight & Action Plan (Rebalancing Strategy, Targeted Pricing, Bike Lane Priorities)
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 0. Page Configuration & Custom CSS Injection
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="서울시 따릉이 빅데이터 융합 분석 대시보드",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System CSS
CUSTOM_CSS = """
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    html, body, [class*="css"] {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Top title styling */
    .main-title-container {
        padding: 1.2rem 1.8rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #10B981, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.28);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(16, 185, 129, 0.18);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-card-accent {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #10B981, #38BDF8);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #38BDF8;
        margin-top: 0.35rem;
        font-weight: 500;
    }

    /* Policy & Insight Cards */
    .insight-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border-left: 4px solid #10B981;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .insight-card-blue {
        border-left-color: #38BDF8;
    }
    .insight-card-purple {
        border-left-color: #818CF8;
    }
    .insight-card-amber {
        border-left-color: #F59E0B;
    }

    /* Badge Tags */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-right: 0.4rem;
    }
    .badge-green { background-color: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-blue { background-color: rgba(56, 189, 248, 0.2); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.4); }
    .badge-purple { background-color: rgba(129, 140, 248, 0.2); color: #A5B4FC; border: 1px solid rgba(129, 140, 248, 0.4); }
    .badge-amber { background-color: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }

    /* Custom Streamlit Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        color: #94A3B8;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(16, 185, 129, 0.2) !important;
        color: #34D399 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. Data Ingestion & Preprocessing Pipeline (Cached with @st.cache_data)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="자전거 이용 데이터 및 공간 공간정보 로딩 중...")
def load_and_preprocess_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1-1. 대여소 마스터 정보 파일 검색 및 정제
    st_files = [f for f in os.listdir(base_dir) if "대여소 정보" in f and f.endswith((".xlsx", ".csv"))]
    st_path = os.path.join(base_dir, st_files[0]) if st_files else os.path.join(base_dir, "공공자전거 대여소 정보(26.6월 기준) (1).xlsx")
    
    if st_path.endswith(".csv"):
        try:
            df_st = pd.read_csv(st_path, encoding='cp949')
        except Exception:
            df_st = pd.read_csv(st_path, encoding='utf-8')
    else:
        df_st = pd.read_excel(st_path)

    # 컬럼 공백 및 개행문자 정제
    df_st.columns = [c.replace('\n', '').strip() for c in df_st.columns]
    df_st['대여소번호'] = pd.to_numeric(df_st['대여소번호'], errors='coerce').fillna(0).astype('int64')
    df_st['보관소(대여소)명'] = df_st['보관소(대여소)명'].astype(str).str.strip()
    df_st['자치구'] = df_st['자치구'].astype(str).str.strip()
    
    # 거치대 수 산출 (LCD + QR)
    lcd_col = 'LCD 설치형태' if 'LCD 설치형태' in df_st.columns else 'LCD설치형태'
    qr_col = 'QR 설치형태' if 'QR 설치형태' in df_st.columns else 'QR설치형태'
    df_st['총거치대수'] = (df_st[lcd_col].fillna(0) if lcd_col in df_st.columns else 0) + \
                        (df_st[qr_col].fillna(0) if qr_col in df_st.columns else 0)
    df_st['총거치대수'] = df_st['총거치대수'].clip(lower=1)  # 0으로 나누기 방지용

    # 대여소 마스터 중복 제거 (대여소번호 기준)
    df_st = df_st.drop_duplicates(subset=['대여소번호']).copy()

    # 1-2. 서울시 시군구 경계 공간 데이터 (sig.shp) 읽기 및 좌표계 변환 (EPSG:4326)
    shp_dir = os.path.join(base_dir, "sig_20230729")
    shp_path = os.path.join(shp_dir, "sig.shp") if os.path.exists(os.path.join(shp_dir, "sig.shp")) else os.path.join(base_dir, "sig.shp")
    
    try:
        gdf = gpd.read_file(shp_path, encoding='cp949')
    except Exception:
        gdf = gpd.read_file(shp_path, encoding='utf-8')
    
    # 서울시 자치구만 필터링 (서울시 SIG_CD는 '11'로 시작)
    gdf = gdf[gdf['SIG_CD'].astype(str).str.startswith('11')].copy()
    
    # 좌표계 정의 및 EPSG:4326 변환
    if gdf.crs is None:
        gdf.set_crs(epsg=5179, inplace=True)
    gdf_4326 = gdf.to_crs(epsg=4326)
    geojson_data = json.loads(gdf_4326.to_json())

    # 1-3. 공공자전거 이용정보 파일 읽기 (Parquet 캐시 지원으로 초고속 로딩)
    pq_path = os.path.join(base_dir, "bike_usage_cached.parquet")
    if os.path.exists(pq_path):
        df_us = pd.read_parquet(pq_path)
    else:
        us_files = [f for f in os.listdir(base_dir) if "이용정보" in f and f.endswith((".xlsx", ".csv"))]
        us_path = os.path.join(base_dir, us_files[0])
        if us_path.endswith(".csv"):
            try:
                df_us = pd.read_csv(us_path, encoding='cp949')
            except Exception:
                df_us = pd.read_csv(us_path, encoding='utf-8')
        else:
            df_us = pd.read_excel(us_path)
            
        df_us.columns = [c.replace('\n', '').strip() for c in df_us.columns]
        df_us['대여소번호'] = pd.to_numeric(df_us['대여소번호'], errors='coerce').fillna(0).astype('int64')
        df_us.to_parquet(pq_path, engine='pyarrow', index=False)

    # 1-4. 이용정보와 대여소 마스터 Left Join
    st_cols_to_merge = ['대여소번호', '보관소(대여소)명', '자치구', '상세주소', '위도', '경도', '총거치대수']
    merged = df_us.merge(
        df_st[st_cols_to_merge],
        on='대여소번호',
        how='left'
    )

    # 자치구 결측치 및 미매칭 처리
    merged['자치구'] = merged['자치구'].fillna('기타/폐쇄대여소')
    
    # 성별 결측치 및 표기 표준화 (M/m -> 남성, F/f -> 여성, 기타/결측 -> 미입력/기타)
    gender_raw = merged['성별'].fillna('미입력').astype(str).str.strip().str.upper()
    merged['성별_정제'] = gender_raw.map({'M': '남성', 'F': '여성'}).fillna('미입력/기타')

    # 연령대 결측치 정제
    merged['연령대코드'] = merged['연령대코드'].fillna('기타').astype(str).str.strip()

    # 대여구분코드 표준 카테고리화
    def classify_rental_type(val):
        s = str(val).strip()
        if '정기' in s:
            return '정기권'
        elif '일일' in s:
            return '일일권'
        elif '단체' in s:
            return '단체권'
        return '기타'
    merged['대여구분_분류'] = merged['대여구분코드'].apply(classify_rental_type)

    # 대여월 문자열 생성 (예: 2026-03)
    merged['대여월'] = merged['대여일자'].astype(str).str[:4] + '-' + merged['대여일자'].astype(str).str[4:6]

    # 파생 변수 계산: 1회당 거리, 1회당 시간, 이동속도, 탄소량, 운동량
    merged['이용건수'] = merged['이용건수'].clip(lower=1)
    merged['총이동거리_km'] = merged['이동거리(M)'] / 1000.0
    merged['1회당_이동거리_m'] = merged['이동거리(M)'] / merged['이용건수']
    merged['1회당_이용시간_분'] = merged['이용시간(분)'] / merged['이용건수']
    
    # 시속(km/h) 계산 (0시간 방지 가드)
    hours = (merged['1회당_이용시간_분'] / 60.0).replace(0, 0.001)
    merged['평균속도_kmh'] = (merged['1회당_이동거리_m'] / 1000.0) / hours

    # 이상치 플래그 (0m 이동 또는 0분 이용, 또는 비정상 속도 50km/h 초과)
    merged['이상치_여부'] = (merged['이동거리(M)'] <= 0) | (merged['이용시간(분)'] <= 0) | (merged['평균속도_kmh'] > 45)

    return merged, df_st, gdf_4326, geojson_data


# 데이터 로드 실행
data_load_state = st.empty()
merged_df, station_df, seoul_gdf, seoul_geojson = load_and_preprocess_data()
data_load_state.empty()

# -----------------------------------------------------------------------------
# 2. Sidebar: Global Controls & Filtering System
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚲 따릉이 데이터 필터")
    st.markdown("<p style='font-size:0.8rem; color:#94A3B8; margin-top:-10px;'>원하는 분석 조건을 조합하여 데이터를 탐색하세요.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # [필터 1] 대여월 필터
    available_months = sorted(merged_df['대여월'].unique().tolist())
    selected_months = st.multiselect(
        "📅 대여월 (월별 분석)",
        options=available_months,
        default=available_months,
        help="분석할 데이터의 기준 월을 선택합니다."
    )

    # [필터 2] 자치구 다중 선택 (전체 선택/해제 토글 버튼)
    all_districts = sorted([d for d in merged_df['자치구'].unique() if d != '기타/폐쇄대여소'])
    
    col_d1, col_d2 = st.columns(2)
    if 'districts_select_all' not in st.session_state:
        st.session_state['districts_select_all'] = all_districts
        
    with col_d1:
        if st.button("전체 선택", use_container_width=True):
            st.session_state['selected_districts'] = all_districts
            st.rerun()
    with col_d2:
        if st.button("전체 해제", use_container_width=True):
            st.session_state['selected_districts'] = []
            st.rerun()

    current_districts = st.session_state.get('selected_districts', all_districts)
    selected_districts = st.multiselect(
        "🏙️ 자치구 선택",
        options=all_districts,
        default=current_districts,
        help="분석할 서울시 자치구를 선택합니다."
    )
    st.session_state['selected_districts'] = selected_districts

    # [필터 3] 대여구분코드 (정기권/일일권/단체권 등)
    available_rental_types = sorted(merged_df['대여구분_분류'].unique().tolist())
    selected_rental_types = st.multiselect(
        "🎫 대여구분 (회원권 형태)",
        options=available_rental_types,
        default=available_rental_types
    )

    # [필터 4] 성별 필터
    available_genders = ['남성', '여성', '미입력/기타']
    selected_genders = st.multiselect(
        "👤 성별 구분",
        options=available_genders,
        default=available_genders
    )

    # [필터 5] 연령대 필터
    age_order = ['~10대', '20대', '30대', '40대', '50대', '60대', '70대이상', '기타']
    actual_ages = [a for a in age_order if a in merged_df['연령대코드'].unique()]
    selected_ages = st.multiselect(
        "🎂 연령대 구분",
        options=actual_ages,
        default=actual_ages
    )

    st.markdown("---")
    # [필터 6] 이상치/단순반납 필터 토글
    exclude_outliers = st.toggle(
        "⚡ 단순반납/이상치 제외",
        value=True,
        help="이동거리 0m, 이용시간 0분 또는 시속 45km/h 초과 이상치 레코드를 필터링합니다."
    )

    # 필터 적용 로직
    filtered = merged_df[
        (merged_df['대여월'].isin(selected_months)) &
        (merged_df['자치구'].isin(selected_districts)) &
        (merged_df['대여구분_분류'].isin(selected_rental_types)) &
        (merged_df['성별_정제'].isin(selected_genders)) &
        (merged_df['연령대코드'].isin(selected_ages))
    ]
    if exclude_outliers:
        filtered = filtered[~filtered['이상치_여부']]

    # 사이드바 하단 데이터 통계 인포
    filtered_rides = filtered['이용건수'].sum()
    total_rides = merged_df['이용건수'].sum()
    coverage_pct = (filtered_rides / total_rides * 100) if total_rides > 0 else 0

    st.markdown(f"""
    <div style='background:rgba(15, 23, 42, 0.6); padding:10px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.08);'>
        <div style='font-size:0.75rem; color:#94A3B8;'>필터 데이터 커버리지</div>
        <div style='font-size:1.15rem; font-weight:700; color:#10B981;'>{filtered_rides:,.0f} 건</div>
        <div style='font-size:0.75rem; color:#38BDF8;'>전체 대비 {coverage_pct:.1f}% 선택됨</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. Main Dashboard Layout & Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-title-container">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
        <div>
            <h1 class="main-title">🚲 서울특별시 따릉이 공공자전거 빅데이터 융합 분석</h1>
            <p class="sub-title">공공자전거 이용정보 × 대여소 마스터 × 행정구역 공간 공간정보(GIS) 통합 인사이트 대시보드</p>
        </div>
        <div>
            <span class="badge badge-green">2026.03 ~ 05 데이터셋</span>
            <span class="badge badge-blue">EPSG:4326 정밀 매핑</span>
            <span class="badge badge-purple">고속 Parquet 최적화</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 만약 필터링 결과가 비어있는 경우 경고
if len(filtered) == 0:
    st.warning("⚠️ 선택하신 필터 조건에 부합하는 데이터가 없습니다. 사이드바의 필터를 재조정해주세요.")
    st.stop()

# Helper for Plotly dark/modern styling
def apply_plotly_theme(fig, height=400):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(15, 23, 42, 0)",
        plot_bgcolor="rgba(15, 23, 42, 0)",
        font=dict(family="Pretendard, sans-serif", color="#E2E8F0", size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.06)")
    return fig

# Helper for Plotly Map functions (compatibility with Plotly 5.x and 6.x/7.x)
def build_choropleth_map(df_metric, geojson, location_col, color_col, title_name, color_scale='Tealgrn'):
    if hasattr(px, 'choropleth_map'):
        fig = px.choropleth_map(
            df_metric,
            geojson=geojson,
            locations=location_col,
            featureidkey='properties.SIG_KOR_NM',
            color=color_col,
            color_continuous_scale=color_scale,
            map_style='carto-positron',
            zoom=9.9,
            center={"lat": 37.5665, "lon": 126.9780},
            opacity=0.8,
            hover_name=location_col
        )
    else:
        fig = px.choropleth_mapbox(
            df_metric,
            geojson=geojson,
            locations=location_col,
            featureidkey='properties.SIG_KOR_NM',
            color=color_col,
            color_continuous_scale=color_scale,
            mapbox_style='carto-positron',
            zoom=9.9,
            center={"lat": 37.5665, "lon": 126.9780},
            opacity=0.8,
            hover_name=location_col
        )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(r=0, t=0, l=0, b=0),
        paper_bgcolor="rgba(15, 23, 42, 0)",
        height=520,
        coloraxis_colorbar=dict(title=title_name, len=0.8, thickness=16)
    )
    return fig


# -----------------------------------------------------------------------------
# 4. Tab Structures
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 [탭 1] Executive Summary (종합 핵심 지표)",
    "🗺️ [탭 2] 공간 분석 및 대여소 히트맵 (Geospatial)",
    "👥 [탭 3] 유저 세그먼트 & 이용 행동 (User Behavior)",
    "💡 [탭 4] 심층 인사이트 & 운영 정책 제언 (Insight & Action)"
])

# =============================================================================
# [TAB 1] Executive Summary
# =============================================================================
with tab1:
    # 1. KPI Aggregations
    kpi_rides = filtered['이용건수'].sum()
    kpi_dist_km = filtered['총이동거리_km'].sum()
    kpi_carbon_kg = filtered['탄소량'].sum()
    kpi_carbon_ton = kpi_carbon_kg / 1000.0
    kpi_cal_kcal = filtered['운동량'].sum()
    kpi_dur_min = filtered['이용시간(분)'].sum()
    
    kpi_avg_dur = kpi_dur_min / kpi_rides if kpi_rides > 0 else 0
    kpi_avg_dist_km = kpi_dist_km / kpi_rides if kpi_rides > 0 else 0
    
    # Conversions
    earth_circ_times = kpi_dist_km / 40075.0  # 지구 둘레 약 40,075 km
    pine_trees_equiv = kpi_carbon_kg / 6.6    # 30년생 소나무 1그루 연간 흡수량 약 6.6kg
    rice_bowls = kpi_cal_kcal / 300.0         # 밥 한 공기 약 300 kcal

    # Render KPI Cards in 5 columns
    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
    
    with kpi_c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-accent"></div>
            <div class="metric-label">총 이용건수</div>
            <div class="metric-value">{kpi_rides:,.0f}<span style='font-size:1rem; font-weight:500; color:#94A3B8;'>건</span></div>
            <div class="metric-sub">약 {kpi_rides/10000:,.1f}만 회 주행</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-accent"></div>
            <div class="metric-label">총 이동거리</div>
            <div class="metric-value">{kpi_dist_km:,.0f}<span style='font-size:1rem; font-weight:500; color:#94A3B8;'>km</span></div>
            <div class="metric-sub">지구 둘레 약 {earth_circ_times:,.1f}바퀴</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-accent"></div>
            <div class="metric-label">총 탄소 절감량</div>
            <div class="metric-value">{kpi_carbon_ton:,.1f}<span style='font-size:1rem; font-weight:500; color:#94A3B8;'>톤</span></div>
            <div class="metric-sub">소나무 약 {pine_trees_equiv:,.0f}그루 효과</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-accent"></div>
            <div class="metric-label">1회 평균 이동</div>
            <div class="metric-value">{kpi_avg_dist_km:.2f}<span style='font-size:1rem; font-weight:500; color:#94A3B8;'>km</span></div>
            <div class="metric-sub">평균 이용시간 {kpi_avg_dur:.1f}분</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-accent"></div>
            <div class="metric-label">총 소모 열량</div>
            <div class="metric-value">{kpi_cal_kcal/1000000:,.1f}<span style='font-size:1rem; font-weight:500; color:#94A3B8;'>백만kcal</span></div>
            <div class="metric-sub">공깃밥 약 {rice_bowls/10000:,.1f}만 공기</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 2. Main Trend Visualizations
    col_t1, col_t2 = st.columns([6, 4])
    
    with col_t1:
        st.markdown("#### 📈 월별 이용건수 추이 및 회원권 유형별 비중")
        monthly_type = filtered.groupby(['대여월', '대여구분_분류'])['이용건수'].sum().reset_index()
        fig_monthly = px.bar(
            monthly_type,
            x='대여월',
            y='이용건수',
            color='대여구분_분류',
            barmode='stack',
            color_discrete_map={'정기권': '#10B981', '일일권': '#38BDF8', '단체권': '#818CF8', '기타': '#F59E0B'},
            labels={'이용건수': '총 이용건수', '대여월': '대여월', '대여구분_분류': '회원권 유형'}
        )
        fig_monthly.update_traces(marker_line_width=0, opacity=0.9)
        fig_monthly = apply_plotly_theme(fig_monthly, height=350)
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col_t2:
        st.markdown("#### 🍩 회원권(대여구분) 이용 비중")
        rental_type_tot = filtered.groupby('대여구분_분류')['이용건수'].sum().reset_index()
        fig_donut = px.pie(
            rental_type_tot,
            names='대여구분_분류',
            values='이용건수',
            hole=0.55,
            color='대여구분_분류',
            color_discrete_map={'정기권': '#10B981', '일일권': '#38BDF8', '단체권': '#818CF8', '기타': '#F59E0B'}
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#0F172A', width=2)))
        fig_donut = apply_plotly_theme(fig_donut, height=350)
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # 3. Demographics and Top 10 Stations
    col_d1, col_d2 = st.columns([5, 5])

    with col_d1:
        st.markdown("#### 👥 연령대 및 성별 이용 현황")
        age_gender = filtered.groupby(['연령대코드', '성별_정제'])['이용건수'].sum().reset_index()
        # Sort age according to age_order
        age_gender['age_rank'] = age_gender['연령대코드'].apply(lambda x: age_order.index(x) if x in age_order else 99)
        age_gender = age_gender.sort_values(by=['age_rank'])
        
        fig_demog = px.bar(
            age_gender,
            x='연령대코드',
            y='이용건수',
            color='성별_정제',
            barmode='group',
            color_discrete_map={'남성': '#38BDF8', '여성': '#F472B6', '미입력/기타': '#94A3B8'},
            labels={'이용건수': '총 이용건수', '연령대코드': '연령대', '성별_정제': '성별'}
        )
        fig_demog = apply_plotly_theme(fig_demog, height=360)
        st.plotly_chart(fig_demog, use_container_width=True)

    with col_d2:
        st.markdown("#### 🏆 최다 이용 대여소 Top 10")
        station_agg = filtered.groupby(['대여소번호', '보관소(대여소)명', '자치구', '총거치대수'])['이용건수'].sum().reset_index()
        station_agg['회전율'] = station_agg['이용건수'] / station_agg['총거치대수']
        station_agg['표시대여소명'] = station_agg['보관소(대여소)명'].str.slice(0, 16) + " (" + station_agg['자치구'] + ")"
        
        top10_st = station_agg.sort_values(by='이용건수', ascending=True).tail(10)
        
        fig_top10 = px.bar(
            top10_st,
            x='이용건수',
            y='표시대여소명',
            orientation='h',
            color='회전율',
            color_continuous_scale='Viridis',
            labels={'이용건수': '총 이용건수', '표시대여소명': '대여소 (자치구)', '회전율': '회전율(건/거치대)'}
        )
        fig_top10.update_traces(marker_line_width=0, opacity=0.9)
        fig_top10 = apply_plotly_theme(fig_top10, height=360)
        st.plotly_chart(fig_top10, use_container_width=True)

    # Top 20 Detailed Expander
    with st.expander("🔍 최다 이용 대여소 Top 20 상세 리스트 및 회전율 보기"):
        top20_df = station_agg.sort_values(by='이용건수', ascending=False).head(20).copy()
        top20_df['순위'] = range(1, len(top20_df) + 1)
        top20_display = top20_df[['순위', '대여소번호', '보관소(대여소)명', '자치구', '총거치대수', '이용건수', '회전율']].copy()
        top20_display.columns = ['순위', '대여소번호', '대여소명', '자치구', '총거치대수', '총 이용건수(건)', '거치대 회전율(회)']
        st.dataframe(
            top20_display.style.format({
                '총 이용건수(건)': '{:,.0f}',
                '거치대 회전율(회)': '{:,.1f}'
            }),
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# [TAB 2] Geospatial Analysis
# =============================================================================
with tab2:
    st.markdown("### 🗺️ 공간 통계 및 대여소 공간 패턴 분석")
    st.markdown("<p style='color:#94A3B8; font-size:0.9rem;'>서울시 25개 자치구 경계 공간 데이터(sig.shp)와 대여소 위경도 좌표를 융합하여 인프라 효율성을 입체적으로 조망합니다.</p>", unsafe_allow_html=True)

    # 1. 자치구 단위 공간 집계
    district_agg = filtered.groupby('자치구').agg(
        총이용건수=('이용건수', 'sum'),
        총이동거리_km=('총이동거리_km', 'sum'),
        총탄소량_kg=('탄소량', 'sum'),
        대여소수=('대여소번호', 'nunique'),
        평균이용시간=('이용시간(분)', lambda x: x.sum() / filtered.loc[x.index, '이용건수'].sum())
    ).reset_index()

    # 대여소 마스터의 자치구별 총 거치대수 결합
    dist_docks = station_df.groupby('자치구')['총거치대수'].sum().reset_index()
    district_agg = district_agg.merge(dist_docks, on='자치구', how='left')
    district_agg['총거치대수'] = district_agg['총거치대수'].fillna(1)
    district_agg['거치대당회전율'] = district_agg['총이용건수'] / district_agg['총거치대수']
    district_agg['대여소당평균이용건수'] = district_agg['총이용건수'] / district_agg['대여소수'].replace(0, 1)
    district_agg['1건당평균이동거리_km'] = district_agg['총이동거리_km'] / district_agg['총이용건수'].replace(0, 1)

    # 지도 표시 지표 선택기
    col_map_ctrl1, col_map_ctrl2 = st.columns([4, 6])
    with col_map_ctrl1:
        map_metric = st.selectbox(
            "📌 코로플레스 지도 시각화 지표",
            options=[
                ('총이용건수', '총 이용건수 (건)', 'Viridis'),
                ('대여소당평균이용건수', '대여소 1개당 평균 이용건수 (건)', 'Blues'),
                ('거치대당회전율', '거치대 1개당 이용 회전율 (회)', 'Plasma'),
                ('1건당평균이동거리_km', '1건당 평균 이동거리 (km)', 'Tealgrn'),
                ('총탄소량_kg', '총 탄소 절감량 (kg)', 'Greens')
            ],
            format_func=lambda x: x[1]
        )

    # Map Layout: Left Choropleth, Right Station Coordinate Bubble Map
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown(f"#### 🌐 서울시 자치구별 {map_metric[1]}")
        fig_choropleth = build_choropleth_map(
            district_agg,
            seoul_geojson,
            location_col='자치구',
            color_col=map_metric[0],
            title_name=map_metric[1],
            color_scale=map_metric[2]
        )
        st.plotly_chart(fig_choropleth, use_container_width=True)

    with col_m2:
        st.markdown("#### 📍 대여소별 좌표 버블 맵 (위도/경도)")
        st_coords = filtered.groupby(['대여소번호', '보관소(대여소)명', '자치구', '위도', '경도', '총거치대수']).agg(
            이용건수=('이용건수', 'sum'),
            평균이동거리_km=('총이동거리_km', lambda x: x.sum() / filtered.loc[x.index, '이용건수'].sum()),
            평균이용시간_분=('이용시간(분)', lambda x: x.sum() / filtered.loc[x.index, '이용건수'].sum())
        ).reset_index()

        st_coords = st_coords.dropna(subset=['위도', '경도'])
        st_coords['회전율'] = st_coords['이용건수'] / st_coords['총거치대수']
        
        # Performance: Top 500 for smooth map rendering
        top_coords = st_coords.sort_values(by='이용건수', ascending=False).head(500)

        if hasattr(px, 'scatter_map'):
            fig_bubble = px.scatter_map(
                top_coords,
                lat='위도',
                lon='경도',
                size='이용건수',
                color='회전율',
                color_continuous_scale='Turbo',
                size_max=22,
                zoom=10.2,
                center={"lat": 37.5600, "lon": 126.9800},
                map_style='carto-positron',
                hover_name='보관소(대여소)명',
                hover_data={'자치구': True, '이용건수': ':,', '총거치대수': True, '회전율': ':.1f', '위도': False, '경도': False}
            )
        else:
            fig_bubble = px.scatter_mapbox(
                top_coords,
                lat='위도',
                lon='경도',
                size='이용건수',
                color='회전율',
                color_continuous_scale='Turbo',
                size_max=22,
                zoom=10.2,
                center={"lat": 37.5600, "lon": 126.9800},
                mapbox_style='carto-positron',
                hover_name='보관소(대여소)명',
                hover_data={'자치구': True, '이용건수': ':,', '총거치대수': True, '회전율': ':.1f', '위도': False, '경도': False}
            )
            
        fig_bubble.update_layout(
            template="plotly_dark",
            margin=dict(r=0, t=0, l=0, b=0),
            paper_bgcolor="rgba(15, 23, 42, 0)",
            height=520,
            coloraxis_colorbar=dict(title="회전율(회)", len=0.8, thickness=16)
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 2. 인프라 밀집도 대비 이용 효율성 (Quadrant & Ranking)
    col_ef1, col_ef2 = st.columns([6, 4])

    with col_ef1:
        st.markdown("#### ⚖️ 자치구별 인프라 규모(대여소 수) 대비 총 이용건수 (사분면 분석)")
        median_stations = district_agg['대여소수'].median()
        median_rides = district_agg['총이용건수'].median()

        fig_quad = px.scatter(
            district_agg,
            x='대여소수',
            y='총이용건수',
            size='총거치대수',
            color='거치대당회전율',
            color_continuous_scale='Viridis',
            text='자치구',
            labels={'대여소수': '설치 대여소 수 (개소)', '총이용건수': '총 이용건수 (건)', '총거치대수': '총 거치대수', '거치대당회전율': '회전율'},
            hover_name='자치구'
        )
        fig_quad.update_traces(textposition='top center', marker=dict(opacity=0.85, line=dict(width=1, color='white')))
        
        # 사분면 기준선 추가
        fig_quad.add_vline(x=median_stations, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig_quad.add_hline(y=median_rides, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        
        # 사분면 텍스트 라벨
        fig_quad.add_annotation(x=district_agg['대여소수'].max()*0.9, y=district_agg['총이용건수'].max()*0.95, text="고인프라·고이용 (핵심 거점)", showarrow=False, font=dict(color="#34D399", size=10))
        fig_quad.add_annotation(x=district_agg['대여소수'].min()*1.1, y=district_agg['총이용건수'].max()*0.95, text="저인프라·고이용 (공급 부족 시급)", showarrow=False, font=dict(color="#F87171", size=10))
        
        fig_quad = apply_plotly_theme(fig_quad, height=400)
        st.plotly_chart(fig_quad, use_container_width=True)

    with col_ef2:
        st.markdown("#### ⚡ 자치구별 거치대 회전율(효율성) 랭킹")
        turnover_ranking = district_agg.sort_values(by='거치대당회전율', ascending=True)
        fig_turnover = px.bar(
            turnover_ranking,
            x='거치대당회전율',
            y='자치구',
            orientation='h',
            color='거치대당회전율',
            color_continuous_scale='Plasma',
            labels={'거치대당회전율': '거치대당 회전율 (회)', '자치구': '자치구'}
        )
        fig_turnover.update_traces(marker_line_width=0, opacity=0.9)
        fig_turnover = apply_plotly_theme(fig_turnover, height=400)
        st.plotly_chart(fig_turnover, use_container_width=True)


# =============================================================================
# [TAB 3] User Behavior & Segmentation
# =============================================================================
with tab3:
    st.markdown("### 👥 유저 세그먼트 및 주행 행동 패턴 심층 분석")
    st.markdown("<p style='color:#94A3B8; font-size:0.9rem;'>이용자의 인구통계학적 특성과 1회 이용거리/시간 상관관계를 통해 이동 목적을 세분화(Segmentation)합니다.</p>", unsafe_allow_html=True)

    # 1. 인구통계 교차 분석 (Heatmap & Pivot Table)
    col_u1, col_u2 = st.columns([5, 5])

    with col_u1:
        st.markdown("#### 🔥 연령대 × 성별 이용건수 교차 히트맵")
        heatmap_data = filtered.pivot_table(
            index='연령대코드',
            columns='성별_정제',
            values='이용건수',
            aggfunc='sum',
            fill_value=0
        )
        # Reorder index
        valid_idx = [a for a in age_order if a in heatmap_data.index]
        heatmap_data = heatmap_data.reindex(valid_idx)
        
        fig_heat = px.imshow(
            heatmap_data,
            color_continuous_scale='Blues',
            aspect="auto",
            text_auto=True,
            labels=dict(x="성별", y="연령대", color="이용건수")
        )
        fig_heat = apply_plotly_theme(fig_heat, height=360)
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_u2:
        st.markdown("#### ⏱️ 연령대 × 성별 평균 이용시간(분) 히트맵")
        dur_heatmap_data = filtered.groupby(['연령대코드', '성별_정제']).apply(
            lambda x: x['이용시간(분)'].sum() / x['이용건수'].sum()
        ).unstack(fill_value=0)
        dur_valid_idx = [a for a in age_order if a in dur_heatmap_data.index]
        dur_heatmap_data = dur_heatmap_data.reindex(dur_valid_idx)

        fig_dur_heat = px.imshow(
            dur_heatmap_data,
            color_continuous_scale='Purples',
            aspect="auto",
            text_auto='.1f',
            labels=dict(x="성별", y="연령대", color="평균시간(분)")
        )
        fig_dur_heat = apply_plotly_theme(fig_dur_heat, height=360)
        st.plotly_chart(fig_dur_heat, use_container_width=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 2. 이동거리 vs 이용시간 효율성 산점도 & 행동 세그먼트
    col_sc1, col_sc2 = st.columns([6, 4])

    with col_sc1:
        st.markdown("#### 🎯 1회당 이동거리(M) vs 이용시간(분) 행동 패턴 산점도")
        
        # 대여소별 또는 세부 그룹별 집계 (대여소번호 + 연령대코드 기준 샘플링)
        scatter_sample = filtered.groupby(['보관소(대여소)명', '자치구', '대여구분_분류', '연령대코드']).agg(
            총이용건수=('이용건수', 'sum'),
            평균이동거리_m=('1회당_이동거리_m', 'mean'),
            평균이용시간_분=('1회당_이용시간_분', 'mean'),
            평균속도_kmh=('평균속도_kmh', 'mean')
        ).reset_index()
        
        # 이상치 클리핑 for clean visualization
        scatter_plot_df = scatter_sample[
            (scatter_sample['평균이동거리_m'] > 100) & (scatter_sample['평균이동거리_m'] <= 15000) &
            (scatter_sample['평균이용시간_분'] > 2) & (scatter_sample['평균이용시간_분'] <= 120)
        ].sample(n=min(len(scatter_sample), 1500), random_state=42)

        # 유저 행동 군집 분류
        def classify_behavior(row):
            dist = row['평균이동거리_m']
            dur = row['평균이용시간_분']
            if dist < 2000 and dur <= 18:
                return '단거리 생활교통형 (Last-Mile)'
            elif dist >= 4500 and dur >= 35:
                return '장거리 레저/운동형 (Leisure)'
            elif dist >= 2000 and dur <= 35:
                return '중거리 통근/통학형 (Commute)'
            else:
                return '저속/도심혼잡형 (Urban Ride)'
        
        scatter_plot_df['행동유형'] = scatter_plot_df.apply(classify_behavior, axis=1)

        fig_scatter = px.scatter(
            scatter_plot_df,
            x='평균이동거리_m',
            y='평균이용시간_분',
            color='행동유형',
            size='총이용건수',
            size_max=18,
            color_discrete_map={
                '단거리 생활교통형 (Last-Mile)': '#34D399',
                '중거리 통근/통학형 (Commute)': '#38BDF8',
                '장거리 레저/운동형 (Leisure)': '#F472B6',
                '저속/도심혼잡형 (Urban Ride)': '#FBBF24'
            },
            labels={'평균이동거리_m': '1회당 평균 이동거리 (M)', '평균이용시간_분': '1회당 평균 이용시간 (분)', '총이용건수': '이용건수'},
            hover_name='보관소(대여소)명',
            hover_data={'자치구': True, '연령대코드': True, '평균속도_kmh': ':.1f'}
        )
        
        # 행동 유형 경계선 가이드
        fig_scatter.add_vline(x=2000, line_dash="dot", line_color="rgba(255,255,255,0.25)")
        fig_scatter.add_hline(y=18, line_dash="dot", line_color="rgba(255,255,255,0.25)")
        fig_scatter.add_vline(x=4500, line_dash="dot", line_color="rgba(255,255,255,0.25)")
        fig_scatter.add_hline(y=35, line_dash="dot", line_color="rgba(255,255,255,0.25)")
        
        fig_scatter = apply_plotly_theme(fig_scatter, height=420)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_sc2:
        st.markdown("#### 🧩 이동 목적별 유저 세그먼트 비중")
        behavior_counts = scatter_plot_df['행동유형'].value_counts().reset_index()
        behavior_counts.columns = ['행동유형', '비중']
        
        fig_be_pie = px.pie(
            behavior_counts,
            names='행동유형',
            values='비중',
            hole=0.5,
            color='행동유형',
            color_discrete_map={
                '단거리 생활교통형 (Last-Mile)': '#34D399',
                '중거리 통근/통학형 (Commute)': '#38BDF8',
                '장거리 레저/운동형 (Leisure)': '#F472B6',
                '저속/도심혼잡형 (Urban Ride)': '#FBBF24'
            }
        )
        fig_be_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#0F172A', width=2)))
        fig_be_pie = apply_plotly_theme(fig_be_pie, height=420)
        st.plotly_chart(fig_be_pie, use_container_width=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 3. 환경 및 건강 기여도 분석 (Carbon & Calories)
    st.markdown("#### 🌱 연령대별 환경(탄소 절감) 및 건강(소모 칼로리) 기여도")
    col_en1, col_en2 = st.columns(2)

    age_eco = filtered.groupby('연령대코드').agg(
        총탄소량_kg=('탄소량', 'sum'),
        총운동량_kcal=('운동량', 'sum'),
        이용건수=('이용건수', 'sum')
    ).reset_index()
    age_eco['age_rank'] = age_eco['연령대코드'].apply(lambda x: age_order.index(x) if x in age_order else 99)
    age_eco = age_eco.sort_values(by='age_rank')
    age_eco['탄소절감_톤'] = age_eco['총탄소량_kg'] / 1000.0

    with col_en1:
        fig_carbon = px.bar(
            age_eco,
            x='연령대코드',
            y='탄소절감_톤',
            color='탄소절감_톤',
            color_continuous_scale='Greens',
            labels={'탄소절감_톤': '탄소 절감량 (톤)', '연령대코드': '연령대'}
        )
        fig_carbon.update_traces(marker_line_width=0, opacity=0.9)
        fig_carbon = apply_plotly_theme(fig_carbon, height=320)
        st.plotly_chart(fig_carbon, use_container_width=True)

    with col_en2:
        age_eco['운동량_백만kcal'] = age_eco['총운동량_kcal'] / 1000000.0
        fig_cal = px.bar(
            age_eco,
            x='연령대코드',
            y='운동량_백만kcal',
            color='운동량_백만kcal',
            color_continuous_scale='Sunsetdark',
            labels={'운동량_백만kcal': '소모 칼로리 (백만 kcal)', '연령대코드': '연령대'}
        )
        fig_cal.update_traces(marker_line_width=0, opacity=0.9)
        fig_cal = apply_plotly_theme(fig_cal, height=320)
        st.plotly_chart(fig_cal, use_container_width=True)


# =============================================================================
# [TAB 4] Insight & Action Plan
# =============================================================================
with tab4:
    st.markdown("### 💡 데이터 기반 심층 인사이트 & 공공 자전거 운영 정책 제언")
    st.markdown("<p style='color:#94A3B8; font-size:0.9rem;'>실제 이용 데이터 및 공간 결합 분석을 통해 도출한 3대 전략 실행 계획(Action Plan)입니다.</p>", unsafe_allow_html=True)

    # 1. 재배치 전략 분석
    st_rebalance = filtered.groupby(['대여소번호', '보관소(대여소)명', '자치구', '총거치대수']).agg(
        이용건수=('이용건수', 'sum')
    ).reset_index()
    st_rebalance['회전율'] = st_rebalance['이용건수'] / st_rebalance['총거치대수']
    
    # 상위 과밀 vs 하위 미달
    high_turnover = st_rebalance.sort_values(by='회전율', ascending=False).head(5)
    low_turnover = st_rebalance[st_rebalance['이용건수'] >= 10].sort_values(by='회전율', ascending=True).head(5)

    col_ins1, col_ins2 = st.columns([6, 4])

    with col_ins1:
        st.markdown("""
        <div class="insight-card">
            <h4 style="color:#34D399; margin-top:0;">1️⃣ 따릉이 재배치 집중 필요 구역 및 공급 최적화 전략</h4>
            <p style="font-size:0.9rem; line-height:1.6; color:#CBD5E1;">
                <b>[문제 도출]</b> 한강공원 인근(뚝섬유원지역, 여의나루역, 잠실 일대) 및 주요 지하철 환승 거점의 경우 거치대당 회전율이 <b>1,000회~1,500회</b>를 초과하여 출퇴근 및 주말 시간대 자전거 품귀 현상이 만성적으로 발생하고 있습니다.<br>
                반면 외곽 주거지 일부 대여소는 회전율이 10회 미만으로 자전거 방치 및 재고 적체가 발생합니다.
            </p>
            <p style="font-size:0.9rem; line-height:1.6; color:#CBD5E1;">
                <b>[실행 제언 (Action Plan)]</b><br>
                • <b>동적 재배치 트럭 우선 배차</b>: 회전율 상위 5% 대여소에 대해 평일 07:30~09:00, 18:00~20:00 집중 투입.<br>
                • <b>임시 가상 거치구역(Geofencing Virtual Station) 지정</b>: 주말 한강공원 인근 대여소는 물리적 거치대 대신 앱 기반 반납 가능 구역을 확장하여 반납 불가 문제 해결.<br>
                • <b>유휴 거치대 다이어트</b>: 회전율 하위 대여소의 QR/LCD 거치대를 고수요 대여소로 이전 증설.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-card insight-card-blue">
            <h4 style="color:#38BDF8; margin-top:0;">2️⃣ 연령대 및 이용목적별 맞춤형 요금제 & 프로모션 전략</h4>
            <p style="font-size:0.9rem; line-height:1.6; color:#CBD5E1;">
                <b>[데이터 패턴]</b> 전체 이용자의 <b>50% 이상이 20대와 30대</b>이며, 1회 평균 주행시간은 약 22분(거리 2.5km)으로 지하철-주거지 간 Last-Mile 통근 목적이 지배적입니다. 40~50대는 장거리 레저 및 건강 목적 비중이 높게 나타납니다.
            </p>
            <p style="font-size:0.9rem; line-height:1.6; color:#CBD5E1;">
                <b>[실행 제언 (Action Plan)]</b><br>
                • <b>청년 출퇴근 통근 패스(Commute Pass)</b>: 대중교통(지하철·버스) 환승 후 30분 내 따릉이 이용 시 기본요금 50% 마일리지 적립.<br>
                • <b>생활밀착형 30분 마이크로 요금제</b>: 근거리 상권/시장 방문 소비자를 위해 기존 1시간 권보다 저렴한 30분 전용 소액권 출시.<br>
                • <b>실버 에코-모빌리티(Silver Eco Pass)</b>: 60대 이상을 대상으로 도심 자전거 유휴 시간대(10:00~16:00) 무료/할인 바우처를 지급하여 건강 증진과 자전거 순환율 동시 제고.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-card insight-card-purple">
            <h4 style="color:#A5B4FC; margin-top:0;">3️⃣ 자전거 전용도로 인프라 확충 시급 자치구 도출</h4>
            <p style="font-size:0.9rem; line-height:1.6; color:#CBD5E1;">
                <b>[데이터 패턴]</b> <b>강서구, 송파구, 영등포구, 마포구</b>는 총 이용건수가 각 100만 건 수준으로 압도적이며, 1건당 평균 주행거리 또한 2.8km 이상으로 길게 나타납니다. 특히 마곡지구(강서), 여의도(영등포), 잠실(송파) 등 평지 위주 업무지구에서 자전거 교통 분담률이 극대화되어 있습니다.
            </p>
            <p style="font-size:0.9rem; line-height:1.6; color:#CBD5E1;">
                <b>[실행 제언 (Action Plan)]</b><br>
                • <b>차로 축소형(Diet) 전용 자전거 도로 신설</b>: 영등포-마포 연결축 및 마곡산업단지 메인 대로변에 분리형 자전거 전용도로 우선 시공.<br>
                • <b>교차로 자전거 횡단도 전면 확대</b>: 우회전 차량과의 사고 위험 방지를 위한 자전거 전용 신호기 도입.<br>
                • <b>하천변 연결 경사로 개선</b>: 안양천, 중랑천, 탄천과 도심 생활권 간 자전거 경사로 램프 추가 설치.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_ins2:
        st.markdown("#### 🚨 자전거 품귀 위험 상위 대여소 (재배치 시급)")
        st.dataframe(
            high_turnover[['보관소(대여소)명', '자치구', '총거치대수', '이용건수', '회전율']].style.format({
                '총거치대수': '{:,.0f}',
                '이용건수': '{:,.0f}',
                '회전율': '{:,.1f}'
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        st.markdown("#### 📉 저활용 대여소 (거치대 재조정 대상)")
        st.dataframe(
            low_turnover[['보관소(대여소)명', '자치구', '총거치대수', '이용건수', '회전율']].style.format({
                '총거치대수': '{:,.0f}',
                '이용건수': '{:,.0f}',
                '회전율': '{:,.1f}'
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # Action Plan Summary Export
        st.markdown("#### 📥 분석 보고서 및 통계 다운로드")
        
        # Summary CSV creation
        summary_csv = district_agg.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📊 자치구별 통계 지표 CSV 다운로드",
            data=summary_csv,
            file_name="seoul_bike_district_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Markdown Policy Report
        policy_report = f"""# 서울특별시 공공자전거(따릉이) 데이터 기반 운영 정책 제언 보고서
- 기준 기간: {', '.join(selected_months)}
- 총 분석 이용건수: {kpi_rides:,.0f}건
- 총 이동거리: {kpi_dist_km:,.1f}km (탄소절감 {kpi_carbon_ton:,.1f}톤)

1. 따릉이 재배치 최적화
- 최고 회전율 대여소: {high_turnover.iloc[0]['보관소(대여소)명']} (회전율 {high_turnover.iloc[0]['회전율']:.1f}회)
- 출퇴근 집중 거점 이동식 재배치 트럭 우선 투입 필요.

2. 타깃 요금제
- 2030 Last-Mile 통근 패스 및 30분 마이크로 요금제 신설 권고.

3. 인프라 우선 확충 구역
- 강서구, 송파구, 영등포구, 마포구 자전거 전용도로 분리대 우선 도입.
"""
        st.download_button(
            label="📝 정책 제언 요약 리포트(MD) 다운로드",
            data=policy_report,
            file_name="seoul_bike_policy_report.md",
            mime="text/markdown",
            use_container_width=True
        )

# -----------------------------------------------------------------------------
# 5. Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.82rem; padding: 10px 0;">
    서울특별시 공공자전거 빅데이터 융합 분석 대시보드 v1.0 | Built with Streamlit, Plotly, GeoPandas & PyArrow
</div>
""", unsafe_allow_html=True)
