# 🚲 서울특별시 공공자전거(따릉이) 빅데이터 융합 분석 대시보드

서울시 공공자전거(따릉이) 이용 데이터와 대여소 마스터 정보, 시군구 공간 데이터(GIS)를 융합한 인터랙티브 분석 대시보드입니다.

---

## 📌 주요 기능
1. **Executive Summary (핵심 현황)**: 총 대여건수, 탄소 절감량, 운동량, 월별 이용 추이, 대여소 TOP 10
2. **Geospatial Analysis (공간 분석)**: 자치구별 대여 현황 단계구분도(Choropleth), 대여소별 버블맵, 거치대 효율성 분석
3. **User Behavior & Segmentation (이용자 행동)**: 연령대/성별 이용 히트맵, 출퇴근 vs 여가 분석, 탄소 절감 효과
4. **Insight & Action Plan (정책 제언)**: 재배치 최적화 전략, 차등 요금제 제안, 자전거 도로 확충 우선순위

---

## 🛠 기술 스택
- **Language**: Python 3.10+
- **Framework**: Streamlit
- **Data & GIS**: Pandas, GeoPandas, Shapely, PyArrow
- **Visualization**: Plotly, Altair

---

## 🚀 실행 방법

### 1. 가상환경 생성 및 활성화
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

### 2. 필수 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 대시보드 실행
```bash
streamlit run app.py
```
