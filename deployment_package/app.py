import streamlit as st
import pandas as pd
from datetime import datetime
import config
from kra_scraper import KRAScraper
from quantitative_analysis import QuantitativeAnalyzer
from gemini_analyzer import GeminiAnalyzer
from pattern_analyzer import PatternAnalyzer
from storage_manager import StorageManager

# 페이지 설정
st.set_page_config(page_title="KRA AI 경마 분석기", page_icon="🐎", layout="wide")

# 캐싱 적용 (속도 향상)
@st.cache_data(ttl=3600)
def load_entries(date, meet):
    scraper = KRAScraper()
    return scraper.fetch_race_entries(date, meet)

@st.cache_data(ttl=3600)
def load_training(date, meet):
    scraper = KRAScraper()
    return scraper.fetch_training_for_week(date, meet)

# 스타일 커스텀
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .sidebar .sidebar-content { background: #ffffff }
    h1 { color: #1e3d59; } 
    h2, h3 { color: #1e3d59; }
    .stButton>button {
        color: white;
        background-color: #ff6e40;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# 제목
st.title("🐎 KRA AI 경마 분석기")
st.markdown("출전표를 먼저 조회한 후, 원하는 경주를 선택하여 **심층 분석**하세요.")

# 사이드바 입력
st.sidebar.header("🔍 설정")
today = datetime.now().strftime("%Y%m%d")
race_date = st.sidebar.text_input("경주 일자 (YYYYMMDD)", value=today)

# [FIX] 지역 코드 일관성 유지 (1:서울, 2:제주, 3:부산)
meet = st.sidebar.selectbox("경마장", ["1 (서울)", "2 (제주)", "3 (부산경남)"])
meet_code = meet.split()[0]
track_condition = st.sidebar.text_input("주로 상태 (AI 참고용)", value="", placeholder="예: 불량, 건조 5%")

# [NEW] AI 모델 선택
st.sidebar.markdown("---")
st.sidebar.header("🤖 AI 설정")
model_choice = st.sidebar.selectbox("Gemini 모델", 
                                  ["Pro (고정밀)", "Flash (빠름/과거분석)"], 
                                  index=0)
selected_model = config.GEMINI_PRO_MODEL if "Pro" in model_choice else config.GEMINI_FLASH_MODEL

# [NEW] API 키 관리 (Persistent)
with st.sidebar.expander("🔑 API 키 설정"):
    g_api_input = st.text_input("Gemini API Key", value=config.GEMINI_API_KEY, type="password")
    k_api_input = st.text_input("KRA API Key (Optional)", value=config.KRA_API_KEY, type="password")
    if st.button("💾 API 키 저장"):
        StorageManager.update_env("GEMINI_API_KEY", g_api_input)
        StorageManager.update_env("KRA_API_KEY", k_api_input)
        st.success("API 키가 저장되었습니다! (재시작 권장)")

# [NEW] 파일 업로드 (User Request)
st.sidebar.markdown("---")
st.sidebar.header("📂 자료 업로드 (선택)")
uploaded_file = st.sidebar.file_uploader("경주 성적표/예상지 (PDF/Excel)", type=["pdf", "xlsx", "xls", "txt"])

if uploaded_file:
    from file_parser import FileParser
    with st.spinner("파일 분석 중..."):
        file_text = FileParser.parse_file(uploaded_file)
        if file_text.startswith("비정상") or file_text.startswith("PDF에서"):
            st.sidebar.error(file_text)
        else:
            st.sidebar.success(f"파일 로드 완료! ({len(file_text)}자)")
            st.session_state['steward_report_ext'] = file_text[:15000] # API 토큰 제한 고려 (약 1.5만자)

# 1. 출전표 조회 (스크래핑 - Single Race)
# [CHANGE] API 대신 웹 스크래핑으로 변경 (User Request: "API 안되니까 기능 없애고 스크래핑만")
def update_race_no():
    # 경주번호 변경 시 기존 데이터 초기화 (자동 로딩 유도)
    st.session_state['scraped_entries'] = None
    st.session_state['entries_loaded'] = True # 로딩 트리거

race_no_input = st.sidebar.number_input("경주 번호", min_value=1, max_value=20, value=1, key='race_no_input', on_change=update_race_no)

if st.sidebar.button("🔍 경주 확정표 조회 (스크래핑)"):
    st.session_state['entries_loaded'] = True
    st.session_state['race_date'] = race_date
    st.session_state['meet_code'] = meet_code
    st.session_state['race_no'] = str(race_no_input)
    st.session_state['scraped_entries'] = None # 초기화

# [NEW] 분석 기록 세션 초기화
if 'history' not in st.session_state:
    st.session_state['history'] = []

# 메인 로직
# 경주번호 입력값(session_state.race_no_input)을 우선 사용
current_race_no = str(st.session_state.get('race_no_input', 1))
if st.session_state.get('race_no') != current_race_no:
    st.session_state['race_no'] = current_race_no
    # 입력값이 바뀌었으면 자동 로딩 트리거가 이미 update_race_no에서 처리됨

if st.session_state.get('entries_loaded'):
    r_no = st.session_state.get('race_no', '1')
    
    # 캐싱 없이 직접 스크래핑 (최신 데이터 보장)
    scraper = KRAScraper()
    
    # 이미 스크래핑된 데이터가 없거나 경주번호가 바뀌었으면 새로 로드
    if st.session_state.get('scraped_entries') is None or st.session_state.get('last_race_no') != r_no:
        with st.spinner(f"{race_date} {meet} {r_no}경주 출전표를 가져오는 중..."):
            entries = scraper.scrape_race_entry_page(race_date, meet_code, r_no)
            st.session_state['scraped_entries'] = entries
            st.session_state['last_race_no'] = r_no
    else:
        entries = st.session_state['scraped_entries']
    
    if entries is None or entries.empty:
        st.error(f"❌ {r_no}경주 출전표 데이터가 없습니다. (날짜/경마장/경주번호 확인 필요)")
        st.info("💡 팁: 마사회 홈페이지에 확정표가 올라오지 않았을 수 있습니다.")
    else:
        st.success(f"✅ {r_no}경주 출전표 로드 완료 ({len(entries)}두)")
        

        # 탭 구성: 현재 분석 | 분석 기록 | 고배당 패턴 | 백테스팅 (NEW)
        tab1, tab2, tab3, tab4 = st.tabs([f"📊 {r_no}경주 분석", "📜 분석 기록", "📈 고배당 패턴", "🧪 백테스팅"])

        # [NEW] Tab 3: 고배당 패턴 분석
        with tab3:
            st.markdown("### 🕵️‍♂️ 최근 3개월 고배당(복승 50배+/삼복 100배+) 패턴 분석")
            st.info("최근 90일간 금/토/일 경주 결과를 분석하여 고배당 경주의 공통점을 찾습니다.")
            
            p_anal = PatternAnalyzer()
            
            if st.button("🚀 최근 3개월 고배당 패턴 분석 시작", key="btn_pattern"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(p, msg):
                    progress_bar.progress(p)
                    status_text.text(msg)
                
                with st.spinner("데이터 수집 중... (약 1~2분 소요)"):
                    result = p_anal.run_analysis(days=90, progress_callback=update_progress)
                
                st.success(result["msg"])
                
                if not result["high_div_races"].empty:
                    df = result["high_div_races"]
                    summary = result["summary"]
                    
                    # Store in session state for Gemini analysis
                    st.session_state['pattern_result'] = result
                    
                    # Display Stats
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("평균 복승 배당", f"{summary['avg_qui']:.1f}배")
                    c2.metric("평균 삼복승 배당", f"{summary['avg_trio']:.1f}배")
                    c3.metric("인기 1위마 탈락률", f"{summary.get('fav1_out_rate', 0):.1f}%")
                    c4.metric("우승마 평균 인기", f"{summary.get('avg_w_odds_rank', 0):.1f}위")
                    
                    st.markdown("#### 💡 분석을 통한 실전 베팅 팁")
                    t1, t2 = st.columns(2)
                    with t1:
                        st.info(f"**패턴 1**: 고배당 경주의 인기 1위마는 **{summary.get('fav1_out_rate', 0):.1f}%** 확률로 3위 안에 못 들었습니다. 인기 1위마를 과감히 제외하는 전략이 유효할 수 있습니다.")
                    with t2:
                        st.info(f"**패턴 2**: 고배당 우승마의 평균 인기 순위는 **{summary.get('avg_w_odds_rank', 0):.1f}위**입니다. 인기 5~10위권 사이의 말을 눈여겨보세요.")

                    st.markdown("#### 1. 고배당 경주 목록")
                    st.dataframe(df)
                    
                    st.markdown("#### 2. 우승마 특성 (Top 5)")
                    k1, k2, k3 = st.columns(3)
                    with k1:
                        st.write("**기수**")
                        st.write(summary['top_jockeys'])
                    with k2:
                        st.write("**조교사**")
                        st.write(summary['top_trainers'])
                    with k3:
                        st.write("**부담중량**")
                        st.write(summary['weight_dist'])
            
            # Gemini Strategy Analysis
            if st.session_state.get('pattern_result'):
                st.markdown("---")
                if st.button("🤖 Gemini에게 필승 전략 분석 의뢰", key="btn_gemini_pattern"):
                    if not config.GEMINI_API_KEY:
                        st.error("API Key가 설정되지 않았습니다.")
                    else:
                        with st.spinner("Gemini가 데이터를 분석하고 전략을 수립 중입니다..."):
                            res = st.session_state['pattern_result']
                            df = res["high_div_races"]
                            summ = res["summary"]
                            
                            # Construct Prompt
                            prompt = f"""
                            최근 3개월간 한국 경마에서 발생한 고배당(복승 50배+, 삼복 100배+) 경주 데이터 통계입니다.
                            이 데이터를 바탕으로 사용자가 바로 참고할 수 있는 '실전 베팅 전략'을 수립해주세요.
                            
                            [통계 요약]
                            - 평균 복승 배당: {summ['avg_qui']:.1f}배 / 삼복승: {summ['avg_trio']:.1f}배
                            - 인기 1위마의 3위 이내 입성 실패율 (탈락률): {summ.get('fav1_out_rate', 0):.1f}%
                            - 고배당 우승마의 평균 인기 순위: {summ.get('avg_w_odds_rank', 0):.1f}위
                            - 주요 우승 기수: {summ['top_jockeys']}
                            - 주요 우승 조교사: {summ['top_trainers']}
                            
                            [상세 경주 데이터 (샘플 20건)]
                            {df.head(20).to_string()}
                            
                            위 데이터를 분석하여 다음을 포함한 '베팅 가이드'를 작성하세요:
                            1. **축마 선정 전략**: 인기마를 믿어야 할 때와 버려야 할 때의 구분.
                            2. **복병마 타겟팅**: 인기 몇 순위권의 어떤 특징(부담중량 등)을 가진 말을 노려야 하는지.
                            3. **구체적인 조합 방법**: "인기 X위마를 축으로 세우고, 기수 Y가 기승한 인기 외 말을 Z두 조합하라"는 식의 실전 예시.
                            """
                            
                            try:
                                import google.genai as genai
                                from google.genai import types
                                client = genai.Client(api_key=config.GEMINI_API_KEY)
                                check_response = client.models.generate_content(
                                    model=config.GEMINI_MODEL,
                                    contents=prompt,
                                    config=types.GenerateContentConfig(temperature=0.7)
                                )
                                st.markdown("### 🧠 Gemini의 고배당 공략 리포트")
                                st.write(check_response.text)
                            except Exception as e:
                                st.error(f"Gemini 분석 중 오류 발생: {e}")

        # [NEW] Tab 4: 3개월 지역별 백테스팅
        with tab4:
            st.markdown("### 🧪 3개월 지역별 백테스팅")
            st.info(f"선택한 지역({meet})에 대해 최근 90일간 분석 적중률과 수익률을 검증합니다.")
            
            c1, c2 = st.columns(2)
            with c1:
                bt_start = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
                bt_end = datetime.now().strftime("%Y%m%d")
                st.write(f"**대상 기간**: {bt_start} ~ {bt_end}")
            
            if st.button(f"🚀 {meet} 3개월 백테스팅 시작", key="btn_backtest"):
                from backtester import Backtester
                bt = Backtester()
                
                status_box = st.empty()
                progress_bar = st.progress(0)
                
                with st.spinner("과거 데이터 수집 및 시뮬레이션 중... (수 분이 소요될 수 있습니다)"):
                    try:
                        # Backtester.run 이 복잡하므로 여기서는 간단한 진행 상황만 표시
                        res = bt.run(bt_start, bt_end, meet_code)
                        
                        if res:
                            st.success("✅ 백테스팅 완료!")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("연승(Top3) 적중률", f"{res.get('hit_rate', 0):.1f}%")
                            m2.metric("VETO 정확도", f"{res.get('veto_accuracy', 0):.1f}%")
                            m3.metric("대상 경주 수", f"{res.get('total_races', 0)}건")
                            
                            st.info("💡 상세 결과는 콘솔(터미널) 로그에서 확인해 주세요.")
                        else:
                            st.warning("데이터가 부족하여 결과를 도출하지 못했습니다.")
                    except Exception as e:
                        st.error(f"백테스팅 중 오류 발생: {e}")

        with tab2:
            st.markdown("### 📜 나의 분석 기록 (History)")
            # [NEW] 로컬 파일에서 히스토리 로드
            db_history = StorageManager.load_all_history()
            
            if not db_history:
                st.info("아직 저장된 분석 기록이 없습니다.")
            else:
                for idx, item in enumerate(db_history):
                    with st.expander(f"[{item.get('saved_at', 'Unknown')}] {item['race_date']} {item['meet']} {item['race_no']}경주 분석 결과"):
                        st.markdown(f"**🏆 추천**: {item['summary']}")
                        st.dataframe(pd.DataFrame(item['result_list']))
                        if item.get('gemini_comment'):
                            st.write(item['gemini_comment'])

        with tab1:
            # [DISPLAY] 출전표 표시
            display_df = entries[['hrNo', 'hrName', 'jkName', 'trName', 'remark', 'rating']].copy()
            st.dataframe(display_df)
            
            # [ACTION] 분석 버튼
            analyze_key = f"analyze_{r_no}"
            if st.button(f"🚀 {r_no}경주 심층 분석 실행", key=analyze_key):
                analyzer = QuantitativeAnalyzer()
                gemini = GeminiAnalyzer()
        
                with st.spinner(f"{r_no}경주 데이터를 정밀 분석 중입니다..."):
                    # 1. 조교 데이터 (Lazy Load)
                    training_data = load_training(race_date, meet_code)
                    
                    # 2. 말 상세 데이터 일괄 수집 (10회 전적 탭 + 심판리포트 탭)
                    score_data = scraper.scrape_race_10score(race_date, meet_code, r_no)
                    steward_data = scraper.scrape_steward_reports(race_date, meet_code, r_no)
                    
                    details_map = {}
                    for _, row in entries.iterrows():
                        h_no = str(row.get("hrNo", ""))
                        hist = score_data.get(h_no, [])
                        steward = steward_data.get(h_no, [])
                        details_map[h_no] = {'hist': hist, 'med': [], 'steward': steward}

                    # 3. 정량 분석
                    training_list = []
                    if isinstance(training_data, pd.DataFrame) and not training_data.empty:
                        training_list = training_data.to_dict('records')
                    elif isinstance(training_data, list):
                        training_list = training_data
                    
                    analyses = []
                    for _, row in entries.iterrows():
                        hr_no = str(row.get("hrNo", ""))
                        hr_name = str(row.get("hrName", "?"))
                        import re
                        w_str = str(row.get("wgBudam", "0"))
                        # 숫자와 소수점만 남기고 제거 (예: *52.5 -> 52.5)
                        w_clean = re.sub(r'[^0-9.]', '', w_str)
                        burden_weight = float(w_clean) if w_clean else 0.0
                        
                        # [FIX] 체중(weight) 컬럼이 있으면 사용, 없으면 0.0 (부담중량 아님)
                        bw_str = str(row.get("weight", "0"))
                        bw_clean = re.sub(r'[^0-9.]', '', bw_str)
                        current_body_weight = float(bw_clean) if bw_clean else 0.0

                        remark = row.get("remark", "") # 스크래핑된 특이사항
                        
                        dt = details_map.get(hr_no, {'hist':[], 'med':[]})
                        # 조교 연결
                        t = [tr for tr in training_list if str(tr.get('hrNo', '')) == hr_no]
                        
                        res = analyzer.analyze_horse(hr_name, dt['hist'], t, 
                                                     current_weight=current_body_weight, 
                                                     steward_reports=dt.get('steward', []))
                        res['medical'] = dt['med']
                        res['remark'] = remark
                        res['steward_reports'] = dt.get('steward', [])
                        res['hrNo'] = hr_no  # 마번 보관
                        analyses.append(res)
                    
                    ranked = analyzer.rank_horses(analyses)
                    
                    # 결과 표시
                    st.markdown("### 📊 분석 결과")
                    df_res = pd.DataFrame(ranked)
                    if not df_res.empty:
                        # 삼복승 추천 생성
                        trio = analyzer.generate_trio_picks(ranked, entries)
                        
                        # 삼복승 추천 표시 (최상단)
                        st.markdown("### 🎯 삼복승 추천")
                        col_trio, col_detail = st.columns([2, 3])
                        with col_trio:
                            st.markdown(f"**축마**: `{','.join(trio['axis'])}번`")
                            st.markdown(f"**상대마**: `{','.join(trio['partners'])}번`")
                            st.markdown(f"### 총 {trio['num_bets']}조합")
                        with col_detail:
                            # 실제 구매 조합 표시
                            st.markdown("**구매 조합 (삼복승):**")
                            combo_text = " / ".join(trio['combinations'])
                            st.code(combo_text)
                            if trio.get('dark_horses'):
                                st.markdown("**💣 복병 마필:**")
                                for dh in trio['dark_horses']:
                                    for reason in dh['reasons']:
                                        st.markdown(f"- **{dh['hrNo']}번 {dh['horse_name']}**: {reason}")
                        
                        st.markdown("---")
                        
                        # 결과 테이블 (방해 보너스/복병 포함)
                        display_cols = ['rank', 'hrNo', 'horse_name', 'total_score', 
                                        'speed_score', 'interference_score', 'g1f_avg', 'g1f_vector']
                        available_cols = [c for c in display_cols if c in df_res.columns]
                        st.dataframe(df_res[available_cols])
                        
                        # 특이사항/VETO/심판리포트
                        c1, c2 = st.columns(2)
                        with c1:
                             st.write("**⚠️ 특이사항 (출전표/기록)**")
                             for r in ranked:
                                 if r.get('remark') and str(r['remark']) != 'nan':
                                     st.warning(f"**{r['horse_name']}**: {r['remark']}")
                                 if r.get('medical'):
                                     st.warning(f"**{r['horse_name']}**: {', '.join(r['medical'][:2])}...")
                        with c2:
                            st.write("**🚫 분석 제외 (VETO)**")
                            for r in ranked:
                                if r.get('veto'):
                                    st.error(f"**{r['horse_name']}**: {r['veto_reason']}")
                        
                        # 심판리포트 섹션
                        st.markdown("### 📋 심판리포트 (주행 방해/진로 문제)")
                        has_reports = False
                        for r in ranked:
                            reports = r.get('steward_reports', [])
                            if reports:
                                has_reports = True
                                with st.expander(f"#{r.get('rank', '?')} {r['horse_name']} ({len(reports)}건)"):
                                    for rpt in reports:
                                        st.markdown(f"- **{rpt['date']}**: {rpt['report']}")
                        if not has_reports:
                            st.info("심판리포트 기록이 없습니다.")
                        
                        # [DEBUG] 데이터 확인용 -> 유저용 상세 보기로 전환
                        with st.expander("📊 분석 데이터 상세 보기 (클릭)"):
                            if ranked:
                                top = ranked[0]
                                st.write(f"**{top['horse_name']}** 데이터 예시:")
                                st.json(top)
    
                        # 4. Gemini (Optional Chain)
                        if config.GEMINI_API_KEY:
                            st.markdown("---")
                            st.markdown("### 🤖 AI 종합 의견")
                            med_map = {r['horse_name']: r.get('medical', []) for r in ranked}
                            
                            # [NEW] 업로드된 심판 리포트/예상지 내용 반영
                            ext_report = st.session_state.get('steward_report_ext', "")
                            
                            # 스크래핑된 특이사항 + 심판리포트를 Gemini에게 전달
                            scraped_remarks = []
                            for r in ranked:
                                if r.get('remark') and str(r['remark']) != 'nan':
                                    scraped_remarks.append(f"- {r['horse_name']}: {r['remark']}")
                            
                            if scraped_remarks:
                                ext_report += "\n\n[출전표 특이사항]\n" + "\n".join(scraped_remarks)
                            
                            # 심판리포트도 Gemini에게 전달
                            steward_lines = []
                            for r in ranked:
                                for rpt in r.get('steward_reports', []):
                                    steward_lines.append(f"- {r['horse_name']}({rpt['date']}): {rpt['report']}")
                            if steward_lines:
                                ext_report += "\n\n[심판리포트 - 주행방해/진로문제 기록]\n" + "\n".join(steward_lines)
                            
                            g_res = gemini.analyze_race(r_no, ranked, ext_report, "", track_condition, med_map, 
                                                       race_date=race_date, model_override=selected_model)
                            
                            st.info(f"💡 **Case**: {g_res.get('case_type')}")
                            st.caption(f"🤖 사용 모델: `{g_res.get('model_used', 'Unknown')}`")
                            st.write(g_res.get('final_comment'))
                            
                            k1, k2 = st.columns(2)
                            with k1:
                                st.error(f"🏆 강선축: {', '.join([x['horse'] for x in g_res.get('strong_axis', [])])}")
                            with k2:
                                st.warning(f"💣 복병: {', '.join([x['horse'] for x in g_res.get('dark_horses', [])])}")
    
                        # [NEW] 분석 결과 자동 저장 (Persistent)
                        summary_text = f"축:{','.join(trio['axis'])} / 도전:{','.join(trio['partners'])}"
                        
                        save_data = {
                            "race_date": race_date, 
                            "meet": meet, 
                            "race_no": r_no,
                            "summary": summary_text,
                            "result_list": df_res[available_cols].to_dict('records'),
                            "gemini_comment": g_res.get('final_comment') if config.GEMINI_API_KEY else "AI 분석 미사용",
                            "model_used": g_res.get('model_used', 'None')
                        }
                        StorageManager.save_analysis(race_date, meet_code, r_no, save_data)
                        st.success(f"✅ 분석 결과가 `data/history/{race_date}/{meet_code}/{r_no}.json`에 자동 저장되었습니다.")



else:
    st.info("👈 왼쪽 사이드바에서 [출전표 조회] 버튼을 먼저 눌러주세요.")
