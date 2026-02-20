"""
gemini_analyzer.py — Gemini API 정성 분석 모듈
파이썬 정량 분석 결과 + 심판 리포트 텍스트를 기반으로
강선축마, 복병, VETO마, Case 판정을 도출합니다.
"""
import json
from datetime import datetime

from google import genai
from google.genai import types

import config


# ─────────────────────────────────────────────
# 전문가 시스템 프롬프트
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 한국 경마 분석 전문가입니다. 아래 용어와 분석 기법을 완벽하게 이해하고 있습니다.

## 핵심 용어 (Core Terms)
- **F (Front/선행)**: 선두 주행 전법. 초반부터 앞서 달리는 마필.
- **M (Middle/선입)**: 내측 선입. 중단에서 경주하며 기회를 노리는 마필.
- **C (Chaser/리베로)**: 후방에서 추격하는 전법. 지구력이 핵심.
- **W (Wide/외곽)**: 외곽 주행. 불리한 바깥 코스로 주행하며 거리 손해를 봄.
  → **W 주행 후 입상 = 실제 능력이 매우 뛰어난 마필** (최고 가산점)

## S1F / G1F 분석
- **S1F**: 초반 200m 구간 기록. 선행력(출발 스피드) 지표.
- **G1F**: 종반 200m 구간 기록. 지구력(마무리 스피드) 지표.
- **G1F 벡터**: G1F 기록의 추세.
  - "Strong": 종반에도 속도 유지/가속 → 지구력 검증
  - "Maintaining": 약간 감속이나 유지 수준 → 양호
  - "Fading": 종반 탈진 패턴 → 지구력 의문

## 복기 데이터 키워드 (Steward Report)
분석 시 아래 키워드에 특히 주목하세요:
- **Blocked(진로 막힘)**: 능력 발휘 못함 → 다음 경주 반등 기대
- **W(외곽 주행)**: 거리 손해 → 입상하면 아주 높은 평가
- **출발 불량**: 게이트 문제 → 일시적 핸디캡, 실력과 무관
- **Strong Finish(강한 마무리)**: 종반 추임새 → 지구력 검증
- **Stumbled(비틀거림)**: 컨디션 문제 가능성

## 강선축마(Strong Axis) 판정 기준
다음 조건을 충족하면 **'예외 없는 축마'**로 지정:
1. 외곽(W) 주행의 불리함을 뚫고 입상한 마필
2. Strong Finish를 보인 마필 (특히 G1F 벡터 "Strong" 이상)
3. 진로 방해(Blocked)를 받고도 착순에 근접한 마필
4. 정량 점수 상위 + 조교 충실 + 체중 적정

## 장구 변화 분석
- **장구 추가(+)**: 혀끈, 그림자롤, 블링커 등 → 단점 보완 시도 = 승부 의지
- **장구 해지(-)**: 이전 장구 제거 → 제어력 자신감 or 변화 시도
→ 장구 변화의 맥락을 해석하여 마필 컨디션 변화 추론

## Case 판정 (경주 유형 분류)
- **Case A**: 독주형 — 강선행 1두만 존재, 단독 도주 가능
- **Case B**: 혼전형 — 선행마 2~3두 경합, 체력 소모전
- **Case C**: 강선행 1 + 약선행 다수 — 승부 경주, 축마 확정에 유리
- **Case D**: 추입 유리형 — 선행마 과다, 후방 추격마 기회

## 출력 형식
분석 결과를 반드시 아래 JSON 형식으로 출력하세요:
```json
{
    "race_no": 경주번호,
    "case_type": "Case A/B/C/D",
    "case_reason": "판정 근거 설명",
    "strong_axis": [
        {
            "horse": "마명",
            "reason": "선정 근거 (W 주행 돌파, G1F Strong 등)",
            "confidence": "상/중/하"
        }
    ],
    "dark_horses": [
        {
            "horse": "마명",
            "reason": "복병 근거 (Blocked 반등, 장구 변화 등)",
            "potential": "상/중/하"
        }
    ],
    "veto_horses": [
        {
            "horse": "마명",
            "reason": "VETO 사유 (체중 초과, 조교 부족 등)"
        }
    ],
    "final_comment": "종합 코멘트 및 추천 마권 구성 (예: 축마 A → 상대 B,C,D)"
}
```"""


class GeminiAnalyzer:
    """Gemini API 기반 정성 분석기"""

    def __init__(self):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        # self.model is distinct per call now

    def analyze_race(self, race_no: int,
                     quantitative_data: list[dict],
                     steward_report: str = "",
                     equipment_changes: str = "",
                     track_condition: str = "",
                     medical_history: dict = None,
                     race_date: str = "",
                     model_override: str = None) -> dict:
        """
        단일 경주에 대한 Gemini 정성 분석.

        Args:
            race_no: 경주 번호
            quantitative_data: 정량 분석 결과 리스트 (마필별)
            steward_report: 심판 리포트 텍스트 (복기 데이터)
            equipment_changes: 장구 변화 정보
            track_condition: 주로 상태 (예: "불량", "건조", "함수율 15%")
            medical_history: {마명: [진료내역, ...]} 딕셔너리
            race_date: 경주 일자
            model_override: 강제 모델 지정 (Flash/Pro)

        Returns:
            dict — 강선축마, 복병, VETO마, Case 판정 결과
        """
        # [NEW] Dynamic Model Selection
        selected_model = model_override if model_override else config.GEMINI_PRO_MODEL
        
        if not model_override and race_date:
            try:
                today_str = datetime.now().strftime("%Y%m%d")
                if race_date < today_str:
                    selected_model = config.GEMINI_FLASH_MODEL
            except Exception:
                pass # Fallback to Pro if date parsing fails

        # Default empty dict
        if medical_history is None:
            medical_history = {}
        # 정량 데이터를 읽기 쉬운 텍스트로 변환
        quant_text = self._format_quantitative(race_no, quantitative_data)

        # [NEW] 진료 내역 포맷팅
        medical_text = ""
        if medical_history:
            medical_text = "\n[주요 진료 내역 (최근 1년)]\n"
            for horse_name, history in medical_history.items():
                if history:
                    history_str = ", ".join(history)
                    medical_text += f"- {horse_name}: {history_str}\n"

        user_prompt = f"""
[분석 대상 경주 데이터]
경주 번호: {race_no}경주
주로 상태: {track_condition if track_condition else "정보 없음"}

[정량 분석 결과 (점수순)]
{quant_text}

[심판/복기 리포트]
{steward_report if steward_report else "심판 리포트 데이터 없음"}

[장구 변화]
{equipment_changes if equipment_changes else "장구 변화 정보 없음"}
{medical_text}

위 데이터를 바탕으로 우승마와 복병을 분석해주세요.

---
위 데이터를 종합하여:
1. **Case 판정** (A/B/C/D)을 먼저 수행하세요.
2. **주로(Track) 변수 분석**:
   - 현재 주로 상태(함수율 등)가 정보에 있다면 반영하고, 없다면 기록(G1F)을 통해 주로 빠르기를 추론하세요.
   - **현재 입력된 주로 상태: {track_condition if track_condition else "정보 없음 (기록으로 추론)"}**
   - 주로가 빠르다면 선행 유리, 무겁다면 추입 유리 등을 고려하여 유불리를 판단하세요.
3. **강선축마(Strong Axis)**를 확정하세요 (W 돌파 입상, Strong Finish, Blocked 반등 등).
4. **복병(Dark Horse)**을 선별하세요 (장구 변화, 과소평가 마필 등).
5. **VETO 마필**을 명시하세요 (체중/조교 결격).
6. 최종 마권 구성 추천을 작성하세요.

JSON 형식으로 응답하세요."""

        try:
            response = self.client.models.generate_content(
                model=selected_model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=config.GEMINI_TEMPERATURE,
                    max_output_tokens=config.GEMINI_MAX_TOKENS,
                )
            )

            result_text = response.text.strip()

            # JSON 파싱 시도
            parsed = self._parse_response(result_text)
            parsed["raw_response"] = result_text
            parsed["model_used"] = selected_model
            return parsed

        except Exception as e:
            print(f"  ⚠ Gemini API 오류: {e}")
            return {
                "error": str(e),
                "race_no": race_no,
                "raw_response": "",
                "model_used": selected_model
            }

    def analyze_full_card(self, all_races: dict, track_condition: str = "", race_date: str = "") -> list[dict]:
        """
        전체 경주 카드에 대한 Gemini 분석.

        Args:
            all_races: {race_no: {"quant_data": [...], "report": "...", "equipment": "..."}}
            track_condition: 사용자 입력 주로 상태
            race_date: 경주 일자 (YYYYMMDD)

        Returns:
            list[dict] — 경주별 분석 결과
        """
        results = []
        for race_no, race_data in sorted(all_races.items()):
            print(f"\n🧠 {race_no}경주 Gemini 분석 중... (주로: {track_condition or '정보없음'}, 모델: {'Flash' if race_date < datetime.now().strftime('%Y%m%d') else 'Pro'})")
            result = self.analyze_race(
                race_no=race_no,
                quantitative_data=race_data.get("quant_data", []),
                steward_report=race_data.get("report", ""),
                equipment_changes=race_data.get("equipment", ""),
                track_condition=track_condition,
                medical_history=race_data.get("medical", {}),
                race_date=race_date
            )
            results.append(result)
            print(f"  ✅ {race_no}경주 분석 완료")

        return results

    def _format_quantitative(self, race_no: int, data: list[dict]) -> str:
        """정량 분석 결과를 Gemini용 텍스트로 포맷팅"""
        lines = [f"[{race_no}경주 출전마 정량 분석]\n"]

        for h in data:
            speed = h.get("speed", {})
            position = h.get("position", {})
            weight = h.get("weight", {})
            training = h.get("training", {})

            lines.append(f"■ {h.get('horse_name', '?')} (종합: {h.get('total_score', 0)}점, 순위: {h.get('rank', '?')})")
            # [FIX] 데이터 부족 시 멘트 수정
            s_vec = speed.get('g1f_vector', 'N/A')
            if s_vec == "기록기반":
                lines.append(f"  속도: S1F/G1F 부재로 '총 주파기록' 기반 분석. 속도점수={speed.get('speed_score', 0)}")
            else:
                lines.append(f"  속도: S1F평균={speed.get('s1f_avg', 0)}, G1F평균={speed.get('g1f_avg', 0)}, "
                             f"G1F벡터={s_vec}, 속도점수={speed.get('speed_score', 0)}")
            lines.append(f"  포지션: 점수={position.get('position_score', 0)}, "
                         f"W보너스={position.get('w_bonus_count', 0)}회")
            lines.append(f"  체중: {weight.get('note', '정보없음')}")
            lines.append(f"  조교: {training.get('detail', '정보없음')}")

            if h.get("is_veto"):
                lines.append(f"  🚫 VETO: {weight.get('note', '')}")

            lines.append("")

        return "\n".join(lines)

    def _parse_response(self, text: str) -> dict:
        """Gemini 응답에서 JSON 추출"""
        # ```json ... ``` 블록 추출
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            json_str = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            json_str = text[start:end].strip()
        else:
            # 전체 텍스트를 JSON으로 시도
            json_str = text.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 텍스트 그대로 반환
            return {
                "parse_error": True,
                "text_response": text
            }


# ─────────────────────────────────────────────
# 단독 실행 테스트
# ─────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = GeminiAnalyzer()

    # 샘플 정량 데이터
    sample_data = [
        {
            "horse_name": "번개호",
            "total_score": 78.5,
            "rank": 1,
            "speed": {"s1f_avg": 12.1, "g1f_avg": 12.3, "g1f_vector": "Strong", "speed_score": 72},
            "position": {"position_score": 90, "w_bonus_count": 1},
            "weight": {"veto": False, "note": "✅ 적정 범위"},
            "training": {"detail": "✅ 충분한 조교 (16회, 강조교 3회)"},
            "is_veto": False,
        },
        {
            "horse_name": "질풍호",
            "total_score": 65.2,
            "rank": 2,
            "speed": {"s1f_avg": 12.3, "g1f_avg": 12.8, "g1f_vector": "Maintaining", "speed_score": 58},
            "position": {"position_score": 60, "w_bonus_count": 0},
            "weight": {"veto": False, "note": "✅ 적정 범위"},
            "training": {"detail": "⚠ 조교 횟수 충분이나 강조교 없음"},
            "is_veto": False,
        },
    ]

    sample_report = """
    번개호: 직전 레이스에서 W(외곽) 주행에도 불구하고 3착 입선. Strong Finish 확인.
    질풍호: 게이트 출발 지연(출발 불량)으로 후미 출발, 이후 중단까지 만회.
    """

    result = analyzer.analyze_race(
        race_no=5,
        quantitative_data=sample_data,
        steward_report=sample_report,
        equipment_changes="번개호: 블링커(+), 질풍호: 혀끈(-)"
    )

    print("\n🧠 Gemini 분석 결과:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
