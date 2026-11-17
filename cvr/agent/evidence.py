"""Evidence package shown to the planner: first-study results and EDA (docs/DESIGN.md §10), all from
files OTHER than the agent's confirmation files (train-10, 11). Each item lists its numbers so the
verifier can check that a report only uses numbers that exist somewhere.
"""

EVIDENCE = [
    {"id": "E1", "title": "데이터 사전 (타오바오 광고 클릭, MAC 공개 벤치마크)",
     "text": "한 행은 광고 클릭 1건, 정답은 마지막 클릭 귀속 전환 여부. 특성 그룹: history_match(과거 구매 시퀀스에 타깃과 "
             "같은 상품·ad_feat1·ad_feat2 값이 나온 횟수, 구매 이력 없음 여부), visual_similarity(과거 구매 상품과 타깃 상품의 "
             "시각 유사도 최댓값·평균), user_profile(익명 사용자 특성 6개), context(익명 맥락 특성 3개), item_attributes"
             "(익명 상품 특성 3개). ad_feat1은 카테고리로 추정되는 익명 특성이다. sparse_G1은 학습 데이터에서 클릭은 충분하지만 "
             "전환이 적은 ad_feat1 그룹이다.", "numbers": []},
    {"id": "E2", "title": "구매 이력 특성의 예측 기여 (1차 실험, 희소 그룹 G1)",
     "text": "LightGBM에서 history_match와 visual_similarity를 함께 뺐을 때 G1 macro LogLoss 차이(있음 − 없음): 검증일 "
             "-0.000118 [95% CI -0.000177, -0.000064], 테스트일 -0.000055 [-0.000122, 0.000001]. 로지스틱 회귀 테스트일 "
             "-0.000062 [-0.000100, -0.000025].",
     "numbers": [-0.000118, -0.000177, -0.000064, -0.000055, -0.000122, 0.000001, -0.000062, -0.000100, -0.000025]},
    {"id": "E3", "title": "구매 이력과 전환의 단순 연관 (EDA, train-17)",
     "text": "클릭 중 과거 구매에 같은 상품이 있는 비율 0.64%, 그때 전환율 6.4% vs 없을 때 1.5%. 같은 ad_feat1 값 13.3%, "
             "1.77% vs 1.50%. 같은 ad_feat2 값 4.8%, 1.85% vs 1.52%. 시각 유사도 최댓값 5분위별 전환율 1.20%에서 1.87%로 증가. "
             "단순 연관이며 인과가 아니다.",
     "numbers": [0.64, 6.4, 1.5, 13.3, 1.77, 1.50, 4.8, 1.85, 1.52, 1.20, 1.87]},
    {"id": "E4", "title": "구매 이력 처리 방식 (1차 실험 신경망)",
     "text": "구매 이력 20개를 타깃 상품 기준 attention으로 가중한 모델이 단순 평균보다 G1 macro LogLoss가 나빴다 "
             "(차이 +0.000650 [0.000522, 0.000770], 테스트일).", "numbers": [0.000650, 0.000522, 0.000770, 20]},
    {"id": "E5", "title": "희소 그룹 보정 (1차 실험)",
     "text": "G1 그룹별 축소 보정이 공통 보정보다 G1 macro LogLoss를 -0.000100 [-0.000154, -0.000049] 낮췄다(테스트일). "
             "그룹별 보정 오차 중앙값의 개선은 검증일과 테스트일에서 일관되지 않았다.",
     "numbers": [-0.000100, -0.000154, -0.000049]},
    {"id": "E6", "title": "날짜별 전환율 변화",
     "text": "양성 비율 train-17 1.54%, train-18 1.70%, train-19 1.69%. 테스트일에는 보정 전 관측/예측 전환 비율이 1.13이었다.",
     "numbers": [1.54, 1.70, 1.69, 1.13, 17, 18, 19]},
    {"id": "E7", "title": "구매 이력 없음",
     "text": "구매 이력이 전부 비어 있는 클릭은 약 3.4%다. 이들에게는 history_match와 visual_similarity 정보가 없다.",
     "numbers": [3.4]},
]


def evidence_text() -> str:
    return "\n".join(f"[{e['id']}] {e['title']}: {e['text']}" for e in EVIDENCE)
