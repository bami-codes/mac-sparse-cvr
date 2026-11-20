# 결과 기록

결과 문서의 표를 만든 원본 기록이다. 실행이 `runs/`에 쓴 파일을 내용 그대로 복사했다(`runs/`는 크기 때문에 git에 넣지 않는다). 서빙 요약만 Linux 서버 부분을 골라 담았다.

| 파일 | 원본 | 내용 | 쓰인 곳 |
|---|---|---|---|
| `official_test_q1.json` | `runs/e_study/frozen/result_q1.json` | 논문 표 4 재현: 시드별 AUC·GAUC, 평균, 표준편차, 판정 | [e_final_test.md](../e_final_test.md) Q1 |
| `official_test_q2.json` | `runs/e_study/frozen/result_q2.json` | 희소 그룹 보정: 모델·시드·보정별 지표(`rows`), 비교와 구간(`primary`, `secondary`), 가중 학습(`q4`) | e_final_test.md Q2–Q4, README의 확률 표 |
| `q2_per_group.json` | `runs/e_study/q2_per_group.json` | 그룹별 (iii) − (ii) 손실의 분포 (테스트 뒤 서술) | e_final_test.md Q2 그룹별 서술 |
| `v2_selection_q2.json` | `runs/e_study/q2/selection.json` | V2에서 고른 설정과 보정 강도, 고를 때의 V2 지표(`v2_rows`) | README의 모델 선택 표 |
| `v2_dryrun_q2.json` | `runs/e_study/frozen/dryrun_q2.json` | 공식 테스트 전 V2 예행: 재현한 값의 수, V2 GAUC | README의 검증 절차, 모델 선택 표 |
| `serving_check_linux.json` | `runs/e_study/serving_v3/check_linux.json` | 서빙 확률과 평가 예측의 최대 차이, 허용 오차 | [e_serving.md](../e_serving.md) 정확성 |
| `serving_summary_linux.json` | `runs/e_study/serving_v3/summary.json`의 `linux` | 세션별·경로별 지연 분포와 구간, 장비 정보 | e_serving.md 지연 |

- 지표와 구간의 계산 코드: [cvr/study_e_eval.py](../../../cvr/study_e_eval.py), [cvr/final_eval_e.py](../../../cvr/final_eval_e.py), [scripts/serving_e.py](../../../scripts/serving_e.py).
- 기록 안의 `code_fingerprint`, `code_sha256`은 실행 당시 코드 파일의 해시다. 이후 린트와 형식을 정리한 커밋이 있어 현재 파일의 해시와는 다를 수 있다. 공식 테스트의 평가 코드는 커밋 `593f03f`, 서빙 측정 코드는 커밋 `7506fd0`의 파일과 해시가 같다.
- 시간 단위는 ms다.
