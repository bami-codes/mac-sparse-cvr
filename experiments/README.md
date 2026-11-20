# 앞선 실험

저장소 첫 화면의 결과(공식 BASE와 MoAE 모델)에 앞서, 같은 MAC 자료에서 LightGBM과 작은 신경망으로 한 실험들입니다. 희소 그룹을 어떻게 정의하고, 보정을 어떻게 비교하고, 테스트를 어떻게 한 번만 열지가 여기서 정해졌고, 최종 실험은 그 방법을 공식 모델에 적용했습니다.

설계와 판정 기준은 [docs/DESIGN.md](../docs/DESIGN.md)의 해당 절에, 실행 명령은 [docs/reproduce.md](../docs/reproduce.md) §5에 있습니다.

| 실험 | 질문 | 결과 | 문서 |
|---|---|---|---|
| 1차 실험 (§10) | LightGBM 위에서 희소 그룹의 그룹별 축소 보정이 공통 보정보다 나은가 | 희소 그룹 LogLoss −0.000100 [−0.000154, −0.000049], 개선 확인 (공식 테스트) | [V2 선택](results/dev_v2_selection.md), [테스트](results/final_test.md), [지연](results/latency.md) |
| 2차 실험 (§13) | 보조 귀속 레이블(`mta`) 예측과 그룹 전환율을 입력으로 넣으면 나아지는가 | −0.000145 [−0.000292, +0.000010], 확인되지 않음 (`train-12`) | [V2 선택](results/p2_dev_v2_selection.md), [테스트](results/p2_final_test.md), [지연](results/p2_latency.md) |
| 연구 A (§15) | 보정 자료의 양에 따라 그룹별 보정의 이득이 어떻게 변하는가 | 개발 자료만 사용, 판정 없음 | [표](results/calibration_data_study.md) |
| 연구 B (§15) | 2차 실험의 이득이 귀속 정보 때문인가, 같은 레이블의 대조로 확인 | 개발 자료만 사용, 판정 없음 | [표](results/aux_label_control.md) |
| 연구 C (§16) | 작은 광고 맥락 그룹에서 도메인별 구조(공유, 헤드, STAR)가 도메인 절편보다 나은가 | +0.000027 [−0.000050, +0.000107], 확인되지 않음 (`train-13`, `14`) | [V2 선택](results/c_dev_v2_selection.md), [예행](results/c_dryrun_v2.md), [테스트](results/c_final_test.md), [지연](results/c_latency.md) |
| 연구 D (§17) | 절편만 축소하는 보정과, 보정 강도를 고르는 지표의 영향 | 이미 연 자료의 백테스트, 확증 아님 | [표](results/study_d.md) |

## 폴더

| 위치 | 내용 |
|---|---|
| `experiments/*.py` | 실험 코드. 보정, 지표, 검사 같은 공통 부분은 `cvr/`의 것을 씁니다. |
| `pipeline.py`, `final_eval.py`, `serving.py` | 1차, 2차 실험의 학습(LightGBM, 로지스틱 회귀, 작은 신경망), 고정과 테스트, 서빙 |
| `study_c.py`, `final_eval_c.py`, `serving_c.py`, `mdnet.py` | 연구 C의 학습, 고정과 테스트, 서빙, 도메인 구조 신경망 |
| [scripts/](scripts) | 탐색(EDA), 연구 A, B, D, 지연 측정 |
| [configs/](configs) | 개발 실행 설정과, 한 파일로 모든 단계를 빠르게 확인하는 시험용 설정(`*smoke*`) |
| [results/](results) | 실행 코드가 만든 결과 표(영어) |

- 신경망은 별도 작업 프로세스(`m3_worker.py`, `study_c_worker.py`, `final_eval_m3.py`, `final_eval_c_nn.py`)에서 돕니다. 이 환경에서는 PyTorch와 LightGBM을 한 프로세스에 함께 불러오면 OpenMP가 충돌하기 때문입니다.
- 각 실험의 테스트는 질문마다 한 번만 실행했습니다. MAC의 학습 파일 20개와 공식 테스트는 모두 어느 실험에서든 이미 열었기 때문에, 새 확증 실험에 쓸 미열람 자료는 남아 있지 않습니다([docs/reproduce.md](../docs/reproduce.md) §4).
- 종단 테스트(`uv run pytest -q -m e2e`)가 1차 실험과 연구 C의 모든 명령을 합성 자료로 끝까지 실행합니다.
