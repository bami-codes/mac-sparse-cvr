# 전환이 드문 광고 그룹을 위한 CVR 예측과 저지연 후보 점수화

광고 전환율(CVR) 모델은 클릭은 많아도 전환이 드문 광고 그룹에서 확률이 쉽게 어긋난다. 이 저장소는 Alimama의 공개 데이터셋 [MAC](https://huggingface.co/datasets/alimamaTech/MAC)([논문](https://arxiv.org/abs/2603.02184))과 논문의 공식 모델(BASE, MoAE)로 세 가지를 한다.

1. **재현.** 논문의 두 모델을 공식 코드 그대로, 전체 규모로 재현한다.
2. **보정.** 전환이 드문 그룹의 확률 오차를 재고 보정한다.
3. **서빙.** 한 사용자의 광고 후보 100개를 CPU 코어 하나로 수 ms 안에 점수화하는 예측기를 만든다.

## 써 보기

```bash
uv sync
uv run python -m cvr.predictor examples/request.json
```

```json
{
  "model": "BASE",
  "calibration": "group-shrunk Platt on the ad group (ad_feat1)",
  "candidates": [
    {"ad_feat0": "bd06e0d08ba7", "ad_feat1": "707e766d45eb", "cvr": 0.0365, "cvr_uncalibrated": 0.0332},
    {"ad_feat0": "db402f022dcf", "ad_feat1": "76af7efae0d0", "cvr": 0.0133, "cvr_uncalibrated": 0.0177},
    {"ad_feat0": "ca752f48f687", "ad_feat1": "76dc15e364ed", "cvr": 0.0224, "cvr_uncalibrated": 0.0268}
  ],
  "values_not_in_vocabulary": {}
}
```

**요청과 응답**

- 요청은 MAC의 원시 열로 쓴다. 사용자 특성, 구매 이력 3열, 후보마다 광고·맥락 특성과 이력 상품별 유사도가 들어간다.
- 사용자 이력은 요청마다 한 번만 인코딩한다.
- 어휘에 없는 값은 학습 때의 결측값으로 처리하고, 응답에 열별 개수를 돌려준다.
- `--jsonl`을 주면 모델을 한 번 띄운 채 한 줄에 요청 하나씩 받는다.

**예제 모델과 실제 모델**

- [examples/model](examples/model)은 합성 자료로 공식 학습 코드를 그대로 돌려 만든 1.3MB짜리 모델이다([scripts/make_example_bundle.py](scripts/make_example_bundle.py)). 그래서 확률 값 자체는 MAC과 관계가 없다.
- 실제 모델도 같은 명령에 모델 폴더와 어휘만 바꿔 쓴다.

```bash
python3 scripts/fetch_mac.py vocabs train/train-19.parquet                       # 어휘(873MB)와 자료 한 파일
uv run python -m cvr.predictor export BASE_lr0.003_lam0.3_s2026 bundles/BASE_train17
uv run python -m cvr.predictor request-from data/raw/mac/train/train-19.parquet 123456 > request.json
uv run python -m cvr.predictor request.json --bundle bundles/BASE_train17 --vocabs data/raw/mac/vocabs
```

체크포인트는 개당 2GB라 저장소에 넣지 않았다. 학습 명령은 [docs/reproduce.md](docs/reproduce.md)에 있다.

## 결과

**희소 그룹(G1)**: 학습 자료(`train-00`~`17`)에서 클릭이 3,069건 이상인데 전환 양성이 123건 미만인 광고 그룹(`ad_feat1`) 979개다.

**평가 방법**

- 공식 테스트(406만 클릭)는 질문마다 한 번만 썼다. 모델 설정과 보정 강도는 그 전에 검증 파일(`train-18`, `train-19`)에서 골라 고정했다.
- 구간은 사용자 단위 부트스트랩 95% 구간이다.
- 시드는 3개다.

### 어떤 모델을 쓰는가

| 학습 범위 | 선택 | 근거 |
|---|---|---|
| `train-17`까지 (마지막 두 파일을 보정 적합과 강도 선택에 씀) | **BASE + 그룹별 축소 보정** | 순위(AUC 0.848)와 희소 그룹 손실(macro LogLoss 0.06248) 모두 MoAE(0.829, 0.06473)보다 낫다. 요청 시간도 절반 이하다(1.89 vs 3.98ms). |
| `train-19`까지 (모든 학습 파일) | MoAE가 순위에서 앞섬 | AUC 0.867 vs 0.851. 이 조건에서는 보정을 맞출 자료가 남지 않아, 확률 보정의 효과는 평가하지 못했다. |

같은 테스트 클릭에서 학습 종료 파일만 바꿨는데 두 모델의 순위가 뒤집혔다(MoAE − BASE AUC −0.019 → +0.016). 한 번의 학습 시점 비교로 모델을 고르면 위험하다는 뜻이다. 같은 학습 실행의 인접한 두 시점이라 독립된 반복은 아니다.

### 재현 (논문 표 4, 공식 테스트, 시드 3개 평균)

| 모델 | AUC | 논문 | GAUC | 논문 |
|---|---|---|---|---|
| BASE | 0.8514 | 0.8494 | 0.7419 | 0.7424 |
| MoAE | 0.8674 | 0.8694 | 0.7604 | 0.7636 |

- 설정은 논문의 탐색 범위(학습률 9개 × λ 4개)에서 골랐다. 고른 값은 MoAE 학습률 0.004·λ 0.2, BASE 학습률 0.003이다.
- 공식 저장소의 실행 예제(`scripts/run.sh`)는 `--learning_rate 4.0e-3`을 넘긴다. 하지만 파서는 `--lr`만 정의하고 모르는 인자를 `parse_known_args`로 버린다. 그래서 예제는 의도한 0.004가 아니라 기본값 0.003으로 학습된다. 이 저장소의 실행기는 알 수 없는 인자를 오류로 처리한다.

### 희소 그룹의 확률 (`train-17` 모델, G1 979개 합산)

| | 관측 양성 100건당 예측 (MoAE) | 관측 양성 100건당 예측 (BASE) | G1 macro LogLoss (MoAE) |
|---|---|---|---|
| 보정 전 | 74.8 | 87.9 | 0.066216 |
| 공통 보정 (Platt) | 101.4 | 102.9 | 0.064743 |
| 그룹별 축소 보정 | 101.3 | 102.9 | 0.064731 |

- **공통 보정이 먼저다.** 강한 모델(MoAE)도 희소 그룹에서 전환을 약 25% 적게 예측했고, 확률 하나로 맞추는 공통 보정이 그 오차의 대부분을 없앴다.
- **그룹별 축소 보정은 작지만 확인된 추가 이득이다.** G1 macro LogLoss가 −0.000011 [−0.000019, −0.000004] 낮아졌고, 시드 3쌍 모두 같은 방향이었다.
  - 그룹별로 보면 548개가 좋아지고 431개가 나빠졌다. 그래서 운영에 쓴다면 그룹별 O/E를 지켜봐야 한다.
  - 같은 방법이 앞선 LightGBM 모델에서는 이득이 약 9배였다(−0.000100 [−0.000154, −0.000049], 희소 그룹 1,083개). 학습 자료와 그룹 목록이 달라 직접 비교는 아니지만, 기반 모델이 강할수록 그룹별 보정의 몫이 줄었다.
- **학습 단계에서 희소 그룹을 2배로 가중하는 방법은 확인되지 않았다.** 차이는 −0.000069 [−0.000231, +0.000088]이고, 시드마다 방향이 달랐다.

### 서빙 (Linux 서버, AMD EPYC 4565P 코어 하나, 세 세션의 중앙값)

| 모델 | 요청 (후보 100개) p50 | p99 | 클릭 한 건 p50 | 사용자 공유 인코딩의 단축 (p50) |
|---|---|---|---|---|
| BASE | 1.89ms | 3.21ms | 0.40ms | 34% |
| MoAE | 3.98ms | 5.64ms | 0.71ms | 20% |

- 서빙한 확률은 평가에 쓴 예측과 같다(배치 경로 2.1e-7 이내).
- 같은 사용자의 구매 이력을 후보마다 다시 인코딩하지 않으면 확률은 그대로이고 시간만 준다.
- 세 세션 사이의 차이는 1.16배 이내다.
- 네트워크와 피처 저장소 조회는 시간에 넣지 않았다.

자세한 표: [공식 테스트](docs/results/e_final_test.md), [서빙](docs/results/e_serving.md).

## 코드

| 위치 | 내용 |
|---|---|
| [third_party/pymal](third_party/pymal_UPSTREAM.md) | 공식 코드(커밋 `3186be8`). 수정하지 않았고, 파일마다 git blob id를 테스트로 대조한다. |
| [cvr/pymal_run.py](cvr/pymal_run.py) | 공식 코드 실행기: 학습, 체크포인트, 이어서 학습, 예측. 공식 `run.py`와 손실·파라미터가 비트 단위로 같은지 테스트한다. |
| [cvr/study_e.py](cvr/study_e.py), [cvr/study_e_eval.py](cvr/study_e_eval.py) | 희소 그룹 정의, 보정 적합과 선택, 지표와 부트스트랩 |
| [cvr/final_eval_e.py](cvr/final_eval_e.py) | 고정 → 검증 자료 예행 → 질문마다 한 번의 공식 테스트 |
| [cvr/calibration.py](cvr/calibration.py) | 공통 Platt, 그룹별 축소 Platt (정확 해법) |
| [cvr/serving_e.py](cvr/serving_e.py), [cvr/predictor.py](cvr/predictor.py) | 원시 행 → 공식 텐서 → 모델 → 보정, 모델 묶음과 명령줄 |
| [cvr/latency_v3.py](cvr/latency_v3.py), [scripts/serving_e.py](scripts/serving_e.py) | 지연 측정과 분석 |
| [scripts/runpod.py](scripts/runpod.py), `scripts/pod_*.sh` | GPU 포드 운영: 예산 상한, 수명, 체크포인트 회수와 해시 대조 |

```bash
uv run pytest -q    # 186개, 약 1분, 원자료 불필요 (합성 MAC 자료)
```

## 앞선 실험

같은 데이터에서 LightGBM과 작은 신경망으로 먼저 한 실험들이다. 위 결과의 출발점이다.

- **LightGBM 희소 그룹 보정** (공식 테스트): 그룹별 축소 보정 −0.000100, 확인됨. [결과](docs/results/final_test.md)
- **보조 귀속 레이블(`mta`)과 그룹 전환율 입력** (`train-12`): −0.000145 [−0.000292, +0.000010], 확인되지 않음. [결과](docs/results/p2_final_test.md)
- **작은 광고 맥락 그룹의 도메인 구조** (GBDT·신경망 7종, `train-13`·`14`): 공유 구조와 그룹별 절편의 차이가 확인되지 않았다. 클릭 단위 손실과 그룹 합계 오차가 서로 다른 방향으로 움직였다. [결과](docs/results/c_final_test.md)
- **전환 단서 가설 생성·검증 에이전트** (HyperCLOVA X 플래너, 코드가 검정): 가설 22개(고유 검정 11개) 중 4개 확인. [보고서](docs/results/agent_final_report.md)

## 문서

- [docs/reproduce.md](docs/reproduce.md): 자료 받기, 학습(GPU 포드), 평가, 서빙 측정 명령
- [docs/DESIGN.md](docs/DESIGN.md): 설계와 사전 고정한 판정 기준, 결정 기록

## 데이터와 라이선스

- MAC 데이터셋과 공식 코드는 Apache-2.0이다. 공식 코드의 저작권은 원저자에게 있다.
- 원자료와 체크포인트는 크기 때문에 저장소에 넣지 않았다.
- 예제 모델은 합성 자료만으로 만들었다.
