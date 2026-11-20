# 재현 방법

이 문서는 각 결과 표를 어떤 명령·설정·자료 파일로 만들었는지와, 무엇까지 재현을 확인했는지를 적는다. 설계와 판정 기준은 [DESIGN.md](DESIGN.md)에 있다.

## 1. 환경

- Python 3.11, 의존성은 `uv.lock`으로 고정한다. `uv sync`로 설치한다.
- 연구 E의 학습과 서빙 지연을 뺀 결과는 모두 Apple M4(10코어), 단일 기계에서 만들었다. 개발 실행 하나의 최대 메모리는 9–15GB였다(1차 실험 14.55GB). 메모리 16GB 이하에서는 개발 실행이 돌지 않을 수 있다.
- 연구 E의 학습(체크포인트 15개)은 RunPod의 Linux 포드(A40 4장)에서 했다. 예측, 보정, 평가는 맥 CPU에서 했다. 장치 차이는 V2에서 따로 확인했다(§5 연구 E). 서빙 지연은 Linux CPU 서버(AMD EPYC 4565P)에서 쟀다.
- 원자료는 파일 하나가 2.6–3.3GB이고, 학습 파일 20개와 공식 테스트를 모두 받으면 약 58GB다.
- PyTorch와 LightGBM은 이 환경에서 한 프로세스에 함께 불러오면 OpenMP 충돌이 난다. 그래서 신경망은 모두 별도 작업 프로세스(`cvr/m3_worker.py`, `cvr/study_c_worker.py`, `cvr/final_eval_m3.py`, `cvr/final_eval_c_nn.py`)에서 돌고, 실행기가 이를 자동으로 띄운다.

## 2. 원자료 없이 확인하기

```bash
uv run pytest -q          # 단위 테스트 (원자료 불필요, 약 1분)
uv run pytest -m e2e -q   # 종단 테스트 (원자료 불필요, 이 기계에서 약 2분)
```

- **단위 테스트**: 리뷰에서 찾은 결함마다 회귀 테스트가 있다. 고친 부분을 되돌리면 실패하는 것을 확인했다.
- **종단 테스트** ([tests/test_e2e.py](../tests/test_e2e.py)): [tests/synthetic_mac.py](../tests/synthetic_mac.py)가 MAC과 같은 열 구조의 작은 파일 7개(`train-08`~`14`, 각 2만 행, SHA-256 사이드카 포함)를 만든다. 그 위에서 실제 명령을 하위 프로세스로 실행한다.
  - 연구 C: 개발 실행 → 진단(독립 재선택, STAR 통계) → 서빙 검사와 지연 → 고정 → V2 예행 → 테스트 → 두 번째 테스트 거부.
  - 테스트 단계에서는 거부돼야 할 경우도 확인한다. 미커밋 파일이 있을 때, 이름은 같지만 내용이 다른 홀드아웃 파일일 때, 홀드아웃이 아닌 파일일 때는 실행권을 쓰지 않고 거부해야 한다. 테스트 뒤 고정 파일을 바꾸면 예행이 거부돼야 한다.
  - 1차 실행기: 개발 실행(GBDT, 로지스틱 회귀, M3 신경망) → 고정 → 예행 → 테스트 → 두 번째 테스트 거부.
  - GPU 포드 학습 대기열([tests/test_pod_queue.py](../tests/test_pod_queue.py)): 합성 학습 파일 20개로 공식 MoAE를 CPU에서 학습 → `train-17`~`19` 체크포인트와 검증 GAUC 기록 → 발산한 실행 표시 → 다시 실행하면 끝난 작업은 건너뜀.
  - 작업 폴더는 임시 git 저장소다. 평가 코드가 작업 트리가 깨끗한지 검사하기 때문이다.
  - 지연 측정은 실제와 같은 채택 규칙을 쓴다. 저장소 가상환경의 다른 파이썬 프로세스가 있거나 다른 프로세스 CPU 합이 400%를 넘으면 측정이 채택되지 않는다. 그러면 테스트는 고정이 그 측정을 거부하는지만 확인하고 건너뛴다(skip). 종단 테스트를 돌리는 동안 이 저장소의 다른 파이썬 작업을 멈춰야 한다.
  - 합성 자료의 수치는 MAC과 관계가 없다. 이 테스트는 명령과 검사가 끝까지 도는지를 확인할 뿐, 결과 값을 확인하지 않는다.
  - 연구 E의 학습·평가·서빙 경로는 단위 테스트가 맡는다: 공식 `run.py`와의 비트 단위 일치, 예측 파일 순서, 고정과 1회 테스트, 서빙과 예측기.
  - 일부러 넣은 결함 두 가지를 단위 테스트는 놓치고 종단 테스트는 잡았다(2026-11-18): 고정 단계가 부르는 함수 이름을 틀리게 바꾼 경우(NameError), 서빙 코드가 `context_feat1` 대신 `ad_feat1`로 보정한 경우(서빙 검사, 차이 0.0155 > 1e-9).

## 3. 자료 받기

```bash
python3 scripts/fetch_mac.py train/train-15.parquet test/test.parquet vocabs ...   # data/raw/mac/<경로>에 저장, 폴더 이름은 그 안의 모든 파일
```

- Hugging Face API가 공개한 값과 받은 파일을 대조한다. LFS 파일(parquet, 큰 어휘 파일)은 크기와 SHA-256, 일반 git 파일(어휘 파일 17개 중 13개)은 크기와 git blob SHA-1을 본다. 맞으면 `<파일>.sha256` 사이드카를 쓰고, 실행 결과(`results.json`)에 이 해시가 기록된다.
- 표준 라이브러리만 쓰므로 macOS와 Linux(GPU 포드)에서 환경 설치 전에 돌릴 수 있다. 예전 셸 스크립트 `scripts/fetch_mac.sh`는 이 스크립트를 부르는 래퍼로 남겼다.
- 라이선스: 데이터 카드 `alimamaTech/MAC`는 Apache-2.0이다 (DESIGN §1). 원자료와 파생 자료는 크기 때문에 저장소에 올리지 않는다.

## 4. 파일별 쓰임

MAC의 학습 파일 20개와 공식 테스트는 모두 이미 어느 연구에서 열었다(학습·선택·평가). 새 확증 실험에 쓸 미열람 파일은 남아 있지 않다. 그래서 연구 D는 백테스트로 분류했다.

| 파일 | 쓰인 곳 (역할) |
|---|---|
| `train-00`~`02` | 2차 실험 학습, 연구 B |
| `train-03` / `train-04` | 2차 실험 V1 / V2, 연구 B |
| `train-05`, `06` / `train-07` | 이 저장소에 없는 다른 실험의 발견 / 확증 |
| `train-08`, `09` / `train-10`, `11` | 같은 실험의 개발 단계 |
| `train-08`~`10` / `train-11` / `train-12` | 연구 C 학습 / V1 / V2 (연구 D 설정 C도 같음) |
| `train-12` | 2차 실험 테스트 (연구 C의 V2보다 먼저) |
| `train-13`, `14` | 연구 C 테스트, 연구 D 설정 C 평가 |
| `train-15`~`17` / `train-18` / `train-19` | 1차 실험 학습 / V1 / V2, 연구 A, 연구 D 설정 1 |
| `train-17`, `19` | 탐색(EDA, 축 발굴) |
| `train-00`~`17` / `train-18` / `train-19` | 연구 E 학습 / V1 / V2 (Q1의 재현 모델은 `train-19`까지 학습) |
| `test` | 1차 실험 테스트, 연구 D 설정 1 평가, 연구 E 테스트 (Q1, Q2) |

## 5. 연구별 명령

`<run>`은 개발 실행이 만든 `runs/<이름>/<UTC 시각>` 폴더다. 표의 실행 폴더는 이 저장소의 결과를 만든 폴더다(`runs/`는 git에 넣지 않는다).

### 고정 → 예행 → 테스트 (1차·2차 실험, 연구 C 공통)

1. **고정** (`freeze`): 모델·보정기·매핑·선택 파일의 SHA-256, 주 비교, 판정 기준을 `freeze.json`에 쓴다. 이미 있으면 다시 고정하지 않는다.
2. **예행** (`dryrun`): 테스트와 같은 코드로 V2를 평가한다. 개발 실행이 기록한 V2 지표와 부트스트랩 값을 1e-9 안에서 재현하지 못하면 멈춘다.
3. **테스트** (`test`): 고정한 해시가 하나라도 다르거나, 작업 트리에 커밋하지 않은 변경이 있거나, 이미 실행한 적이 있으면 거부한다. 연구 C는 여기에 더해 다음을 확인한다: 예행이 같은 `freeze.json`과 같은 평가 코드(cvr/*.py의 지문)로 통과했는지, 홀드아웃 파일의 이름·크기·공식 SHA-256. 그다음 잠금 파일로 실행권을 한 번만 얻는다. 실행권을 얻은 뒤의 실패는 기록되고, 실행권은 쓴 것으로 본다.

### 1차 실험: 희소 광고 그룹(`ad_feat1`)의 보정 (DESIGN §10)

실행 폴더 `runs/dev_train15_17_m3/20261117T110713Z`, 개발 실행 3,564초.

```bash
uv run python -m cvr.pipeline configs/dev_train15_17_m3.toml
uv run python -m cvr.final_eval freeze <run>
uv run python -m cvr.final_eval dryrun <run>
uv run python -m cvr.final_eval test   <run> data/raw/mac/test/test.parquet
PYTHONPATH=. uv run python scripts/latency.py <run> data/raw/mac/train/train-19.parquet
```

결과: [V2 선택](results/dev_v2_selection.md), [테스트](results/final_test.md), [지연](results/latency.md).

### 2차 실험: 모델 차원의 희소 그룹 개선 (DESIGN §13)

실행 폴더 `runs/p2_train00_02/20261117T131443Z`, 개발 실행 2,793초.

```bash
uv run python -m cvr.pipeline configs/p2_train00_02.toml
uv run python -m cvr.final_eval freeze <run>
uv run python -m cvr.final_eval dryrun <run>
uv run python -m cvr.final_eval test   <run> data/raw/mac/train/train-12.parquet
PYTHONPATH=. uv run python scripts/latency.py <run> data/raw/mac/train/train-04.parquet              # 선택한 보정
PYTHONPATH=. uv run python scripts/latency.py <run> data/raw/mac/train/train-04.parquet --method ii  # (ii) 보정
```

결과: [V2 선택](results/p2_dev_v2_selection.md), [테스트](results/p2_final_test.md), [지연](results/p2_latency.md).

### 연구 A·B: 작동 조건 (DESIGN §15, 개발 자료만)

```bash
PYTHONPATH=. uv run python scripts/calibration_data_study.py runs/dev_train15_17_m3/20261117T110713Z runs/calib_study \
    --md docs/results/calibration_data_study.md
PYTHONPATH=. uv run python scripts/aux_label_control.py runs/p2_train00_02/20261117T131443Z runs/aux_control \
    --md docs/results/aux_label_control.md
```

결과: [연구 A](results/calibration_data_study.md), [연구 B](results/aux_label_control.md).

### 연구 C: 작은 광고 맥락 그룹과 도메인 구조 (DESIGN §16)

실행 폴더 `runs/c_dev_train08_10/20261118T061335Z`, 개발 실행 3,295초, 최대 메모리 9.33GB.

```bash
uv run python -m cvr.study_c configs/c_dev_train08_10.toml
PYTHONPATH=. uv run python scripts/study_c_diagnostics.py calib <run>   # 독립 재선택 (고정에 필요)
PYTHONPATH=. uv run python scripts/study_c_diagnostics.py star  <run>
PYTHONPATH=. uv run python scripts/latency_c.py <run>                  # 서빙 검사 + 지연 (고정에 필요)
uv run python -m cvr.final_eval_c expect <run>                         # 홀드아웃의 공식 크기·SHA-256 (메타데이터만)
uv run python -m cvr.final_eval_c freeze <run>
uv run python -m cvr.final_eval_c dryrun <run>
uv run python -m cvr.final_eval_c test   <run> data/raw/mac/train/train-13.parquet data/raw/mac/train/train-14.parquet
```

- 지연 측정은 다른 작업이 없을 때 돌린다. 채택 규칙(§2)을 못 맞추면 다시 재되, 계열마다 모두 3번까지다. 고정은 3번을 넘긴 기록과 채택되지 않은 기록을 거부한다.

결과: [V2 선택](results/c_dev_v2_selection.md), [지연](results/c_latency.md), [예행](results/c_dryrun_v2.md), [테스트](results/c_final_test.md).

### 연구 D: 그룹 절편만 축소하는 보정 (DESIGN §17, 재사용 자료 백테스트)

```bash
PYTHONPATH=. uv run python scripts/study_d.py runs/c_dev_train08_10/20261118T061335Z runs/dev_train15_17_m3/20261117T110713Z runs/study_d
PYTHONPATH=. uv run python scripts/study_d.py --render runs/study_d/<UTC 시각>   # 저장 결과에서 표만 다시 만들기
```

- 실행 전에 연구 C와 1차 실험에 기록된 보정 후 지표 30개를 다시 계산해 맞는지 확인한다.

결과: [표](results/study_d.md).

### 연구 E: 공식 PyMAL 모델(BASE, MoAE)의 재현과 희소 그룹 보정 (DESIGN §18)

공식 코드(`third_party/pymal`, 커밋 `3186be8`, 수정 없음)는 [cvr/pymal_run.py](../cvr/pymal_run.py)가 감싸 부른다. 학습 루프, 옵티마이저, 스케줄은 공식 `run.py`와 비트 단위로 같다(`tests/test_pymal_run.py`).

```bash
# 1. 그룹 목록 (train-00~17 통계만, 모델 결과 전에 한 번 고정)
uv run python -m cvr.study_e groups
# 2. GPU 포드 (Linux, A40 4장): 환경, 자료, 공식 전처리, 탐색 격자 45개 실행
python3 scripts/runpod.py budget --usd 60 --max-price 2.0
python3 scripts/runpod.py create --name e-main --gpu "NVIDIA A40" --count 4 --hours 15 --image <PyTorch 이미지>
bash scripts/pod_setup.sh && python3 scripts/pod_queue.py run runs/e_runs/jobs_grid.json --gpus 4      # 포드에서
# 3. 맥: 격자가 끝나면 설정 선택(train-18 체크포인트의 train-19 GAUC), 시드 2개와 가중 학습 3개, 체크포인트 회수,
#    포드 종료, V1/V2 예측, 보정, 고정, 예행 (각 단계는 결과가 있으면 건너뛴다)
POD="-p <port> root@<host>" POD_ID=<pod id> bash scripts/e_autopilot.sh
bash scripts/e_local_finish.sh                 # 포드가 이미 종료된 뒤라면 이것으로 이어서
# 4. 공식 테스트 (질문마다 한 번만; 깨끗한 트리와 같은 평가 코드의 예행이 필요)
#    먼저 고정한 체크포인트로 테스트 예측을 만든다. --official_test가 고정 파일의 체크포인트 집합과 테스트 SHA-256을
#    확인하고 실행권을 한 번 잡는다. Q1은 train-19 체크포인트 6개, Q2는 train-17 체크포인트 9개다.
uv run python -m cvr.pymal_run preprocess test.parquet          # 공식 전처리 캐시 (train-18, train-19도 같은 방식)
uv run python -m cvr.pymal_run predict --device cpu --num_workers 0 --out runs/e_runs/q1 \
    --official_test runs/e_study/frozen/freeze_q1.json \
    --ckpt runs/e_runs/<실행 id>/model_after_train-19.pt ...  test.parquet   # --ckpt를 6번
uv run python -m cvr.pymal_run predict --device cpu --num_workers 0 --out runs/e_runs/research \
    --official_test runs/e_study/frozen/freeze_q2.json \
    --ckpt runs/e_runs/<실행 id>/model_after_train-17.pt ...  test.parquet   # --ckpt를 9번
uv run python -m cvr.final_eval_e test q1    # 예측 완료 기록, 파일·체크포인트·원본 해시, 행 수, 확률 범위를 확인한 뒤 평가
uv run python -m cvr.final_eval_e test q2
# 5. 테스트 뒤 서술 (판정 없음)
PYTHONPATH=. uv run python scripts/q2_per_group.py
```

- 학습 실행 기록은 `runs/e_runs/<실행 id>/run.json`에 있다. 인자, 학습 파일의 SHA-256, 손실, 체크포인트 SHA-256이 들어 있다. 예측 파일은 체크포인트 해시로 찾는다(`<체크포인트>_<SHA-256 앞 12자>/<파일>.npz`). 그래서 다른 체크포인트의 예측을 잘못 읽을 수 없다.
- GPU와 맥 CPU의 예측 차이: 같은 체크포인트, 같은 V2 행에서 `p_last` 최대 7.86e-7, 보조 출력 포함 최대 4.47e-6이었다(`scripts/device_compare.py`, 허용 오차 1e-5).
- 포드 비용은 학습 $28.21(파일럿 포함), 서빙 측정 $1.01이다(`runs/runpod/ledger.json`).

결과: [공식 테스트](results/e_final_test.md), 기록 [results/records](results/records).

### 연구 E 서빙: 원시 입력 → 보정 확률과 지연 (DESIGN §18.12)

```bash
uv run pytest -q tests/test_pymal_run.py -k serving               # 원자료 없이: 공식 텐서와 같은지, 신규 입력, 공유 인코딩
PYTHONPATH=. uv run python scripts/serving_e.py prepare            # 측정 행과 저장 예측을 한 번 고정 (V2 필요)
bash scripts/serving_sessions.sh linux                             # 측정할 서버에서: 일치 검사, 20분 간격 세션 3개, 요약
PYTHONPATH=. uv run python scripts/serving_e.py render linux       # 결과 표
```

- 측정 규약(v3)은 DESIGN §18.12.1에 측정 전에 고정했다. 측정 중에는 이 저장소의 다른 작업을 돌리지 않는다.

결과: [서빙](results/e_serving.md).

## 6. 재현을 확인한 범위

- **확인함**:
  - 저장된 모델로 다시 평가한 값: 예행이 각 개발 실행의 V2 지표와 부트스트랩 값을 1e-9 안에서 재현했다(1차·2차·C). 연구 A·B·D도 저장된 기록값을 다시 계산해 대조했다.
  - 같은 실행 안의 결정성: 모델을 저장했다가 다시 불러온 예측이 1e-9 안에서 같았다. 연구 C의 신경망은 튜닝 때의 시드 1 체크포인트를 최종 학습이 1e-9 안에서 재현했다.
  - 명령 전체: 종단 테스트가 합성 자료에서 모든 명령을 끝까지 실행한다.
- **확인하지 않음**:
  - 다른 기계에서 처음부터 다시 학습했을 때 같은 값이 나오는지. LightGBM·PyTorch의 다중 스레드 계산은 기계와 스레드 수에 따라 마지막 자리가 달라질 수 있다.
  - 지연 수치. 같은 기계에서도 다른 프로세스의 부하에 따라 달라진다. 그래서 1차–연구 C는 채택 규칙을 따로 뒀다. 연구 E는 세션 세 개 사이의 차이를 함께 보고한다(DESIGN §18.12.1).
- 테스트 단계는 실행 폴더마다 한 번만 돈다. 다시 재현하려면 개발 실행부터 새 실행 폴더를 만들어야 하고, 그 테스트 결과는 이미 열어 본 파일에 대한 것이므로 확증이 아니다.
