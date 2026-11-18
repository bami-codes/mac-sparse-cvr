# 직접 검토 목록 (REVIEW.md를 쓰기 위한 준비)

`REVIEW.md`에는 **본인이 직접 코드를 따라가고, 일부러 깨 보고, 이유를 자기 말로 설명할 수 있는 항목만** 적는다 (DESIGN §9). 이 목록은 그 확인 순서다. 항목마다 30분 안팎이면 된다.

각 항목은 같은 형식이다.
1. 무엇이 왜 맞아야 하는가
2. 코드 위치
3. 실행해서 통과를 확인
4. 일부러 틀리게 바꿔서 실패를 확인 (바꾼 뒤에는 `git checkout -- <파일>`로 되돌린다)
5. 설명할 것

---

## 1. 조기 종료 사용자의 레이블이 학습 절편에 새지 않는다 (G-int)

- **왜**: G-int는 도메인 전환율을 로짓 절편으로 넣는다. 조기 종료용 사용자(학습 사용자의 약 10%)를 떼기 전에 교차 절편을 계산하면, 그 사용자들의 레이블이 다른 학습 행의 절편에 들어간다. 그러면 조기 종료 성능이 낙관적이 되고 반복 수 선택이 오염된다.
- **코드**: [cvr/domain_models.py](../cvr/domain_models.py)의 `early_stopping_offsets`, `GBDTOffset.fit`
- **실행**: `uv run pytest -q tests/test_domain_models.py -k early_stopping`
- **깨 보기**: `early_stopping_offsets`에서 `off_fit = oof_domain_offset(fit_side, folds)`를 `off_fit = oof_domain_offset(train, folds)[~es]`로 바꾼다. 테스트가 실패해야 한다 (조기 종료 사용자의 레이블을 뒤집으면 학습 절편이 달라짐).
- **설명할 것**: 왜 "먼저 떼고, 남은 쪽 안에서만 교차 계산"인가. 반복 수가 정해진 뒤 전체 자료로 다시 적합할 때는 왜 다시 교차 계산하는가.

## 2. 한 도메인의 배치가 다른 도메인의 파라미터를 움직이지 않는다 (STAR·N-head)

- **왜**: 도메인별 파라미터를 한 텐서에 두고 일반 Adam으로 갱신하면, 이번 배치에 없는 도메인도 이전 모멘텀 때문에 값이 바뀐다. 그러면 "도메인 파라미터는 그 도메인의 예제로만 갱신된다"는 STAR의 전제가 깨진다.
- **코드**: [cvr/mdnet.py](../cvr/mdnet.py)의 `_table`, `make_optimizers` (희소 임베딩 + SparseAdam)
- **실행**: `uv run pytest -q tests/test_mdnet.py -k "domain_a"`
- **깨 보기**: 테스트 `test_dense_adam_would_move_domain_a_through_momentum`이 이미 일반 Adam에서는 값이 움직인다는 것을 보여 준다. 이 테스트의 `_isolation(variant, sparse=False)` 결과에서 기울기는 0인데 값이 바뀐다는 점을 직접 출력해 본다.
- **설명할 것**: Adam의 업데이트가 `m / (sqrt(v) + eps)`라서 현재 기울기가 0이어도 움직인다. SparseAdam(lazy Adam)은 배치에 나온 행의 값과 모멘트만 갱신한다.

## 3. STAR의 도메인 ID 임베딩 차원은 학습·평가 모두 전체 통계로 정규화한다

- **왜**: 도메인 ID 임베딩은 한 단계 안에서는 상수(분산 0)라서 학습 때는 전체 통계로 정규화된다. 그런데 평가 때 쓰는 이동 평균·제곱 평균은 여러 단계의 값을 섞으므로, 분산이 기준(전체의 1%)을 넘으면 자기 통계로 정규화되어 학습 때와 달라진다.
- **코드**: [cvr/mdnet.py](../cvr/mdnet.py)의 `PartitionedNorm.stats`, `global_dims`
- **실행**: `uv run pytest -q tests/test_mdnet.py -k domain_id`
- **깨 보기**: `stats` 안의 `& ~self.global_dims`를 지운다. 테스트가 실패해야 한다.
- **실제 자료 확인**: `PYTHONPATH=. uv run python scripts/study_c_diagnostics.py star runs/c_dev_train08_10/20261118T061335Z`. 시드 2026에서 592쌍 중 1쌍이 옛 규칙의 경계를 넘는다.
- **설명할 것**: 이것은 "실제 모델이 잘못 예측했다"가 아니라 "옛 규칙이면 경계를 넘는 경우가 실제 자료에 있었다"이다.

## 4. 그룹별 보정을 정확해로 풀자 개선 추정치가 35% 줄었다

- **왜**: 기존 결합 L-BFGS-B는 `converged=True`였지만 목적함수를 충분히 정밀하게 풀지 못했다. 공통 계수와 λ를 고정하면 그룹마다 2변수 볼록 문제로 나뉘어 뉴턴법으로 정확히 풀 수 있다.
- **코드**: [cvr/calibration.py](../cvr/calibration.py)의 `GroupShrunkPlatt.fit`(기존), `fit_group_shrunk_exact`(정확해)
- **실행**: `uv run pytest -q tests/test_calibration_exact.py`
- **수치 확인**: `runs/c_dev_train08_10/20261118T061335Z/diagnostics_calib.json`의 `per_fit`에서 `G-shared_s*|lam1`과 `G-int_s*|lam1`의 `old`·`exact`로 주 비교(G-shared − G-int)를 직접 계산한다. 기존 −0.0000278, 정확해 −0.0000181이 나와야 한다.
- **설명할 것**: 왜 목적함수가 그룹별로 나뉘는가. 기울기 하한(a + da ≥ 0.001)은 어떻게 처리했는가. 차이가 효과 크기와 같은 크기라는 것이 왜 중요한가.

## 5. 서빙 확률이 평가 확률과 같다 (잘못된 보정 키를 잡는다)

- **왜**: 서빙 코드가 1차 실험처럼 `ad_feat1`로 보정하면 다른 모델을 서빙하게 된다. 두 경로가 같은 실수를 공유하면 서로 비교하는 검사도 통과한다. 그래서 기준값은 서빙 코드를 쓰지 않고, 저장된 예측과 따로 불러온 보정기로 만든다.
- **코드**: [cvr/serving_c.py](../cvr/serving_c.py)의 `ServerC.calibrate`, [scripts/latency_c.py](../scripts/latency_c.py)의 `measure_family`
- **깨 보기**: `ServerC.calibrate`의 `codes[DOMAIN_KEY]`를 `codes["ad_feat1"]`로 바꾸고 `PYTHONPATH=. uv run python scripts/latency_c.py runs/c_smoke/20261118T061015Z --family G-shared --out /tmp/x.json`을 실행한다. 시간 측정 전에 `CheckFailed`로 멈춰야 한다(스모크에서 차이 0.0128).
- **설명할 것**: 왜 기준값이 서빙 코드와 독립이어야 하는가. GBDT는 1e-9, 신경망은 1e-6으로 허용 오차가 다른 이유(float32 단건 대 배치).

## 6. 두 테스트 파일에 모두 나온 사용자는 한 군집이다

- **왜**: 같은 사용자의 클릭은 서로 상관된다. 두 날짜를 합쳐 평가할 때 같은 사용자를 두 사람으로 보면 구간이 너무 좁아진다.
- **코드**: [cvr/domain_eval.py](../cvr/domain_eval.py)의 `paired_bootstrap` (`np.unique(users)`로 군집)
- **실행**: `uv run pytest -q tests/test_domain_eval.py -k one_cluster`
- **설명할 것**: 부트스트랩이 다루는 불확실성은 "학습된 모델을 고정했을 때 평가 사용자의 변동"뿐이다. 재학습 변동은 들어가지 않는다.

---

`REVIEW.md` 형식 (항목마다): 확인한 날짜, 무엇을 확인했는가(자기 말), 실행한 명령과 결과, 깨 봤을 때의 결과, 아직 이해가 불확실한 점.
