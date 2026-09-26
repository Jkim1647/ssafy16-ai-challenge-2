# Qwen3.5 smoke test 재현 가이드 (5대 PC용)

작성: 2026-09-17 · 대상: 나머지 4대 PC에서 `python -m src.run smoke -c 9b_base_1024 --synthetic`를
그대로 통과시키기 위한 절차. `HANDOFF_SMOKE.md`의 첫 실행에서 실제로 걸렸던 지점만 정리했다.

## 결론 먼저

1대(<호스트명>)에서 로드 → visual token 실측 → PEFT 저장/재로드까지 전부 통과했다.
5대 모두 아래 순서를 그대로 따르면 재현된다.

## 1. baseline 가상환경 + 설치

```powershell
.\baseline\Scripts\python.exe -m pip install -e .
```

`pyproject.toml`의 `transformers` 의존성이 PyPI 최신 릴리스가 아니라
**git 커밋 고정**으로 바뀌어 있다 — 아래 2번 때문이다. `pip install -e .`만 실행하면
자동으로 그 커밋을 받으니 별도 조치는 필요 없다. 단 최초 설치 시 git 클론 + wheel
빌드가 있어 1~2분 더 걸린다.

## 2. 왜 transformers를 git main으로 고정했는가 — 절대 스킵하지 말 것

`configs/models/qwen35_9b.yaml`의 `hf_id: Qwen/Qwen3.5-9B`는 **정확하다** (실제 HF repo,
2026-02 공개, apache-2.0). 하지만 이 모델의 `model_type`은 `qwen3_5`(Gated DeltaNet +
Attention 하이브리드, Qwen3.5 고유 신규 아키텍처)이고, PyPI 최신 정식 릴리스인
`transformers==4.57.6`의 `CONFIG_MAPPING`에는 아직 등록돼 있지 않다. 그대로 실행하면:

```
ValueError: The checkpoint you are trying to load has model type `qwen3_5` but
Transformers does not recognize this architecture.
```

HF 모델 카드도 "latest transformers from source" 설치를 명시한다. 이미 검증된 커밋으로
`pyproject.toml`에 고정해뒀으니, **버전을 임의로 올리거나 내리지 말 것** — 5대가 서로
다른 nightly를 받으면 processor 동작이 달라질 수 있다(AGENTS.md 고정 설정 원칙).

버전 확인:

```powershell
.\baseline\Scripts\python.exe -c "import transformers; print(transformers.__version__)"
# 5.18.0.dev0 이 나와야 정상
```

## 3. HF_HOME / 디스크

`HF_HOME`을 지정하지 않으면 기본값(`C:\Users\<사용자>\.cache\huggingface`)에 받는다.
9B 체크포인트가 약 19GB이므로 그 드라이브에 여유 공간이 있는지 먼저 확인한다.

```powershell
Get-PSDrive -PSProvider FileSystem | Select-Object Name, @{N='FreeGB';E={[math]::Round($_.Free/1GB,1)}}
```

넉넉한 드라이브가 따로 있으면 다운로드 전에 지정한다.

```powershell
$env:HF_HOME = "D:\hf_cache"
```

## 4. 실행

```powershell
.\baseline\Scripts\python.exe -m src.run smoke -c 9b_base_1024 --synthetic -n 4
```

`-v`(verbose)를 쓰려면 서브커맨드 앞에 와야 한다: `-m src.run -v smoke ...`
(argparse 전역 플래그라 뒤에 붙이면 `unrecognized arguments` 에러가 난다).

첫 실행은 가중치 다운로드(19GB)가 있어 네트워크에 따라 수 분 걸린다. 두 번째부터는
캐시를 쓰므로 로드까지 1분 내외다.

### 정상 종료 시 마지막 로그

```
... smoke test 통과. visual_tokens=768 (config 계산=768)
```

`runs/smoke_9b_base_1024/events.jsonl`에 `visual_token_check`(expected==actual==768)와
`peft_smoke_test`(ok=true)가 남는다. `summary.json`의 `peak_vram_gb`도 확인한다.

## 5. 이 세션에서 고친 코드 버그 2건 (이미 반영됨, 참고용)

- **`src/checkpoint.py` — PEFT 재로드 이중 래핑.** `smoke_test_roundtrip`이 이미
  LoRA가 붙은 학습용 모델 위에 `PeftModel.from_pretrained`를 또 호출해서
  `base_model.model.base_model.model...`으로 이중 래핑되고 있었다. 새로 붙는 adapter가
  항등(identity)이라 재로드 검증이 실질적으로 무의미한 "거짓 통과"였다. 저장한 adapter를
  `unload()`로 내려 순수 base로 되돌린 뒤 다시 올리도록 고쳤다.
- **`src/run.py` — `smoke` 커맨드가 `peak_vram_gb`를 아예 기록하지 않았다.** 해상도·crop
  sweep의 핵심 산출물이 이 값인데 `cmd_smoke`에는 `resources.reset_peak()`/`peak_gb()`
  호출이 없었다. 로드~PEFT 왕복 구간을 감싸 `metrics.jsonl`·`summary.json`에 남기도록
  추가했다.

## 6. 여기서 막히면

`HANDOFF_SMOKE.md`의 "터질 만한 곳" 표를 순서대로 확인한다. 단, 표의 1순위(`hf_id` 오탈자
의심)는 **이미 확인 완료 — hf_id는 정확하다.** 실제 1순위 원인은 여기 문서의 2번(transformers
버전)이었다.
