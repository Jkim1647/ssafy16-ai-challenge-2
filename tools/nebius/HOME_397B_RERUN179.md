# 집 PC에서 397B 이어받기 — 검수 수정분 179건 (2026-09-23)

교육장에서 여기까지 끝났다: 데이터 준비, 스크립트 작성, **35B 절반 완료**.
남은 건 **397B를 같은 179건에 돌리는 것** 하나다.

## 왜 하는가

지금 모든 모델 비교가 잡음이다(9월 23일 오전 진단, paired bootstrap 10쌍 전부 CI가 0 포함).
원인은 평가셋이 작아서다. 검수로 문항이 수정된 179건을 다시 추론해 **H229 → H408**로 키우면
1문항의 무게가 0.44%p → 0.25%p가 된다.

**397B가 필요한 이유**: 최종 제출 2개를 "앙상블"로 할지 "397B 단독"으로 할지가 남은 최대 결정인데,
앙상블은 35B와 397B의 로그확률 평균이라 397B 예측 없이는 새 179건에서 앙상블을 평가할 수 없다.

## 교육장에서 이미 끝난 것

| 항목 | 상태 |
|---|---|
| `runs/rerun_dev179/` (179건, 원본 dev 이미지) | 생성 완료 (git 제외 — 아래 1번) |
| `tools/nebius/run_397b_rerun179.sh` | 커밋됨 |
| **35B v1/v2 예측** | **완료** — `shared_predictions/runpod_35b_rerun179/` |
| 35B 결과 | dev179 159/179 = 0.8883 |
| RunPod 파드 | 정지 완료 |
| Nebius VM | 없음 (전부 삭제된 상태). 모델 디스크 `c031-397b-model` 930GiB만 보존 |

## 1. 준비 (집 PC)

```bash
git checkout feat/9b-setup-dev-relabel && git pull --ff-only
```

**`dev.csv`·`gold.json`은 이제 git에 들어 있다**(2026-09-23 커밋). 검수 원본
`review_gold_final_20260923/.../audit_all_350.csv`도 함께 올렸다. 기기 간 수동 전송은 필요 없다.

git에 없는 것은 **이미지뿐**이다. 179장 전부 원본 dev 이미지라 링크 하나면 끝난다:

```bash
ln -s "$PWD/ssafy-16-2-ai/dev" runs/rerun_dev179/dev
# Windows cmd:  mklink /D runs\rerun_dev179\dev ..\..\ssafy-16-2-ai\dev
```

`dev.csv`의 `path` 열이 `dev/<파일명>`이라 이 링크만 있으면 해결된다.
(`review_gold_final_20260923/` 원본에서 다시 만들고 싶으면 `baseline/Scripts/python.exe tools/build_rerun200.py`.
multistep 이미지는 git에 없으므로 그 절반은 자동으로 건너뛴다.)

검증(어느 쪽이든):
```bash
baseline/Scripts/python.exe -c "import json,pandas as pd,os; d=pd.read_csv('runs/rerun_dev179/dev.csv',encoding='utf-8-sig'); g=json.load(open('runs/rerun_dev179/gold.json',encoding='utf-8')); print(len(d),len(g),all(os.path.exists('runs/rerun_dev179/'+p) for p in d.path))"
# 179 179 True 가 나와야 한다
```

## 2. Nebius VM 만들기

콘솔 → Compute → Virtual machines → Create VM

| 항목 | 값 |
|---|---|
| Region | **eu-north1** (모델 디스크가 여기 있다 — 다른 리전이면 397GB 재다운로드) |
| Platform / Preset | **NVIDIA H200 NVLink, 4 GPUs** ($4.50/GPU·h → **$18/h**) |
| 부팅 디스크 | 기본 |
| **추가 디스크** | 기존 디스크 `c031-397b-model` (930GiB) **attach** |
| Public IP | Auto |
| Configuration | username `ubuntu` + `~/.ssh/nebius.pub` 공개키. **cloud-init은 끈다** |

> 지난번 8×H100($30.80/h)은 FP8 397GB에 과했다. 4×H200 = 564GB면 충분하고 42% 싸다.

## 3. 보안그룹 인바운드 22 열기 ← 지난번 교육장에서 막힌 지점

콘솔 → Networking → Virtual Networks → `default-network` → Security groups →
`default-security-group-...` → **Ingress → Add rule**

- Access **Allow** / Protocol **TCP** / Dest. ports **22** / Source CIDR **본인 공인 IP/32** (https://ifconfig.me)

`0.0.0.0/0`보다 `/32`가 안전하다. 실험이 끝나면 규칙을 지운다.

> 참고: 교육장 PC의 **아웃바운드** 22는 열려 있음을 확인했다(github.com:22 배너 수신).
> 즉 지난번 실패 원인은 네트워크가 아니라 이 보안그룹이었다.

## 4. 디스크 마운트 + 데이터 업로드

```bash
IP=<공용 IP>
SSH="ssh -i ~/.ssh/nebius ubuntu@$IP"

# 모델 디스크 마운트 (포맷하지 말 것 — 이미 ext4에 가중치가 들어 있다)
$SSH 'lsblk'                       # 930G 장치 확인 (예: vdb)
$SSH 'sudo mkdir -p /mnt/m && sudo mount /dev/vdb /mnt/m && ls /mnt/m'
#   /mnt/m 에 venv, hf, AI_2_CHALLENGE, ssafy-16-2-ai 가 보여야 한다

# 코드·데이터 (저장소가 비공개라 VM에서 clone하지 않는다)
tar czf - tools | $SSH 'mkdir -p ~/AI_2_CHALLENGE && tar xzf - --no-same-owner -C ~/AI_2_CHALLENGE'
tar czhf - -C runs/rerun_dev179 . | $SSH 'mkdir -p ~/rerun_dev179 && tar xzf - --no-same-owner -C ~/rerun_dev179'
#   -h 는 심볼릭 링크를 따라가서 이미지를 실제로 담는다 (A 방법으로 링크를 만든 경우 필수)

$SSH 'ls ~/rerun_dev179/dev | wc -l'   # 179 여야 한다
```

## 5. 실행

```bash
$SSH 'cd ~/AI_2_CHALLENGE/tools/nebius && REPO=$HOME/AI_2_CHALLENGE EVAL=$HOME/rerun_dev179 \
      HF_HOME=/mnt/m/hf SKIP_INSTALL=1 bash run_397b_rerun179.sh 2>&1 | tee ~/run397.log'
```

스크립트가 하는 일: 입력 검증 → 8건 스모크 → 179건 본실행 → tar 묶기.
지난 실행과 **같은 플래그·같은 revision `ea5b4f8`**을 쓴다(바뀌면 비교가 깨진다).

- 예상: 로드 5~15분 + 추론 2~3분 → **총 30~45분, 약 $11**
- `SKIP_INSTALL=1`은 모델 디스크의 venv를 쓴다. venv가 비었으면 이 옵션을 빼면 설치부터 한다.
- **로드 단계에서 GPU util 0%가 10~15분 지속되면 중단하고 알릴 것** (vLLM 멀티-GPU hang 전례)

## 6. 회수 → 끄기

```bash
scp -i ~/.ssh/nebius ubuntu@$IP:~/nebius_397b_rerun179.tar.gz .
mkdir -p shared_predictions/nebius_397b_rerun179 && tar xzf nebius_397b_rerun179.tar.gz -C shared_predictions/nebius_397b_rerun179
```

⚠️ **VM을 지우기 전에 모델 디스크를 반드시 detach한다.** 삭제 확인창이
`The VM and its managed disks will be permanently deleted`이고 버튼이 `Delete VM and disks`라서,
그냥 지우면 397GB 가중치가 같이 날아간다. VM → Disks 탭 → 모델 디스크 ⋮ → **Detach** → 그 다음 VM 삭제.

보안그룹 22 규칙도 지운다.

## 7. 결과 합치기 (GPU 불필요)

```bash
baseline/Scripts/python.exe tools/rescore_gold_v4.py     # 기존 H229 기준
# H408 비교는 아래를 새로 만들어 돌린다 (아직 없음)
```

**아직 없는 것**: H229 + dev179을 합쳐 408건에서 6개 모델(35B v1/v2, 397B, 앙상블, 9B CE/KD)을
채점하는 스크립트. `tools/rescore_gold_v4.py`를 복제해 gold를
`hard_eval_gold_v5_H.json` ∪ `runs/rerun_dev179/gold.json`로, 예측을
`runpod_35b_read2` ∪ `runpod_35b_rerun179`, `nebius_397b` ∪ `nebius_397b_rerun179`로 합치면 된다.
9B CE/KD는 새 179건 예측이 없으므로 H229에서만 보고한다.

## 확인할 질문

1. **H408에서 앙상블과 397B 단독의 차이가 유의해지는가?** H229에서는 불일치 16문항(10:6)으로
   동전 던지기와 구별되지 않았다. n이 늘면 갈릴 수 있다.
2. 갈리면 그쪽으로 최종 제출을 정하고, 여전히 잡음이면 **2개를 앙상블 / 397B 단독으로 나눠 지정**한다
   (현재 Kaggle 최종 선택이 0/2로 비어 있다).
