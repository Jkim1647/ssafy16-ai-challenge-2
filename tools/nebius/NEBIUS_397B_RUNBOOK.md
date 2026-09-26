# Nebius H100×8 자체 호스팅 397B FP8 천장 테스트 (형 SSH 운영용)

규칙 OK: GPU를 빌려 **가중치를 직접 로드해 in-process 추론**(호스팅 API 아님). 결과는 제출이 아니라
"397B가 35B/122B보다 실제로 높은가(Q1)" 판정용.

브라우저만으로는 Nebius VM을 못 몰아서(자동 Jupyter 없음, SSH 필요, 포트 노출 차단),
**형이 본인 PC에서 SSH로 실행**하고 결과를 넘겨주면 Claude가 비교표를 만든다.

## 0) 옛 VM 삭제(선택)
`c031-397b-h100x8`(SSH 없이 만든 것)은 못 쓴다. Virtual machines 목록 → 행 우측 ⋮ → Delete.

## 1) SSH 키 생성 (형 PC, 1회)
```bash
ssh-keygen -t ed25519 -f ~/.ssh/nebius -N ""
cat ~/.ssh/nebius.pub    # 이 공개키를 VM 생성 시 붙여넣는다
```

## 2) 새 VM 생성 (Nebius 콘솔)
Compute → Virtual machines → Create VM
- Platform: **NVIDIA H100 NVLink**, Preset **8 GPUs - 128 CPUs - 1600 GiB** (eu-north1, $30.93/hr)
- Storage: 부팅 디스크 기본 1280 GiB (그대로)
- Network: Public IP = **Auto (dynamic)**
- Configuration → **Username and SSH key**: username `ubuntu`, 위 `nebius.pub` 붙여넣기
  - ⚠️ custom cloud-init는 **끄기**(SSH만 쓴다)
- Create VM → 공용 IP 확인 (예: 89.x.x.x). 기본 보안그룹이 22(SSH)를 연다.

## 3) 접속 + 데이터/코드 올리기 (형 PC)
```bash
IP=<공용IP>
ssh -i ~/.ssh/nebius ubuntu@$IP        # 접속 확인 후 exit

# 코드(스크립트+splits+gold): 저장소 clone
ssh -i ~/.ssh/nebius ubuntu@$IP 'git clone https://github.com/Jkim1647/AI_2_CHALLENGE ~/AI_2_CHALLENGE && cd ~/AI_2_CHALLENGE && git checkout feat/9b-setup-dev-relabel'

# 데이터(이미지+csv): 형 PC의 ssafy-16-2-ai 를 scp (dev+train만 있으면 됨)
scp -i ~/.ssh/nebius -r C:/ssafy/AI_2_CHALLENGE/ssafy-16-2-ai ubuntu@$IP:~/ssafy-16-2-ai
#   느리면 dev/ + train/ + *.csv 만: 최소 851장이지만 통째 scp가 편하다
```
(대안) 데이터가 팀 Drive `ai2_data_extracted.tar`에 있으면 VM에서 `pip install gdown; gdown <ID>; tar -xf ...` 로 받아 `~/ssafy-16-2-ai` 로 둬도 된다.

## 4) 실행 (VM 안에서, tmux 권장)
```bash
ssh -i ~/.ssh/nebius ubuntu@$IP
tmux new -s r                 # 세션 끊겨도 계속 돌게
cd ~/AI_2_CHALLENGE/tools/nebius
REPO=$HOME/AI_2_CHALLENGE DATA=$HOME/ssafy-16-2-ai bash run_397b_ssh.sh 2>&1 | tee ~/run.log
```
- 스크립트가: vLLM 설치 → FP8 397B 다운로드(~397GB) → H184 → val667 채점 → 결과 tar + accuracy 요약 출력.
- 예상 ~1~1.5시간. **로드 단계에서 hang(가중치 로드 안 되고 GPU util 0%)이 10~15분 지속되면 Ctrl-C 후 알려줄 것**(RunPod처럼 vLLM 멀티-GPU 문제면 여기서 판정).

## 5) 결과 회수 → Claude에게
- 화면 맨 끝 **accuracy 요약 두 줄**을 복사해 Claude에 붙여넣기(그것만으로 Q1 판정 가능).
- 상세 비교/맞춘ID 분석을 원하면 예측 파일도:
```bash
scp -i ~/.ssh/nebius ubuntu@$IP:~/nebius_397b_results.tar.gz .
```
그 tar를 Claude에 첨부하면 35B/122B와 문항별 비교표 작성.

## 6) 끝나면 반드시 VM 삭제(과금 중단)
Virtual machines → ⋮ → Delete. (Stop만 하면 스토리지 소액 과금 지속)

## 참고 (비교 기준값, 동일 H184=184 / val667 동일 id 해시)
| 모델 | val667 | H184 |
|---|---|---|
| 35B(3.6) read+tiles | 0.9655 | 0.837 |
| 122B(3.5) BF16 read+tiles | 0.9580 | 0.8478 |
| **397B(3.5) FP8 read+tiles** | **?** | **?** ← 이번 |
- 397B가 H184에서 122B(0.848)를 유의미하게(±오차 넘어) 넘으면 크기 확대가 가치 있음. 비슷하면 35B 학습·앙상블에 집중.
