# 집에서 397B 이어서 하기 (Nebius, 교육장에서 준비 끝난 상태)

교육장에서 여기까지 됨: Nebius 계정/$50 충전, H100×8 VM `c031-397b-ssh` 생성(Stopped),
SSH 키(username `ubuntu`) 등록, 실행 스크립트/배치/데이터 파이프라인 준비 완료.
**남은 건 딱 두 개: (1) 보안그룹 인바운드 22 열기, (2) 배치 실행.**

## 왜 교육장에서 막혔나
- 원인 1: **보안그룹 기본 규칙이 인바운드를 사실상 전부 차단**(Ingress Allow인데 Source CIDR 없음) → SSH(22) timeout.
- 원인 2(가능): 교육장 네트워크가 아웃바운드 22를 막았을 수 있음(집은 보통 열림).

## 준비물 확인
- 집 PC에 이 저장소가 있어야 함 → `git pull` (브랜치 `feat/9b-setup-dev-relabel`).
- 대회 데이터 `ssafy-16-2-ai/`가 집 PC에 있어야 함(배치가 scp로 올림).
- **SSH 키 선택 (둘 중 하나)**
  - (A) 교육장 PC의 키를 지참: `C:\Users\<user>\.ssh\nebius` 와 `nebius.pub` 두 파일을 집 PC의 같은 경로(`%USERPROFILE%\.ssh\`)에 복사 → 기존 VM 그대로 사용.
  - (B) 키를 안 가져옴: 집에서 배치 실행 시 **새 키가 생성**됨 → 그 새 공개키로 **VM을 새로 만들어야** 함(기존 `c031-397b-ssh`는 교육장 키라 접속 불가). 이 경우 아래 3-대안 참고.

## 순서 (집)
### 1. 보안그룹 인바운드 22 열기 (핵심)
콘솔 → Networking → Virtual Networks → 해당 network → Security groups →
`default-security-group-...` → **Ingress → Add rule**:
- Access: **Allow**, Protocol: **TCP**, Dest. ports: **22**, Source CIDR: **0.0.0.0/0**
  (더 안전하게 하려면 집 공인 IP만: https://ifconfig.me 값/32)
- (선택) 나중에 필요하면 같은 방식으로 8888 등도. 지금은 22만 있으면 됨.

### 2. VM 시작 + 공용 IP 확인
콘솔 → Compute → Virtual machines → `c031-397b-ssh` → **Start** →
Running 되면 **Public IPv4** 복사(정지 후 재시작이라 IP가 새로 바뀜).

### 3. 배치 실행
`C:\ssafy\AI_2_CHALLENGE\tools\nebius\nebius_397b.bat` 더블클릭 →
`Enter VM public IP:` 에 새 IP 입력 → 자동으로 clone·scp·397B(FP8) 실행 → 로그 표시.
- 끝의 `ALL DONE` + accuracy 두 줄을 Claude에 붙여넣기.

### 3-대안 (키 B를 골랐을 때: 새 VM 생성)
1. 배치를 먼저 한 번 실행 → `[2/5]`에 새 공개키가 출력되면 복사(또는 `type %USERPROFILE%\.ssh\nebius.pub`).
2. 콘솔에서 VM 새로 생성: **H100 NVLink / Preset 8 GPUs / eu-north1 / Public IP Auto /
   Configuration: username `ubuntu` + 그 공개키 / cloud-init OFF** → Create.
3. 위 1(보안그룹 22) 확인 → 2(IP) → 3(배치에 IP 입력).

## 끝나면
- 결과: `scp -i %USERPROFILE%\.ssh\nebius ubuntu@<IP>:~/nebius_397b_results.tar.gz .`
- **반드시 VM 삭제**(과금 중단): Compute → Virtual machines → 행 ⋮ → Delete.
  안 쓰는 옛 VM(`c031-397b-h100x8`)도 같이 삭제.
- Claude가 35B(0.9655/0.837)·122B(0.9580/0.8478)와 비교표 작성.

## 실패 판정 (돈 아끼기)
- 로드 중 GPU util 0%로 10~15분 멈추면(RunPod식 hang) Ctrl+C 후 Claude에 알림. 깨끗한 노드라 정상일 가능성이 높음.
