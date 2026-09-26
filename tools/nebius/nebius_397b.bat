@echo off
REM Nebius H100x8  Qwen3.5-397B-A17B-FP8  ceiling test  one-click launcher
REM Double-click. It: makes SSH key -> shows public key (create VM) -> asks IP
REM -> clone repo + scp data + run(nohup) + tail log.  (Create the VM in console.)
setlocal
set KEY=%USERPROFILE%\.ssh\nebius
set DATA=C:\ssafy\AI_2_CHALLENGE\ssafy-16-2-ai
set SSHOPT=-o StrictHostKeyChecking=no

where ssh >nul 2>nul || (echo [ERROR] OpenSSH Client not found. Install: Settings ^> Apps ^> Optional features ^> OpenSSH Client. & pause & exit /b 1)

echo.
echo [1/5] Ensuring SSH key ...
if not exist "%USERPROFILE%\.ssh" mkdir "%USERPROFILE%\.ssh"
if not exist "%KEY%" (ssh-keygen -t ed25519 -f "%KEY%" -N "" -C nebius397b) else (echo     using existing key %KEY%)

echo.
echo ============================================================
echo [2/5] Copy the PUBLIC KEY line below into the Nebius VM form:
echo       Configuration ^> Username and SSH key
echo       username = ubuntu   ^|   cloud-init = OFF   ^|   Public IP = Auto
echo       Platform = H100 x8 (eu-north1), boot disk 1280 GiB default
echo ------------------------------------------------------------
type "%KEY%.pub"
echo ------------------------------------------------------------
echo Create the VM, wait for Running, copy its Public IPv4.
echo ============================================================
echo.
set /p IP=Enter VM public IP:
if "%IP%"=="" (echo No IP entered. & pause & exit /b 1)

echo.
echo [3/5] Testing SSH ...
ssh -i "%KEY%" %SSHOPT% ubuntu@%IP% "echo OK; nvidia-smi -L | wc -l" || (echo [ERROR] SSH failed. Check VM Running, key pasted, IP correct. & pause & exit /b 1)

echo.
echo [4/5] Clone repo + upload data (~1GB, a few minutes) ...
ssh -i "%KEY%" %SSHOPT% ubuntu@%IP% "git clone https://github.com/Jkim1647/AI_2_CHALLENGE ~/AI_2_CHALLENGE 2>/dev/null; cd ~/AI_2_CHALLENGE && git checkout feat/9b-setup-dev-relabel && git pull --ff-only"
scp -i "%KEY%" %SSHOPT% -r "%DATA%" ubuntu@%IP%:~/ssafy-16-2-ai || (echo [ERROR] scp failed. Check path: %DATA% & pause & exit /b 1)

echo.
echo [5/5] Launching 397B run (nohup - keeps running even if you close this) ...
ssh -i "%KEY%" %SSHOPT% ubuntu@%IP% "cd ~/AI_2_CHALLENGE/tools/nebius && chmod +x run_397b_ssh.sh && REPO=$HOME/AI_2_CHALLENGE DATA=$HOME/ssafy-16-2-ai nohup bash run_397b_ssh.sh > ~/run.log 2>&1 & echo started"

echo.
echo ============================================================
echo Showing live log. Ctrl+C stops the VIEW only (job keeps running).
echo ~1-1.5h total. If weight-load stalls at GPU 0%% for 10-15min = hang; tell Claude.
echo At the end, copy the "ALL DONE" and accuracy lines to Claude.
echo ============================================================
echo.
ssh -i "%KEY%" %SSHOPT% ubuntu@%IP% "tail -n +1 -f ~/run.log"
echo.
echo Re-view log later:  ssh -i "%KEY%" ubuntu@%IP% "tail -f ~/run.log"
echo Get results:        scp -i "%KEY%" ubuntu@%IP%:~/nebius_397b_results.tar.gz .
echo IMPORTANT: delete the VM in the Nebius console when done (stops billing).
pause
endlocal
