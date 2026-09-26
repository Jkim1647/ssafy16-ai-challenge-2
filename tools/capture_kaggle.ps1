# Kaggle 대회 페이지를 README용 이미지로 캡처한다 (Windows 전용).
#
# 로그인된 Chrome 프로필로 새 창을 띄우고, 마우스 휠로 스크롤한 뒤 PrintWindow 로 창을 찍는다.
# 브라우저 UI·왼쪽 메뉴·쿠키 배너는 잘라내 Kaggle 본문만 남긴다. 찍은 창만 닫고 기존 창·탭은 건드리지 않는다.
# 캡처하는 몇 초 동안 새 창이 화면 맨 앞에 뜨고 마우스 커서가 움직인다.
#
# 사용 예 (대회 종료 후 Private 리더보드):
#   powershell -ExecutionPolicy Bypass -File tools/capture_kaggle.ps1 -Page leaderboard -Wheel 6 `
#       -Out reports/figures/kaggle/02_leaderboard_private_20261002.png
param(
    [Parameter(Mandatory)] [string]$Page,          # overview | leaderboard | submissions ...
    [Parameter(Mandatory)] [string]$Out,
    [int]$Wheel = 0,                               # 아래로 굴릴 휠 칸 수
    [string]$Competition = "ssafy-16-2-ai-9-21-9-28",
    [string]$Chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe",
    [int]$LoadSeconds = 9
)
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System; using System.Text; using System.Collections.Generic; using System.Runtime.InteropServices;
public class KaggleCap {
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc f, IntPtr l);
  [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint f);
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int ht, bool r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, int d, UIntPtr e);
  public static List<IntPtr> All() { var l = new List<IntPtr>(); EnumWindows((h, p) => { if (IsWindowVisible(h)) l.Add(h); return true; }, IntPtr.Zero); return l; }
  public static string Title(IntPtr h) { var s = new StringBuilder(512); GetWindowText(h, s, 512); return s.ToString(); }
}
"@
[KaggleCap]::SetProcessDPIAware() | Out-Null

$url = "https://www.kaggle.com/competitions/$Competition/$Page"
$before = @([KaggleCap]::All())
Start-Process $Chrome -ArgumentList "--new-window", $url
Start-Sleep -Seconds $LoadSeconds
$win = [KaggleCap]::All() | Where-Object { $before -notcontains $_ -and ([KaggleCap]::Title($_)) -like "*Chrome*" } | Select-Object -First 1
if (-not $win) { throw "새 Chrome 창을 찾지 못했다. Chrome 경로와 로그인 상태를 확인할 것." }

[KaggleCap]::MoveWindow($win, 0, 0, 1500, 1000, $true) | Out-Null
Start-Sleep -Seconds 2
if ($Wheel -gt 0) {
    [KaggleCap]::SetForegroundWindow($win) | Out-Null; Start-Sleep -Milliseconds 400
    [KaggleCap]::SetCursorPos(1380, 700) | Out-Null
    for ($i = 0; $i -lt $Wheel; $i++) { [KaggleCap]::mouse_event(0x0800, 0, 0, -120, [UIntPtr]::Zero); Start-Sleep -Milliseconds 120 }
    [KaggleCap]::SetCursorPos(1495, 1090) | Out-Null   # 체크박스 툴팁이 찍히지 않게 커서를 치운다
    Start-Sleep -Seconds 2
}

$r = New-Object KaggleCap+RECT; [KaggleCap]::GetWindowRect($win, [ref]$r) | Out-Null
$full = New-Object System.Drawing.Bitmap ($r.R - $r.L), ($r.B - $r.T)
$g = [System.Drawing.Graphics]::FromImage($full)
$hdc = $g.GetHdc(); [KaggleCap]::PrintWindow($win, $hdc, 2) | Out-Null; $g.ReleaseHdc($hdc)
[KaggleCap]::PostMessage($win, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null   # WM_CLOSE: 방금 연 창만 닫는다

# 1500x1000 창 기준: 위 182px = 탭·주소창·Chrome 알림, 왼쪽 266px = Kaggle 메뉴, 아래 945px 이후 = 쿠키 배너
$crop = New-Object System.Drawing.Rectangle 266, 182, ([Math]::Min(1234, $full.Width - 266)), ([Math]::Min(763, $full.Height - 182))
$img = $full.Clone($crop, $full.PixelFormat)
New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
$img.Save((Resolve-Path -LiteralPath (Split-Path $Out)).Path + "\" + (Split-Path $Out -Leaf), [System.Drawing.Imaging.ImageFormat]::Png)
"saved $Out ($($img.Width)x$($img.Height))"
