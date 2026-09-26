"""Colab Drive의 실험 산출물(예측·로그·meta·adapter)을 팀 저장소 브랜치에 올린다.

토큰은 Colab 보안 비밀 `GH_TOKEN`(노트북 액세스 허용)에서 읽고, git에는 `http.extraHeader`로만 넘긴다.
remote URL·.git/config·출력에 토큰을 남기지 않는다. 강제 push는 하지 않는다.

    !python colab_drive_to_git.py --src /content/drive/MyDrive/ai2_35b_out --repo /content/repo \
        --dest shared_predictions/colab_ai2_35b_out --branch feat/9b-setup-dev-relabel

- GitHub 파일 한도(100MB) 때문에 --max-mb(기본 45MB)를 넘는 파일은 복사하지 않고
  `<dest>/LARGE_FILES.md`에 경로·크기·SHA-256만 기록한다(원본은 Drive에 남는다).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import subprocess
from pathlib import Path

SKIP_SUFFIX = {".tar", ".zip", ".pt", ".pth", ".bin"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--repo", required=True, help="이미 clone된 저장소 경로")
    ap.add_argument("--dest", required=True, help="저장소 안의 복사 위치")
    ap.add_argument("--branch", required=True)
    ap.add_argument("--max-mb", type=float, default=45.0)
    ap.add_argument("--message", default="Add Colab Drive experiment outputs (predictions, logs, meta)")
    args = ap.parse_args()

    from google.colab import userdata
    tok = userdata.get("GH_TOKEN")
    hdr = "Authorization: Basic " + base64.b64encode(("x-access-token:" + tok).encode()).decode()

    def git(*a: str) -> str:
        r = subprocess.run(["git", "-c", "http.extraHeader=" + hdr, *a], capture_output=True, text=True, cwd=args.repo)
        out = (r.stdout + r.stderr).replace(tok, "***")
        if r.returncode:
            raise SystemExit(f"git {a[0]} 실패: {out[-500:]}")
        return out

    src, dest = Path(args.src), Path(args.repo) / args.dest
    large = []
    n = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        size_mb = p.stat().st_size / 1e6
        if p.suffix in SKIP_SUFFIX or size_mb > args.max_mb:
            large.append((str(rel), round(size_mb, 1), sha256(p)))
            continue
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest / rel)
        n += 1
    lines = ["# Drive에만 있는 큰 파일", "",
             f"원본 위치: 팀 계정 Drive `{args.src.replace('/content/drive/', '')}`. GitHub 한도 때문에 git에는 올리지 않았다.", "",
             "| 경로 | MB | SHA-256 |", "|---|---|---|"]
    lines += [f"| `{r}` | {mb} | `{h}` |" for r, mb, h in large]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "LARGE_FILES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"복사 {n}개, 큰 파일 {len(large)}개(목록만 기록)")

    git("pull", "--ff-only", "origin", args.branch)
    git("add", args.dest)
    status = git("status", "--short")
    if not status.strip():
        print("변경 없음")
        return 0
    git("-c", "user.name=JKim1647", "-c", "user.email=<email>",
        "commit", "-q", "-m", args.message + "\n\nCo-Authored-By: Claude Opus 5 <<email>>")
    git("push", "origin", f"HEAD:refs/heads/{args.branch}")
    print("push 완료", git("log", "--oneline", "-1").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
