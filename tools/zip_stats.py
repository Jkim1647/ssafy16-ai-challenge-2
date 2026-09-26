"""Inspect ssafy-16-2-ai.zip without extracting: central-directory size stats per split,
plus an attempt to extract every entry that is NOT password-protected."""
import zipfile
import os
import sys

ZIP_PATH = os.path.join(os.path.dirname(__file__), "..", "ssafy-16-2-ai.zip")
OUT_TXT = os.path.join(os.path.dirname(__file__), "zip_stats.txt")
EXTRACT_DIR = os.path.join(os.path.dirname(__file__), "..", "extracted_unlocked")


def pct(sorted_list, p):
    if not sorted_list:
        return 0
    k = (len(sorted_list) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_list) - 1)
    if f == c:
        return sorted_list[f]
    return sorted_list[f] + (sorted_list[c] - sorted_list[f]) * (k - f)


def describe(name, sizes, log):
    if not sizes:
        log(f"{name:8} n=0")
        return
    s = sorted(sizes)
    n = len(s)
    log(f"{name:8} n={n:7} min={s[0]:8} p10={int(pct(s,0.10)):8} p25={int(pct(s,0.25)):8} "
        f"med={int(pct(s,0.50)):8} p75={int(pct(s,0.75)):8} p90={int(pct(s,0.90)):8} "
        f"max={s[-1]:9} mean={sum(s)//n:8}")


def main():
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    if not os.path.exists(ZIP_PATH):
        log(f"ZIP not found at {ZIP_PATH}")
        sys.exit(1)

    z = zipfile.ZipFile(ZIP_PATH)
    infos = z.infolist()
    log(f"Total entries: {len(infos)}")

    by_split = {"train": [], "dev": [], "test": [], "other": []}
    csv_nb = []
    for i in infos:
        if i.is_dir():
            continue
        name = i.filename
        lname = name.lower()
        if lname.endswith(".jpg") or lname.endswith(".jpeg"):
            placed = False
            for split in ("train", "dev", "test"):
                if name.split("/")[0].startswith(split) or name.lower().startswith(split):
                    by_split[split].append(i.file_size)
                    placed = True
                    break
            if not placed:
                by_split["other"].append(i.file_size)
        elif lname.endswith(".csv") or lname.endswith(".ipynb"):
            csv_nb.append((name, i.file_size, i.compress_size))

    log("\n--- JPEG size distribution (bytes) by split ---")
    for split in ("train", "dev", "test", "other"):
        describe(split, by_split[split], log)

    log("\n--- CSV / notebook files ---")
    for name, size, csize in csv_nb:
        log(f"{name}  size={size} compressed={csize}")

    # resolution estimate assuming ~10:1 jpeg-quality-dependent bpp guess is unreliable;
    # just report bytes/pixel-ish heuristic ranges for common qualities
    log("\n--- rough resolution guess (assumes ~0.5-1.5 bytes/px at typical JPEG quality) ---")
    all_sizes = [s for split in ("train", "dev", "test", "other") for s in by_split[split]]
    if all_sizes:
        med = sorted(all_sizes)[len(all_sizes) // 2]
        for bpp in (0.3, 0.5, 0.8, 1.2, 1.5):
            px = med / bpp
            log(f"  if {bpp} bytes/px -> ~{px:,.0f} px total (e.g. {int((px*4/3)**0.5)}x{int((px*3/4)**0.5)})")

    # encryption check + selective extraction of unlocked files
    log("\n--- encryption / extraction attempt ---")
    encrypted = 0
    unlocked = 0
    extract_errors = 0
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    for i in infos:
        if i.is_dir():
            continue
        is_enc = bool(i.flag_bits & 0x1)
        if is_enc:
            encrypted += 1
            continue
        try:
            z.extract(i, EXTRACT_DIR)
            unlocked += 1
        except Exception as e:
            extract_errors += 1

    log(f"encrypted entries: {encrypted}")
    log(f"unlocked entries extracted: {unlocked}")
    log(f"extraction errors: {extract_errors}")
    log(f"extracted to: {os.path.abspath(EXTRACT_DIR)}")

    with open(OUT_TXT, "w", encoding="utf-8-sig") as f:  # BOM: 한글 포함 .txt를 메모장에서 열 때 안 깨지게
        f.write("\n".join(lines))
    log(f"\nSaved report to {OUT_TXT}")


if __name__ == "__main__":
    main()
