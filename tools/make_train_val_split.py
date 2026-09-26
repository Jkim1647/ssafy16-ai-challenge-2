"""train에서 순수 holdout 검증셋을 떼어 data_meta/train_val_split.csv로 고정한다.

왜 train에서 떼는가: dev는 검증셋으로 못 쓴다. 주최측 방송에 따르면 5명 라벨러 중
4명 이상이 합의한 문항만 train/test로 채택되고, 미달 문항이 dev로 간다. 따라서 dev는
'어려운/합의 안 된' 문항만 모인 편향 표본이라 train/test와 분포가 다르다.

왜 베이스라인 valid와 다른가: 베이스라인 valid는 학습 루프 안에서 로스 계산에 쓰여
성능이 과대평가된다(주최측 방송에서 직접 지적). 이 split의 val은 학습에 일절 쓰지 않는
순수 holdout이다.

중복 처리: train 내부 이미지 중복(같은 이미지가 다른 질문으로 두 번)을 train/val로
가르면 누수다. SHA-256 완전 동일 + 재인코딩(dHash<=2를 픽셀 MAE로 확정) 쌍을 찾아
같은 폴드(train)에 함께 둔다. 2026-09-21 실측: 확정 9쌍(18장, 0.33%).

재현: 결정론적이다. seed 고정 + id 정렬 후 층화 추출. 팀 전원이 이 CSV를 그대로 쓴다.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "ssafy-16-2-ai"
OUT_CSV = REPO / "data_meta" / "train_val_split.csv"
SEED = 1              # base.yaml seed와 동일. split 재현 기준.
VAL_FRACTION = 0.10
PIXEL_MAE_MAX = 4.0   # 32x32 gray MAE가 이 이하면 재인코딩 중복으로 확정


def dhash(path: Path, hash_size: int = 8) -> int:
    im = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    diff = (a[:, 1:] > a[:, :-1]).flatten()
    bits = 0
    for b in diff:
        bits = (bits << 1) | int(b)
    return bits


def fingerprint(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L").resize((32, 32), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32)


def find_duplicate_pairs(paths: dict[str, Path]) -> list[tuple[str, str]]:
    ids = list(paths)
    # 1) SHA-256 완전 동일
    sha: dict[str, list[str]] = {}
    for i in ids:
        sha.setdefault(hashlib.sha256(paths[i].read_bytes()).hexdigest(), []).append(i)
    pairs: set[tuple[str, str]] = set()
    for g in sha.values():
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                pairs.add(tuple(sorted((g[a], g[b]))))
    # 2) dHash<=2 후보를 픽셀 MAE로 확정 (재인코딩 중복)
    dh = np.array([dhash(paths[i]) for i in ids], dtype=np.uint64)
    fp: dict[str, np.ndarray] = {}
    for a in range(len(ids) - 1):
        xor = np.bitwise_xor(dh[a + 1:], dh[a])
        dist = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for k in np.nonzero(dist <= 2)[0]:
            ia, ib = ids[a], ids[a + 1 + int(k)]
            for x in (ia, ib):
                if x not in fp:
                    fp[x] = fingerprint(paths[x])
            if float(np.abs(fp[ia] - fp[ib]).mean()) <= PIXEL_MAE_MAX:
                pairs.add(tuple(sorted((ia, ib))))
    return sorted(pairs)


def union_find_groups(ids: list[str], pairs: list[tuple[str, str]]) -> dict[str, int]:
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    roots = {i: find(i) for i in ids}
    # 그룹 id 부여: 2장 이상인 그룹만 번호를 매기고, 단독은 빈 값
    from collections import Counter
    size = Counter(roots.values())
    group_num, next_id = {}, 1
    out = {}
    for i in ids:
        r = roots[i]
        if size[r] > 1:
            if r not in group_num:
                group_num[r] = next_id
                next_id += 1
            out[i] = group_num[r]
    return out


def main() -> None:
    df = pd.read_csv(ROOT / "train.csv", encoding="utf-8-sig").sort_values("id").reset_index(drop=True)
    paths = {r["id"]: ROOT / r["path"] for _, r in df.iterrows()}

    pairs = find_duplicate_pairs(paths)
    dup_group = union_find_groups(list(paths), pairs)
    dup_ids = set(dup_group)
    print(f"확정 중복 쌍: {len(pairs)}쌍, 중복 이미지 {len(dup_ids)}장")

    # 중복 이미지는 전부 train으로 (같은 폴드에 묶어 누수 차단). 나머지는 층화 추출.
    singles = df[~df["id"].isin(dup_ids)]
    rng_val_ids: list[str] = []
    for ans, grp in singles.groupby("answer"):
        n_val = round(len(grp) * VAL_FRACTION)
        rng_val_ids += grp.sample(n=n_val, random_state=SEED)["id"].tolist()
    val_ids = set(rng_val_ids)

    df["split"] = df["id"].apply(lambda i: "val" if i in val_ids else "train")
    df["dup_group"] = df["id"].map(dup_group).astype("Int64")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # 커밋 대상 파일에는 정답(train 라벨)·경로를 넣지 않는다 — 경쟁 데이터를 저장소에
    # 올리지 않는다는 원칙(AGENTS.md)에 맞춘다. 필요하면 각자 train.csv에 id로 join한다.
    # utf-8-sig: data_meta 산출물 — 사람이 Excel로 열 수 있어 정책상 BOM을 붙인다.
    df[["id", "split", "dup_group"]].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # 검증 출력
    print(f"\n저장: {OUT_CSV}")
    print("split 분포:", df["split"].value_counts().to_dict())
    print("val 비율:", round((df["split"] == "val").mean(), 4))
    print("\n정답 분포 (train / val 층화 확인):")
    for ans in sorted(df["answer"].unique()):
        tr = int(((df["split"] == "train") & (df["answer"] == ans)).sum())
        va = int(((df["split"] == "val") & (df["answer"] == ans)).sum())
        tr_p = tr / (df["split"] == "train").sum()
        va_p = va / (df["split"] == "val").sum()
        print(f"  {ans}: train {tr} ({tr_p:.3f}) | val {va} ({va_p:.3f})")
    # 누수 자기검증: val에 든 이미지 중 중복쌍 상대가 train에 있는지
    leak = [ (a,b) for a,b in pairs if (a in val_ids) != (b in val_ids) ]
    print("\n중복쌍이 train/val로 갈린 건수(0이어야 함):", len(leak))
    print("확정 중복 쌍 목록:")
    for a, b in pairs:
        print(f"  {a} ~ {b}")


if __name__ == "__main__":
    main()
