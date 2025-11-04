# scripts/p02_generate_feature_manifest.py
import hashlib
import json
import random
import re
from pathlib import Path

import yaml

RFI_JSON = Path("output/reports/diag_bundle/data_schema_and_missing.json")
CATS_YAML= Path("configs/atft/feature_categories.yaml")
OUT_YAML = Path("output/reports/feature_manifest_306.yaml")
SEED     = 42

def load_schema():
    j = json.loads(RFI_JSON.read_text())
    # 期待：schema 全列, missing 率（可能なら全列）。無い列は1.0扱いで後段スコアに反映。
    schema = j.get("schema", {})  # col -> dtype
    # 可能なら full missing map を優先。無ければ top_missing から補完（不足は1.0）
    miss_map = {k: v for k,v in j.get("missing_by_col", {}).items()} if "missing_by_col" in j else {}
    for k,v in j.get("top_missing", []):
        miss_map[k] = v
    return list(schema.keys()), miss_map

def load_categories():
    y = yaml.safe_load(CATS_YAML.read_text())
    # 例：{'core': ['adjustment_*','return_*d'], 'technical':['rsi*','macd*',...], 'fundamental':['stmt_*'], ...}
    return y

def pat(p):  # glob-like -> regex
    return re.compile("^" + p.replace("*", ".*").replace("?", ".") + "$")

def select_306(cols, miss_map, cats):
    random.seed(SEED)
    # 1) 必須コア（順序固定）
    core_pats = cats.get("core", ['adjustment_*','return_*d','volume*','turnover*'])
    core = []
    for gp in core_pats:
        rg = pat(gp)
        core += [c for c in cols if rg.match(c)]
    core = sorted(set(core))  # 安定順

    # 2) 除外候補（本当に定数か/全欠損かはRFI-2に準拠、無ければ missing>=0.98 を強除外）
    hard_drop = set([c for c,v in miss_map.items() if v>=0.98])

    # 3) 残りカテゴリの配分
    quotas = cats.get("quotas", {
        "technical": 160,   # RSI/ADX/ATR/EMA/MACD/NATR...（環境に合わせて調整可）
        "flow": 40,         # 流動性/需給
        "graph": 16,        # graph_* などの外因特徴
        "fundamental": 20,  # stmt_*
        "misc": 70          # x_*, dmi_*, margin_* 等をここへ
    })

    # カテゴリのパターン
    cat_pats = {
        "technical": cats.get("technical", ['rsi*','adx*','atr*','ema*','macd*','natr*']),
        "flow": cats.get("flow", ['turnover*','*volume*','*liquidity*']),
        "graph": cats.get("graph", ['graph_*']),
        "fundamental": cats.get("fundamental", ['stmt_*']),
        "misc": cats.get("misc", ['dmi_*','x_*','margin_*','*zscore*'])
    }
    def bucketize(name):
        for cat, pats in cat_pats.items():
            if any(pat(p).match(name) for p in pats):
                return cat
        return "misc"

    # 4) スコアリング（低欠損・高分散・短名優先）: 分散は未提供の可能性があるため欠損率中心に
    def score(c):
        m = miss_map.get(c, 1.0)
        return (1 - m, -len(c))  # 高スコアが先

    picked = []
    picked += [c for c in core if c not in hard_drop]
    picked = sorted(set(picked))
    # 5) コアで306超えたら切る（稀）。足りなければカテゴリ配分で埋める
    def fill_from_pool(pool, need):
        pool = [c for c in pool if c not in hard_drop and c not in picked]
        pool = sorted(pool, key=score, reverse=True)
        return [c for c in pool[:need]]

    remaining = 306 - len(picked)
    by_cat = {k:[] for k in quotas}
    if remaining > 0:
        # バケット候補
        bucketed = {k:[] for k in quotas}
        for c in cols:
            if c in picked or c in hard_drop: continue
            bucketed.setdefault(bucketize(c), []).append(c)
        for cat, q in quotas.items():
            take = min(q, remaining)
            sel = fill_from_pool(bucketed.get(cat, []), take)
            by_cat[cat] = sel
            remaining -= len(sel)
            picked += sel
            if remaining <= 0: break

    # 6) 足りない場合は全体から補充
    if remaining > 0:
        leftovers = [c for c in cols if c not in picked and c not in hard_drop]
        picked += fill_from_pool(leftovers, remaining)

    # 7) 安定順に最終整列（カテゴリー→名前）で determinismを担保
    def cat_order(c):
        order = ["core"] + list(quotas.keys())
        return (0 if c in core else 1 + order.index(bucketize(c)))
    picked = sorted(set(picked), key=lambda c: (cat_order(c), c))
    assert len(picked) == 306, f"selected {len(picked)} != 306"
    # 8) 指紋（ABI）
    abi = hashlib.sha1((",".join(picked)).encode()).hexdigest()
    return picked, {"abi_sha1": abi, "n_total": len(cols), "n_drop": len(hard_drop), "n_core": len(core)}

def main():
    cols, miss = load_schema()
    cats = load_categories()
    feats, meta = select_306(cols, miss, cats)
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({"features": feats, "meta": meta}, OUT_YAML.open("w"), sort_keys=False, allow_unicode=True)
    print("Saved:", OUT_YAML, "ABI:", meta["abi_sha1"])

if __name__ == "__main__":
    main()
