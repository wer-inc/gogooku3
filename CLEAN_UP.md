わかる、その状況で一気に消すのは事故のもと…。
**“絶対にミスしない”ためのベルト＆サスペンダー方式**で、実行ファイル2つを起点に「実行に本当に必要なファイル＝許可リスト（allowlist）」を機械的に作り、段階的にクリーンアップする手順を用意しました。すべてローカルで完結・可逆です。

# 手順（安全度MAXの順に実行）

## 0) 退避と検証用ブランチ

```bash
git switch -c cleanup/safe
git tag pre-cleanup-$(date +%Y%m%d)
```

## 1) 実行時に本当に使ったファイルを収集（動的トレース）

**coverage** を使うと、実行されたPythonファイル一覧が取れます。入口スクリプトが `run_a.py` と `run_b.py` と仮定：

```bash
pip install coverage

# 実行1
coverage run --parallel-mode run_a.py <必要なら引数>
# 実行2（別の入口）
coverage run --parallel-mode run_b.py <必要なら引数>

# レポートを統合してjson出力
coverage combine
coverage json
```

生成された `coverage.json` の `"files"` に「実際に走ったコード」のパスが入ります。
（＝**最低限の“確実に必要”セット**）

> 入力パターンが複数あるなら、代表的なパターンを全部流しておくと安全度UP。

## 2) import 依存も拾う（静的解析）

実行時には通らなかったが **import されうる** 依存も拾います。例：`pydeps` か `modulegraph`。

```bash
pip install pydeps
# パッケージ単位で依存を吐く。package_name はあなたのパッケージ名 or ルート
pydeps package_name --max-bacon=3 --show-deps --noshow
# 生成物（.dot/.json）からファイル一覧を抽出（後述スクリプトでOK）
```

## 3) 非Python系の同梱資産も加える

* `pyproject.toml` / `setup.cfg` / `requirements*.txt`
* `configs/`, `migrations/`, `templates/`, `assets/`, `py.typed`
* 実行時に読み込む `.yaml/.json/.csv/.sql` 等
  → **設定ファイルのパスはコードから grep** で拾うのが確実です。

例：

```bash
rg -n "open\(|Path\(|yaml\.safe_load|json\.load|read_csv|read_parquet" -g '!venv' -g '!**/__pycache__' -S .
```

## 4) “残すべきファイル”の許可リスト（allowlist）を作る

下のスクリプトが、(1)coverage、(2)pydeps、(3)手動追加の固定ファイル群をマージして **allowlist.txt** を作ります。

```python
# scripts/build_allowlist.py
import json, subprocess, sys, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]

# 1) coverage.json
cov_files = set()
cov = json.loads((ROOT/"coverage.json").read_text())
for f in cov.get("files", {}):
    p = str(pathlib.Path(f).resolve())
    if p.startswith(str(ROOT)):
        cov_files.add(str(pathlib.Path(p).relative_to(ROOT)))

# 2) pydeps出力（.json or .dot）を素朴に抽出
pydeps_files = set()
for path in ROOT.glob("**/*.dot"):
    lines = path.read_text(errors="ignore").splitlines()
    for line in lines:
        m = re.findall(r'"([^"]+\.py)"', line)
        for hit in m:
            try:
                r = str(pathlib.Path(hit).resolve().relative_to(ROOT))
                pydeps_files.add(r)
            except Exception:
                pass

# 3) 常備ファイルやディレクトリ
fixed = {
    "pyproject.toml", "setup.cfg", "setup.py", "requirements.txt",
    "README.md", "LICENSE", ".gitignore",
}
# よくあるディレクトリ
fixed_dirs = ["configs", "config", "migrations", "templates", "assets", "scripts"]

fixed_files = set()
for f in fixed:
    if (ROOT/f).exists():
        fixed_files.add(f)
for d in fixed_dirs:
    if (ROOT/d).exists():
        for p in (ROOT/d).rglob("*"):
            if p.is_file():
                fixed_files.add(str(p.relative_to(ROOT)))

allow = sorted(cov_files | pydeps_files | fixed_files)
(out := ROOT/"allowlist.txt").write_text("\n".join(allow) + "\n")
print(f"Wrote {out} with {len(allow)} files")
```

> 必要に応じて “手動で残したいもの” を `fixed` / `fixed_dirs` に足してください。

## 5) 候補の削除対象一覧を作る（まだ消さない）

**Gitで管理中の全ファイル** から allowlist を引いて、**削除候補リスト** を作るだけ：

```bash
# 追跡中ファイル一覧
git ls-tree -r --name-only HEAD > all_tracked.txt

# 正規化（ユニーク）
sort -u all_tracked.txt -o all_tracked.txt
sort -u allowlist.txt -o allowlist.txt

# 候補 = 追跡中 - allow
comm -23 all_tracked.txt allowlist.txt > deletion_candidates.txt
echo "候補数: $(wc -l < deletion_candidates.txt)"
```

ここで **人の目でサッと見て**、消してよいものか感触をつかむ（ログ・実験ノート・古いノートブック等が並ぶはず）。

## 6) “隔離” にとどめる（削除ではなく移動）

誤りゼロのため、まず **リポジトリ内の `trash/` へ mv**（= いつでも戻せる）＋CIで動作確認。

```bash
mkdir -p trash
xargs -I{} -a deletion_candidates.txt bash -c 'd=$(dirname "{}"); mkdir -p "trash/$d"; git mv "{}" "trash/{}"'
git commit -m "chore(cleanup): quarantine unused files into trash/ (no deletions)"
```

## 7) 自動テスト & 実行確認

* 入口2つを **CIで実行**（代表的な引数・入力で）
* 成功するまで `trash/` から戻すものがないか確認
* 問題なければ PR 作成

## 8) 一定期間後に完全削除

「1～2週間問題なし」を確認できたら **削除**：

```bash
git rm -r trash/
git commit -m "chore(cleanup): remove quarantined files after soak period"
```

---

# オプション（さらに堅牢に）

* **“起動時にimport監査”**：`sitecustomize.py` に import ログを挟み、どのモジュールが読み込まれたか毎回記録（予期せぬパスを検出可能）。
* **プリフライトCI**：`deletion_candidates.txt` が空であること、`allowlist.txt` が存在することをPRでチェック。
* **ブロックリストの自動生成**：`notebooks/`, `experiments/`, `old/` など歴史的フォルダは最初から `trash/` 候補に。

---

# これで得られること

* **可逆**：最初は「移動」だけ。いつでも戻せる。
* **機械的**：実行トレース＋依存解析＋設定 grep の **和集合**＝安全側。
* **段階的**：隔離 → soak → 削除で **事故ゼロ** を担保。

---

必要なら、あなたのリポ構成（入口2つのパス、パッケージ名、設定ディレクトリ）に合わせて **`build_allowlist.py` をそのまま使える形に調整**して出します。
