# 外部レビュー(2026-09-04)の整理

作成日: 2026-09-04
対象: PR #3 コメント「メモ: 外部での会話まとめ」(head `c2855c3` に対する静的レビュー)
状態: **決定済み(§5)・A 完了・B 完了**(結果は `controlled_eval.md`)

レビューの指摘をコードと PR の状態に照合し、(A) 本 PR で閉じるもの、
(B) 次 PR の対照実験、(C) その後の生成器 v2、(D) 保留 — に分類した。

## 1. 指摘の照合結果

すべて事実として確認できた。過大主張になっていた箇所は §3-A-2 で是正する。

| # | 指摘 | 照合 | 根拠 |
|---|---|---|---|
| 1-1 | 案 2″ は REF 窓の上限を除去していない | ✓ | `training/prepare_tokens.py` Pass 1 が**全 split** を走査して窓を決める。`train_ar.candidate_mask` は `dist <= max_k`(`train_ar.py:135`)。文脈距離ヘッドは `Linear(d_model, max_k + 1)`(`:203`)。入力は `REF_k` 別トークン埋め込みのまま。`model_input.tokenize` は窓超過で `ValueError`(`model_input.py:116-120`) |
| 1-2 | 案 2″ の「非制約 WF」は baseline と同条件でない | ✓ | `--pointer-legal` は学習・teacher-forced 評価(`token_logprobs`、`:270-276`)と生成(`_sample_pointer_step`、`:421`、`allowed=None` でも)の両方で合法候補ゼロ時に REF 種別をマスクする。accuracy は pointer 時 type-level(`run_epoch`、`:306`) |
| 1-3 | 単一学習 seed、同一 val の反復利用 | ✓ | 全 run が学習 seed 1 本。test split なし。案 1→3′ の判定に同じ val を使用 |
| 1-4 | 「難度は k の関数」に頻度交絡が残る | ✓ | REF NLL-by-k 表に件数・p(k) の併記なし(`eval_axes.ref_nll_by_k` は `min_count=10` で切るのみ)。unigram 対照なし |
| 1-5 | generator 名が固定 | ✓ | `build_dataset(generator=...)` は注入可(`dataset.py:95`)だが manifest / provenance は `GENERATOR_NAME` 固定(`:136`, `:162`) |
| 1-6 | 同型判定が孤立ノードを無視 | ✓ | `_digraph` は edges のみから構築(`dataset.py:55`)。現生成器は線形骨格を先に張る(`generate.py:34`)ので孤立ノードは出ない — 影響は生成器 family 追加後 |
| 1-7 | PR がそのままではマージ不可 | ✓ | 本文は Schema draft / 21 tests 時点のまま。38 commits / 45 files / +5,701 −112。`mergeable: CONFLICTING`(3 ファイル)。CI なし(`.github/` 自体がない)。training テストは意図的に torch-free(pointer の mask / NLL / sampling / 勾配は未テスト) |

文書側の確認: `representation_experiments.md` は案 2′ 以降「語彙窓内」と明記して
おり実装との齟齬はないが、`structure_sweep.md:151` の動機(「REF 窓の上限も消える」)
は**未達**のまま結論に至っている。

## 2. 追加で判明した事実(main との関係)

競合の原因を調べた結果、レビューにない事実が見つかった。

- **MetaGraph 本体は 2026-06-29 に PR #1 として main へマージ済み**(`34abf1b`)。
  内容は本ブランチの `438ef97` と同一(`metagraph.py` / `types.py` に差分なし)。
- 本ブランチの merge-base は `e1801ba`(PR #1 より前)。PR #3 は MetaGraph 本体を
  重複して含んでいる。
- main 側の追加(本ブランチに未取り込み):
  - `3a214a8`: `tests/test_metagraph.py` に 4 テスト追加(linear chain, multi-exit
    loop, DAG invariant, build_cfg integration)、`main.py` の `build_cfg` を
    engine 省略可 + `GraphEngine` 返却に変更
  - `f3acb9a`: `docs/discussion_log.md` に Session 3(階層展開 Graph Transformer
    の設計議論 + 外部レビュー、未決事項 A–F)
- 競合 3 ファイルの性質:
  - `cfg_reducer/__init__.py`: export の差(本ブランチが `generate` / `model_input`
    を追加)— 合併で解決
  - `main.py`: main は `build_cfg` を改変、本ブランチは `generate.py` へ移設 —
    本ブランチ側を採用
  - `tests/test_metagraph.py`: add/add(main 7 tests vs 本ブランチ 3 tests)—
    main の 7 テストを取り込み、`from main import build_cfg` 依存の integration
    テストは `generate_cfg` 経由に書き換え
- `pyproject.toml` / `uv.lock`: main も ipykernel / ipython を含まない →
  A1-b(未コミット維持)は継続。
- `docs/discussion_log.md` は本ブランチで未変更 → 競合なし。

## 3. 分類と対応方針

### A. 本 PR で閉じる(マージ可能化 + 記録の是正)

| # | 作業 | 備考 |
|---|---|---|
| A-1 | main を `feat/metagraph` へ **merge**(rebase しない) | PR コメントが commit hash を参照しているため履歴を保つ。競合解決は §2 の方針 |
| A-2 | 文書の是正 | 案 2″ の「非制約 WF」→ **reference-constrained WF** に改名し、baseline の raw WF と同条件でない旨を注記。結論を「採用」→「**暫定第一候補**(baseline と同等以上の可能性が高い)」に弱める。「REF 窓の上限が消える」は未達と明記。WF の名称を raw / reference-constrained / fully-constrained の 3 種に統一(`eval_axes.md`) |
| A-3 | 小修正(テスト付き) | generator descriptor(name / version / config / commit / dirty)を `build_dataset` に渡し manifest・provenance に記録。`fingerprint` / `is_structural_duplicate` を nodes + edges から構築 |
| A-4 | PR 本文の更新 | 到達点、再現条件(データ生成 → tokenize → Colab 学習 → 評価)、除外事項 |
| A-5 | CI 追加 | GitHub Actions: `uv sync` → `pytest` → `ty check`。torch 系は対象外 |
| A-6 | REF NLL-by-k に件数と p(k) を併記 | `eval_axes` の出力拡張のみ。B-6 の前提 |

### B. 次 PR: 対照実験と評価の厳密化(生成器変更の前)

| # | 作業 | 目的 |
|---|---|---|
| B-1 | 語彙・モデル形状を **train のみ**から決定 | `prepare_tokens` の窓を train で固定。val / test の窓超過サンプルは除外数を meta に記録 |
| B-2 | train / dev / test の三分割 | seed 範囲を追加(例: test = `[base+2200, base+2400)`)。test は設計判断に使わない |
| B-3 | 対照 5 構成 | raw baseline / **baseline + 参照合法マスク** / distance-only / pointer-only / 案 2″(参考: 案 3′)。優先は baseline・baseline+mask・案 2″ |
| B-4 | 学習 seed 3 本(可能なら 5) | 同一 dataset・同一 seed 番号を構成間で対応。サンプリング乱数も対応 |
| B-5 | 統計 | 同一 test サンプル上の paired ΔNLL、WF の Wilson 区間、NLL-by-k の件数 + bootstrap 区間、**edge / pointer accuracy**(baseline と pointer を同じ指標に変換) |
| B-6 | k の頻度交絡の分離 | p(k) 併記、unigram / 条件付き頻度 baseline、`NLL(k) − (−log p(k))`、k 同数サンプリング、候補数・階層位置・in-degree を揃えた matched pair |
| B-7 | 窓なし pointer(案 2‴) | 入力は汎用 `REF` + 選択候補の表現、距離項は共有関数 φ(k)、語彙から `REF_k` を除去。C の offset OOD セルの前提 |

概算: 優先 3 構成 × 3 seed × 2 サイズ(n24 / n32)= 18 run。各 run は早期停止まで
60〜100 epoch(既存 run と同規模)。

### C. その後の PR: 生成器 v2 と OOD マトリクス

- C-1 MetaGraph 直接生成器: Motif 数・token 長・depth・階層内位置・in-degree・
  合法候補数・kind を揃えたペアで参照 1 本の k だけを変える(表現機構の因果検証)
- C-2 CFG 生成器 v2: `GeneratorSpec`(family / width / degree / loop / goto /
  span / depth)。指定値と縮約後特徴は一対一でないため **realized feature の
  bucket 採用**、試行数・採用率を manifest へ
- C-3 実験マトリクス: ID / balanced-k / offset OOD / depth OOD / degree OOD /
  family OOD
- C-4 案 2″ の採用条件: untouched test で baseline+mask に勝つ、offset OOD で
  急落しない、KIND / LOOP / EOS canary を悪化させない、短距離 k を犠牲にしない、
  複数 seed で方向一致

### D. 保留・吸収

- 案 3′ の **EOS 要検証**(patience 延長 / EOS 損失重み): B-4 の複数 seed 化で
  seed 揺らぎか系統的かが判別できるため、B に吸収する。
- **pyClangAST 実 CFG loader**: 引き続き保留。C-3 の family OOD に「実データ」
  セルとして接続する。
- **Session 3 の階層展開 GT 設計論点 A–F**(main の discussion_log): 本 AR
  baseline 系列とは別系統。AR baseline は「参照表現の因果検証ベンチマーク」の
  位置づけであり、GT 本体(順序 p(G,π)、PE 選択、`ATTACH_EXISTING`)への接続は
  未定 → Q6-7。

## 4. 判断が必要な点(Q6)

| # | 論点 | 推奨 |
|---|---|---|
| Q6-1 | マージ方式 | main を merge(履歴保持)。rebase は PR コメントの hash 参照が切れる |
| Q6-2 | スコープ分割 | A を本 PR で閉じ、B・C は別 PR。A-3 の小修正は小さくテスト付きなので本 PR に含める |
| Q6-3 | 案 2″ の位置づけ | 「採用」→「暫定第一候補」に改める。B の結果で確定 |
| Q6-4 | CI 追加 | GitHub Actions を追加してよいか(リポジトリ設定に触れるため確認) |
| Q6-5 | B の規模 | 3 構成 × 3 seed × 2 サイズ = 18 run から開始。結果次第で 5 構成 / 5 seed へ拡張 |
| Q6-6 | 窓なし pointer(B-7)の位置 | B の最後に実装し、ID セルで案 2″ と同等を確認してから C へ |
| Q6-7 | Session 3 の GT 設計との接続 | AR baseline の知見(難度 = 参照距離、pointer 化で解消)を GT 側の生成文法(`ATTACH_EXISTING` = pointer)へ引き継ぐ方針のみ確認したい。本シリーズでは扱わない |

## 5. 決定(2026-09-04、A6)

| # | 決定 | 反映 |
|---|---|---|
| A6-1 | main を merge、競合は本ブランチ優先 | `2a3e9db`(main の追加テスト 4 本は `generate_cfg` 経由で取り込み) |
| A6-2 | A は本 PR、B・C は別 PR に分割 | 本文書 §3 の区分どおり |
| A6-3 | 案 2″ は「暫定第一候補」。成績と工夫は評価する | `representation_experiments.md` 結論を改稿 |
| A6-4 | GitHub Actions は**延期** | A-5 は保留 |
| A6-5 | B は 3 構成 × 3 seed × 2 サイズ = 18 run | 次 PR |
| A6-6 | 窓なし pointer(B-7)は **C へ** | offset OOD セルと同時に実装 |
| A6-7 | Session 3 の GT 設計との接点は Issue にメモとして残す | Issue 参照は PR コメントに記載 |

A の実施記録: A-1 merge、A-3 generator descriptor + nodes 込み同一性 + manifest `code`、
A-6 NLL-by-k の件数・−log p(k) 併記(副産物: `structure_sweep.md` 追記の頻度交絡)、
A-2 文書是正(本節・`representation_experiments.md`・`eval_axes.md`・`structure_sweep.md`・
`dataset_generation.md`)、A-4 PR 本文更新。

## 6. B の結果(2026-09-04)

18 run(base / mask / ptr × 3 seed × n24 / n32)の判定: **案 2″ の改善は参照合法マスク
で説明でき、pointer 機構の寄与は帰属できない**(レビュー §2 の懸念どおり)。既定構成は
baseline + 参照合法マスク。「難度 = k の関数」は頻度交絡(§1-4)が主因と確認。
詳細と表は `docs/design/controlled_eval.md`。
