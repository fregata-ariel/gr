# 可変長・長系列に耐える事前学習モデル(pyClangAST 適用に向けて)

作成: 2026-09-26(決定レベル。詳細設計は `pretrain_longseq_detailed.md`、Codex 依頼)。

## 1. 目的

実 CFG(pyClangAST 由来、関数単位)は基本ブロック数が数個〜数百と幅広い。現行の AR baseline は
(1) ノード数固定のデータで学習し、(2) 学習可能な絶対位置埋め込み(`nn.Embedding(max_len)`)を使い、
(3) REF 語彙の窓 `max_offset` が bundle ごとに変わる、ため長さの外挿ができない(`mixture_doe.md` §8.3 の
要検証事項)。ここでは **ノード数を混合した合成データで事前学習し、学習時より長い系列にも劣化なく
使える checkpoint** を作り、実 CFG での微調整・評価の土台にする。

## 2. 決定事項

1. **データ**: 混合族(重心 1:1:1、A10)、ノード数は候補集合 {8, 12, 16, 24, 32, 48, 64, 96, 128} からの
   重み付き乱択(小さい n を厚く、実コードの分布を意識: 例 8–24 に 60%、32–64 に 30%、96–128 に 10%)。
   family パラメタは n に比例(loop 0.15n、goto 0.1n、spaghetti_rate 0.1 など、`experiments/sweep` の
   spec と同じ規則)。外挿用の held-out として n = 192(と可能なら 256)の test を別 dataset で作る。
   規模: train 20k / val 1k、test は n ごとに 200(bucket 評価)。生成は既存 v2 生成器で、n の乱択は
   dataset レベル(spec の `num_nodes` を分布で与え、サンプルごとの実現 n を provenance に記録)を第一候補
   とし、family plugin は変えない。
2. **語彙**: REF 窓を固定 `--max-offset 128`(語彙 137)。既存 `prepare_tokens --max-offset` を使う。
   窓を超える REF を含むサンプルは除外し、件数を meta に残す(既存機構)。
3. **位置表現**: `--pos {learned, sinusoidal, alibi, none}` を追加。`learned` は互換用。事前学習では
   sinusoidal / alibi / none を同一データ・同一 seed で比較し(各 1 run)、n=192 への外挿(NLL、REF 違反、
   WF)が最も良いものを既定にする。ALiBi は `nn.TransformerEncoder` の float attention mask
   (形 (B·H, L, L)、head ごとの傾き)で実装し、既存の因果マスクに加算する。
4. **参照表現**: 既定は B の結論どおり `--ref-legal-mask`(語彙型 + 合法マスク)。pointer(案 2″)は
   長系列で有利な可能性があるため、位置表現を決めた後に 1 run だけ比較する(任意)。
5. **バッチ**: 長さでバケット化してからシャッフル(padding 削減)。`--max-len` を学習時の系列上限として
   明示(既定は train 最大長)、超えるサンプルは学習から除外して件数を記録。評価は上限なし。
6. **モデル**: 128d / 4L(容量増は床を下げない、`scale_experiments.md`)から始め、データ 20k で
   val が改善し続けるなら 256d / 6L を 1 run 追加。epochs は early stopping(patience 10、上限 60)。
7. **checkpoint の再利用**: run dir に `config.json`(args、vocab、meta の要約、git commit)を保存し、
   `--init-from <run_dir>` で重みを読み込んで微調整できるようにする(語彙・位置表現が一致することを検証)。
   Colab / Docker 両バックエンドで同じ wrapper が使えること(`training/runner` の `TrainJob.extra`)。
8. **評価**: (a) n ごとの bucket NLL/token(ID: 学習範囲、OOD: 192/256)、(b) family 別、(c) REF の k 別
   NLL と違反(`controlled_eval`)、(d) WF(制約あり / なし)、(e) 専門家との比較: n48 の bucket を
   スイープ §14 の n48 専用モデル(0.77 / 0.61)と比べ、汎用化の代償を測る。基準符号長(`info_baseline`)
   を n ごとに当てて超過 NLL でも報告する。
9. **実行**: `training/runner` の Plan で Colab(既定)。見積り: n48 単独 2150 サンプル ≈ 5 分/run
   (2080 Ti)なので、20k サンプル・平均長 2 倍で 1 run ≈ 1〜1.5 h(T4)。比較 3 run + 最終 3 seed +
   容量 1 run ≈ 8–10 h の Colab 時間。

## 3. 段階

| 段階 | 内容 |
|---|---|
| P1 | 生成器: ノード数分布 + provenance、spec/データ生成スクリプト(`experiments/pretrain/`) |
| P2 | `train_ar`: `--pos`、ALiBi、長さバケット、`--max-len`、`config.json`、`--init-from`、テスト |
| P3 | 位置表現の比較 run(3)→ 既定決定 → 最終 3 seed → 評価レポート |
| P4 | (任意)pointer 比較、256d/6L |

## 4. 成果物

生成器・`train_ar` の変更とテスト、`experiments/pretrain/`(spec、Plan、評価)、事前学習 checkpoint
(`runs/pretrain_*`、バックアップ対象)、本書 §5 の実行記録、`generator_v2.md` §16 の要約。

## 5. 実行記録

### 5.1 予備計測(2026-09-26、混合 1:1:1、family params は n 比例、100 サンプル/n、seed 910000〜)

| n | loop / goto | 受理 / 試行 | トークン数 平均 / 最大 | 必要な REF 窓 | 生成 + 縮約(秒) |
|---|---|---|---|---|---|
| 8 | 1 / 1 | 40 / 100 | 22 / 26 | 5 | 0.2 |
| 12 | 2 / 1 | 97 / 100 | 33 / 38 | 9 | 0.2 |
| 16 | 2 / 2 | 100 / 100 | 43 / 49 | 10 | 0.2 |
| 24 | 4 / 2 | 100 / 100 | 65 / 72 | 12 | 0.3 |
| 32 | 5 / 3 | 100 / 100 | 86 / 97 | 17 | 0.3 |
| 48 | 7 / 5 | 100 / 100 | 130 / 143 | 22 | 0.5 |
| 64 | 10 / 6 | 100 / 100 | 173 / 195 | 29 | 0.7 |
| 96 | 14 / 10 | 100 / 100 | 258 / 291 | 44 | 1.5 |
| 128 | 19 / 13 | 100 / 100 | 344 / 382 | 53 | 3.3 |
| 192 | 29 / 19 | 100 / 100 | 515 / 563 | 83 | 9.1 |
| 256 | 38 / 26 | 100 / 100 | 683 / 763 | 105 | 20.9 |

- 系列長 ≈ 2.7n、必要な REF 窓 ≈ 0.41n。固定窓 128 は n ≈ 300 まで除外なし(§2-2 のとおり)。
- 生成は安価(n256 でも 100 サンプル 21 秒)。train 20k の生成は数分。
- n = 8 は同型重複で受理率 40%(トポロジー空間が小さい)。n = 12 も 8000 で飽和した経験
  (`scale_experiments.md`)があるので、小さい n の重みは重複を避ける範囲(n8 数百、n12 2000 程度)に抑え、
  「小さい n を厚く」は n16〜24 で実現する。
- 学習時間の見積り: n48 単独(2150 × 133 トークン ≈ 0.29M トークン/epoch)で 2.8 秒/epoch(2080 Ti)。
  train 20k × 平均 ≈ 150 トークン ≈ 3M トークン/epoch → 30〜40 秒/epoch、60 epoch で 30〜40 分(T4 は
  2080 Ti の 0.7〜0.8 倍の速度、L4 は 1.5〜2 倍)。注意: 長い系列の attention は二乗で効くので n128 の
  比率が高いほど遅くなる。

### 5.2 実装と P1 データ生成(2026-09-26)

- 実装(詳細設計 §7 の T1〜T3、Codex gpt-6-astra low、各回オーケストレータが再検証): e1ec9f9(dataset のノード数分布、
  `--target-count`、per-n 統計、`experiments/pretrain/make_spec.py`)、1984c8a(`--pos` 4 種、ALiBi、長さバケット、`--max-len`、
  `prepare_tokens --eval-length-policy`)、a16f9a9(`config.json`、`--init-from`、`--ref-diagnostics`、`--wf-probes`、ランナーの
  checkpoint 転送、`--colab-inner-timeout-s`)。`train_ar` の変更は `training/smoke_longseq.py` を pinned イメージ内で実行して受入
  (既定経路の bit 一致、ALiBi の手計算一致、バケットの再現性、4 方式の学習と学習最大長超の採点、init-from の一致/不一致、probe の RNG 分離。
  記録 `experiments/pretrain/smoke_longseq_t{2,3}_cpu.json`)。
- データ生成(`experiments/pretrain/gen_data.sh`、生成 commit 547caa0、所要 7 分): test は各 n 200 件を先に確保
  (8: 200 (940 試行) / 12: 200 (211 試行) / 16: 200 (202 試行) / 24: 200 (201 試行) / 32: 200 (200 試行) / 48: 200 (200 試行) / 64: 200 (200 試行) / 96: 200 (200 試行) / 128: 200 (200 試行) / 192: 200 (200 試行) / 256: 200 (200 試行))。混合 train 20000 / val 1000(試行 24521 / 1286、全 test を exclude)。
  train の n 別採用数: 8: 298 / 12: 2852 / 16: 3328 / 24: 3722 / 32: 2417 / 48: 2531 / 64: 2444 / 96: 1180 / 128: 1228。val: 8: 9 / 12: 128 / 16: 169 / 24: 195 / 32: 147 / 48: 123 / 64: 110 / 96: 69 / 128: 50。
- bundle: `data/tok_pretrain_mix`(train/val)、`data/tok_pretrain_n<N>`(同じ train/val + 各 n の test、`--max-offset 128`、
  `--eval-length-policy unlimited`)。REF 窓 128 による除外は 0。train の系列長: 平均 107.7、中央 69、p95 326、最大 389、
  1 epoch 2.13M トークン(n48 参照の 7.5 倍 → 2080 Ti 換算で 60 epoch ≈ 20〜25 分、L4 でも同程度と見込む)。
- パイロット(2026-09-26、Colab L4、`plan_pilot.json`: 3 方式 × 2 epoch、サンプル 2 + 2): 1 run 54〜81 秒(転送・構築・採点込み)。
  1 epoch は 20 秒未満で、60 epoch + 生成 100 + 100(予算 778 トークン)でも 1 run 30 分前後の見込み(inner timeout 5400 秒内)。
  Colab の torch は 2.11.0+cu128(pinned イメージの 2.14 と異なる)だが ALiBi 経路も有限で学習が進む(2 epoch で val 0.96、
  sinusoidal 1.02、none 1.03)。比較 Plan(`plan_compare.json`)を L4 で開始。
