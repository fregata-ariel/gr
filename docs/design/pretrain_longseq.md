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

### 5.3 位置表現の比較と選択(2026-09-26、Colab L4、seed 0、`experiments/pretrain/compare_eval.md`)

NLL/token(bucket 別、REF 窓 128 による除外 0、`eval_buckets`):

| n | sinusoidal | alibi | none | 基準符号長(alibi の行) |
|---|---|---|---|---|
| 8 | 0.732 | 0.605 | 0.819 | 0.999 |
| 12 | 0.540 | 0.502 | 0.630 | 0.932 |
| 16 | 0.490 | 0.475 | 0.559 | 0.967 |
| 24 | 0.584 | 0.569 | 0.604 | 1.081 |
| 32 | 0.576 | 0.566 | 0.591 | 1.078 |
| 48 | 0.579 | 0.580 | 0.603 | 1.129 |
| 64 | 0.583 | 0.594 | 0.627 | 1.149 |
| 96 | 0.558 | 0.567 | 0.616 | 1.115 |
| 128 | 0.595 | 0.604 | 0.661 | 1.119 |
| 192 | 0.885 | 0.658 | 0.756 | 1.156 |
| 256 | 1.058 | 0.697 | 0.841 | 1.222 |

| 指標 | sinusoidal | alibi | none |
|---|---|---|---|
| ID token micro | 0.5780 | 0.5795 | 0.6293 |
| ID bucket macro | 0.5816 | 0.5624 | 0.6344 |
| 無制約 WF(reference-constrained、生成数) | 0.63 (100) | 0.84 (100) | 0.9 (10) |
| epochs / best val | 60 / 0.5877 | 45 / 0.5883 | 43 / 0.6335 |

対比較(token 加重 paired bootstrap 95% CI、左 − 右、nats/token):
- sinusoidal − alibi: ID -0.0014 [-0.0028, +0.0000]、n192 +0.227 [+0.216, +0.239]、n256 +0.361 [+0.349, +0.375]
- sinusoidal − none: ID -0.0513 [-0.0534, -0.0491]、n192 +0.129 [+0.119, +0.140]、n256 +0.217 [+0.206, +0.227]
- alibi − none: ID -0.0499 [-0.0520, -0.0477]、n192 -0.098 [-0.105, -0.092]、n256 -0.145 [-0.154, -0.135]

読み:
- **選択: ALiBi**(`selection.json`)。選択規則の n192 で 0.658 と最小、sinusoidal(0.885)・none(0.756)に対する
  paired CI はいずれも 0 を含まない。n256 でも 0.697 と最小で、学習範囲(≤ 128)の 2 倍の長さまで劣化が
  最も小さい(n128 → n256 で +0.09。sinusoidal は +0.46、none は +0.18)。
- ID では sinusoidal と同等(token micro の差 0.0014、CI が 0 に接する)。bucket macro は ALiBi が最良
  (小さい n で有利: n8 0.605 vs 0.732)。none は ID で 0.05 劣る。
- 無制約生成の整形率は ALiBi 0.84、sinusoidal 0.63。none の 0.9 は生成 10 本のみの参考値。
- n48 bucket(0.580)は §14 の n48 専用モデル(混合 ID 0.61、旧窓・旧 test の参考値)より小さく、汎用化の
  代償は見えない。基準符号長も n とともに増える(n8 1.00 → n256 1.22)ので、超過 NLL で見ると ALiBi の
  n256 は −0.53、sinusoidal は −0.16。
- 実行上の注意: none の比較 run は生成 100 + 100(予算 778 トークン)が 90 分を超えて inner timeout に達した
  (EOS を出しにくく予算いっぱいまで生成したと推定)。生成 10 + 10 で再実行(9 分)。方式選択は再スコアの
  NLL に基づくので影響なし。ALiBi / sinusoidal は 100 + 100 で 13〜16 分。
- 最終 3 seed(`pretrain_final_alibi_s{0,1,2}`、生成 200 + 200、`--wf-probes 5`)は 3 セッション並列で実行
  (`run_final_parallel.sh alibi`)。結果は §5.4。

### 5.4 最終 checkpoint(ALiBi、seed 0–2、2026-09-26、`experiments/pretrain/final_eval.md`)

3 セッション並列で実行(`run_final_parallel.sh alibi`、L4)。seed 0 / 1 は 18:09〜18:27 に同時、seed 2 は 3 本目の
L4 が 412 で 2 回拒否され 20 分遅れて開始(§6 の注記)。学習 + 生成 400 本 + probe で 1 run 14〜20 分、
3 run と再スコア 33 件で 1.33 CU。epochs / best epoch / best val: 45 / 35 / 0.5880 、58 / 48 / 0.5815 、60 / 52 / 0.5767(early stopping patience 10、上限 60)。

| n | NLL/token 平均 ± SD(3 seed) | 基準符号長 | 超過 NLL | 続き生成 WF(prefix 半分、5 本 × 3 seed、参照制約) |
|---|---|---|---|---|
| 8 | 0.713 ± 0.093 | 0.999 | -0.286 | 0.33 |
| 12 | 0.539 ± 0.063 | 0.932 | -0.393 | 0.47 |
| 16 | 0.483 ± 0.016 | 0.967 | -0.483 | 0.53 |
| 24 | 0.559 ± 0.008 | 1.081 | -0.522 | 0.53 |
| 32 | 0.559 ± 0.008 | 1.078 | -0.519 | 1.00 |
| 48 | 0.575 ± 0.005 | 1.129 | -0.554 | 0.40 |
| 64 | 0.586 ± 0.008 | 1.149 | -0.563 | 0.60 |
| 96 | 0.558 ± 0.010 | 1.115 | -0.557 | 0.47 |
| 128 | 0.592 ± 0.012 | 1.119 | -0.527 | 0.73 |
| 192 | 0.644 ± 0.012 | 1.156 | -0.511 | 0.00 |
| 256 | 0.682 ± 0.012 | 1.222 | -0.540 | 0.20 |

- ID token micro 0.5791, 0.5754, 0.5677(bucket macro 0.5630, 0.5662, 0.5919)。seed 間の SD は n ≥ 16 で
  0.005〜0.016、n8 / n12 は 0.09 / 0.06(トークン数が少なく、学習データでも希少)。
- 外挿: n192 0.644、n256 0.682(学習最大 n128 の 0.592 から +0.05 / +0.09)。超過 NLL は全 n で −0.29〜−0.56 と
  安定で、n256 でも −0.54。位置埋め込みの未学習問題(`mixture_doe.md` §8.3)は解消。
- 無制約生成の整形率(参照制約のみ、200 本): 0.86, 0.81, 0.69。文法制約付きは 1.0。
- gold-prefix の続き生成(prefix = 系列の前半、予算 2 倍、5 本): 参照制約のみでは n ≤ 128 で 0.4〜1.0 だが
  n192 で 0.0〜0.2、n256 で 0.2 前後。NLL の外挿は良いが、長い系列の自由生成は括弧(LOOP_END)や EOS の
  管理で崩れる。文法制約付きなら 1.0(生成の既定は制約付き)。
- teacher-forced の REF 合法性違反は全 n で 0(`--ref-legal-mask` により構成的に 0。診断の分母は gold REF)。
- **成果物**: `runs/pretrain_final_alibi_s{0,1,2}/`(`model.pt`、`config.json`、`length_stats.json`、`vocab` は
  `data/tok_pretrain_mix/vocab.json` = REF 窓 128 の 137 語)。微調整は `--init-from runs/pretrain_final_alibi_s0`
  (語彙・位置表現・形状を検証して strict に読み込む)。バックアップ対象(`runs/`)。
- 次の候補(P4、任意): pointer 参照(案 2″)の長系列比較、256d/6L、epochs 上限の引き上げ(seed 2 は 60 で停止)。
