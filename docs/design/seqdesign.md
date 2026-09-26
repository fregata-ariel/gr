# 逐次(ベイズ)実験計画: 全パターンからの乱択(A9-4)

作成: 2026-09-26(決定レベル。詳細設計は `seqdesign_detailed.md`、Codex 依頼)。
前提: `mixture_doe.md` §7、A9-4「今後の転用では全パターンから乱択するベイズ的な方法を原則とする」。

## 1. 目的

固定設計点(simplex-centroid など)の代わりに、**有限の候補パターン集合**(例: simplex 格子 231 点、
モデル構成の因子格子)の上で、観測に応じて次に試す点を確率的に選ぶ逐次設計を、`training/mixture_doe.py`
と `training/runner` に接続できる形で作る。目的は探索(モデルの特徴そのものを調べる)であって
最適化だけではないので、事後分布からの乱択(Thompson sampling)を既定にし、決定的な獲得関数は比較用に持つ。

## 2. 決定事項

1. **候補空間**: 明示的な有限集合。パターン = 因子値の dict(例 `{"lay":0.35,"str":0.30,"spa":0.35}`)。
   連続因子は格子化して与える(mixture は `simplex_grid(step)`)。因子は数値のみ(カテゴリは one-hot で数値化)。
2. **代理モデル**: ガウス過程回帰(numpy)。カーネル = 定数 × ARD RBF(または Matérn 5/2)+ 観測ノイズ。
   超パラメタは周辺尤度の最大化(座標探索か L-BFGS 相当の簡易法、依存追加なし)。応答は複数
   (mixture では lay / str / spa の超過 NLL)を**独立の GP** で持ち、複合(bal / max)は事後サンプル上で計算する。
3. **獲得**: 既定 = **Thompson sampling**(各応答の GP から関数を 1 本ずつサンプルし、複合を最小にする
   候補を選ぶ。バッチ k 点は k 回の独立サンプル、重複は許す = 反復 seed として意味を持つ)。比較用に
   EI と UCB、ベースラインとして一様乱択と固定設計。
4. **反復(seed)**: 同一パターンの複数 seed は GP のノイズ項で扱う。各提案は seed 数を指定(既定 2)。
5. **状態**: `campaign.json`(空間定義、応答名、複合の定義、観測履歴、乱数状態、提案履歴)を 1 ファイルで
   持ち、`experiments/<campaign>/` に置く(git 管理)。観測の追加は `mixture_doe collect` の JSON を取り込む。
6. **実験への接続**: パターン → データ生成 → トークン化 → Plan JSON の「実体化」は campaign ごとの小さな
   Python ドライバ(mixture は `experiments/mixture_doe/materialize.py`)。runner の Plan 形式を再利用する。
7. **検証(A9「実験的な仕組みの Validation」)**:
   (a) 合成曲面(既知の Scheffé 二次 + ノイズ)で、TS / EI / 一様乱択 / 固定 10 点設計の regret を回数に
   対して比較(乱数 seed 20 本)。(b) 実データの遡及: n24 の 12 点 × 3 seed を順に与えたときの事後と
   提案を確認し、平坦な bal を GP が再現するか。(c) 実機: n24 で 2 ラウンド × 3 提案 × 2 seed を Colab で
   回し、既存 DoE の結論(bal 平坦、spaghetti が最悪)と矛盾しないことを確認。
8. **非依存**: 実行時依存は numpy(matplotlib 経由で導入済み)のみ。torch なし。テストは合成データで
   決定的(seed 固定)。

## 3. インタフェース(案)

```
python -m training.seqdesign init     --campaign C.json --space space.json --responses lay,str,spa --objective bal
python -m training.seqdesign observe  --campaign C.json --obs obs.json           # mixture_doe collect の出力
python -m training.seqdesign propose  --campaign C.json --k 3 --seeds 2 --acq ts  # 提案を campaign に記録、JSON 出力
python -m training.seqdesign report   --campaign C.json --out report.md [--fig-dir]
python -m training.seqdesign simulate --surface scheffe --rounds 8 --k 3 --repeats 20 --out sim.md
```

## 4. 成果物

`training/seqdesign.py`、`tests/test_seqdesign.py`、`experiments/mixture_doe/materialize.py`、
`experiments/seqdesign_sim/`(シミュレーション結果)、本書 §5 に実行記録。

## 5. 実行記録

### 5.1 実装と遡及評価(2026-09-26)

- 実装(詳細設計 §9 の T1〜T3、Codex gpt-6-astra low、各回オーケストレータが再検証): e3d41d9(型・検証・JSON・決定的 RNG・
  numpy GP コア)、b1bf2e4(獲得関数 ts / ei / ucb / random / fixed、提案と seed 予約、CLI 5 コマンド、日本語報告と三角図、
  合成検証と遡及評価)、547caa0(`experiments/mixture_doe/materialize.py`: 提案 → spec・generate/tokenize/collect スクリプト・
  runner Plan・manifest)、95b8355(正規乱数生成のベクトル化。`normal_draw` を繰り返した列と完全一致。`simulate --summary-samples`
  既定 512、report は 4096 のまま)。テスト 873 件、`ty` 通過。
- 遡及評価(n24 の 12 点 × 3 seed、超過 NLL(target 固有基準)、`obs_xt.json`、TS k=3、seed 2):
  - 6 点(純 3 点 + 辺 3 点)投入後の bal 提案: (0.45, 0.30, 0.25)、(0.35, 0.00, 0.65)、(0.60, 0.20, 0.20)。内部の点を選び、頂点や辺には戻らない。
  - 12 点投入後の bal 提案: (0.35, 0.15, 0.50)、(0.00, 0.30, 0.70)、(0.60, 0.05, 0.35)。max 提案: (0.25, 0.25, 0.50)、(0.05, 0.35, 0.60)、(0.25, 0.30, 0.45)。いずれも spaghetti 寄りの内部で、
    「正規化後の最悪ケースは spaghetti」「bal は内部で平坦」という DoE の読み(`mixture_doe.md` §9.1)と整合する。
  - 6 点時点から残り 6 点(全て内部)への 95% 予測区間の被覆: lay 15/18、
    str 12/18、spa 15/18、
    bal 15/18。等分散ノイズの GP では純 structured の大きな seed 分散が内部の
    区間幅に混ざる一方、str の区間はやや狭い(詳細設計 §4.1 の注記どおり)。
- 合成検証(`simulate`、10 条件 × 目的 2 × 手法 5 × 8 ラウンド)は 1 ラウンド 1〜3 秒かかるため、反復 10 で実行中
  (結果は §5.2 に追記)。実装時の 20 反復は `GR_SLOW_TESTS=1` の試験にのみ残す。

### 5.2 合成検証の結果(2026-09-26、`experiments/seqdesign_sim/sim_summary.md`、反復 10)

- **regret**: 滑らかな面(base)では TS / random / fixed が 48 観測で 0.001 以下に収束し、EI は同程度、UCB(下側信頼
  境界)は 0.003〜0.01 と遅い。頂点急変・不連続では固定 10 点設計の bal regret が 0.018〜0.022 と悪化する
  (頂点の値が内部の推奨を歪める)のに対し、TS は 0.001 以下(無雑音)〜0.009(SD 0.03)。目的 max でも同様。
- **予算配分**: 最適近傍(真値差 ≤ 0.01)に使った seed の比率は TS 0.65〜0.69(低雑音)、EI 0.44〜0.57、
  random 0.29、fixed 0.08。random の regret が小さいのは面が平坦で内部のどこでも最適に近いためで、
  TS は最適近傍を集中的に反復している(探索目的に合う)。
- **校正**: 観測前に凍結した 95% 予測区間(提案点)の被覆率は TS で 0.82〜0.94(base:0.005 で 0.86)。
  一方、全候補に対する潜在 95% 信用区間の被覆率は内部 0.42〜0.84、頂点 0.12〜0.49 と低い。無雑音条件では
  ノイズ分散が下限に張り付き、RBF カーネルの外挿が過信になる(頂点は設計どおり過信を記録)。
- **判定**(§11 の見送り条件): 提案点の予測区間被覆 0.86 ≥ 0.80 で実機に進む。潜在区間の過信は要検証事項
  として残す(対策候補: ノイズ分散の下限、長さスケールの事前分布、Matérn 5/2)。
