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

(未着手)
