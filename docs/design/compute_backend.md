# 計算バックエンドの抽象化(Colab / ローカル GPU コンテナ)

作成: 2026-09-16。状態: 設計中。判断事項は `docs/handoff_questions.md` Q8 / A8。
決定(A8): 既定の実行先は Colab。先にルーターと現行 colab CLI フローのアダプタを作り、
ローカル GPU コンテナのバックエンドは後続。詳細設計は `compute_backend_detailed.md`(Codex high)。
補足(2026-09-16、ユーザー指示): 既定は Colab だが、いま実際に動かして検証できるのは
ローカル GPU のみ。したがって (1) テストは FakeBackend だけで Colab に触れない、
(2) コード上の既定は colab のまま、このマシンでは環境変数 `GR_BACKEND=local` で上書きする、
(3) Docker バックエンドは Colab アダプタの直後に実装し、実機の受け入れ試験(小さな Plan の
end-to-end)はローカルで行う。イメージは `pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime`
(CUDA 12.6、Turing sm_75 対応)を pin する。


## 1. 背景

節目スイープ(`generator_v2.md` §14)は n12〜n24 まで完了したが、n32 / n48 の 12 run は
Colab 無料枠の T4 が 2 日連続で確保できず(`Service Unavailable` × 96 回、2 周期とも
GIVE-UP)止まっている。学習・再スコアの実行先が Colab に直接書かれているため、
実行先を変えるには run スクリプトを書き直す必要がある。

## 2. 現状の確認(2026-09-16)

### 2.1 ローカル環境

| 項目 | 結果 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2080 Ti, 11 GB, compute capability 7.5(Turing), driver 595.84 |
| コンテナ | Docker 29.7.2、`nvidia` ランタイム登録済み、NVIDIA Container Toolkit 1.20.0 |
| コンテナから GPU | `docker run --gpus all debian:bookworm nvidia-smi` で 2080 Ti を認識(確認済み) |
| torch 入りイメージ | なし(pull または build が必要、数 GB。ディスク空き 228 GB、レジストリ到達可) |
| Kubernetes | このマシンに kubectl なし(自宅 K8s は別ホスト) |
| ホスト側 torch | なし(方針どおり: torch は実行環境側のみ) |

参考: T4 では n24 mask の 1 run が約 4 分(学習 70〜80 epoch)。2080 Ti は T4 より
FP32 演算が速く、モデルは小さいので、同等以上の速度が見込める。無料枠固有の
1 時間切断・割当待ち・セッション登録簿の破損はすべて消える。

### 2.2 Colab への結合点

リポジトリ側のコードはほぼ実行先に依存しない。結合は scratchpad のシェル
スクリプト(21 本)と、その生成物である Python ラッパー(69 本)に集中している。

| # | 結合点 | 現状の実装 |
| --- | --- | --- |
| 1 | セッション寿命 | `colab new --gpu T4 -s NAME` / `colab sessions` / `colab stop`(再試行ループ、直列化の制約) |
| 2 | ステージング | `colab upload` でコード(`train_ar.py`, `grammar_mask.py`)、バンドル 5 ファイル、再スコア用チェックポイントを平坦な `/content/…` へ |
| 3 | 実行 | `colab exec -s NAME -f w_<run>.py --timeout` 。ラッパーは `sys.path` に `/content` を足し、`train_ar.main([...])` を呼ぶ |
| 4 | 回収 | `colab download` で run あたり 6 ファイル |
| 5 | パス規約 | `/content` が `train_ar.py` の既定引数、ラッパー、`rescore_c.py`(`/content/rescore_jobs.json`)に埋め込み |
| 6 | 障害対策 | `Service Unavailable` 再試行、1 時間切断、`test_scores.jsonl` による冪等性、部分ディレクトリ削除 |

`training/train_ar.py` 本体は `main(argv)` と `torch.cuda.is_available()` による
デバイス選択のみで、既定パス以外に Colab 依存はない。`rescore_c.py` は
`/content` を直書きしている。

## 3. 提案: ルーター + バックエンド

### 3.1 構成(`training/runner/`、torch 非依存)

```
JobSpec      — 凍結 dataclass。run 名、バンドル dir、train_ar 引数(パスを含まない)、
               再スコア対象バンドル、seed。ラッパー 69 本はこれから生成する。
Backend      — Protocol: acquire() / release() / alive() / put(local, remote) /
               get(remote, local) / run(script, args, timeout) / root(実行側のパス根)
ColabBackend — colab CLI を timeout 付きで包む。直列化・再試行・1 時間切断の再取得を内包。
DockerBackend— `docker run --gpus all -v <staging>:/work <image>`。put/get はステージング
               dir へのコピー(実質不要)。GPU 名・イメージ digest を run のメタに記録。
Router       — 方針 `--backend colab|local|auto`。auto = ローカル GPU が空いていれば local、
               確保できなければ colab へ(逆も可)。
Plan 実行     — 冪等(`test_scores.jsonl` があれば skip)、run ごとに実行先を記録。
CLI          — `python -m training.runner --plan plan.json --backend auto`
```

- 実行側スクリプト(ラッパー・再スコア)はパス根を環境変数(`GR_ROOT`)で受け、
  `/content` / `/work` を切り替える。`train_ar.py` の既定引数も `GR_ROOT` を参照する。
- テストは FakeBackend で Router / Plan の分岐と冪等性を検証する(torch 不要)。
- 将来の K8s バックエンドは同じ Protocol の 3 つ目の実装(Job + PVC)として追加する。
  今回は対象外(このマシンに kubectl がない)。

### 3.2 イメージ

`training/docker/Dockerfile` を `FROM pytorch/pytorch:<pin>-cuda12.x-cudnn9-runtime`
のみで置き、リポジトリの `training/` はバインドマウントする。AGENTS.md の
「依存追加禁止」は Python プロジェクト側の規則で、イメージには影響しない。

### 3.3 再現性への注記

GPU が変わると数値は一致しない(CUDA カーネル・アーキテクチャ差)。節目スイープの
比較は同一 n・同一 seed・同一実行先の layered 対 mixed なので、n32 / n48 をローカルで
学習しても比較の妥当性は保たれる。n 方向の傾向は実行先が混ざるが、差の大きさ
(+3.5 対 +0.1)に対して GPU 差は小さい。確認として n24 s0 の 2 run(lay / mix)を
ローカルで再学習し、T4 結果との差を §14 に記録する(約 10 分)。

## 4. 進め方(案)

1. イメージ pull とローカル 1 run(n12)のスモーク。設計とは独立に先行できる。
2. Codex(high)に §3 の詳細設計を依頼 → 判断 → Codex(low)で実装(2〜3 タスク)。
3. n32 / n48 をルーター経由で実行し、ルーターの受け入れ試験とする。
4. `sweep_summary` → `generator_v2.md` §14 を完成。scratchpad の run スクリプト群は
   ルーターに置き換え、`CLAUDE.md` / `AGENTS.md` の Colab 注記を更新する。

代替: 先に最小のローカル runner(既存ラッパーを `/work` を `/content` に見せる
マウントで流す)で n32 / n48 を今日中に終え、その後ルーターを作る。結果は早いが
使い捨てが増える。
