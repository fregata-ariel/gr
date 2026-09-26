# 可変ノード数・長系列事前学習の詳細設計

作成: 2026-09-26。状態: 実装前。上位決定は `pretrain_longseq.md`。

## 0. 範囲と既存契約

P1 は dataset レベルの n 乱択、P2 は単一ファイル trainer の長さ汎化、P3 は比較と集計である。
本書は仕様のみであり、実験結果・性能保証ではない。P4(pointer、256d/6L)と実 CFG の取り込みは別タスク。
128d/4L/4 heads/FF512、`--ref-legal-mask`、混合重み 1:1:1 を固定する。
根拠は `scale_experiments.md` の容量増でも下がらない床、`representation_experiments.md`「結論と推奨」の
B 後の追記、`mixture_doe.md` §8.3/A10。位置容量だけ増やしても未学習位置は改善しない。

- `generate.py`、`store.py`、`build_dataset` の既存 legacy 分岐と既存整数 spec の出力は維持する。
- `GeneratorSpec.num_nodes`、`spec_from_json`、全 family plugin は整数契約のまま。分布は別の dataset codec が扱う。
- Colab に渡す実装は引き続き `train_ar.py` と `grammar_mask.py` のみ。前者の依存は stdlib + torch。
  notebook の `-f kernel.json` を許す `parse_known_args` は残す。runner 側で新フラグの綴りをテストする。
- JSON は sorted keys、有限数値のみ。欠測は `null`。新しい生成コードは新 commit version で全データを作り直す。
  旧 dataset の混載や `--allow-version-mismatch` は本実験では行わない。

## 1. P1: dataset spec とサンプル生成

### 1.1 入力形式と params の展開

`cfg_reducer/dataset_v2.py` に `dataset_spec_from_json`、`resolve_sample_config` を追加する。
旧 `{family, num_nodes: int, params}` は既存 codec にそのまま渡す。新形式は次の 2 通りとし、未知キーを拒否する。

```json
{"family":"mixture","num_nodes":{"choices":[8,12,16,24,32,48,64,96,128],"weights":[3,3,3,3,2,2,2,1,1]},"params_by_num_nodes":{"8":{"components":[]},"12":{"components":[]}}}
```

上例の params は構造説明用の省略であり、そのままでは不正。実際には全 choices のキーが必要で、各 components は非空。
新形式のキー集合は `{family,num_nodes,params}` または `{family,num_nodes,params_by_num_nodes}`。
前者は全 n で同じ params、後者は n ごとに具体的な params を選ぶ。両方の指定は拒否する。
choices は昇順・重複なしの正整数、weights は同じ長さの有限正数(bool 不可)、空配列不可。
`params_by_num_nodes` は choices の十進文字列キーと完全一致すること。固定整数 spec にこのキーは認めない。
codec の保持型は frozen `DatasetSpec` / `NodeCountDistribution`、choices/weights/展開済み spec は tuple。
読み込み時に全 n の具体的 spec を検証し、実行時には選択した整数 spec を family normalisation に渡す。

**テンプレート方式は Python の規則表を採用**する。新規 `experiments/pretrain/make_spec.py` の
`params_for_n(n)` が全 family の concrete components を生成し、`params_by_num_nodes` を JSON に保存する。
式文字列の評価や family 内の n 分岐は導入しない。mixture の component に独自 `num_nodes` は追加しない。
component は現行どおり親 `spec.num_nodes` を継承するため、異なる n を components に混ぜる方式は使わない。

| family | 任意の整数 n に対する具体的規則 |
|---|---|
| layered | `L=3*n//20`, `G=n//10`, `max_layer_width=3`, `edge_prob=0.18`, `span_mode="uniform"` |
| structured | `L0=3*n//20`, `G0=n//10`, `H=11*n//10`; `L=min(L0,(H-3)//3)`, `G=min(G0,(H-3-3*L)//2)` |
| spaghetti | structured と同じ L/G/構造 params に `spaghetti_rate=0.1` を加える |

規則表の対応範囲は `n>=3`(それ未満は明示エラー)。structured/spaghetti の残りは
`max_layer_width=3, branch_degree=2, merge_degree=3, span_mode="uniform", target_depth=null`。
この切下げ順は loops を優先し `3+3*L+2*G <= floor(1.1*n)` を満たす。`max(1,...)` は使わない。
n=3..6 は L/G=0/0、n8 は 1/0、n12 は 1/1、n48 は 7/4、n192 は 28/19、n256 は 38/25。
n12/n48 の既存 spec と normalise 後に一致することをテストする。L/G は要求値であり実測ループ/辺数ではない。
structured/spaghetti は実ノード数が `[ceil(.9*n),floor(1.1*n)]` となる現行仕様を保持する。

### 1.2 RNG と builder の接続

分布の重みは上記の比率(8–24:60%、32–64:30%、96–128:10%)を **候補生成時** に適用する。
`resolve_sample_config(config, seed) -> dict` は `Random("gr:num_nodes:v1:" + str(seed))` を専用に作り、
`weights/max(weights)` の累積和と `rng.random()*math.fsum(weights_scaled)` で最初の該当 n を選ぶ。
末尾を丸め誤差の fallback とする。Python の `hash()`、global RNG、生成用 RNG の先頭消費を使わない。
出力は `{"spec": spec_to_json(normalize_spec(GeneratorSpec(family,n,params_n)))}`。
整数入力では乱数を消費せず既存 config を返す。選択は split、試行順、dedup 結果に依存しない。

`dataset.build_dataset` に keyword-only `resolve_config=None, target_counts=None, record_node_counts=False` を追加。
いずれかが指定された場合だけ `_build_selected` に流し、旧呼出しの legacy 本体は変更しない。
`dataset_v2.main` は分布入力で resolver を渡す。descriptor は選択候補の共通 family の `descriptor_for` を使う。
`_build_selected` は seed ごとに **生成前に一度** config を解決し、同じ resolved config を
`desc.fn`、provenance、`Candidate.requested` に渡す。ここに family 名による分岐を置かない。
`mixture.component_for(resolved_spec, seed)` は既存の `Random(seed)` を再現するため変更不要。
provenance を読む `training.mixture_doe.realized_composition` もそのまま動くことを受入条件とする。

### 1.3 provenance、manifest、重複排除

UUID の入力は **ちょうど次のオブジェクト全体** とする。分布や採用判断は入れない。

```json
{"source":"synthetic","generator":{"name":"cfg_v2:mixture","version":"<commit>","seed":123,"config":{"spec":{"family":"mixture","num_nodes":48,"params":{"components":["<normalized components>"]}}}}}
```

実際の components は辞書配列。実現した抽選値は `provenance.generator.config.spec.num_nodes`、解決済み整数 spec は
同 `.spec` に記録する。`sample_id = UUIDv5(store.SAMPLE_NAMESPACE, canonical_json(provenance))`。
namespace は現行の URL namespace から導出する定数、canonical JSON は再帰的キー昇順、
`separators=(",",":"), ensure_ascii=False`。split、出力パス、code.dirty、分布の重み、quota、bucket、実測特徴は含めない。
同じ version/seed/解決 spec なら固定 n dataset と UUID が一致する。分布の変更自体は manifest hash を変える。

manifest の `generator.config.spec` は要求した分布 spec、`generator.name/version` は従来どおり。
各採用 entry に `requested`(resolved config)、`realized.num_nodes`(抽選 n)、`realized.cfg_num_nodes`(縮約前の実数)、
`bucket="n48"` 等を保存する。他の measured features は保持する。family は文字列なので Features 型に混ぜず、
評価時に resolved spec と seed から plugin の公開機能で決める。整数 spec + 新 count モードでも同じ entry を作る。
分布入力では `record_node_counts=True` を自動設定する。各 split に `per_num_nodes` を追加し、全 choices(0 件も含む)について
`{attempts,generated,accepted,rejected,rejected_by_reason,cfg_num_nodes_histogram}` を記録する。
rejections.jsonl にも `num_nodes` を含め、生成失敗でも集計できるよう resolution を先に行う。
histogram は採用 CFG の実ノード数ごとの件数。この n bucket モードと既存多次元 `--plan` の併用は当面拒否する。

iso-dedup は従来の WL 候補絞込み + 有向同型確認を、split 全体と exclude 全体に対して行う。
異なる名目 n でも実 CFG が同型なら除外する。UUID 一致だけでは構造重複を判定しない。
`load_references` は分布 manifest の場合、各 entry の seed から resolver を再実行し、entry.requested と一致、
上の provenance から UUID 一致を確認して CFG を再生成する。旧 manifest は従来経路を保持する。
新 entry の sample ファイルにある provenance とも照合する。不一致は破損として中止する。

`dataset_v2 --target-count NAME=N`(反復可、正整数・既知split・重複指定不可) は split ごとの **dedup 後採用件数**。seed range は試行上限。
目標に達した split は直ちに停止し、`attempts` は実際の試行数、`seed_range` は指定上限、
`last_seed`、`target`、`missing`、`complete` を記録する。不足時 exit 2、`--allow-incomplete` のみ続行可能。
quota は合計件数にのみかける。小さい n の飽和による分布変化を隠すための重み再調整はしない。

## 2. P1: データ作成手順と計測

以下は **実装後** のコマンド。`make_spec.py --out` は `mixed.json` と整数 `n<N>.json`(11 個)を出力する。
全 test を先に予約し train/val から除外する。n8 の空間不足を先に検出でき、train/val 間は builder 内で dedup する。

```bash
uv run python experiments/pretrain/make_spec.py --out experiments/pretrain/specs
PRETRAIN_VERSION=$(git rev-parse HEAD)
for n in 8 12 16 24 32 48 64 96 128 192 256; do
  uv run python -m cfg_reducer.dataset_v2 --spec experiments/pretrain/specs/n${n}.json \
    --out data/pretrain_test_n${n} --version "$PRETRAIN_VERSION" \
    --split test=$((2000000+n*10000)):$((2010000+n*10000)) --target-count test=200 || exit
done
PRETRAIN_EXCLUDE=()
for n in 8 12 16 24 32 48 64 96 128 192 256; do
  PRETRAIN_EXCLUDE+=(--exclude-dataset data/pretrain_test_n${n})
done
uv run python -m cfg_reducer.dataset_v2 --spec experiments/pretrain/specs/mixed.json \
  --out data/pretrain_mix --version "$PRETRAIN_VERSION" --split train=1000000:1200000 \
  --split val=1200000:1250000 --target-count train=20000 --target-count val=1000 \
  "${PRETRAIN_EXCLUDE[@]}"
uv run python -m training.prepare_tokens --dataset data/pretrain_mix \
  --out data/tok_pretrain_mix --max-offset 128
for n in 8 12 16 24 32 48 64 96 128 192 256; do
  uv run python -m training.prepare_tokens --dataset data/pretrain_mix \
    --test-dataset data/pretrain_test_n${n} --out data/tok_pretrain_n${n} \
    --max-offset 128 --eval-length-policy unlimited || exit
done
```

`--target-count` を指定した整数 spec でも `record_node_counts=True`。本番は clean な生成 commit に固定する。
20k/1k/200 は canonical dataset の受理件数であり token 化後の件数ではない。REF 窓による除外を補充しない。
vocab は `REF_1..128` を含む 137 語。全 bundle の token→id 辞書 hash と train/val JSONL hash の一致を検査する。

**必要な tokenizer 変更**: 現在 `_prepare_cross_dataset` は `capacity=2*train_max_len` を超える val/test を
`over_length` として落とす。`prepare(..., eval_length_policy="legacy")` と CLI
`--eval-length-policy {legacy,unlimited}` を追加し、unlimited ではこの判定だけを無効化する。
meta の `sequence_capacity=null, eval_length_policy="unlimited"`、`max_len_source="train"` を記録する。
既定 legacy の出力は維持する。REF 窓判定・`excluded_over_window` の count/seeds・index.exclusions は共通。
通常 prepare の meta.max_len は全 split の最大値なので、trainer の training cap には使用しない。
`evaluation_index.json` の bucket/realized を使い、per-n raw/retained/excluded と理由別件数を報告する。

生成予算は学習の 4.7 分と分ける。既存 n48 sweep は約 2,370 train/val 候補を処理する規模だが、
参照文書には n48 の生成専用 wall time がない。`t48` を同規模生成の実測秒数とすると、23,200 採用は
同速度・棄却なしで `9.79*t48`。仮に t48=5–10 分なら 49–98 分、長い n と dedup を含め暫定 1–3 時間枠を確保する。
これは仮定による予算であり実測ではない(`mixture_doe.md` §7 の生成15分は n48専用計測ではない)。
最初に各 n × family 20 seeds の pilot を別ディレクトリで測り、n48 は 2,370 候補まで拡大して比較する。
生成/縮約/測定/iso照合/書込の秒数、棄却理由、採用 n/family、系列長 p50/p95/max、REF>128 率を記録する。
`sum(attempts[n,f]*seconds_per_attempt[n,f])` で更新し、n8/12 の採用率の逓減も見る。
既存 n12 mixed train 1631/2150、structured test 144 の経験から、200 件に届く保証はない。
不足時は不完全 manifest を残して報告し、重複許可・test 縮小・重み変更を実装者が勝手に行わない。

## 3. P2: 位置表現と attention

### 3.1 分岐の位置と系列長

`ARBaseline.__init__` の末尾引数に `pos="learned", pos_table_len=4096` を追加し、`build_parser` に
`--pos {learned,sinusoidal,alibi,none}` と正整数 `--pos-table-len` を追加する。
`build_model` は古い samples.json の Namespace にも使えるよう `getattr(args,"pos","learned")` 等で読む。

| pos | `tok_emb` 直後に作るもの | `encode(x)` の hidden / 長さ上限 |
|---|---|---|
| learned(既定) | 現行と同じ `nn.Embedding(max_len,d_model)` を `pos_emb` に置く | `tok_emb(x)+pos_emb(arange(L))[None]`。L が表容量を超えたら説明付きエラー |
| sinusoidal | `sinusoidal_table(pos_table_len,d_model)` を非永続 buffer `pos_table` に登録、pos_emb=None | 同じ加算位置で `pos_table[:L]` を足す。L>表長なら `max(L,2*表長)` に決定論的再生成 |
| alibi | pos_emb=None、非永続 buffer `alibi_slopes` | tok_emb のみ、下記の attention bias。位置表の上限なし |
| none | pos_emb=None、位置 buffer なし | tok_emb のみ、通常の causal attention。位置表の上限なし |

sinusoidal の初期表は 4096、CPU float32 の既存 `sinusoidal_table` で作り device/dtype を hidden に合わせる。
拡張は乱数を消費しない。非永続 buffer なので評価時の拡張が checkpoint shape を変えない。
`pos_table_len` は初期割当量であって scoring cap ではない。奇数 d_model の既存処理も維持する。
新3方式では dummy learned embedding を作らず、同じ seed なら token/encoder/head の初期値を方式間で揃えられる。
learned との RNG 一致は要求しないが、旧 learned 経路との完全一致は §4.3 で守る。

`model.max_len` の用途を分離する: learned の位置表容量、旧 structural lpos 表容量、生成予算の計算元。
`score_rows`/`token_logprobs` は max_len で切断・除外しない。learned の容量超過は失敗にして件数を隠さない。
learned の範囲内でも train 最大入力位置を超えた位置は未学習であり、外挿比較の候補にはしない。
`use_struct=True` では depth clamp は現状維持。learned lpos は `lpos_emb.num_embeddings-1` に clamp
(旧値 max_len-1 と同じ)し clamp 件数を新評価 metadata に記録する。
sinusoidal lpos は learned pos 経路では旧 frozen embedding/clamp を維持し、新方式では独立の拡張可能 buffer とする。
`depth_only` は lpos 表なし。`--pos none --struct-pos` は位置信号なしの定義と矛盾するため拒否する。
本比較は struct-pos を全て無効、`sequence_aux` の pointer context は `pointer or ref_legal_mask` のまま使う。

### 3.2 ALiBi の定義と PyTorch 2.14 の検証根拠

head 数 H が 2 の冪のとき `a=2**(-2**(-(log2(H)-3)))`, `slopes[h]=a**(h+1)`。
H=4 なら `[1/4,1/16,1/64,1/256]`。それ以外は `p=2**floor(log2(H))` とし
`slopes(p) + slopes(2*p)[0::2][:H-p]`。これは [ALiBi 著者の実装](https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L693) の順序である。
`H>0` と `d_model%H==0` を検証する。全層で同じ slopes、追加の学習パラメタなし。

`encode` で i=query, j=key、`distance=i-j`、`bias[h,i,j]=-slopes[h]*distance`。
既存 `generate_square_subsequent_mask(L)` の float causal(対角以下0、上側 -inf)を加算し、
`(H,L,L) -> expand(B,H,L,L) -> reshape(B*H,L,L)`。並びは batch-major/head-minor。
mask の dtype/device は hidden と一致させる。padding mask は `(B,L)` の float、PAD のみ -inf、他は0。
`encoder(hidden, mask=mask, src_key_padding_mask=padding, is_causal=False)` と明示する。
有限 bias を bool に変換しない。右 PAD の query 行は loss の valid mask で除外し、全キーを消す query mask は作らない。

2.14 の [MultiheadAttention API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html) は
`batch_first=True` の `(B,L,E)` 入力と `(B*H,L,S)` の float additive mask を認める。
[v2.14.0 functional.py](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/functional.py#L6450) は
3-D shape を検証し、padding mask を加算する。`is_causal=True` は条件によって明示 mask を捨てて
SDPA の causal hint に置換するため、bias 付き mask では False が必須である。

ALiBi の encoder は `nn.TransformerEncoder(layer,num_layers,enable_nested_tensor=False)`。
さらに encoder 呼出しを `get_fastpath_enabled()` の退避 → `set_fastpath_enabled(False)` → `try/finally` 復元で囲む。
[v2.14.0 TransformerEncoderLayer.forward](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/modules/transformer.py#L775)
はこのフラグで fused encoder 経路を回避し、`_sa_block` から明示 mask を MHA に渡す。
nested 無効だけで fused layer が無効になると仮定しない。runner は逐次実行であり、この scoped global 設定を共有スレッドで使わない。
既定 learned の encoder 設定、float causal + bool padding の旧経路、causal 自動判定には手を入れない。
新 sinusoidal/none は nested 無効、float causal/padding、`is_causal=False` に統一する。

以上は対象 tag の API/ソース確認。設計作成環境では Docker socket が permission denied のため、
`pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime` 内の実行確認は未了。§7 の smoke を完了条件にする。
dropout=0、B=2/H=4/L=7、padding 有無、train/backward と eval/no_grad の双方で shape/有限性を確認し、
head ごとの手計算 `softmax(QKᵀ/sqrt(d)-slope*(i-j)+causal+padding)V` と一致させる。
未来 token の変更で過去出力が変わらないこと、slopes=0 と非零で出力が変わること、H=3 の slope 順も確認する。

## 4. P2: バッチ・系列長・checkpoint

### 4.1 長さ管理とバッチ順

`--max-len` は非負整数、既定0=読み込んだ train の最大 `len(tokens)`(BOS/EOS 込み)。
`main` で train rows を保持し、`len(tokens)>effective_max_len` を aux 作成前に丸ごと落とす。
全除外/空 train/空 val は失敗。val/test はこの閾値で除外せず、early stopping も val 全件で行う。
token_logprobs の入力長は L-1、scored targets は BOS を除き EOS を含む L-1。切断による偽 EOS は作らない。
新 `length_stats.json` に raw/kept/dropped 件数・token 数・除外 sample_id/seed・閾値を保存し、
history 各 epoch に `train_kept,train_dropped_over_max_len,train_tokens` を追加する。eval_buckets も同じ件数を載せる。

`iter_batches(...,rng=None,*,length_buckets=False,bucket_size=0)` を追加。
CLI は `--length-buckets`(既定 False)、`--bucket-size`(既定0、有効時は20*batch_size)。
有効時は `(len(seq),original_index)` で安定ソートし、bucket_size 個ごとに区切る。
学習では各 bucket 内を順番に rng.shuffle、次に bucket 配列を rng.shuffle、各 bucket 内で batch に分割する。
bucket 境界で短い末尾 batch を落とさず、隣 bucket と結合しない。seq と aux は同じ index で取り出す。
`rng=None` の val は順序付き bucket で padding を削減する。`score_rows` は従来の1行ずつの採点で順序維持。
`run_epoch` はフラグを転送するだけで token weighting/accuracy の2回目 encode/optimizer 順序は変更しない。
`main` の単一 `random.Random(seed)` を epoch 間で持ち回る。履歴全体が同一 seed・データ・設定で再現する。
無効時は現在の `rng.shuffle(order)` を含む分岐そのものを残す。bucket 操作に torch RNG は使わない。

`sample_stream` は明示 generation budget で停止し、scoring cap と共有しない。
learned の既定 `gen_max_len=args.gen_max_len or 2*meta["max_len"]` と
`build_model` の容量 `max(meta["max_len"],gen_max_len)` は旧計算どおり。
新3方式の既定生成予算は `2*max(len(s) for retained_train)`、指定値があればそれを使い、
解決値を args.gen_max_len に保存する。sinusoidal は必要なら表を伸ばし、ALiBi/none は予算以外の制限なし。
learned の `--init-from` では元の容量を保持し、要求入力/生成予算が収まらなければ事前エラー。表の自動補間はしない。

### 4.2 config と微調整

`main` は optimizer 作成前に `config.json`(schema_version=1)を書き、次を含める。
`args`(全 CLI と effective 長)、`vocab`(token→id 全体)、`vocab_sha256`、`model`(constructor 全値、表容量)、
`meta_summary`(max_offset/fixed、split counts、sources、除外数、train 長統計)、`git_commit`(取得不能なら null)、
`torch_version`、`init_from`(元 config/model の SHA256)。hash は UTF-8 の sorted/compact JSON、vocab の空白差に依存しない。
stdlib subprocess による git 取得は任意情報であり Colab に .git がなくても動作する。
runner の backend.json の実行元と混同しない。config はモデル再構築、meta はデータ由来の記録を担う。

`--init-from <run_dir>` は config.json/model.pt を必須とし、実行前に vocab 辞書/hash、pos、
d_model/nhead/layers/FF/dropout、struct と pointer の全 mode、max_k/ref-legal-mask、learned/depth 表容量を比較する。
`state_dict` の key と shape を全比較し `load_state_dict(strict=True)`。vocab の同サイズで id が違う場合も拒否する。
sinusoidal/ALiBi の非永続表は shape 比較対象外、pos_table_len は記録するが伸長可能。
不足キーや旧 run(config.json なし)からの微調整は明示エラー。旧 checkpoint の通常 rescore は従来どおり可能。
AdamW は新規作成し epoch/RNG/early stopping を新 seed から開始する。「再開」ではなく重み初期化である。
out と init-from が同一なら拒否する。元 model.pt は上書きしない。

元 checkpoint の解決済み `model_max_len` を args に保持して `build_model` が優先利用する。
`samples.json["config"]` にもこの args を保存し、既存 rescore script が source meta と Namespace から同じ形を復元できるようにする。
`TrainJob.argv` の順序と `train_wrapper` の本文は変更しない。新 flags は末尾 extra だけで渡す。
`runner/executor.py` の TRAIN_OUTPUTS に config.json/length_stats.json を追加し、古い run で欠けても回収可能にする。
`--init-from runs/<name>` の runner 対応は、executor がローカル config/model を検証して
`<root>/init_<name>/` に stage、実行用 job の extra の値だけ置換する(元 Plan は保持)。
dry-run も転送先を表示し、絶対ローカルパスを Colab にそのまま渡さない。新 Plan schema は不要。

### 4.3 互換性

`--pos` 未指定、追加 flags 無効、init-from なしの旧 run は、同じ torch/device/seed で数値結果を bit 同一にする。
tok_emb → pos_emb → depth/lpos(有効時) → layer → encoder clone → head の初期化順を保持する。
追加 metadata/hash 計算は RNG を消費しない。モデル parameter key/order、batch 順、dropout 呼出し回数も保持する。
既定の bucket 有効化や learned 表4096化は互換性を壊すため行わない。追加 JSON fields の byte 一致は要求しない。
`tests/test_training.py` は現状 train_ar を import せず AST で依存境界を検査し、runner tests は wrapper の byte 比較を行う。
通常 pytest はこの torch-free 方針を継続する。数値試験は `training/smoke_longseq.py` を別プロセスで実行し、
`importlib.util.find_spec("torch")` が None なら pytest 側 wrapper は clean skip(install/import しない)。
torch がある VM 内だけで旧版を参照し、初期 state_dict/RNG state、1 epoch 更新、score、sample IDs の完全一致を検証する。

## 5. P3: 評価の入力と指標

### 5.1 集計契約

新規 `training/eval_buckets.py` は torch-free。CLI は `--index experiments/pretrain/eval_index.json --out ...`。
index の schema_version=1、`source_dataset`、`source_bundle`、`buckets`(n/dataset/bundle)、
`runs`(name/pos/seed/source_run/scores_by_n)、`selection_protocol` を必須にする。
各 scores_by_n は n→`runs/<out_run>/test_scores.jsonl`。make_plan.py が Plan と対で生成する。
dataset manifest から `sample_id -> resolved spec,seed,n` を読み、mixture は `component_for` で family を得る。
スコア行順やファイル名から n/family を推定しない。manifest hash は bundle index.sources と照合する。
全要求 run/seed/n のファイル、重複なし、token rows と scores の ID 完全一致、n_tokens=L-1、有限 nll を検証する。
`controlled_eval.validate_scores/ref_records/by_k/offset_by_k/edge_accuracy` を再利用する。
token-weighted paired CI は同じsample対を `Random(0)` で1000回 bootstrap し、各回 `sum(Δnll)/sum(n_tokens)`。
既存 `paired_delta` の sample平均Δは補助欄に限り、token micro の CI と混同しない。
不足は失敗。空 bucket は raw/retained と `null` の指標を出し、0 NLL や0%違反にしない。

| 出力単位 | 指標と分母 |
|---|---|
| n、n×family、family 全体 | `sum(nll)/sum(n_tokens)`(nats/token)、samples/tokens/raw/retained/excluded、長さ統計 |
| gold REF の k ごと | 件数、full REF NLL、`NLL(REF_k)-ref_type_nll`、edge accuracy。低件数も残し n<30 を注記 |
| ID 全体 | token micro と9個の n の等重み macro を別欄。OOD192/256 を混ぜない |
| seed 集計 | 各 seed 値、mean/SD、同じ ID による paired ΔNLL。標本数不足を seed 平均で隠さない |

score_rows の既存 `ref_correct` は正解距離との一致であり、合法性違反ではない。
`--ref-diagnostics`(既定False)を追加し、build_model 経由で model に保持する。
診断だけでも `score_rows` 内の sequence_aux は `pointer or model.ref_diagnostics` で参照 context を作る。
`main.needs_grammar` に ref-diagnostics/wf-probes も含める。run_epoch の aux 契約は既存どおり。
有効時、各 gold REF 位置について effective logits の REF 内 argmax の `ref_pred_k` と
`ref_pred_legal`(`klast<k<=plpos-1`)を ref_pos と同順で追加し、by-k 違反率の分母を gold REF 件数にする。
REF が全て -inf となる gold REF 位置はデータ契約違反として失敗。pointer では選択 target を距離に変換する。
これは teacher-forced/legal-mask 後の診断であり、生成違反や raw logits の合法率とは区別する。

`InfoBaseline.fit(train_rows,vocab,128,alpha=.5)` と `info_baseline.score_rows` で同じ test IDs を採点する。
ID は source train の **同じ名目 n** の retained rows だけで fit。n/family の出力でも同じ per-n baseline を使う。
OOD192/256 は該当 train がないので、全 retained source train で fit した baseline をそれぞれの bucket に適用し、
`baseline_fit_scope="pooled_id"` と明記する(ID は `"n<N>"`)。test/val で fit しない。
超過は `sum(model_nll-baseline_nll)/sum(n_tokens)`、負値も許す。OOD は target 固有の情報量推定と呼ばない。
ID の fit rows が0なら excess は null。基準値・fit件数・ID/hash・alpha も保存する。

### 5.2 WF と外挿の生成診断

無条件の samples.json は source run の全体 WF であり、rescore だけでは target n に条件付けられない。
runner が rescore 出力に複製する eval.json を各 n の WF として扱わない。
各 run の samples.json と samples_constrained.json を `eval_samples.evaluate` で別々に評価し、
`reference-constrained`(ref-legal-mask は有効)と `grammar-constrained` を明記する。
EOS 未到達も分母200に含め、`no_eos:would_close_cleanly` を構造違反と分ける。

n192 の WF を比較可能にするため **gold-prefix continuation** を追加する(元 CFG の n 条件ではない)。
`--wf-probes 20`(既定0)、`score_rows` が ID 昇順の先頭 min(20,N) 件だけで実施する。
prefix は `tokens[:max(1,(len(tokens)-1)//2)]`、予算は BOS 込み `2*len(tokens)`、temperature=1、top_k=0。
`sample_stream(...,prefix=None)` を追加し、prefix 指定時はコピーから `max_len-len(ids)` 回まで生成する(未指定は従来と同じ)。
制約ありの GrammarState は prefix の BOS を除く各 token を push して復元する。
raw/制約ありを同じ seed で1本ずつ生成し、score 行の `wf_probe` に prefix_len/budget/seed と両 tokens を保存する。
seed は `int.from_bytes(sha256(f"{args.seed}:{sample_id}:wf-v1".encode()).digest()[:8],"big")`。
`torch.random.fork_rng` で CPU と使用 CUDA RNG を保存/復元し、通常 scoring/sampling の RNG を変えない。
build_model が probe 設定と seed を model に付けるので、既存 rescore wrapper でも同じ probe を再生成できる。
`eval_buckets` は prefix+completion 全体の WF と Wilson CI、生成 suffix の REF 違反を predicted k ごとに集計する。
合法 prefix を順次 parse し、最初の文法違反までに現れた REF を分母とする。そこで停止し、以後の token を合法扱いしない。
違反の種類、分母、途中停止件数を出す。k 別 TF 診断・continuation 診断・無条件診断を別欄にする。
全体の生成 n は指定できないため `prefix_source_n` と表示し、成功した生成だけから bucket を逆推定しない。

n48 は family 別に sweep §14 の専用モデル layered≈0.77/spaghetti≈0.61 と併記する。
旧窓・旧 test・旧 version の数値なので参考値。厳密な差には同じ window128/test の再学習が必要で、P3 には含めない。
REF>128 除外率も必ず併記し、「長い全 CFG に汎化した」という主張はしない。

## 6. P3: Plan と計算予算

### 6.1 配置と実行順

`experiments/pretrain/make_plan.py` は `--phase compare|final --pos <chosen> --out-dir ...` を受け、
`plan_compare.json` / `plan_final.json` と対応する `eval_index_<phase>.json` を出力する。
Plan は既存型だけを使う。各 Plan は source_bundle=`data/tok_pretrain_n48` の **1 Block**。
これには同一20k/1kの train/val と n48 test があり、test は early stopping に使わない。
比較は sinusoidal/alibi/none を seed=0 で3本。最終は選択 pos の seed=0,1,2 を別 run 名で3本、warm-start しない。
最終 seed0 を比較 run と共有せず再実行するため、予定は6 training runs。以下は1 job/1 rescore の構造例。

```json
{
  "plan_id":"pretrain_compare_v1",
  "blocks":[{
    "source_bundle":"data/tok_pretrain_n48",
    "train":[{
      "name":"pretrain_cmp_sinusoidal_s0","epochs":60,"patience":10,
      "num_samples":200,"constrained_samples":200,"seed":0,"sample_seed":1000,
      "extra":["--pos","sinusoidal","--pos-table-len","4096","--ref-legal-mask",
               "--length-buckets","--max-len","0","--ref-diagnostics","--wf-probes","20",
               "--d-model","128","--num-layers","4","--nhead","4","--dim-feedforward","512"]
    }],
    "rescore":[{
      "source_run":"pretrain_cmp_sinusoidal_s0","target_bundle":"data/tok_pretrain_n192",
      "out_run":"pretrain_cmp_sinusoidal_s0__n192"
    }]
  }]
}
```

生成される完全 Plan は train 3件 × rescore 11件(n8..128,192,256)、計33件。n48 も rescore し本採点と一致を確認する。
最終も同じ33件。names は `pretrain_final_<pos>_s<seed>`、sample_seed=1000+seed。
RescoreJob.source_run は同 Block の TrainJob に必ず存在させる。run/out_run は一意、extra で --out/--seed 等を重複上書きしない。
runner は target の test だけを転送するため、make_plan 時点で全 bundle の vocab/hash/index を照合する。

```bash
uv run python experiments/pretrain/make_plan.py --phase compare --out-dir experiments/pretrain
uv run python -m training.runner run --plan experiments/pretrain/plan_compare.json --backend colab --dry-run
uv run python -m training.runner run --plan experiments/pretrain/plan_compare.json --backend colab --session pretrain
uv run python -m training.eval_buckets --index experiments/pretrain/eval_index_compare.json --out experiments/pretrain/compare.json
# selection.json の chosen_pos を確定後、同じ dry-run → run → eval を final に実施する。
uv run python experiments/pretrain/make_plan.py --phase final --pos alibi --out-dir experiments/pretrain
```

末行の alibi は選択値の例であり既定決定ではない。選択規則は n192 retained 全件の NLL/token 最小、
paired ΔNLL の95% CIが0を跨ぐ方式は同等候補とし、REF違反が少ない、prefix WFが高い、秒/epochが短い順で決める。
異常値/NaN/制約生成の文法違反は選択前に実装検査する。n192 は方式選択用 OOD dev と位置付け、n256 は選択に使わない。
全 rescoring 結果を保存しても n256 を開くのは選択固定後。ID9 bucket の代償も selection.json に記録する。
「同長比で劣化なし」を採用条件にするには別の同長専門家が必要なので、本比較では3方式間の順位のみ結論にする。

### 6.2 時間・メモリ・early stopping

参照は RTX 2080 Ti、2150 samples × 平均133 tokens = **285,950 tokens/epoch**、99 epochs/4.7分。
新規20kの平均長を μ とすると token 比は `20000*μ/285950`、60 epoch の線形換算時間は
`4.7*(20000*μ/285950)*(60/99)` 分。実計測は BOS/PAD を除く `sum(len(tokens)-1)` に統一する。

| 仮定 | μ / epoch tokens | 参照比 | 60 epoch の2080 Ti線形換算 |
|---|---|---|---|
| 指定重み・長さ∝n・棄却なし | E[n]=34.6、μ≈95.9 / 1.92M | 6.71倍 | 約19.1分 |
| 上位設計の保守仮定(平均長2倍) | μ=266 / 5.32M | 18.60倍 | 約53.0分 |

したがって20kだから必ず平均長2倍になるわけではない。小 n の重複除外で実現平均が大きくなる可能性がある。
T4 は同じ tokens/sec ではないため、暫定 **1–1.5時間/run**(比較3 + 最終3 = 6–9時間)を予約し、
rescore/転送/400本の逐次生成と prefix probes に追加1–2時間を見込む。任意の容量1 run は別に1–2時間以上を確保する。
上位設計の総額8–10時間は pilot 後に更新し、保証値として扱わない。

attention は `sum(B*Lmax_batch²)` に依存する。ALiBi mask のみでも fp32 は `4*B*H*L²` bytes
(B64/H4/L1024 で1GiB)。通常 scoring はB1だが、学習の長い末尾 batch は別に peak memory を測る。
最初の tiny Plan 後、各方式2 epochsで秒/epoch、非PAD token/s、padding率、Lmax、peak VRAM、
sample/probe/rescore 時間を記録する。OOM なら比較全方式の batch_size を同じ32または16にして新 run 名で再実行する。
学習中に自動的に batch_size を変えない。既存 run_epoch の accuracy 用2回目 encode も見積りに含める。

early stopping は既存の `val_loss < best_val`(min_delta=0)、patience=10、上限60、best model.pt を採用する。
test は checkpoint の選択・early stopping に不使用、best_epoch/epochs_run を保存。容量256d/6Lは20kでval改善が続く場合だけP4。
runner の Colab inner timeout は現状5400秒、outer は6000秒。`--train-timeout-s` だけでは inner は延びない。
pilot 推定が75分を超えたら長時間 run を投入する前に実行枠/timeout をオーケストレータが決める。
`--init-from` は optimizer resume ではなく、セッション切断による完全再開機能は今回追加しない。

## 7. 実装タスクと受入条件

各タスクは Codex low / OpenCode 1 run の所有範囲。前タスクの受入後に進み、無関係な改修はしない。

| タスク | 所有ファイル・範囲 | 必須受入条件 |
|---|---|---|
| T1 / P1 | dataset_v2.py、dataset.py の選択経路、make_spec.py、対応 tests | 分布検証/RNG固定、全 n の scaling、旧整数/v1出力不変、resolved provenance UUID、mixture composition、exclude再生、quota/不足/全n集計 |
| T2 / P2長さ | train_ar.py の位置/batches/長さCLI、prepare_tokens.py、smoke_longseq.py、対応 tests | 4 pos、4096超伸長、ALiBi手計算/causal/padding/train・eval、learned数値互換、bucket被覆/epoch再現、max-len除外数、評価上限撤去 |
| T3 / P2運用 | train_ar.py の config/init/score/sample、runner/executor.py、対応 tests | config回収、init成功/不一致拒否、REF/probe診断、RNG非干渉、wrapper互換 |
| T4 / P3 | eval_buckets.py、make_plan.py、eval index/Plan fixtures、対応 tests | 完全性検査、手計算NLL/excess、family/n分母、窓除外、WF失敗を含む分母、3+3 run/66 rescore生成、両backend dry-run |

T1 は `PYTHONHASHSEED=1,77` の subprocess 結果を比較し、distribution の重み変更で同じ resolved spec の UUID が
変わらないこと、同型の別seedを採用しないことを検証する。小さい n の全探索/大規模生成は `GR_SLOW_TESTS=1` のみ。
T3 は長さの異なる短い fixtures、vocab id 入替え、pos/head shape の不一致、sinusoidal score L>4096 を検証する。
T4 は score欠落/重複/NaN/誤ったsource hashを拒否し、TF正解誤りとREF合法性違反を区別する fixture を持つ。
どのタスクも `uv run pytest -q`、`uv run ty check`、`git diff --check` を必須とし、依存を追加しない。

T2/T3 は通常pytestだけでは完了としない。旧 train_ar.py を実装前に `/tmp/train_ar_before.py` に退避し、
torch のある対象イメージ内の CPU smoke、または training/runner 経由の tiny Colab Plan を必ず実行する。
smoke_longseq.py は stdlib+torch の独立 acceptance harness(通常のアップロード対象には追加しない)とする。
tiny Plan は合法train8/val2/test2、4 pos各1epoch、sample2+constrained2、長いtargetへのrescore、
さらに新runへのinit-fromを含み、test_scores/config/length_statsと有効なprobeを回収する。

```bash
# ホスト側の起動も uv 経由。image は事前配置済み、pull/install はしない。
uv run python -c 'import subprocess; subprocess.run(["docker","run","--rm","--pull=never","-v","/home/fischeri/Projects/Compiler/gr:/work:ro","-v","/tmp/train_ar_before.py:/tmp/train_ar_before.py:ro","-w","/work","pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime","python","training/smoke_longseq.py","--legacy","/tmp/train_ar_before.py","--device","cpu"],check=True)'
uv run pytest -q
uv run ty check
git diff --check
```

smoke は結果を `/tmp` に書き、torch.__version__/image ID と差分を保存する。skip は通常 repo 検証の成功であって
torch smoke 合格の代用ではない。変更ファイルと検証結果を報告し、実装者は commit/本番生成/本番学習をしない。

## 8. 決定とオーケストレータへの確認事項

実装契約は上記の dataset 専用分布 codec、規則表による concrete params、resolved spec の UUID、
独立 n RNG、既定 learned 互換、新方式のみ位置表伸長、無制限の評価 token 化、既存 Plan 型とする。
以下は実験投入前の確認事項であり、実装者が仕様を曖昧に補完する箇所ではない。

1. pilot で n8/12 の test200 が満たせない、または小 n の受理比率が60%から大幅に落ちる場合、件数/分布をどう改訂するか。
2. n192 を方式選択用、n256 を最終外挿評価用とする運用、および prefix WF 20件の不確実性で十分か。
3. OOD の超過 NLL は pooled-ID 基準で開始する。target固有基準が必要なら、別seedのcalibration dataset追加を別途承認するか。
4. pilot の実測に基づく Colab枠/5400秒timeout、256 test の実行可能性、任意P4の予算をどこまで確保するか。

## 9. オーケストレータの回答(2026-09-26)

1. n8 / n12 の test が 200 に届かない場合: 得られた件数(最低 100)で確定し、100 未満なら ID の macro 平均から
   その n を外して注記する。重み・分布は変えない。
2. n192 を方式選択用 OOD dev、n256 を最終外挿評価用とする運用を採用。prefix WF は生成が逐次で高価
   (予算 2×len、KV cache なし)なので、比較 phase は `--wf-probes 0`、最終 phase は `--wf-probes 5`。
   無条件生成も比較 phase は num_samples / constrained_samples = 100、最終は 200。
3. OOD の超過 NLL は pooled-ID 基準で開始する。target 固有基準は必要になったら別途。
4. pilot(各方式 2 epoch)の実測後に Colab 枠を決める。75 分/run を超えるなら runner の inner timeout(5400 秒)を
   CLI から延ばせるようにする(T3 の範囲に `--colab-inner-timeout-s` を追加してよい)。
5. 実装順: T1(P1)→ T2 → T3 → T4。T1 は main checkout で、seqdesign の T1 と並行。Docker smoke は
   このマシン(2080 Ti、image 配置済み)でオーケストレータが実行する。
