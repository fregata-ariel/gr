**1. ガイドの全体要約**

GPT-5.6 Sol 向けプロンプトガイドの中心は、「手順を細かく縛る」よりも、成果物・制約・根拠・完了条件を明確にし、モデルに効率的な経路選択を任せることです。

構成と主要トピックは以下です。

- まずプロンプトを簡素化する: 重複ルール、効かない例、不要なプロセス指示、タスク外ツールを削る。残すのは成果物、成功条件、停止条件、安全・権限・根拠制約、出力形式。
- outcome-first: 「何を達成すべきか」「成功とは何か」「いつ止まるか」を書き、探索やツール利用の細かい手順は過剰指定しない。
- 応答長・人格・協調スタイル: GPT-5.6 は GPT-5.5 より簡潔になりがちなので、広い “be concise” は再評価。既定の詳細度は `text.verbosity` で制御。
- 自律性と承認境界: 調査・計画だけの依頼では変更しない、修正依頼ならローカル変更と非破壊検証を進める、外部書き込み・破壊的操作・高コスト操作は確認、という短い方針を一箇所に置く。
- ツールルーティング: タスク関連ツールだけを出し、前提取得・検証が必要なら明記。独立取得は並列、依存関係があれば逐次。
- Programmatic Tool Calling: 大量結果の絞り込み、集約、検証など、境界の明確な処理に使う。意味判断、承認、最終検証、引用保持は直接ツール呼び出しが向く。
- grounding / citation: 何を根拠づけるか、十分な根拠とは何か、根拠欠落時にどう狭めるかを明記。
- 長時間ワークフロー: 最初に短い可視更新、以後は大きなフェーズ変化だけ報告。compaction、persisted reasoning、prompt caching は測定しながら扱う。
- reasoning effort: 現行 5.4/5.5 の設定を基準にし、同設定と一段下を評価。`max` は全体標準ではなく最難関向け。
- 検証: コードでは対象テスト、型・lint、ビルド、スモークテスト。実行不可なら理由と次善策を説明。
- 推奨構造: `Role / Personality / Goal / Success criteria / Constraints / Tools / Output / Stop rules`。

**2. GPT-5.4/5.5 向け作法との差分**

大きな変化は、XML ブロックを厚く積むより、短い契約を一度だけ書く方向です。ガイドは XML 形式を禁止していません。実際、Programmatic Tool Calling の例では `<tool_orchestration>` のような XML 風ブロックも出ます。ただし推奨の基本構造は XML ブロック名ではなく、`Role`、`Goal`、`Success criteria`、`Constraints`、`Tools`、`Output`、`Stop rules` です。

既存の Codex 風 XML ブロックは、次のように圧縮して残すのが自然です。

- `<task>` → `Goal`
- `<action_safety>` → `Constraints` または承認境界
- `<grounding_rules>` → `Constraints` と `Tools`
- `<verification_loop>` → `Success criteria` と `Stop rules`
- `<completeness_contract>` → `Success criteria`
- `<structured_output_contract>` → `Output`

不要になりやすいものは、同じ安全ルールの反復、常に質問せよ・常に検索せよのような広すぎる絶対規則、モデルが安定して実行できる細かな手順、効果のない例示、タスク外ツール説明です。

新設・強調点は、`text.verbosity`、Programmatic Tool Calling、persisted reasoning、明示的 prompt caching、`reasoning.mode: "pro"`、`reasoning.effort: "max"`、モデルファミリー内の Sol/Terra/Luna 選択です。

reasoning effort は「上げればよい」ではありません。5.4/5.5 の現行値を維持してベースラインを取り、同じ値と一段下を代表タスクで比較します。`medium` はバランス開始点、`low` は latency-sensitive、`high`/`xhigh` は測定上の品質改善がある場合、`max` は最難関の品質優先ワークロードだけです。上げる前に、成功条件・依存関係・ツールルーティング・検証ループの欠落を疑うべき、とされています。

検証ループと完了契約は不要化ではなく、短く明示する方向です。コード変更後の対象テスト、型・lint、ビルド、スモークテストは引き続き推奨されています。実行不可なら理由と次善チェックを書く、という契約も維持すべきです。

**3. Sol / Terra / Luna の位置づけと使い分け**

取得した prompt guidance 本体は Sol を対象にしつつ GPT-5.6 family 全体への適用を述べ、詳細は model guide と併用するよう案内しています。model guide による位置づけは以下です。

- `gpt-5.6-sol`: flagship capability。`gpt-5.6` エイリアスは Sol にルーティングされる。最難関の計画、設計判断、深いレビュー、高価値な失敗分析向き。
- `gpt-5.6-terra`: strong performance at a lower price、知能とコストのバランス。通常の実装、修正、TDD 反復、レビュー補助の第一候補。
- `gpt-5.6-luna`: efficient, high-volume workloads。大量・反復・低単価が重要な分類、整形、軽量変換、下準備向き。

ガイド上は「計画向き」「実装向き」という明示ラベルではなく、能力・価格・高ボリューム効率で区分されています。したがって我々のワークフローに当てるなら、計画や最終判断は Sol、実装の標準は Terra、大量補助タスクは Luna、という割り当てが文書からの妥当な推論です。

**4. 委譲ワークフローへの具体的変更点**

- 最上位計画モデルは `gpt-5.6-sol`、中位実装モデルはまず `gpt-5.6-terra` を候補にする。`gpt-5.6` alias は Sol なので、コスト制御したい実装層では明示的に Terra を指定する。
- 既存 XML プロンプトは全面書き換えせず、まずモデルだけ切り替え、reasoning effort を現行値のまま評価する。
- その後、重複した XML ブロック、反復 safety 文、過剰な verification 手順、効いていない例を一群ずつ削って eval する。
- `verification_loop` と `completeness_contract` は削除対象ではなく、`Success criteria` と `Stop rules` に短く統合する。
- TDD 指示は維持する。ただし「必ず全部回す」ではなく、変更範囲に最も関連するテスト、型・lint、ビルド、スモークの優先順位を書く。
- 構造化出力契約は維持する。Programmatic Tool Calling を使う場合は `program_output` と最終 `message` の両方が必須フィールド・引用・ caveat を満たすかテストする。
- reasoning effort は Sol/Terra それぞれで、現行値と一段下を比較する。`max` や pro mode は計画・レビュー・高リスク設計など、測定で効果があるケースだけに限定する。
- 長時間 Codex タスクの進捗報告は、各ツール呼び出しの実況ではなく、フェーズ変化と判断変更時だけに寄せる。
- 検索・根拠づけタスクでは、根拠不足を「存在しない」と即断しないルールを残す。取得済みソースだけを引用し、推論は推論として明示する。
- PTC は「大量候補の絞り込み・集約・重複除去」など境界の明確な段階だけに使い、承認、意味判断、最終引用付き回答は直接モデル判断に戻す。

sources: https://developers.openai.com/api/docs/guides/prompt-guidance-gpt-5p6 ; https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.6

