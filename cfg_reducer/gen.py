from __future__ import annotations

import random

from .engine import GraphEngine


def build_cfg(engine: GraphEngine | None = None,
              num_nodes: int = 12,
              edge_prob: float = 0.5,
              seed: int = 42) -> GraphEngine:
    """
    構造ベース＋スパゲティ化のハイブリッドCFG生成。
    DAG（有向非巡回グラフ）で基本構造を作り、後から安全なループとジャンプを追加する。
    """
    if engine is None:
        engine = GraphEngine()

    rng = random.Random(seed)
    ids = [f"N{i:02d}" for i in range(num_nodes)]

    for nid in ids:
        engine.add_node(nid)

    if num_nodes < 3:
        for i in range(num_nodes - 1):
            engine.add_edge(ids[i], ids[i+1])
        return engine

    # ──────────────────────────────────────────────
    # 1. 前方への基本構造構築 (DAG)
    # ──────────────────────────────────────────────
    start_node = ids[0]
    exit_node = ids[-1]

    current_layer = [start_node]
    unassigned_nodes = ids[1:-1]

    while unassigned_nodes:
        # 次のレイヤーのサイズを決定
        # 1なら直列または合流、2〜3なら分岐(if/switch)を模倣
        max_next_size = min(len(unassigned_nodes), rng.choice([1, 1, 2, 2, 3]))
        next_layer = unassigned_nodes[:max_next_size]
        unassigned_nodes = unassigned_nodes[max_next_size:]

        # current_layer から next_layer へエッジを張る
        # 孤立ノードや行き止まりを作らないための制約を課す
        for u in current_layer:
            # 親は必ず最低1つの子を持つ
            v = rng.choice(next_layer)
            engine.add_edge(u, v)

        for v in next_layer:
            # 子は必ず最低1つの親を持つ（上で繋がらなかった場合をフォロー）
            if not any(v in engine.successors(u) for u in current_layer):
                u = rng.choice(current_layer)
                engine.add_edge(u, v)

        # さらにランダムで交差エッジを足し、より複雑な合流を作る
        for u in current_layer:
            for v in next_layer:
                if rng.random() < edge_prob and v not in engine.successors(u):
                    engine.add_edge(u, v)

        current_layer = next_layer

    # 残った末端をすべて EXIT ノードに繋いで収束させる
    for node in current_layer:
        if exit_node not in engine.successors(node):
            engine.add_edge(node, exit_node)

    # ──────────────────────────────────────────────
    # 2. スパゲティ化（ループ・大ジャンプの追加）
    # ──────────────────────────────────────────────

    # [A] ループ（後方エッジ）: while / for の表現
    num_loops = max(1, int(num_nodes * 0.15))
    for _ in range(num_loops):
        j = rng.randint(2, num_nodes - 2)
        # 戻る先は最低でも2つ前。直前(j-1)には戻さない（双方向エッジの禁止）
        if j - 2 >= 0:
            i = rng.randint(0, j - 2)
            u, v = ids[j], ids[i]
            if v not in engine.successors(u):
                engine.add_edge(u, v)

    # [B] 大ジャンプ（前方への遠距離エッジ）: break / goto / 例外処理の表現
    num_gotos = max(1, int(num_nodes * 0.1))
    for _ in range(num_gotos):
        i = rng.randint(1, num_nodes - 3)
        if i + 2 <= num_nodes - 1:
            j = rng.randint(i + 2, num_nodes - 1)
            u, v = ids[i], ids[j]
            if v not in engine.successors(u):
                engine.add_edge(u, v)

    return engine
