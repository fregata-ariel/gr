"""
Interactive CFG reduction visualizer.

Controls:
    Space / Enter   step forward
    U               undo
    R               redo
    S               save step history to JSON
    Q               quit

Layout:
    cfg_reducer/        (package)
    main.py             (this file)
"""

import random
import matplotlib.pyplot as plt
import networkx as nx

from cfg_reducer import GraphEngine, ReductionAlgorithm, store


# ──────────────────────────────────────────────
#  Graph construction
# ──────────────────────────────────────────────

def build_cfg(engine: GraphEngine,
              num_nodes: int = 12,
              edge_prob: float = 0.5,
              seed: int = 42) -> list[str]:
    """
    構造ベース＋スパゲティ化のハイブリッドCFG生成。
    DAG（有向非巡回グラフ）で基本構造を作り、後から安全なループとジャンプを追加する。
    """
    rng = random.Random(seed)
    ids = [f"N{i:02d}" for i in range(num_nodes)]

    for nid in ids:
        engine.add_node(nid)

    if num_nodes < 3:
        for i in range(num_nodes - 1):
            engine.add_edge(ids[i], ids[i+1])
        return ids

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

    return ids


# ──────────────────────────────────────────────
#  Layout (networkx used ONLY here)
# ──────────────────────────────────────────────

def compute_layout(engine: GraphEngine, seed: int = 42) -> dict:
    """Compute spring layout from the initial graph snapshot."""
    G = nx.DiGraph()
    for nid in engine.node_ids():
        G.add_node(nid)
        for s in engine.successors(nid):
            G.add_edge(nid, s)
    return nx.spring_layout(G, seed=seed)


# ──────────────────────────────────────────────
#  Drawing
# ──────────────────────────────────────────────

def _build_draw_graph(engine: GraphEngine,
                      algo: ReductionAlgorithm) -> nx.DiGraph:
    """Build a temporary nx.DiGraph for the visible portion."""
    if algo.scope.active:
        visible = algo.scope.focus & engine.node_ids()
    else:
        visible = engine.node_ids()

    G = nx.DiGraph()
    for nid in visible:
        G.add_node(nid)
        for s in engine.successors(nid):
            if s in visible:
                G.add_edge(nid, s)
    return G


def draw(ax, engine, algo, pos, log_msg):
    ax.clear()

    zoom = algo.scope.active
    prefix = "[ZOOM] " if zoom else ""
    title_color = "red" if zoom else "black"
    ax.set_title(f"{prefix}{log_msg}",
                 fontsize=11, loc="left", color=title_color)

    if engine.is_empty():
        ax.text(0.5, 0.5, "Reduction Complete!",
                fontsize=20, ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        _draw_footer(ax, engine, algo)
        return

    G = _build_draw_graph(engine, algo)
    if not G.nodes():
        _draw_footer(ax, engine, algo)
        return

    draw_pos = {n: pos[n] for n in G.nodes() if n in pos}

    # ── labels: show weight when > 1 ──
    labels = {}
    for nid in G.nodes():
        w = engine.weight(nid)
        labels[nid] = f"{nid}\nw={w}" if w > 1 else nid

    # ── node fill ──
    base_color = "#ffb3b3" if zoom else "#99ccff"
    node_colors = [base_color] * len(G)

    nx.draw(G, pos=draw_pos, ax=ax,
            labels=labels,
            node_color=node_colors,
            node_size=900,
            font_size=9,
            font_weight="bold",
            arrows=True,
            arrowsize=15)

    # ── heap highlight (green ring) ──
    heap_set = {nid for _, nid in algo.heap if nid in G}
    if heap_set:
        nx.draw_networkx_nodes(
            G, pos=draw_pos, ax=ax,
            nodelist=sorted(heap_set),
            node_color="none",
            edgecolors="green",
            linewidths=3,
            node_size=900)

    _draw_footer(ax, engine, algo)


def _draw_footer(ax, engine, algo):
    heap_count = sum(1 for _, n in algo.heap if engine.has_node(n))
    ax.text(
        0.01, 0.01,
        (f"Step: {engine.cursor}  |  "
         f"Nodes: {len(engine.node_ids())}  |  "
         f"Heap: {heap_count}  |  "
         "[Space] step  [U] undo  [R] redo  [S] save"),
        transform=ax.transAxes, fontsize=8, color="gray",
        verticalalignment="bottom")


# ──────────────────────────────────────────────
#  Log message formatters
# ──────────────────────────────────────────────

def _fmt_step(op) -> str:
    if op.kind == "remove_node":
        t = op.forward["target"]
        w = op.inverse["weight"]
        return f"Removed [{t}]  (weight={w})"

    if op.kind == "remove_edges":
        header = op.meta.get("header", "?")
        scc = op.meta.get("scc", [])
        return f"Loop {scc} → header [{header}], back-edges cut"

    return f"{op.kind}: {op.forward}"


def _fmt_undo(op) -> str:
    if op.kind == "remove_node":
        return f"Undo remove [{op.forward['target']}]"
    if op.kind == "remove_edges":
        return f"Undo break-edges (header [{op.meta.get('header', '?')}])"
    return f"Undo {op.kind}"


def _fmt_redo(op) -> str:
    if op.kind == "remove_node":
        return f"Redo remove [{op.forward['target']}]"
    if op.kind == "remove_edges":
        return f"Redo break-edges (header [{op.meta.get('header', '?')}])"
    return f"Redo {op.kind}"


# ──────────────────────────────────────────────
#  Key handler
# ──────────────────────────────────────────────

def make_key_handler(engine, algo, pos, fig, ax):
    log_msg = "Press Space to start."

    def on_key(event):
        nonlocal log_msg

        if event.key in (" ", "enter"):
            op = algo.step()
            log_msg = _fmt_step(op) if op else "Reduction complete!"

        elif event.key == "u":
            op = algo.undo()
            log_msg = _fmt_undo(op) if op else "Nothing to undo."

        elif event.key == "r":
            op = algo.redo()
            log_msg = _fmt_redo(op) if op else "Nothing to redo."

        elif event.key == "s":
            path = "reduction_steps.json"
            store.save(engine.history, path)
            log_msg = f"Saved {len(engine.history)} ops → {path}"

        elif event.key == "q":
            plt.close(fig)
            return

        else:
            return

        draw(ax, engine, algo, pos, log_msg)
        fig.canvas.draw()
        print(log_msg)

    return on_key, log_msg


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    engine = GraphEngine()
    build_cfg(engine, num_nodes=12, edge_prob=0.18, seed=42)
    pos = compute_layout(engine, seed=42)
    algo = ReductionAlgorithm(engine)

    fig, ax = plt.subplots(figsize=(10, 8))
    on_key, init_msg = make_key_handler(engine, algo, pos, fig, ax)
    fig.canvas.mpl_connect("key_press_event", on_key)

    draw(ax, engine, algo, pos, init_msg)
    plt.show()


if __name__ == "__main__":
    main()
