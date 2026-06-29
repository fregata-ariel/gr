"""
Interactive CFG reduction visualizer with MetaGraph views.

Controls (CFG view):
    Space / Enter   step forward
    U               undo
    R               redo
    S               save step history to JSON
    M               switch to MetaGraph view (after reduction completes)
    Q               quit

Controls (MetaGraph view):
    M               toggle motif_kind / motif_depth coloring
    Left / Right    select Loop node
    D               drill into selected Loop
    B / Escape      back to parent level (or CFG view)
    Q               quit

Layout:
    cfg_reducer/        (package)
    main.py             (this file)
"""

import random
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from cfg_reducer import GraphEngine, ReductionAlgorithm, store, motif, metagraph
from cfg_reducer.types import MetaGraph


# ──────────────────────────────────────────────
#  Graph construction
# ──────────────────────────────────────────────

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
#  MetaGraph utilities
# ──────────────────────────────────────────────

def _compute_depths(mg: MetaGraph) -> dict[int, int]:
    steps = {m.step for m in mg.motifs}
    adj: dict[int, list[int]] = {s: [] for s in steps}
    in_deg: dict[int, int] = {s: 0 for s in steps}
    for src, dst in mg.edges:
        adj[src].append(dst)
        in_deg[dst] += 1
    depths: dict[int, int] = {s: 0 for s in steps}
    queue = [s for s in sorted(steps) if in_deg[s] == 0]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            depths[v] = max(depths[v], depths[u] + 1)
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    return depths


def _compute_metagraph_layout(
    mg: MetaGraph, depths: dict[int, int],
) -> dict[int, tuple[float, float]]:
    layers: dict[int, list[int]] = {}
    for step, d in depths.items():
        layers.setdefault(d, []).append(step)
    pos: dict[int, tuple[float, float]] = {}
    for d, members in layers.items():
        members.sort()
        n = len(members)
        for i, step in enumerate(members):
            x = (i - (n - 1) / 2) * 1.5
            y = -d * 1.5
            pos[step] = (x, y)
    return pos


def _build_metagraph_nx(mg: MetaGraph) -> nx.DiGraph:
    G = nx.DiGraph()
    for m in mg.motifs:
        G.add_node(m.step, kind=m.kind, motif=m)
    for src, dst in mg.edges:
        G.add_edge(src, dst)
    return G


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
         "[Space] step  [U] undo  [R] redo  [S] save  [M] metagraph"),
        transform=ax.transAxes, fontsize=8, color="gray",
        verticalalignment="bottom")


# ──────────────────────────────────────────────
#  MetaGraph drawing
# ──────────────────────────────────────────────

MOTIF_KIND_COLORS = {
    "entry": "#4CAF50",
    "linear": "#2196F3",
    "merge": "#FF9800",
    "loop": "#F44336",
}


def _draw_metagraph(ax, mg, G, pos, depths, color_mode,
                    selected_step, nav_path):
    ax.clear()

    path_str = " > ".join(["Top"] + nav_path) if nav_path else "Top"
    mode_label = "Motif Kind" if color_mode == "motif_kind" else "Motif Depth"
    ax.set_title(f"[{mode_label}]  {path_str}",
                 fontsize=11, loc="left", color="purple")

    if not mg.motifs:
        ax.text(0.5, 0.5, "No motifs extracted.",
                fontsize=16, ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return

    motif_by_step = {m.step: m for m in mg.motifs}
    loop_steps = [m.step for m in mg.motifs if m.kind == "loop"]
    other_steps = [m.step for m in mg.motifs if m.kind != "loop"]

    labels = {}
    for m in mg.motifs:
        if m.kind == "loop":
            scc = m.meta.get("scc", [])
            scc_str = ",".join(scc[:4])
            if len(scc) > 4:
                scc_str += "..."
            labels[m.step] = f"loop\n{{{scc_str}}}"
        else:
            labels[m.step] = f"{m.kind}\n{m.node}"

    max_depth = max(depths.values()) if depths else 0

    if color_mode == "motif_kind":
        colors_other = [MOTIF_KIND_COLORS.get(motif_by_step[s].kind, "#ccc")
                        for s in other_steps]
        colors_loop = [MOTIF_KIND_COLORS["loop"]] * len(loop_steps)
    else:
        cmap = matplotlib.colormaps["YlOrRd"]
        norm = max(max_depth, 1)
        colors_other = [cmap(depths[s] / norm) for s in other_steps]
        colors_loop = [cmap(depths[s] / norm) for s in loop_steps]

    if other_steps:
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=other_steps,
            node_color=colors_other, node_size=900, node_shape="o")

    if loop_steps:
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=loop_steps,
            node_color=colors_loop, node_size=1200, node_shape="s",
            edgecolors="darkred", linewidths=3)

    if selected_step is not None and selected_step in pos:
        is_loop = motif_by_step[selected_step].kind == "loop"
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=[selected_step],
            node_color="none",
            edgecolors="black", linewidths=4,
            node_size=1200 if is_loop else 900,
            node_shape="s" if is_loop else "o")

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                           arrowsize=15, edge_color="#666666")
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=7, font_weight="bold")

    if color_mode == "motif_kind":
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label=kind,
                   markerfacecolor=color, markersize=10)
            for kind, color in MOTIF_KIND_COLORS.items()
        ]
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=8, framealpha=0.8)

    _draw_metagraph_footer(ax, mg, color_mode, depths)


def _draw_metagraph_footer(ax, mg, mode, depths):
    max_depth = max(depths.values()) if depths else 0
    n_loops = sum(1 for m in mg.motifs if m.kind == "loop")
    ax.text(
        0.01, 0.01,
        (f"Motifs: {len(mg.motifs)}  |  "
         f"Edges: {len(mg.edges)}  |  "
         f"Loops: {n_loops}  |  "
         f"Max depth: {max_depth}  |  "
         "[M] mode  [←→] select  [D] drill  [B] back"),
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

    state = {
        "view_mode": "cfg",
        "selected_step": None,
        "nav_path": [],
    }
    cache = {"mg": None, "G": None, "pos": None, "depths": None}
    nav_stack = []

    def _invalidate():
        cache["mg"] = cache["G"] = cache["pos"] = cache["depths"] = None
        nav_stack.clear()
        state["view_mode"] = "cfg"
        state["selected_step"] = None
        state["nav_path"] = []

    def _ensure_metagraph():
        if cache["mg"] is not None:
            return True
        ops = engine.history
        if not ops:
            return False
        motifs = motif.extract(ops)
        mg = metagraph.build(motifs)
        cache["mg"] = mg
        cache["depths"] = _compute_depths(mg)
        cache["pos"] = _compute_metagraph_layout(mg, cache["depths"])
        cache["G"] = _build_metagraph_nx(mg)
        return True

    def _current_view():
        if nav_stack:
            return nav_stack[-1]
        return cache

    def _redraw():
        if state["view_mode"] == "cfg":
            draw(ax, engine, algo, pos, log_msg)
        else:
            cv = _current_view()
            _draw_metagraph(ax, cv["mg"], cv["G"], cv["pos"],
                            cv["depths"], state["view_mode"],
                            state["selected_step"], state["nav_path"])
        fig.canvas.draw()

    def on_key(event):
        nonlocal log_msg

        key = event.key
        vm = state["view_mode"]

        # ── CFG-only keys ──
        if vm == "cfg":
            if key in (" ", "enter"):
                _invalidate()
                op = algo.step()
                log_msg = _fmt_step(op) if op else "Reduction complete!"

            elif key == "u":
                _invalidate()
                op = algo.undo()
                log_msg = _fmt_undo(op) if op else "Nothing to undo."

            elif key == "r":
                _invalidate()
                op = algo.redo()
                log_msg = _fmt_redo(op) if op else "Nothing to redo."

            elif key == "s":
                path = "reduction_steps.json"
                store.save(engine.history, path)
                log_msg = f"Saved {len(engine.history)} ops → {path}"

            elif key == "m":
                if not algo.is_done:
                    log_msg = "Reduction not complete — finish before viewing MetaGraph."
                elif not _ensure_metagraph():
                    log_msg = "No reduction history available."
                else:
                    state["view_mode"] = "motif_kind"
                    log_msg = "MetaGraph: colored by Motif kind"

            elif key == "q":
                plt.close(fig)
                return

            else:
                return

        # ── MetaGraph keys ──
        else:
            cv = _current_view()
            current_mg = cv["mg"]

            if key == "m":
                if vm == "motif_kind":
                    state["view_mode"] = "motif_depth"
                    log_msg = "MetaGraph: colored by depth"
                else:
                    state["view_mode"] = "motif_kind"
                    log_msg = "MetaGraph: colored by Motif kind"

            elif key in ("left", "right"):
                loop_steps = sorted(
                    m.step for m in current_mg.motifs if m.kind == "loop"
                )
                if not loop_steps:
                    log_msg = "No loop nodes to select."
                else:
                    sel = state["selected_step"]
                    if sel is None or sel not in loop_steps:
                        idx = 0 if key == "right" else len(loop_steps) - 1
                    else:
                        cur = loop_steps.index(sel)
                        if key == "right":
                            idx = (cur + 1) % len(loop_steps)
                        else:
                            idx = (cur - 1) % len(loop_steps)
                    state["selected_step"] = loop_steps[idx]
                    m = next(m for m in current_mg.motifs
                             if m.step == state["selected_step"])
                    scc = m.meta.get("scc", [])
                    log_msg = f"Selected loop {{{','.join(scc)}}}"

            elif key == "d":
                sel = state["selected_step"]
                if sel is None:
                    loop_steps = [m.step for m in current_mg.motifs
                                  if m.kind == "loop"]
                    if loop_steps:
                        state["selected_step"] = loop_steps[0]
                        m = next(m for m in current_mg.motifs
                                 if m.step == loop_steps[0])
                        scc = m.meta.get("scc", [])
                        log_msg = f"Selected loop {{{','.join(scc)}}} — press D again to drill in"
                    else:
                        log_msg = "No loop nodes to drill into."
                else:
                    m = next((m for m in current_mg.motifs
                              if m.step == sel), None)
                    if m is None or m.kind != "loop":
                        log_msg = "Selected node is not a loop."
                    else:
                        sub_mg = current_mg.subgraphs.get(sel)
                        if sub_mg is None or not sub_mg.motifs:
                            log_msg = "No internal structure to show."
                        else:
                            scc = m.meta.get("scc", [])
                            sub_depths = _compute_depths(sub_mg)
                            sub_pos = _compute_metagraph_layout(
                                sub_mg, sub_depths)
                            sub_G = _build_metagraph_nx(sub_mg)
                            nav_stack.append({
                                "mg": sub_mg, "G": sub_G,
                                "pos": sub_pos, "depths": sub_depths,
                            })
                            state["nav_path"].append(
                                f"loop{{{','.join(scc)}}}")
                            state["selected_step"] = None
                            log_msg = (f"Drilled into loop"
                                       f" {{{','.join(scc)}}}")

            elif key in ("b", "escape"):
                if nav_stack:
                    nav_stack.pop()
                    if state["nav_path"]:
                        state["nav_path"].pop()
                    state["selected_step"] = None
                    log_msg = "Returned to parent level."
                else:
                    state["view_mode"] = "cfg"
                    state["selected_step"] = None
                    state["nav_path"] = []
                    log_msg = "Returned to CFG view."

            elif key == "q":
                plt.close(fig)
                return

            else:
                return

        _redraw()
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
