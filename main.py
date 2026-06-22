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
              edge_prob: float = 0.18,
              seed: int = 42) -> list[str]:
    """
    Populate *engine* with a forward-biased directed graph.
    Returns the list of node ids in topological order.
    """
    rng = random.Random(seed)
    ids = [f"N{i:02d}" for i in range(num_nodes)]

    for nid in ids:
        engine.add_node(nid)          # node_type="basic", weight=1

    for i, u in enumerate(ids):
        for j, v in enumerate(ids):
            if i == j:
                continue
            if i < j:                  # forward edge — normal branch
                if rng.random() < edge_prob:
                    engine.add_edge(u, v)
            else:                      # back edge — loop (rare)
                if rng.random() < edge_prob * 0.15:
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
