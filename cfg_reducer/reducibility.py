"""Reducibility by recursively removing single-entry SCC headers."""

from .algorithm import tarjan_scc


def is_reducible(
    nodes: tuple[str, ...], edges: tuple[tuple[str, str], ...], *, entry: str,
) -> bool:
    vertices = set(nodes)
    if not vertices or entry not in vertices:
        raise ValueError("CFG must be nonempty and contain entry")
    pred: dict[str, set[str]] = {v: set() for v in sorted(vertices)}
    succ: dict[str, set[str]] = {v: set() for v in sorted(vertices)}
    for u, v in edges:
        if u not in vertices or v not in vertices:
            raise ValueError("unknown edge endpoint")
        pred[v].add(u)
        succ[u].add(v)
    reached: set[str] = set()
    pending = [entry]
    while pending:
        v = pending.pop()
        if v not in reached:
            reached.add(v)
            pending.extend(sorted(succ[v] - reached))
    if reached != vertices:
        raise ValueError("all CFG nodes must be reachable from entry")

    stack = [vertices]
    while stack:
        region = stack.pop()
        components = sorted(tuple(sorted(c)) for c in tarjan_scc(region, succ.__getitem__))
        for component in components:
            if len(component) == 1 and component[0] not in succ[component[0]]:
                continue
            members = set(component)
            # Keep the original predecessors, including removed headers.
            entries = [v for v in component if v == entry or pred[v] - members]
            if len(entries) != 1:
                return False
            remainder = members - {entries[0]}
            if remainder:
                stack.append(remainder)
    return True
