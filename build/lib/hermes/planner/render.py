"""Reading plans: expressions as SPARQL-like text, a plan as a tree or a formula."""

from rdflib import BNode, Literal, URIRef, Variable
from rdflib.paths import Path
from rdflib.plugins.sparql.parserutils import CompValue

from hermes.planner.algebra import Join, Scan, children, field

SYMBOL = {
    "Join": "⋈", "LeftJoin": "⟕", "Filter": "σ", "Project": "π", "Distinct": "δ",
    "Reduced": "δ", "Union": "∪", "Minus": "−", "Extend": "ε", "OrderBy": "τ", "Slice": "λ",
    "Group": "γ", "AggregateJoin": "γ", "values": "values", "ToMultiSet": "multiset",
}
_ARGUMENTS = ("arg", "arg1", "arg2", "arg3", "vars")


def expression(expr, namespaces=None):
    """SPARQL-like text of an rdflib term or expression."""
    show = lambda e: expression(e, namespaces)  # noqa: E731
    if isinstance(expr, Variable):
        return f"?{expr}"
    if isinstance(expr, (URIRef, Literal, BNode, Path)):
        return expr.n3(namespaces) if namespaces is not None else expr.n3()
    if isinstance(expr, (list, tuple)):
        return ", ".join(show(e) for e in expr)
    if not isinstance(expr, CompValue):
        return str(expr)
    name = expr.name
    if name == "RelationalExpression":
        other = expr.other
        right = f"({show(other)})" if isinstance(other, list) else show(other)
        return f"{show(expr.expr)} {expr.op} {right}"
    if name in ("ConditionalAndExpression", "ConditionalOrExpression"):
        op = " && " if name.startswith("ConditionalAnd") else " || "
        return op.join(show(e) for e in [expr.expr, *(field(expr, "other") or [])])
    if name in ("AdditiveExpression", "MultiplicativeExpression"):
        text = show(expr.expr)
        for op, other in zip(field(expr, "op") or [], field(expr, "other") or []):
            text += f" {op} {show(other)}"
        return text
    if name == "UnaryNot":
        return f"!{show(expr.expr)}"
    if name == "UnaryMinus":
        return f"-{show(expr.expr)}"
    if name == "OrderCondition":
        order = field(expr, "order")
        return f"{order}({show(expr.expr)})" if order else show(expr.expr)
    if name.startswith(("Builtin_", "Aggregate_")):
        args = [expr[k] for k in _ARGUMENTS if k in expr and expr[k] is not None]
        distinct = "DISTINCT " if field(expr, "distinct") else ""
        return f"{name.split('_', 1)[1]}({distinct}{', '.join(show(a) for a in args)})"
    if name == "Function":
        return f"{show(expr.iri)}({show(field(expr, 'expr') or [])})"
    if name == "TrueFilter":
        return "true"
    return name


def describe(node, namespaces=None):
    """A short name for an operand: a scan's pattern, or its operator."""
    if isinstance(node, Scan):
        return "(" + " ".join(expression(t, namespaces) for t in node.terms) + ")"
    if isinstance(node, Join):
        return "⋈"
    if node.kind == "Filter" and node.children:
        return "σ" + describe(node.children[0], namespaces)
    return SYMBOL.get(node.kind, node.kind)


def label(node, namespaces=None):
    """One line for a node: its operator and arguments."""
    if isinstance(node, Scan):
        pruned = "" if node.keep is None else \
            "  π " + " ".join("?" + v for v in sorted(node.keep, key=str))
        return "scan " + describe(node, namespaces) + pruned
    if isinstance(node, Join):
        return "⋈"
    part, kind = node.part, node.kind
    if kind == "Filter":
        return "σ " + expression(part.expr, namespaces)
    if kind == "Project":
        return "π " + " ".join(expression(v) for v in part.PV)
    if kind == "Extend":
        return f"ε ?{part.var} ← {expression(part.expr, namespaces)}"
    if kind == "OrderBy":
        return "τ " + expression(part.expr, namespaces)
    if kind == "Slice":
        length = field(part, "length")
        return f"λ offset {field(part, 'start') or 0}" + \
            (f" limit {length}" if length is not None else "")
    if kind == "LeftJoin":
        expr = field(part, "expr")
        shown = "" if expr is None or getattr(expr, "name", "") == "TrueFilter" \
            else " " + expression(expr, namespaces)
        return "⟕" + shown
    if kind == "Group":
        by = field(part, "expr") or []
        return "γ by " + (expression(by, namespaces) if by else "()")
    if kind == "AggregateJoin":
        return "γ " + ", ".join(f"?{a.res} ← {expression(a, namespaces)}" for a in part.A)
    if kind == "values":
        return f"values ({len(part.res)} rows)"
    return SYMBOL.get(kind, kind)


def tree(node, namespaces=None, notes=None):
    """The plan as an indented tree; `notes(node)` adds text after each node's line."""
    lines = [label(node, namespaces) + (notes(node) if notes else "")]
    kids = children(node)
    for index, child in enumerate(kids):
        last = index == len(kids) - 1
        sub = tree(child, namespaces, notes).splitlines()
        lines.append(("└─ " if last else "├─ ") + sub[0])
        lines.extend(("   " if last else "│  ") + line for line in sub[1:])
    return "\n".join(lines)


def formula(node, namespaces=None):
    """The plan as one relational-algebra expression."""
    if isinstance(node, Scan):
        return describe(node, namespaces)
    if isinstance(node, Join):
        return f"({formula(node.left, namespaces)} ⋈ {formula(node.right, namespaces)})"
    kids = [formula(c, namespaces) for c in node.children]
    if node.kind in ("LeftJoin", "Union", "Minus"):
        return f"({kids[0]} {SYMBOL[node.kind]} {kids[1]})"
    head = label(node, namespaces)
    symbol, _, rest = head.partition(" ")
    sub = f"_{{{rest}}}" if rest else ""
    return f"{symbol}{sub}({kids[0]})" if kids else head
