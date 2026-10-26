"""Rewrite rules: each one a command that rewrites a plan and says what it did.

A rule takes the plan's root and a `Context` (statistics, estimator, the
query, and a log) and returns the new root. Rules keep the query's answers:
they only move selections, reorder joins, and narrow what scans fetch.

- `PushSelections` moves each σ down to the lowest operator that binds all of
  its variables in every solution, through joins, left joins (their left
  side), unions, δ, τ, ε and π.
- `OrderJoins` reorders each join group, its operands and the selections
  between them, by dynamic programming over left-deep orders that avoid
  cartesian products, minimizing estimated C_out (Selinger et al., SIGMOD
  1979); each selection goes right after the operand that binds its last
  variable. Groups with more than `dp_limit` operands are ordered greedily.
- `PruneScans` pushes projections into scans: a scan fetches only the
  variables used above it, never fewer than one so that multiplicities stay.
"""

import itertools
from dataclasses import dataclass, field

from hermes.planner.algebra import (
    Join, Op, Scan, certain, copy_part, expression_variables, field as part_field, has_pattern,
    variables,
)
from hermes.planner.estimate import Estimate, Estimator, access, cap, join, selectivity
from hermes.planner.render import describe, expression


@dataclass
class Context:
    statistics: object
    query: object
    log: list = field(default_factory=list)

    @property
    def namespaces(self):
        return self.query.namespaces

    def estimator(self):
        return Estimator(self.statistics)


class Rule:
    name = ""

    def apply(self, root, context):
        raise NotImplementedError


class PushSelections(Rule):
    name = "push selections"

    def apply(self, root, context):
        return self._push(root, context)

    def _push(self, node, context):
        if isinstance(node, Scan):
            return node
        if isinstance(node, Join):
            return Join(self._push(node.left, context), self._push(node.right, context))
        node.children = [self._push(c, context) for c in node.children]
        if node.kind != "Filter" or has_pattern(node.part.expr):
            return node
        wanted = expression_variables(node.part.expr)
        moved = self._sink(node.children[0], node, wanted)
        if moved is None:
            return node
        names = sorted("?" + v for v in wanted)
        context.log.append((self.name, f"σ {expression(node.part.expr, context.namespaces)} "
                                       f"moves down to where {', '.join(names)} "
                                       f"{'is' if len(names) == 1 else 'are'} bound"))
        return moved

    def _sink(self, child, select, wanted):
        """The child with the selection placed inside it, or None if it cannot go lower."""
        if isinstance(child, Join):
            for side in ("left", "right"):
                branch = getattr(child, side)
                if wanted <= certain(branch):
                    setattr(child, side, self._sink(branch, select, wanted)
                            or selection(select, branch))
                    return child
            return None
        if not isinstance(child, Op):
            return None
        kind = child.kind
        if kind == "LeftJoin" and wanted <= certain(child.children[0]):
            child.children[0] = self._sink(child.children[0], select, wanted) \
                or selection(select, child.children[0])
            return child
        if kind == "Union":
            child.children = [self._sink(c, select, wanted) or selection(select, c)
                              for c in child.children]
            return child
        if (kind in ("Filter", "Distinct", "Reduced", "OrderBy")
                or (kind == "Extend" and child.part.var not in wanted)
                or (kind == "Project" and wanted <= set(child.part.PV))):
            child.children[0] = self._sink(child.children[0], select, wanted) \
                or selection(select, child.children[0])
            return child
        return None


class OrderJoins(Rule):
    name = "order joins"

    def __init__(self, dp_limit=10):
        self.dp_limit = dp_limit

    def apply(self, root, context):
        return self._order(root, context)

    def _order(self, node, context):
        if isinstance(node, Scan):
            return node
        if isinstance(node, Op):
            node.children = [self._order(c, context) for c in node.children]
            if node.kind != "Filter" or not isinstance(node.children[0], Join):
                return node
        operands, selections = [], []
        _flatten(node, operands, selections)
        if len(operands) < 2:
            return node
        estimate = context.estimator()
        before = estimate(node).cost
        order = (self._dynamic if len(operands) <= self.dp_limit else self._greedy)(
            operands, selections, estimate)
        tree = _build(order, operands, selections)
        after = context.estimator()(tree).cost
        if after >= before - 1e-9:
            return node
        names = ", ".join(describe(operands[i], context.namespaces) for i in order)
        context.log.append((self.name, f"⋈ in the order {names}: estimated C_out "
                                       f"{before:,.0f} → {after:,.0f}"))
        return tree

    def _dynamic(self, operands, selections, estimate):
        leaves = [estimate(o) for o in operands]
        bound = [certain(o) for o in operands]
        best = {}
        for i, leaf in enumerate(leaves):
            rows, distinct = _with_selections(leaf.rows, leaf.distinct, bound[i], selections, set())
            best[frozenset([i])] = (leaf.rows, rows, distinct, (i,), bound[i])
        for size in range(2, len(operands) + 1):
            for subset in itertools.combinations(range(len(operands)), size):
                options = []
                for j in subset:
                    rest = frozenset(subset) - {j}
                    if rest not in best:
                        continue
                    cost, rows, distinct, order, vars_ = best[rest]
                    connected = bool(vars_ & set(leaves[j].distinct))
                    prefix = Estimate(rows, 0, distinct)
                    fetched = access(prefix, leaves[j], operands[j])
                    joined, merged = join(prefix, leaves[j])
                    applied = {k for k, (wanted, _) in enumerate(selections) if wanted <= vars_}
                    joined, merged = _with_selections(joined, merged, vars_ | bound[j],
                                                      selections, applied)
                    options.append((not connected, cost + fetched + joined, joined,
                                    merged, order + (j,), vars_ | bound[j]))
                if options:
                    _cartesian, *value = min(options, key=lambda o: (o[0], o[1]))
                    best[frozenset(subset)] = tuple(value)
        return list(best[frozenset(range(len(operands)))][3])

    def _greedy(self, operands, selections, estimate):
        leaves = [estimate(o) for o in operands]
        remaining = set(range(len(operands)))
        first = min(remaining, key=lambda i: leaves[i].rows)
        order, vars_ = [first], set(certain(operands[first]))
        remaining.discard(first)
        while remaining:
            connected = [i for i in remaining if vars_ & set(leaves[i].distinct)] or list(remaining)
            nxt = min(connected, key=lambda i: leaves[i].rows)
            order.append(nxt)
            vars_ |= certain(operands[nxt])
            remaining.discard(nxt)
        return order


class PruneScans(Rule):
    name = "prune scans"

    def apply(self, root, context):
        query = context.query
        needed = set(query.variables) if query.form == "SELECT" else set()
        self._prune(root, needed or certain(root), context)
        return root

    def _prune(self, node, needed, context):
        if isinstance(node, Scan):
            keep = node.bound & needed
            if not keep and node.bound:  # a scan still binds something, to keep multiplicity
                keep = {sorted(node.bound, key=str)[0]}
            if keep != node.bound:
                node.keep = frozenset(keep)
                dropped = ", ".join(sorted("?" + v for v in node.bound - keep))
                context.log.append((self.name, f"π {describe(node, context.namespaces)} fetches "
                                               f"only {', '.join(sorted('?' + v for v in keep))}"
                                               f" (not {dropped})"))
            return
        if isinstance(node, Join):
            shared = variables(node.left) & variables(node.right)
            self._prune(node.left, needed | shared, context)
            self._prune(node.right, needed | shared, context)
            return
        kind, part = node.kind, node.part
        if kind == "Project":
            return self._prune(node.children[0], set(part.PV), context)
        if kind in ("Filter", "OrderBy"):
            return self._prune(node.children[0], needed | expression_variables(part.expr), context)
        if kind == "Extend":
            return self._prune(node.children[0],
                               (needed - {part.var}) | expression_variables(part.expr), context)
        if kind == "Slice":
            return self._prune(node.children[0], needed, context)
        if kind == "LeftJoin":
            shared = variables(node.children[0]) & variables(node.children[1])
            extra = expression_variables(part_field(part, "expr"))
            for child in node.children:
                self._prune(child, needed | shared | extra, context)
            return
        if kind == "Union":
            for child in node.children:
                self._prune(child, needed, context)
            return
        for child in node.children:  # anything else needs every variable below it
            self._prune(child, variables(child), context)


DEFAULT_RULES = (PushSelections(), OrderJoins(), PruneScans())


def selection(select, child):
    """The selection `select` applied to another child."""
    return Op("Filter", copy_part(select.part), [child], ("p",))


def _flatten(node, operands, selections):
    """Operands and selections of a join group: joins, with selections between them."""
    if isinstance(node, Join):
        _flatten(node.left, operands, selections)
        _flatten(node.right, operands, selections)
    elif (isinstance(node, Op) and node.kind == "Filter" and not has_pattern(node.part.expr)
          and _in_group(node.children[0])):
        selections.append((expression_variables(node.part.expr), node))
        _flatten(node.children[0], operands, selections)
    else:
        operands.append(node)


def _in_group(node):
    return isinstance(node, Join) or (isinstance(node, Op) and node.kind == "Filter"
                                      and _in_group(node.children[0]))


def _with_selections(rows, distinct, bound, selections, applied):
    for index, (wanted, select) in enumerate(selections):
        if index not in applied and wanted <= bound:
            rows *= selectivity(select.part.expr)
    return rows, cap(distinct, rows)


def _build(order, operands, selections):
    placed, tree, bound = set(), None, set()
    for i in order:
        tree = operands[i] if tree is None else Join(tree, operands[i])
        bound |= certain(operands[i])
        for index, (wanted, select) in enumerate(selections):
            if index not in placed and wanted <= bound:
                tree = selection(select, tree)
                placed.add(index)
    for index, (_wanted, select) in enumerate(selections):
        if index not in placed:
            tree = selection(select, tree)
    return tree
