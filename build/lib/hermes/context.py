"""What a system may know besides the question and the graph."""

from dataclasses import dataclass, field


@dataclass
class Example:
    nl: str
    sparql: str


@dataclass
class Context:
    """Extra input for one question.

    `examples` are question/query pairs. `answers` are reference answers when
    an experiment provides them, as in UniQGen's gold-oracle ablation; leave
    it empty in a fair evaluation.
    """

    text: str = ""
    examples: list[Example] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)


def as_context(context):
    if context is None:
        return Context()
    if isinstance(context, Context):
        return context
    raise TypeError("context must be a Context or None.")
