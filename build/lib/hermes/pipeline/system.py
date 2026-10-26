"""The pipeline as a registered system, for benchmark runs and the challenge API."""

import threading
import weakref

from hermes.pipeline import Hermes
from hermes.pipeline.fewshot import store_path
from hermes.pricing.budget import Budget
from hermes.systems import register


@register("hermes")
class HermesSystem:
    """Budget fields (llm_calls, tokens, usd, queries, rows) cap every question's run."""

    def __init__(self, llm=None, llm_calls=3, tokens=None, usd=None, queries=None, rows=None,
                 fewshots="auto", examples=2, retriever="lexical", executor="plan",
                 optimize=True, max_repairs=2, revisions=1):
        self.llm = llm
        self.budget = Budget(llm_calls, tokens, usd, queries, rows)
        self.options = {"examples": examples, "retriever": retriever, "executor": executor,
                        "optimize": optimize, "max_repairs": max_repairs, "revisions": revisions}
        self.fewshots = fewshots
        self._pipelines = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    def pipeline(self, kg):
        with self._lock:
            hermes = self._pipelines.get(kg)
            if hermes is None:
                fewshots = store_path(kg.name) if self.fewshots == "auto" else self.fewshots
                hermes = self._pipelines[kg] = Hermes(kg, self.llm, self.budget, fewshots,
                                                      **self.options)
            return hermes

    def __call__(self, question, kg, context=None):
        answer = self.pipeline(kg).ask(question)
        llm = self.pipeline(kg).llm
        return {
            # A run that failed or went over budget gives no query, so it is not scored.
            "sparql": answer.sparql if answer.ok else "",
            "draft_sparql": answer.sparql,
            "error_reason": answer.error,
            "repairs": answer.repairs,
            "problems": answer.problems,
            "rewrites": [f"{r.rule}: {r.text}" for r in answer.plan.rewrites] if answer.plan else [],
            "examples": [e.question for e in answer.examples],
            "entities": [e.label for e in answer.slice.entities] if answer.slice else [],
            "spend": answer.spend.to_dict(),
            "model": llm.model,
            "provider": llm.provider,
        }
