"""The systems Hermes is compared with, behind the same system interface.

Importing this package registers them: training-free (one prompted draft),
uniqgen (UniQGen: a constraint table refined with Chase and Backchase),
algebra-agent (an LLM applies one algebra operator per step), prover (the
algebra agent with the verifier) and budgeted (parse, bind, check, price,
select under a budget).
"""

from hermes.baselines import algebra_agent, budgeted, prover, training_free, uniqgen  # noqa: F401
