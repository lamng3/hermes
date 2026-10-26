# Hermes: the ideas

UniQGen's idea is **Chase & Backchase**: guess the answers first, then remove
constraints until the query covers those answers (Chase) and add them back
until it returns only those answers (Backchase). The answers it guesses are
the reference that steers the search.

**Hermes' idea is to compile the question like a database compiles SQL.** The
model only *parses*. Everything after that is symbolic and checked before it
costs anything: *bind* the words to the ontology, *type-check* the result,
*price* it, and only then *spend* a budget running the candidates most likely
to be right. Nothing needs a reference answer.

```
question ──parse (LLM, 1 call)──> plan with [phrases]
         ──bind──> candidate plans ──check──> well-typed ones
         ──price──> estimated cost ──select──> run the best value for price
         ──accept, or re-parse with what was learned (within the budget)
```

## One question, end to end

*"Which department does Ms. Brant belong to?"* (CK26, Llama 3.1 8B, a real run)

1. **Parse.** The model writes `?e [name] "Ms. Brant" . ?e [memberOf] ?d`,
   with the goal "one `?d` of type `[pv:Department]`". It never has to know an IRI.
2. **Bind.** `[name]` → `pv:name` or `foaf:name`; `[memberOf]` →
   `pv:memberOf` (or read backwards); `"Ms. Brant"` → the literal, or the
   entity it names (Karen Brant).
3. **Check.** The verifier rejects 8 candidates as ill-typed, for example an
   entity where `pv:name` holds only strings, and keeps 8. The rejected ones
   never run.
4. **Rewrite.** The label triple `?e pv:name "Ms. Brant"` collapses into the
   entity itself: `empl-Karen.Brant pv:memberOf ?d`.
5. **Select.** The best candidate runs, returns exactly one department as the
   goal asked, and passes the confidence threshold (0.81). The answer is
   correct, after one model call and one query.

The algebra agent needs about 15 model calls per question to get there by
trial and error.

## The decisions, problem by problem

| Problem | What we do | Where the idea comes from |
|---|---|---|
| The LLM doesn't know the graph's IRIs and invents them | It writes **phrases in brackets**; binding resolves them. A guessed IRI the graph lacks is turned back into a phrase | The SQL binder resolving names against the catalog |
| A phrase could mean several terms | **Position decides the kind**: predicate slot → property, after `a` → class, elsewhere → entity or value. Candidates are ranked by label and local-name similarity, and joint assignments are enumerated best-first | Name resolution; KB-BINDER's draft-then-bind |
| Models write names as strings (`?x name "Ms. Brant"`) | **Label collapse**: replace the variable with the entity that label names, and drop the triple | Query rewriting |
| Models read relations backwards ("X's manager") | Each property is also tried **inverted**, at a lower score | — |
| Many candidates are nonsense | The **verifier type-checks** before running: domain/range, literal vs entity, unbound variables, grouping | Semantic analysis in a compiler |
| Ontologies leave ranges out | **Read the range from the data**: if every object of `foaf:name` is a literal, it is literal-valued | Statistics as part of the catalog |
| The answer's type is stated but not enforced | **Close the goal by construction**: add `?d a pv:Department` when the type goal is still open | — |
| Some queries explode or time out | **Price before running**: System R estimates of rows and C_out, with seconds fitted online. Median q-error 1.05 on CK26 | Selinger et al., SIGMOD 1979 |
| Which candidate do we run next? | Run the one with the highest **p(correct) − λ·price**; running it updates p; stop once p passes a threshold | Cost-based plan choice; Bao's learned steering |
| A bad first reading eats the whole budget | Each reading gets an **even share of the executions**; then **re-parse**, shown what the earlier readings returned | Budgeted test-time compute |
| Wrong answers look like right ones | Return **p as a confidence** and abstain below a threshold | Selective prediction |

## What is ours, and what isn't

- **Borrowed:** draft-then-bind (KB-BINDER), checked plans scored by an LM
  (Pangu), System R estimation, and candidate selection (CHASE-SQL).
- **Ours:**
  - typing candidates against the ontology, with ranges read from the data
    where the ontology is silent;
  - pricing every candidate before it is sent;
  - choosing what to run next by value net of price, under a hard budget of
    model calls and executions, with abstention.

  As far as we found, no prior work does all of this for SPARQL.

## What the numbers say so far (Llama 3.1 8B)

- **CK26, held-out questions 11–50:** 0.089 F1, against 0.081 for the algebra
  agent, with 2.15 model calls per question instead of 14.5. That is a small
  gain on 80 items.
- **Selection already works.** The chosen answers score 0.111, and the best
  candidate ever generated would score 0.113. When the planner is confident,
  it is right far more often (F1 0.41 on the 27% it answers).
- **Generation is the bottleneck.** On 40 of 100 CK26 questions nothing binds
  to a well-typed plan. On DB26, the model's readings are wrong, so it scores
  0.007.

So the next improvements belong in **parse and bind**, not in selection.
