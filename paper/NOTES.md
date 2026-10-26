# Research notes: framing, decisions, and survey

Working notes behind the write-up. The write-up (`hermes.tex`) states what is
done; this file records what is still being decided and the literature the
decisions rest on.

## 1. Framing: NL-to-SPARQL as query compilation

A database compiles a query in fixed phases. For

```sql
SELECT e.name FROM emp e JOIN dept d ON e.dept_id = d.id WHERE d.city = 'Paris';
```

| Phase | Database | Output |
|---|---|---|
| Parse | grammar to syntax tree | names still unresolved |
| Bind | resolve every name and type against the catalog | `π_name(σ_city='Paris'(emp ⋈ dept))` |
| Rewrite | equivalence-preserving rules (push selections, decorrelate) | `π_name(emp ⋈ σ_city='Paris'(dept))` |
| Cost-based search | statistics choose join order and operators | index scan on `dept.city`, hash join |
| Execute | run the plan | rows |

SPARQL engines do the same. Jena ARQ compiles a query to algebra
(`(project (?name) (filter (= ?city "Paris") (bgp ...)))`), applies algebra
transforms, and reorders BGPs by selectivity; RDF-3X, QLever and Blazegraph
order joins from statistics.

A database is trustworthy because meaning is fixed at binding and every later
step chooses among equivalent plans. NL-to-SPARQL has no such fixed point: the
hard step is choosing *which* query, not *how* to run it, and the endpoint
already optimizes whatever we send.

Hermes' mapping, with the ontology as catalog and the KG as a black-box endpoint:

| Database | Hermes |
|---|---|
| Parse | LLM parses the question into an unbound logical form (algebra with phrase leaves) |
| Bind | phrases bound to ontology terms and KG entities; only referenced slots are retrieved |
| Type check | domain/range and subclass checks (`src/hermes/kernel.py`) |
| Rewrite | rewrites that preserve meaning under the ontology (subclass, subproperty, inverse) |
| Cost model | price of each query and probe, and a learned estimate of whether it is correct |
| Plan | SPARQL plus a certificate (span → operator → binding, typing, probe evidence) |

## 2. What the "kernel" is

`kernel.py` is not a process. It is a pure, side-effect-free checker:
`Kernel.check(plan, goal)` returns errors, a typing context Γ and the open
goals, and never touches the graph. "Kernel" is meant in the theorem-prover
sense (the small trusted core every step must pass), not the operating-system
sense. In database terms it is the binder and semantic analyzer. The write-up
already calls it the verifier (Algorithm `Verify`).

## 3. Candidate contributions

**C1. Ontology-guided query optimization (how to run a query).** Use ontology
axioms (domain/range, disjointness, functional properties) for semantic query
optimization: prune joins that are provably empty, drop redundant type
patterns, and order joins from cardinality estimates over an endpoint whose
statistics are hidden. This is engine work. It competes with mature systems
(RDF-3X, QLever, learned estimators such as GNCE), and it would belong inside
an engine (Jena ARQ in Java, QLever in C++), not in a Python wrapper.

**C2. Budgeted NL-to-SPARQL (which query to send).** Under a budget B (LLM
tokens, regenerations, endpoint calls, dollars), choose the next action
(regenerate, refine an operator, rebind a slot, probe with ASK/COUNT, answer,
abstain) to maximize the probability that the final answer is right:

```
max_π  P(answer correct)   s.t.   Σ_t c(a_t) ≤ B
c(a) = λ_tok · tokens(a) + λ_ex · ĉ_exec(a)        (price before sending)
p̂(Q correct | type check, goal status, cardinality match,
              candidate agreement, probe results)   (learned estimate)
```

The policy can be greedy value per cost, or a bandit with Thompson sampling
(as Bao does for hint sets). Report accuracy against budget (anytime curves),
selective accuracy at each coverage level, and dollars or GPU-seconds per
correct answer (`src/hermes/costs.py` already records cost).

**How they connect.** C1's cost model is a component of C2: `ĉ_exec` prices a
probe or candidate before it is sent, the way an optimizer costs a rewrite
before choosing it. *Leaning:* C2 as the main contribution, with query pricing
(C1's estimator, not a new engine) as its enabling component.

## 4. Implementation language

For C2, stay in Python. One LLM call takes seconds and an endpoint call takes
milliseconds to seconds, so the planner's own overhead does not matter. The
engine stays a black box behind SPARQL; rdflib or pyoxigraph serve local
graphs. Move to Java (a Jena ARQ transform, or a `ReorderTransformation` fed by
TDB statistics) or C++ (QLever) only if C1 becomes a contribution in its own
right.

## 5. Decisions

| Date | Decision | Status |
|---|---|---|
| 2026-10-25 | README frames Hermes as query compilation, not "the paper" | done |
| 2026-10-25 | Main contribution: C1 (optimizer) or C2 (budgeted selection) | open, leaning C2 |
| 2026-10-25 | Implementation stays in Python unless C1 is pursued | proposed |
| 2026-10-25 | Rename `kernel.py` to `verifier.py` to match the write-up | proposed |
| 2026-10-25 | Planning order: parse → bind → check → price → select (not trial and error on execution) | proposed |

## 6. Survey

✓ means the venue and authors were checked against the publisher or arXiv on
2026-10-25. Unmarked entries are cited from memory: verify them before they
enter `hermes.bib`. Preprints are not citable in the write-up under the venue
rule, but they matter for novelty.

### A. Classical query compilation

- **Selinger et al., SIGMOD 1979.** System R access-path selection: dynamic programming over join orders with a cost model. The template for "price, then choose". (in `hermes.bib`)
- **Graefe & McKenna, ICDE 1993 (Volcano); Graefe, IEEE Data Eng. Bull. 1995 (Cascades).** Rule-based search over equivalent plans with memoization. The model for a search space of rewrites.
- **Leis et al., PVLDB 2015, "How Good Are Query Optimizers, Really?"** Cardinality errors, not the cost model, dominate plan quality. For C2, `p̂` and `ĉ_exec` are cardinality problems in disguise. (in `hermes.bib`)
- **King, VLDB 1981; Chakravarthy, Grant & Minker, TODS 1990.** Semantic query optimization: use integrity constraints to rewrite or prune queries. This is C1 with ontology axioms as the constraints.

### B. SPARQL algebra and engines

- **Pérez, Arenas & Gutierrez, TODS 2009.** Compositional semantics of SPARQL and its complexity. (in `hermes.bib`)
- **Schmidt, Meier & Lausen, ICDT 2010.** Algebraic equivalences for SPARQL: which rewrites are sound, especially around OPTIONAL.
- **Stocker et al., WWW 2008.** BGP reordering by selectivity estimation. This is what Jena ARQ does.
- **Neumann & Weikum, PVLDB 2008 (RDF-3X); Neumann & Moerkotte, ICDE 2011 (characteristic sets).** Statistics-driven join ordering for RDF and estimates for star queries.
- **Bast & Buchhold, CIKM 2017 (QLever).** A C++ engine. Its group also wrote GRASP (below).
- ✓ **Verborgh et al., J. Web Semantics 37–38, 2016, Triple Pattern Fragments.** Clients evaluate SPARQL over a cheap server interface. The reference for treating the KG as a black box and pricing what the client asks of it. [link](https://ruben.verborgh.org/publications/verborgh_jws_2016/)

### C. Learned cost and cardinality

- **Kipf et al., CIDR 2019 (MSCN); Marcus et al., PVLDB 2019 (Neo); Yang et al., SIGMOD 2022 (Balsa).** Learned cardinality estimation and learned optimizers.
- ✓ **Marcus et al., SIGMOD 2021, Bao.** Steers an existing optimizer with hints, chosen by Thompson sampling over a learned cost. The closest template for a C2 policy that learns which candidate to try. [link](https://people.csail.mit.edu/tatbul/publications/bao_sigmod21.pdf)
- ✓ **Davitkova, Gjurovski & Michel, EDBT 2022, LMKG.** Learned cardinality for KG queries, for star and path shapes. [doi](https://doi.org/10.48786/edbt.2022.07)
- ✓ **Schwabe & Acosta, SIGMOD 2024 (PACMMOD 2(1):44), GNCE.** GNN cardinality estimation over KGs with entity embeddings, inductive to unseen entities, under 1 ms per estimate. A candidate for `ĉ_exec`. [arXiv](https://arxiv.org/abs/2303.01140)
- ✓ **Acosta, Qin & Schwabe, arXiv 2411.14277 (book chapter).** Neuro-symbolic query optimization for KGs. A good map of C1. [arXiv](https://arxiv.org/abs/2411.14277)

### D. Ontology-based data access

- **Calvanese et al., J. Automated Reasoning 2007 (DL-Lite).** Query rewriting under ontology axioms with tractable guarantees. The formal basis for "rewrites that preserve meaning under the ontology".
- ✓ **Calvanese et al., Semantic Web Journal 8(3), 2017, Ontop.** OBDA in practice: SPARQL rewritten through OWL 2 QL and mappings. [link](https://www.inf.unibz.it/~calvanese/papers-html/SWJ-2017.html)

### E. KBQA: from parsing to binding

- **Yih et al., ACL 2015 (STAGG).** Staged query-graph generation. Early "build the query in steps".
- ✓ **Li et al., ACL 2023, KB-BINDER.** The LLM drafts a logical form and BM25 binds it to the KB. **Closest prior to "parse, then bind"**; Hermes must add typing, certificates and pricing. [ACL](https://aclanthology.org/2023.acl-long.385)
- ✓ **Gu, Deng & Su, ACL 2023, Pangu (Outstanding Paper).** A symbolic agent proposes valid plans and the LM only scores them. **Closest prior to "the LLM chooses among checked plans".** [ACL](https://aclanthology.org/2023.acl-long.270)
- ✓ **Xiong, Bao & Zhao, ACL 2024, Interactive-KBQA.** Multi-turn tool use over the KB. Its own limitations section notes the cost grows with turns, which motivates C2. [ACL](https://aclanthology.org/2024.acl-long.569)
- ✓ **Liu et al., Findings of EMNLP 2024, SPINACH.** A Wikidata agent that grows queries from fragments, using empty results as feedback. The trial-and-error pattern Hermes wants to replace with pricing. [arXiv](https://arxiv.org/abs/2407.11417)
- ✓ **Walter & Bast, ISWC 2025, GRASP.** Generic SPARQL generation across KGs without fine-tuning (venue taken from a citing reference list; confirm in LNCS 16140).
- **ARUQULA, SPARQL-LLM, UniQGen.** Already in `hermes.bib`; Text2SPARQL peers.

### F. Text-to-SQL: candidates, selection, cost

- ✓ **Liu et al., arXiv 2408.05109.** Survey of NL2SQL with LLMs: the best single overview of the pipeline (schema linking, generation, correction, selection). [arXiv](https://arxiv.org/abs/2408.05109)
- **Pourreza & Rafiei, NeurIPS 2023 (DIN-SQL).** Decomposed generation with self-correction.
- ✓ **Pourreza et al., ICLR 2025, CHASE-SQL.** Several generators and a trained pairwise selector. The state of the art on selection, but with no budget. [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/974ff7b5bf08dbf9400b5d599a39c77f-Abstract-Conference.html)
- ✓ **Huang et al., PVLDB 18, ThrifLLM.** Choosing LLMs under a cost budget to maximize correctness, shown NP-hard-like. A formal template for C2's budget constraint. [PVLDB](https://www.vldb.org/pvldb/vol18/p4410-huang.pdf)
- ✓ **Peng et al., arXiv 2608.02876 (2026), BAP-SQL.** Budget-aware observation planning: estimate query risk, rewrite, cap with a runtime shield. **Closest prior to C2**; it controls observation cost, not which candidate to try, and its gains shrink with larger budgets. [arXiv](https://arxiv.org/abs/2608.02876)
- ✓ **arXiv 2505.14174; EllieSQL arXiv 2503.22402.** Confidence-aware selection that cuts LLM calls by about 60%, and routing by question complexity. Preprints, but they set the baselines C2 must beat.

### G. Budgeted inference in general

- **Wang et al., ICLR 2023 (self-consistency); Ni et al., ICML 2023 (LEVER).** Majority voting, and learning to verify programs from their execution results.
- **Chen, Zaharia & Zou, FrugalGPT (TMLR).** LLM cascades under a cost budget.
- ✓ **Snell et al., arXiv 2408.03314.** Compute-optimal test-time scaling: spend per question according to its difficulty (venue unconfirmed). [arXiv](https://arxiv.org/abs/2408.03314)

### Reading order

1. Leis 2015, then Bao: why estimates matter, and how a learned policy can steer a system it doesn't own.
2. Pangu and KB-BINDER: what "checked plans" and "draft then bind" already cover.
3. CHASE-SQL, ThrifLLM and BAP-SQL: selection and budgets in text-to-SQL. C2 must state its difference from these three.
4. GNCE: a ready-made `ĉ_exec`.
5. DL-Lite and Ontop: if the ontology-rewrite angle is kept.

### Where the gap appears to be

No work found so far combines (i) candidates checked against an ontology,
(ii) a price for each LLM call and endpoint probe before it is spent, and
(iii) a learned estimate of correctness that decides the next action or
abstains, evaluated as accuracy against budget on Text2SPARQL. BAP-SQL
(budget, SQL, observation only) and Pangu (checked plans, no budget) are
closest. This needs another search of ISWC/ESWC 2026 and PVLDB 19 before it is
claimed.
