# Architecture docs

Read these in order:

1. [00-overview.md](./00-overview.md) — top-level architecture and data flow
2. [01-language.md](./01-language.md) — the DAG IR, types, evaluation
3. [02-neural.md](./02-neural.md) — embedding network, heads, caching
4. [03-search.md](./03-search.md) — search algorithm and frontier
5. [04-library.md](./04-library.md) — compression / abstraction sleep
6. [05-tasks.md](./05-tasks.md) — task families, including ARC
7. [06-training.md](./06-training.md) — wake/sleep loop
8. [07-testing.md](./07-testing.md) — testing strategy
9. [08-roadmap.md](./08-roadmap.md) — milestones and risks
10. [09-questions.md](./09-questions.md) — design choices to confirm + open questions

If you only read two: [00-overview.md](./00-overview.md) and
[09-questions.md](./09-questions.md).

## Per-milestone decisions

Implementation logs from each milestone live in
[`decisions/`](./decisions/). They record the calls made while shipping
that milestone — open the relevant file when picking up where the
previous milestone left off.

- [decisions/m1-lang.md](./decisions/m1-lang.md) — `lang` crate.
- [decisions/m2-search-tasks.md](./decisions/m2-search-tasks.md) — `tasks` and `search`.
- [decisions/m2-strip-static-types.md](./decisions/m2-strip-static-types.md) — mid-M2 deletion of the static type system.
