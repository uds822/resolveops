
## Graphify — Token-Efficient Rules

- Use `graphify query "<question>"` for relevant codebase exploration when graphify-out/graph.json exists.
- Reuse information already retrieved in the current task.
- Use `graphify path` or `graphify explain` only when relationships or concepts need further investigation.
- Never read the entire graph.json.
- Read GRAPH_REPORT.md only for broad architecture analysis when smaller queries are insufficient.
- Read source files selectively, based on the task.
- Pass Graphify instructions to subagents only when they explore the codebase.
- Avoid redundant subagent exploration.
- Run `graphify update .` once after completing a batch of code changes, not after every individual edit.
- Avoid displaying full generated graph diffs unless explicitly requested.
- Exclude virtual environments, caches, generated outputs and large PDFs from broad searches.
- Keep terminal output concise.
