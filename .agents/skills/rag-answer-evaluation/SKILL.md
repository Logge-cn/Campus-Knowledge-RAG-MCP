---
name: rag-answer-evaluation
description: Run and repeat this repository's fixed RAG answer-level evaluation with one fresh Codex subagent per question and the local MCP server, validate run artifacts, organize human review, and generate the reviewed report. Use when asked to execute or rerun the first-priority answer evaluation, inspect answer predictions, review citations or refusals, or calculate final answer-level metrics. Do not use for dataset changes, retrieval tuning, ingestion, index rebuilding, or generic RAG experiments.
---

# RAG Answer Evaluation

Run the repository's isolated-subagent evaluation workflow. Use the skill for orchestration and the checked-in runner for deterministic task preparation, validation, aggregation, and reporting.

## Preserve the evaluation boundary

- Keep `gpt-5.6-sol`, Prompt version `answer-eval-v1`, tool-description version `answer-eval-v1`, and retrieval limit 5 unchanged.
- Treat `data/sources/南京邮电大学学生手册（2023版）+.pdf` and `data/sources/奖学金细则.pdf` as approved public inputs. When the user directly asks to run the evaluation and the dataset references only these inputs, give a brief outbound-data notice and proceed without a separate blocking authorization question.
- Stop and request confirmation if the dataset introduces another source or the user identifies any input as non-public.
- Send only one fixed question and its MCP-returned evidence through each subagent. Do not upload whole PDFs, model directories, the index, gold answers, or another case's context.
- Do not modify the dataset, Prompt, output schema, runner, evaluator, MCP description, or retrieval configuration during a run.
- Do not start retrieval tuning or other optimization after seeing failures unless the user separately requests it.
- Preserve existing worktree changes and never clean or overwrite unrelated files.
- Never label model-generated review judgments as human review.

## Select the workflow

- For a new or repeated run, perform Preflight, Prepare, Run isolated subagents, Finalize, Validate, then Prepare human review.
- For an existing `predictions.json`, skip execution and start at Validate.
- For a final reviewed report, require completed human review fields before running the evaluator.

## Preflight

1. Resolve the repository root with `git rev-parse --show-toplevel` and run all commands there.
2. Confirm `evaluation/answer/dataset.json`, `evaluation/answer/prompt_v1.md`, `evaluation/answer/output_schema.json`, `evaluation/answer/case_result_schema.json`, `evaluation/answer/run.py`, and `evaluation/answer/evaluate.py` exist.
3. Read the dataset and verify that it contains 12 unique cases and that every non-null `source_file` is one of the two approved public PDFs.
4. Check Git status for `evaluation/answer/dataset.json`, `evaluation/answer/prompt_v1.md`, `evaluation/answer/output_schema.json`, and `src/mcp_server.py`. If any is modified, report the exact files and ask whether this should be treated as a new evaluation cycle. Runner, Skill, test, documentation, and unrelated worktree changes are not blockers.
5. Confirm the current Codex host exposes `njupt-rag.search_knowledge_base` and can create subagents. Do not run `codex login status`: fresh subagents use the current Codex host and its authentication. If either capability is absent, stop and report which capability is unavailable.
6. Verify `runtime/storage/index/metadata.json` reports schema 4, 2 documents, and 731 chunks, and verify the four model directories documented in `README.md` exist.
7. If the user gives an output directory, require it to be a new path under `runtime/reports/answer/`. Otherwise let the runner create its timestamped output directory so repeated runs never overwrite earlier evidence.

## Prepare

Briefly state that the fixed public questions and at most five retrieved evidence chunks per question will be sent to the configured Codex service. Then run:

```powershell
uv run python evaluation/answer/run.py prepare
```

When the user explicitly requests a named output directory, add:

```powershell
--output-dir runtime/reports/answer/<new-run-name>
```

Read the generated `agent-tasks.json`. It must contain 12 unique tasks, each with `model=gpt-5.6-sol`, `fork_turns=none`, `limit=5`, a unique result path, and a prompt containing only that task's id and query. Do not pass the dataset or conversation history to a child.

## Run isolated subagents

Use a bounded pool of at most three active subagents. For each task:

1. Spawn a new subagent with `fork_turns=none`, model `gpt-5.6-sol`, and a unique lowercase task name derived from the case id by replacing every non-alphanumeric character with `_`; task names may contain only letters, digits, and underscores.
2. Pass the task's `prompt` verbatim as the complete task message. Do not prepend summaries, expected answers, diagnoses, prior results, or another case's context.
3. Require the subagent to call `njupt-rag.search_knowledge_base` exactly once and write only its unique `case-results/<id>.json` artifact as directed by the prompt.
4. When a subagent completes, run:

```powershell
uv run python evaluation/answer/run.py validate-case `
  --output-dir runtime/reports/answer/<run-name> `
  --case-id <case-id>
```

5. Start the next pending task when a slot becomes free. Continue until all 12 cases validate.

Do not ask a completed subagent to revise a result and do not fill, repair, or infer missing MCP fields in the parent. If any subagent fails or its result does not validate, stop launching new tasks, allow already-running tasks to finish, preserve the incomplete output directory, and report the exact case and validation error. Use a new output directory for a retry.

## Finalize

After all 12 case results validate, run:

```powershell
uv run python evaluation/answer/run.py finalize `
  --output-dir runtime/reports/answer/<run-name>
```

The runner must aggregate results in dataset order, generate `predictions.json` and `report.json`, and set the manifest status to `completed`.

## Validate

Require `agent-tasks.json`, exactly 12 files under `case-results/`, `predictions.json`, `report.json`, and `manifest.json`. Verify that:

- the manifest status is `completed`, `execution_mode` is `one-case-per-fresh-subagent`, and all 12 subagents are validated;
- the model and versioned settings match the frozen boundary;
- the dataset contains 12 cases and predictions contain the same 12 unique IDs;
- all citations and cited chunk IDs agree;
- every task records `fork_turns=none`, one case, the original query, and retrieval limit 5;
- the runner reports no result-contract failure.

Treat `case-results/` as subagent-submitted MCP payloads validated for structure and internal consistency. Do not describe them as host-captured raw MCP event logs; this mode does not provide the independent JSONL trace that the former nested `codex exec` runner provided.

Report actual failure-stage counts without interpreting pending human review as a passed result.

## Prepare human review

For each answerable case, present a compact review record containing:

- question and generated answer;
- gold answer and required facts from the dataset;
- cited source file, page, chunk ID, and relevant retrieved evidence text;
- the four fields requiring a human decision: `correct`, `complete`, `citation_supported`, and `uses_model_memory_or_guess`.

Call out obvious mismatches, but leave all four values unset until a human confirms them. The current evaluator counts manual-review coverage only for answerable cases. For no-answer cases, summarize whether the automatic checks observed `evidence_sufficient=false`, `refused=true`, and empty citations.

After the user supplies the human decisions, edit only the corresponding `review` values in `predictions.json`. Do not change generated answers, retrieval results, citations, or refusal decisions.

## Generate the reviewed report

Run:

```powershell
uv run python evaluation/answer/evaluate.py `
  --dataset evaluation/answer/dataset.json `
  --predictions runtime/reports/answer/<run-name>/predictions.json `
  --report runtime/reports/answer/<run-name>/report-reviewed.json
```

Verify full manual-review coverage for answerable cases. Summarize correctness, completeness, citation precision and recall, false-answer rate, false-refusal rate, insufficient-evidence refusal compliance, model-memory/guess rate, and failure-stage counts.

Update planning or implementation documents only when the user explicitly asks for documentation changes.
