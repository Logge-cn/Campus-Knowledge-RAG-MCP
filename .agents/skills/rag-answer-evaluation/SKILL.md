---
name: rag-answer-evaluation
description: Run and repeat this repository's fixed RAG answer-level evaluation through Codex CLI and the local MCP server, validate run artifacts, organize human review, and generate the reviewed report. Use when asked to execute or rerun the first-priority answer evaluation, inspect answer predictions, review citations or refusals, or calculate final answer-level metrics. Do not use for dataset changes, retrieval tuning, ingestion, index rebuilding, or generic RAG experiments.
---

# RAG Answer Evaluation

Run the repository's existing evaluation workflow. Reuse the checked-in runner and evaluator; do not recreate their logic in the skill.

## Preserve the evaluation boundary

- Keep `gpt-5.6-sol`, Prompt version `answer-eval-v1`, tool-description version `answer-eval-v1`, and retrieval limit 5 unchanged.
- Treat `data/南京邮电大学学生手册（2023版）+.pdf` and `data/奖学金细则.pdf` as approved public inputs. When the user directly asks to run the evaluation and the dataset references only these inputs, give a brief outbound-data notice and proceed without a separate blocking authorization question.
- Stop and request confirmation if the dataset introduces another source or the user identifies any input as non-public.
- Send only the fixed questions and MCP-returned evidence through the existing runner. Do not upload whole PDFs, model directories, or the index.
- Do not modify the dataset, Prompt, output schema, runner, evaluator, MCP description, or retrieval configuration during a run.
- Do not start retrieval tuning or other optimization after seeing failures unless the user separately requests it.
- Preserve existing worktree changes and never clean or overwrite unrelated files.
- Never label model-generated review judgments as human review.

## Select the workflow

- For a new or repeated run, perform Preflight, Run, Validate, then Prepare human review.
- For an existing `predictions.json`, skip execution and start at Validate.
- For a final reviewed report, require completed human review fields before running the evaluator.

## Preflight

1. Resolve the repository root with `git rev-parse --show-toplevel` and run all commands there.
2. Confirm `evaluation/answer_eval_dataset.json`, `evaluation/answer_eval_prompt_v1.md`, `evaluation/answer_eval_output_schema.json`, `evaluation/run_answer_evaluation.py`, and `evaluation/evaluate_answers.py` exist.
3. Read the dataset and verify that it contains 12 unique cases and that every non-null `source_file` is one of the two approved public PDFs.
4. Check Git status for the frozen inputs above and `src/mcp_server.py`. If any is modified, report the exact files and ask whether this should be treated as a new evaluation cycle. Unrelated dirty files are not blockers.
5. Verify Codex CLI availability and saved authentication with `codex login status`. On Windows, prefer the discovered `codex.cmd` when PowerShell blocks `codex.ps1`. If not logged in, stop and provide the device-login command; do not start an interactive login invisibly.
6. Verify `storage/index/metadata.json` reports schema 4, 2 documents, and 731 chunks, and verify the four model directories documented in `README.md` exist.
7. If the user gives an output directory, require it to be a new path under `evaluation/reports/`. Otherwise let the runner create its timestamped output directory so repeated runs never overwrite earlier evidence.

## Run

Briefly state that the fixed public questions and at most five retrieved evidence chunks per question will be sent to the configured Codex service. Then run:

```powershell
uv run python evaluation/run_answer_evaluation.py
```

When the user explicitly requests a named output directory, add:

```powershell
--output-dir evaluation/reports/<new-run-name>
```

Allow the long-running command to finish and provide concise progress updates. If it fails, preserve its output directory and inspect `manifest.json` and `client-stderr.log`. Do not delete or reuse a partially created directory; choose a new directory for a retry.

## Validate

Require `raw-events.jsonl`, `client-stderr.log`, `client-output.json`, `predictions.json`, `report.json`, and `manifest.json`. Verify that:

- the manifest status is `completed` and `client_exit_code` is 0;
- the model and versioned settings match the frozen boundary;
- the dataset contains 12 cases and predictions contain the same 12 unique IDs;
- all citations and cited chunk IDs agree;
- the runner reports no parsing or MCP-call contract failure.

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
uv run python evaluation/evaluate_answers.py `
  --dataset evaluation/answer_eval_dataset.json `
  --predictions evaluation/reports/<run-name>/predictions.json `
  --report evaluation/reports/<run-name>/report-reviewed.json
```

Verify full manual-review coverage for answerable cases. Summarize correctness, completeness, citation precision and recall, false-answer rate, false-refusal rate, insufficient-evidence refusal compliance, model-memory/guess rate, and failure-stage counts.

Update planning or implementation documents only when the user explicitly asks for documentation changes.
