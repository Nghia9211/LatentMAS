#!/usr/bin/env python3
"""Read-only audit of full MedQA A-latent/B-text outputs; print JSON to stdout.

No torch/model/GPU/API calls. Only tokenizers is optional for decoding patterns.
Question numbers are one-based positions in the common 300-question dataset.
"""
import argparse
import hashlib
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def repetition(ids):
    single = {"length": 0, "start": 0, "token_id": None}
    offset = 0
    for token, group in itertools.groupby(ids):
        length = sum(1 for _ in group)
        if length > single["length"]:
            single = dict(length=length, start=offset, token_id=token)
        offset += length
    periodic = {"length": 0, "period": 0, "start": 0, "pattern_ids": []}
    # A segment satisfies ids[t] == ids[t-period] at every overlapping position.
    # Its length includes a possible partial final repetition.
    for period in range(1, 17):
        start = period
        for index in range(period, len(ids) + 1):
            if index == len(ids) or ids[index] != ids[index - period]:
                length = index - start + period
                if length >= 64 and length >= 8 * period and length > periodic["length"]:
                    periodic = dict(length=length, period=period, start=start - period,
                                    pattern_ids=ids[start - period:start])
                start = index + 1
    return dict(loop_flag=single["length"] >= 32 or periodic["length"] > 0,
                longest_identical=single, longest_qualifying_periodic=periodic)


def self_test():
    assert not repetition([1, 2, 3, 4])["loop_flag"]
    assert repetition([7] * 32)["longest_identical"]["length"] == 32
    result = repetition([99] + [1, 2, 3] * 30)
    assert result["loop_flag"]
    assert result["longest_qualifying_periodic"]["start"] == 1
    assert result["longest_qualifying_periodic"]["length"] == 90
    assert not repetition(list(range(16)) * 4)["loop_flag"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 20, 40, 60, 80])
    parser.add_argument("--cases", type=int, nargs="+", default=[43, 57, 68, 125, 129, 188, 205, 235])
    parser.add_argument("--tokenizer-file", type=Path)
    args = parser.parse_args()
    if any(not 1 <= case <= 300 for case in args.cases):
        parser.error("--cases must be one-based positions from 1 through 300")
    self_test()
    tokenizer = None
    if args.tokenizer_file:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer_file))
    runs = []
    for path in sorted(args.output_dir.glob("latent_mas*medqa*.json")):
        raw = path.read_bytes()
        doc = json.loads(raw)
        summary, rows = doc["summary"], doc["predictions"]
        config = summary["run_config"]
        if (len(rows) != 300 or config.get("judger_mode", "text") != "text"
                or config["latent_steps"] not in args.budgets):
            continue
        identity = digest([(r["question"], r["gold"], [a["input"] for a in r["agents"]]) for r in rows])
        metrics = []
        for index, row in enumerate(rows, 1):
            generation = row["generation"]
            ids = generation["output_token_ids"]
            assert len(ids) == generation["generated_tokens"], (path.name, index)
            boxes = re.findall(r"\\boxed\{([^}]*)\}", row["raw_prediction"])
            choice = boxes[-1].strip().lower() if boxes else None
            prediction = choice if choice in ("a", "b", "c", "d") else None
            correct = prediction is not None and prediction == str(row["gold"]).strip().lower()
            assert correct == row["medqa_boxed_v1"]["correct"], (path.name, index)
            m = repetition(ids)
            m.update(question=index, capped=generation["finish_reason"] == "length",
                     correct=correct, parse_failed=prediction is None, prediction=prediction,
                     generated_tokens=len(ids), token_ids_sha256=digest(ids))
            if tokenizer:
                p = m["longest_qualifying_periodic"]
                p["pattern_text"] = tokenizer.decode(p["pattern_ids"], skip_special_tokens=False)
                token = m["longest_identical"]["token_id"]
                m["longest_identical"]["token_text"] = tokenizer.decode([token], skip_special_tokens=False) if token is not None else ""
            metrics.append(m)
        runs.append(dict(file=path.name, file_sha256=hashlib.sha256(raw).hexdigest(),
                         budget=config["latent_steps"], run_config=config,
                         identity_sha256=identity, metrics=metrics))
    if not runs:
        parser.error("No full 300-question B-text runs found for requested budgets")
    identities = {r["identity_sha256"] for r in runs}
    assert len(identities) == 1, "Question/gold/agent prompts differ; do not pair these runs"
    groups = defaultdict(list)
    for run in runs:
        groups[run["budget"]].append(run)
    summaries = []
    for budget, group in sorted(groups.items()):
        loop_sets = [{m["question"] for m in r["metrics"] if m["loop_flag"]} for r in group]
        cap_sets = [{m["question"] for m in r["metrics"] if m["loop_flag"] and m["capped"]} for r in group]
        intersection, union = set.intersection(*loop_sets), set.union(*loop_sets)
        repeat_available = len(group) >= 2
        summaries.append(dict(budget=budget, run_files=[r["file"] for r in group],
            loop_counts=[len(s) for s in loop_sets], capped_loop_counts=[len(s) for s in cap_sets],
            loop_cases_per_run=[sorted(s) for s in loop_sets],
            capped_loop_cases_per_run=[sorted(s) for s in cap_sets],
            repeat_available=repeat_available,
            loop_in_all_repeats=sorted(intersection) if repeat_available else None,
            loop_in_some_not_all=sorted(union - intersection) if repeat_available else None,
            capped_loop_in_all_repeats=sorted(set.intersection(*cap_sets)) if repeat_available else None,
            exact_whole_output_in_all_repeats_among_common_loops=[i for i in sorted(intersection)
                if len({r["metrics"][i-1]["token_ids_sha256"] for r in group}) == 1] if repeat_available else None))
    cases = []
    for index in args.cases:
        row = json.loads((args.output_dir / runs[0]["file"]).read_bytes())["predictions"][index - 1]
        cases.append(dict(question_number=index, question=row["question"], gold=row["gold"],
                          observations=[dict(file=r["file"], budget=r["budget"], **r["metrics"][index-1]) for r in runs]))
    public_runs = [{k: v for k, v in r.items() if k != "metrics"} for r in runs]
    print(json.dumps(dict(detector_version="literal_token_periodic_v1", question_numbering="one-based",
        detector="identical token >=32 OR contiguous periodic segment length >=64 and >=8*period, period 1..16",
        caveats=["heuristic, not semantic repetition detection", "cap alone is not loop",
                 "repeated seed42 runs are not independent-seed replications", "saved A states absent from historical full runs"],
        runs=public_runs, same_question_gold_agent_prompts=True,
        budget_summaries=summaries, selected_cases=cases), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
