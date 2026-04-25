import csv
import itertools
import json
import os
import re
import subprocess
import time


TOP_K = 10
RANK_METRIC = "last_final_ood_test"

SUMMARY_PATTERNS = {
    "highest_train": r"Highest Train:\s*([0-9.]+)\s*±\s*([0-9.]+)",
    "highest_valid": r"Highest Valid:\s*([0-9.]+)\s*±\s*([0-9.]+)",
    "highest_in_test": r"Highest In Test:\s*([0-9.]+)\s*±\s*([0-9.]+)",
    "final_train": r"Final Train:\s*([0-9.]+)\s*±\s*([0-9.]+)",
    "final_in_test": r"Final In\s+Test:\s*([0-9.]+)\s*±\s*([0-9.]+)",
}


def extract_summary_metrics(log_text):
    metrics = {}
    for metric_name, pattern in SUMMARY_PATTERNS.items():
        match = re.search(pattern, log_text)
        if match:
            metrics[metric_name] = float(match.group(1))
            metrics[f"{metric_name}_std"] = float(match.group(2))
        else:
            metrics[metric_name] = None
            metrics[f"{metric_name}_std"] = None

    highest_ood_matches = re.findall(r"Highest OOD Test:\s*([0-9.]+)\s*±\s*([0-9.]+)", log_text)
    final_ood_matches = re.findall(r"Final OOD Test:\s*([0-9.]+)\s*±\s*([0-9.]+)", log_text)

    for idx, (mean_value, std_value) in enumerate(highest_ood_matches, start=1):
        metrics[f"highest_ood_test_{idx}"] = float(mean_value)
        metrics[f"highest_ood_test_{idx}_std"] = float(std_value)

    for idx, (mean_value, std_value) in enumerate(final_ood_matches, start=1):
        metrics[f"final_ood_test_{idx}"] = float(mean_value)
        metrics[f"final_ood_test_{idx}_std"] = float(std_value)

    return metrics


def safe_metric_value(record, metric_name):
    value = record.get(metric_name)
    return float("-inf") if value is None else value


def get_last_final_ood_metric(record):
    ood_indices = []
    for key in record.keys():
        match = re.fullmatch(r"final_ood_test_(\d+)", key)
        if match:
            ood_indices.append(int(match.group(1)))

    if not ood_indices:
        return None

    return record.get(f"final_ood_test_{max(ood_indices)}")


def get_final_ood_keys(record):
    return sorted(
        (key for key in record.keys() if re.fullmatch(r"final_ood_test_\d+", key)),
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )


def write_all_results_csv(results, csv_path):
    if not results:
        return

    fieldnames = []
    for result in results:
        for key in result.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def write_topk_summary(results, summary_path, dataset, backbone, stage, hp_keys):
    ranked_results = sorted(
        results,
        key=lambda item: (
            safe_metric_value(item, RANK_METRIC),
            safe_metric_value(item, "final_in_test"),
            safe_metric_value(item, "highest_valid"),
            safe_metric_value(item, "highest_in_test"),
        ),
        reverse=True,
    )
    top_results = ranked_results[:TOP_K]

    with open(summary_path, "w") as summary_file:
        summary_file.write(
            f"Top {len(top_results)} front-door experiments on {dataset} "
            f"({backbone}, {stage}) ranked by {RANK_METRIC}\n"
        )
        summary_file.write("=" * 120 + "\n\n")

        for rank, result in enumerate(top_results, start=1):
            summary_file.write(f"Rank {rank}\n")
            summary_file.write(f"log_file: {result['log_file']}\n")
            summary_file.write(f"result_name: {result['result_name']}\n")
            summary_file.write(f"rank_metric ({RANK_METRIC}): {result.get(RANK_METRIC)}\n")
            summary_file.write(
                f"highest_valid: {result.get('highest_valid')} ± {result.get('highest_valid_std')}\n"
            )
            summary_file.write(
                f"highest_in_test: {result.get('highest_in_test')} ± {result.get('highest_in_test_std')}\n"
            )
            summary_file.write(
                f"final_in_test: {result.get('final_in_test')} ± {result.get('final_in_test_std')}\n"
            )
            summary_file.write("hyperparameters:\n")
            for key in hp_keys:
                summary_file.write(f"  {key}: {result.get(key)}\n")

            final_ood_keys = get_final_ood_keys(result)
            if final_ood_keys:
                summary_file.write("final_ood_metrics:\n")
                for key in final_ood_keys:
                    summary_file.write(
                        f"  {key}: {result.get(key)} ± {result.get(f'{key}_std')}\n"
                    )

            summary_file.write("\n")


def _append_param(cmd, key, value):
    flag = f"--{key}"
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    cmd.extend([flag, str(value)])


def run_grid_search(
    dataset,
    backbone,
    search_stage,
    base_cmd,
    stage_grids,
    search_root="search_logs_frontdoor",
):
    if search_stage not in stage_grids:
        raise ValueError(f"Unsupported search stage: {search_stage}")

    param_grid = stage_grids[search_stage]
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    log_dir = os.path.join(search_root, dataset, backbone, search_stage)
    os.makedirs(log_dir, exist_ok=True)

    all_results = []

    print(
        f"Start front-door {search_stage} grid search on {dataset} "
        f"with {backbone}, total runs: {len(combinations)}"
    )
    print("-" * 80)

    for idx, combination in enumerate(combinations, start=1):
        current_params = dict(zip(keys, combination))
        cmd = base_cmd.copy()

        result_name_parts = [dataset, backbone, "frontdoor", search_stage]
        for key, value in current_params.items():
            _append_param(cmd, key, value)
            result_name_parts.append(f"{key}{value}")

        result_name = "_".join(str(part) for part in result_name_parts)
        cmd.extend(["--result_name", result_name])
        cmd_str = " ".join(cmd)

        print(f"[{idx}/{len(combinations)}] {cmd_str}")

        log_file_path = os.path.join(log_dir, f"{idx:03d}_{result_name}.log")
        start_time = time.time()
        completed = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        log_text = completed.stdout + completed.stderr

        with open(log_file_path, "w") as log_file:
            log_file.write(log_text)
            log_file.write(f"\nCommand: {cmd_str}\n")
            log_file.write(f"Return code: {completed.returncode}\n")
            log_file.write(f"Elapsed: {elapsed:.2f}s\n")

        metrics = extract_summary_metrics(log_text)
        record = {
            "index": idx,
            "dataset": dataset,
            "backbone": backbone,
            "stage": search_stage,
            "result_name": result_name,
            "log_file": log_file_path,
            "command": cmd_str,
            "return_code": completed.returncode,
            "elapsed_seconds": round(elapsed, 2),
        }
        record.update(current_params)
        record.update(metrics)
        record[RANK_METRIC] = get_last_final_ood_metric(record)
        all_results.append(record)

        print(
            f"Finished {idx}/{len(combinations)} in {elapsed:.2f}s, "
            f"{RANK_METRIC}={record.get(RANK_METRIC)}, log: {log_file_path}"
        )

        all_results_csv = os.path.join(
            log_dir,
            f"{dataset}_{backbone}_{search_stage}_all_results.csv",
        )
        all_results_json = os.path.join(
            log_dir,
            f"{dataset}_{backbone}_{search_stage}_all_results.json",
        )
        topk_summary = os.path.join(
            log_dir,
            f"{dataset}_{backbone}_{search_stage}_top{TOP_K}.txt",
        )

        write_all_results_csv(all_results, all_results_csv)
        with open(all_results_json, "w") as json_file:
            json.dump(all_results, json_file, indent=2)
        write_topk_summary(all_results, topk_summary, dataset, backbone, search_stage, keys)

    print(f"{search_stage} grid search finished.")
    print(f"Saved full results to: {all_results_csv}")
    print(f"Saved top-{TOP_K} summary to: {topk_summary}")
