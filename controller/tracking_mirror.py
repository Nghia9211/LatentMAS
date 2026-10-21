"""Mirror a task's scalar training log without racing file creation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def read_complete_rows(path):
    """Writers append JSON plus newline; ignore an unfinished last record."""
    path = Path(path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines(keepends=True)
        if line.endswith("\n") and line.strip()
    ]


def mirror(training, job, wait_seconds=30000):
    training, job = Path(training).resolve(), Path(job).resolve()
    repo = Path(__file__).resolve().parents[1]
    boundary = repo.parent
    if not training.is_relative_to(boundary) or not job.is_relative_to(boundary):
        raise ValueError("Tracking paths must stay in the workspace")
    identity_path = training / "wandb_tracking.json"
    metrics_path = training / "training_metrics.jsonl"
    exit_path = job / "exit-code"

    def state(status, **fields):
        payload = dict(
            status=status,
            utc=datetime.now(timezone.utc).isoformat(),
            **fields,
        )
        temporary = job / "wandb_online_mirror.tmp"
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(job / "wandb_online_mirror.json")
        print(json.dumps(payload), flush=True)

    deadline = time.monotonic() + wait_seconds
    while not identity_path.exists() or not metrics_path.exists():
        if exit_path.exists():
            state("training_failed_before_metrics", training_exit_code=int(exit_path.read_text()))
            return
        if time.monotonic() > deadline:
            state("tracking_wait_expired")
            return
        time.sleep(10)

    from dotenv import load_dotenv
    from controller.tracking import PUBLIC_CONFIG_KEYS
    import wandb

    load_dotenv(repo / ".env", override=False)
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Configured tracking credential unavailable")
    identity = json.loads(identity_path.read_text())
    manifest = json.loads((training / "manifest.json").read_text())
    config = {
        key: value
        for key, value in manifest["config"].items()
        if key in PUBLIC_CONFIG_KEYS
        and isinstance(value, (str, int, float, bool, type(None)))
    }
    config.update(
        controller_normalization=manifest["config"]["controller_normalization"],
        observation_dim=manifest["observation"]["input_dim"],
        live_A_B_rollout=True,
        grid_training=False,
        scalar_log_mirror=True,
    )
    folder = job / "wandb-online-mirror"
    folder.mkdir(exist_ok=True)
    run = None
    code = 1
    cursor = 0
    try:
        run = wandb.init(
            entity=identity["entity"],
            project=identity["project"],
            id=identity["run_id"],
            resume="allow",
            mode="online",
            name=manifest["wandb"]["name"].replace("-offline", "-live"),
            dir=str(folder),
            config=config,
            save_code=False,
            settings=wandb.Settings(
                console="off", disable_git=True, disable_code=True, init_timeout=60
            ),
        )
        run.define_metric("optimizer_step")
        run.define_metric("iteration")
        for pattern in ("loss/*", "ppo/*"):
            run.define_metric(pattern, step_metric="optimizer_step")
        for pattern in (
            "iteration/*", "train/*", "time/*", "budget/*", "handoff/*",
            "rollout/*", "output/*", "data/*", "validation/*", "progress/*",
        ):
            run.define_metric(pattern, step_metric="iteration")
        previous = None
        while True:
            rows = read_complete_rows(metrics_path)
            if len(rows) < cursor:
                raise RuntimeError("Training scalar log was truncated")
            for row in rows[cursor:]:
                run.log(row)
                cursor += 1
            events = read_complete_rows(training / "events.jsonl")
            ends = [event for event in events if event["stage"] == "B_cohort_complete"]
            committed = [
                event for event in events if event["stage"] == "PPO_iteration_complete"
            ]
            progress = {
                "iteration": committed[-1]["iteration"] if committed else 0,
                "progress/PPO_iterations_committed": len(committed),
                "progress/train_B_completed": 16 * sum(
                    event.get("purpose") == "live_train_reward" for event in ends
                ),
                "progress/validation_B_completed": 16 * sum(
                    event.get("purpose") == "validation_PPO" for event in ends
                ),
            }
            if progress != previous:
                run.log(progress)
                previous = progress
            state("online", url=run.url, metric_rows_mirrored=cursor, progress=progress)
            if exit_path.exists():
                code = int(exit_path.read_text())
                break
            time.sleep(15)
    except BaseException as error:
        state("mirror_failed", error_type=type(error).__name__, metric_rows_mirrored=cursor)
        raise
    finally:
        if run is not None:
            run.finish(exit_code=code)
    cloud = wandb.Api(timeout=25).run(
        f"{identity['entity']}/{identity['project']}/{identity['run_id']}"
    )
    (job / "wandb_cloud_verified.json").write_text(
        json.dumps(
            dict(
                run_id=identity["run_id"], url=run.url, state=cloud.state,
                summary=dict(cloud.summary), verified_via_API=True,
                training_exit_code=code,
            ),
            indent=2,
        )
    )
    state(
        "finished" if code == 0 else "training_failed",
        url=run.url, metric_rows_mirrored=cursor, training_exit_code=code,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--job", required=True, type=Path)
    options = parser.parse_args()
    mirror(options.training, options.job)
