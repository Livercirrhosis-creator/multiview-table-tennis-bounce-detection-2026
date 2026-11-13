"""Fit the unchanged 102-parameter head once on development121 for deployment."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from train_evaluate import EXP, ROOT, PATHS, CONFIG, FEATURE_NAMES, read, lines, sha, save, train_fold


def main():
    parent = read(EXP / "protocol_v001.json")
    assert all(sha(PATHS[k]) == value for k, value in parent["input_sha256"].items())
    assert all(sha(ROOT / k) == value for k, value in parent["source_sha256"].items())
    assert read(EXP / "runs/lofo_v001/run_status.json")["completed"]
    allowed = parent["video_ids"]
    plan = dict(schema="masked_gap_deployment_fit.v1", created_utc=datetime.now(timezone.utc).isoformat(),
        research_protocol_sha256=sha(EXP / "protocol_v001.json"), config=CONFIG, feature_names=FEATURE_NAMES,
        video_ids=allowed, training_runs=1, evaluation_runs=0, checkpoint_selection="Fixed final epoch, no in-sample scoring",
        input_sha256={k: sha(PATHS[k]) for k in ("features", "index", "policy")},
        implementation_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in (Path(__file__), EXP / "code/train_evaluate.py")},
        rationale="13 recovered GT, 15 net extra FP, small positive micro/macro on source-held-out development; prepare coordinate/cost regression without replacing main 2D",
        protected33_used=False, public15_used=False, ssh=False)
    save(EXP / "deployment_protocol_v001.json", plan)
    out = EXP / "runs/deployment_v001"
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = dict(started_utc=datetime.now(timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
                  pid=os.getpid(), completed=False, exit_code=1, training_runs_started=0)
    with (out / "console.log").open("x", encoding="utf-8") as console:
        try:
            with redirect_stdout(console), redirect_stderr(console):
                torch.set_num_threads(CONFIG["cpu_threads"])
                torch.use_deterministic_algorithms(True)
                x, rows = np.load(PATHS["features"]), lines(PATHS["index"])
                fold = dict(fold_id=10, held_out_family="deployment_all121_no_evaluation", train_video_ids=allowed, held_out_video_ids=[])
                status["training_runs_started"] = 1
                predictions, result = train_fold(x, rows, fold, out)
                assert predictions == []
                target = EXP / "weights/masked_gap_v001/model.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                assert not target.exists()
                shutil.copyfile(ROOT / result["weights_path"], target)
                report = dict(completed=True, deployment_protocol_sha256=sha(EXP / "deployment_protocol_v001.json"),
                    weight_path=target.relative_to(ROOT).as_posix(), weights_sha256=sha(target),
                    fit=result, training_runs=1, epochs=200, evaluations=0, protected33_used=False,
                    public15_used=False, ssh=False, full_video_inference_runs=0)
                save(out / "report.json", report)
                save(EXP / "analysis/deployment_v001.json", report)
            status.update(completed=True, exit_code=0)
        except BaseException:
            status["error"] = traceback.format_exc()
            console.write(status["error"])
            raise
        finally:
            status.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started)
            save(out / "run_status.json", status)
            (out / "exit_code.txt").write_text(str(status["exit_code"]) + "\n", encoding="utf-8")
    print(json.dumps(dict(completed=True, weights_sha256=report["weights_sha256"], training_runs=1, evaluation_runs=0)))


if __name__ == "__main__":
    main()
