"""Fixed source-held-out, paired-observation training of the existing event models.

The two observation views share a physical event, source fold and event weight.
Old per-fold train-source normalization is reused unchanged. This script never
reads public15/protected33 labels or selects a checkpoint from held-out scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
RUNTIME = ROOT / "experiments/20261105_coverage_aware_events/versions/v001/participant/pingpang"
OLD = dict(dense=ROOT / "experiments/20261105_coverage_aware_events",
           window=ROOT / "experiments/20261104_event_presence_location")
ALLOW = ROOT / "experiments/20261102_research_foundation/data/development_videos.json"
PROTOCOL = EXP / "protocol_v001.json"
VIEWS = ("original", "augmented")
TRAINING = dict(epochs=80, seed=20261103, event_batch_size=64, optimizer="AdamW",
                learning_rate=.001, weight_decay=.0001, threads=2, interop_threads=1,
                checkpoint="fixed_final_epoch", new_fusion_head=False)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def digest_rows(rows):
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def runtime_modules():
    sys.path.insert(0, str(RUNTIME))
    from participant.dense_context_v001.model import DenseContextModel
    from participant.event_window_v001.model import WindowEventModel
    from participant.event_decision_v001 import normalization, trajectory_features
    path = OLD["dense"] / "code/train_dense_context.py"
    spec = importlib.util.spec_from_file_location("fixed_dense_training_loss", path)
    loss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loss_module)
    return {"dense": DenseContextModel, "window": WindowEventModel}, normalization, trajectory_features, loss_module


def load_dataset(dataset, kind):
    path = Path(dataset) / kind
    manifest = read(path / "manifest.json")
    for name, expected in manifest["output_sha256"].items():
        assert sha(path / name) == expected, "Dataset changed: " + name
    index, windows = lines(path / "index.jsonl"), read(path / "windows.json")
    supervision, folds = read(path / "supervision.json"), read(path / "folds.json")["folds"]
    allowed = set(read(ALLOW)["video_ids"])
    assert len(allowed) == 121 and len(folds) == 10
    assert {r["video_id"] for r in index} <= allowed
    assert [r["row_index"] for r in index] == list(range(len(index)))
    assert [w["window_index"] for w in windows] == list(range(len(windows)))
    if kind == "dense":
        assert len(index) == len(windows)
        assert [w["center_row_index"] for w in windows] == list(range(len(index)))
    for w in windows:
        assert w["center_row_index"] in w["member_rows"]
        assert all(index[i]["video_id"] == w["video_id"] and index[i]["view"] == w["view"]
                   for i in w["member_rows"])
    for fold in folds:
        train, held = set(fold["train_rows"]), set(fold["held_out_rows"])
        assert train.isdisjoint(held) and train | held == set(range(len(index)))
        assert set(fold["train_video_ids"]).isdisjoint(fold["held_out_video_ids"])
        assert set(fold["train_video_ids"]) | set(fold["held_out_video_ids"]) == allowed
        assert all(index[i]["camera_family"] != fold["held_out_family"] for i in train)
        assert all(index[i]["camera_family"] == fold["held_out_family"] for i in held)
        for w in windows:
            side = train if w["center_row_index"] in train else held
            assert set(w["member_rows"]) <= side
    data = np.load(path / "features.npz", allow_pickle=False)
    sequence, summary = data["sequence"], data["summary"]
    assert sequence.shape == (len(index), 15, 23) and summary.shape == (len(index), 44)
    return path, manifest, index, windows, supervision, folds, sequence, summary


def membership(kind, index, windows, supervision, fold):
    if kind == "dense":
        variants = [supervision["bags"][i] for i in fold["train_bag_indices"]]
        background = fold["train_background_indices"]
        legal = sorted({i for v in variants for i in v["indices"]} | set(background))
        bg_view = lambda i: index[i]["view"]
    else:
        variants = [supervision["events"][i] for i in fold["train_event_indices"]]
        background = fold["train_background_windows"]
        active = {supervision["positive_windows"][i]["window_index"]
                  for v in variants for i in v["positive_window_indices"]} | set(background)
        legal = sorted({i for wi in active for i in windows[wi]["member_rows"]})
        bg_view = lambda i: windows[i]["view"]
    groups = {}
    for v in variants:
        assert v["video_id"] in fold["train_video_ids"] and v["view"] in VIEWS
        key = (v["video_id"], v["physical_event_id"])
        if key not in groups:
            groups[key] = dict(video_id=v["video_id"], physical_event_id=v["physical_event_id"],
                               class_id=v["class_id"], variants=[])
        g = groups[key]
        assert g["class_id"] == v["class_id"] in (0, 1)
        assert v["view"] not in {old["view"] for old in g["variants"]}
        g["variants"].append(v)
    groups = [groups[k] for k in sorted(groups)]
    bg = {view: [i for i in background if bg_view(i) == view] for view in VIEWS}
    assert groups and all(bg.values())
    assert set(legal) <= set(fold["train_rows"]) and not set(legal) & set(fold["held_out_rows"])
    return groups, bg, legal


def old_normalization(kind, fold):
    directory = OLD[kind] / f"runs/lofo_v001/folds/{fold['fold_id']:02d}"
    member_path = directory / "membership.json"
    previous = read(member_path)
    assert set(previous["fold"]["train_video_ids"]) == set(fold["train_video_ids"])
    assert set(previous["fold"]["held_out_video_ids"]) == set(fold["held_out_video_ids"])
    assert set(previous["normalization_rows"]) <= set(previous["fold"]["train_rows"])
    assert not set(previous["normalization_rows"]) & set(previous["fold"]["held_out_rows"])
    return directory / "normalization.json", member_path


def prepare_plan(dataset, kind):
    path, manifest, index, windows, supervision, folds, _, _ = load_dataset(dataset, kind)
    sources = [Path(__file__), PROTOCOL, ALLOW, path / "manifest.json", OLD["dense"] / "code/train_dense_context.py"]
    for package in ("dense_context_v001", "event_window_v001", "event_decision_v001"):
        sources.extend(sorted((RUNTIME / "participant" / package).glob("*.py")))
    rows = []
    for fold in folds:
        groups, bg, legal = membership(kind, index, windows, supervision, fold)
        norm, old_member = old_normalization(kind, fold)
        sources.extend([norm, old_member])
        rows.append(dict(fold_id=fold["fold_id"], held_out_family=fold["held_out_family"],
            train_video_ids=fold["train_video_ids"], held_out_video_ids=fold["held_out_video_ids"],
            physical_training_events=len(groups), event_variants=sum(len(g["variants"]) for g in groups),
            event_view_count={str(k):v for k,v in Counter(len(g["variants"]) for g in groups).items()},
            background_by_view={view:len(bg[view]) for view in VIEWS},
            training_legal_rows=len(legal), training_legal_membership_sha256=digest_rows(legal),
            normalization_source=relative(norm), normalization_sha256=sha(norm),
            normalization_membership_source=relative(old_member),
            optimizer_steps=TRAINING["epochs"] * math.ceil(len(groups)/TRAINING["event_batch_size"]),
            held_out_rows=len(fold["held_out_rows"])))
    return dict(schema="channel_event_adaptation_training_plan.v1", kind=kind, prepared=True,
        created_utc=datetime.now(timezone.utc).isoformat(), configuration=TRAINING,
        dataset=relative(path), dataset_counts=manifest.get("counts"), folds=rows,
        objective="0.5 mean_physical_events(mean_available_views(original_event_loss)) + 0.5 mean_views(mean_background_CE)",
        normalization="Reuse frozen old per-fold train-source statistics; no new fitting",
        views_are_independent_samples=False, public15_used=False, protected33_used=False,
        source_sha256={relative(p):sha(p) for p in sorted(set(sources))})


def mean_by_owner(values, owners, count):
    import torch
    ids = torch.tensor(owners, dtype=torch.long, device=values.device)
    counts = torch.bincount(ids, minlength=count).to(values.dtype)
    assert torch.all(counts > 0)
    return values.new_zeros(count).index_add(0, ids, values) / counts


def background_mean(logp, positions_by_view):
    import torch
    losses = [-logp[positions, 2].mean() for positions in positions_by_view if positions]
    return torch.stack(losses).mean() if losses else logp.sum()*0


def dense_batch(model, x, z, windows, groups, bg, old_loss):
    variants = [(owner, v) for owner,g in enumerate(groups) for v in g["variants"]]
    centers = sorted({i for _,v in variants for i in v["indices"]} | {i for rows in bg.values() for i in rows})
    active = sorted({i for c in centers for i in windows[c]["member_rows"]})
    local, center_local = {i:j for j,i in enumerate(active)}, {i:j for j,i in enumerate(centers)}
    embeddings = model.encode(x[active], z[active])
    logits = model.candidate_logits(embeddings, [[local[i] for i in windows[c]["member_rows"]] for c in centers],
                                    [local[c] for c in centers])
    mil, other = old_loss.event_loss_terms(logits, [[center_local[i] for i in v["indices"]] for _,v in variants],
                                          [v["class_id"] for _,v in variants])
    per_event = mean_by_owner(mil+other, [owner for owner,_ in variants], len(groups))
    import torch
    logp = torch.log_softmax(logits, -1)
    background = background_mean(logp, [[center_local[i] for i in bg[view]] for view in VIEWS])
    return .5*(per_event.mean()+background), per_event.mean(), background, len(active)


def window_batch(model, x, z, windows, supervision, groups, bg):
    import torch
    variants = [(owner,v) for owner,g in enumerate(groups) for v in g["variants"]]
    positives = [(vi,supervision["positive_windows"][pi]) for vi,(_,v) in enumerate(variants)
                 for pi in v["positive_window_indices"]]
    negative = [(view,wi) for view in VIEWS for wi in bg[view]]
    window_ids = [p["window_index"] for _,p in positives] + [wi for _,wi in negative]
    active = sorted({i for wi in window_ids for i in windows[wi]["member_rows"]})
    local = {i:j for j,i in enumerate(active)}
    members = [[local[i] for i in windows[wi]["member_rows"]] for wi in window_ids]
    embeddings = model.encode(x[active], z[active])
    logp = torch.log_softmax(model.window_logits(embeddings, members), -1)
    location = model.location(embeddings)
    n, width = len(positives), max(len(m) for m in members[:len(positives)])
    padded = np.zeros((n,width), dtype=np.int64)
    valid, targets = np.zeros((n,width), dtype=bool), np.zeros((n,width), dtype=bool)
    for j,(_,p) in enumerate(positives):
        size = len(members[j])
        padded[j,:size], valid[j,:size] = members[j], True
        target_rows = set(p["positive_rows"])
        targets[j,:size] = [i in target_rows for i in windows[p["window_index"]]["member_rows"]]
    assert targets.any(axis=1).all()
    kinds = torch.tensor([p["class_id"] for _,p in positives], dtype=torch.long)
    gathered = location[torch.from_numpy(padded)].gather(2,kinds[:,None,None].expand(-1,width,1)).squeeze(-1)
    location_nll = torch.logsumexp(gathered.masked_fill(~torch.from_numpy(valid),-torch.inf),1) - \
                   torch.logsumexp(gathered.masked_fill(~torch.from_numpy(targets),-torch.inf),1)
    presence_ce = -logp[torch.arange(n),kinds]
    by_variant = mean_by_owner(presence_ce+location_nll,[vi for vi,_ in positives],len(variants))
    per_event = mean_by_owner(by_variant,[owner for owner,_ in variants],len(groups))
    background = background_mean(logp, [[n+j for j,(v,_) in enumerate(negative) if v == view] for view in VIEWS])
    return .5*(per_event.mean()+background), per_event.mean(), background, len(active)


def predict_held(model, kind, x, z, windows, fold, output):
    import torch
    held = np.asarray(fold["held_out_rows"], dtype=np.int64)
    local = {int(i):j for j,i in enumerate(held)}
    model.eval()
    with torch.inference_mode():
        embeddings = torch.cat([model.encode(x[chunk],z[chunk]) for chunk in
                                np.array_split(held,max(1,math.ceil(len(held)/512)))])
        if kind == "dense":
            logits = model.candidate_logits(embeddings,[[local[j] for j in windows[int(i)]["member_rows"]] for i in held],
                                             [local[int(i)] for i in held])
            arrays = dict(held_rows=held, logits=logits.numpy(), probabilities=torch.softmax(logits,-1).numpy())
        else:
            held_windows = np.asarray(fold["held_out_window_indices"],dtype=np.int64)
            presence = []
            for chunk in np.array_split(held_windows,max(1,math.ceil(len(held_windows)/512))):
                logits = model.window_logits(embeddings,[[local[i] for i in windows[int(wi)]["member_rows"]] for wi in chunk])
                presence.append(torch.softmax(logits,-1).numpy())
            arrays = dict(held_rows=held,held_windows=held_windows,location_logits=model.location(embeddings).numpy(),
                          presence_probabilities=np.concatenate(presence))
    assert all(np.isfinite(a).all() for a in arrays.values())
    np.savez_compressed(output / "held_out.npz",**arrays)
    return arrays


def fit_fold(kind, sequence, summary, index, windows, supervision, fold, output, runtime):
    import torch
    model_classes, normalization, trajectory_features, old_loss = runtime
    output.mkdir(parents=True,exist_ok=False)
    groups,bg,legal = membership(kind,index,windows,supervision,fold)
    norm_path,old_membership = old_normalization(kind,fold)
    norm = read(norm_path)
    x = torch.from_numpy(normalization.transform(sequence,norm["sequence"],"tcn"))
    z = torch.from_numpy(trajectory_features.transform(summary,norm["summary"]))
    (output / "normalization.json").write_bytes(norm_path.read_bytes())
    save(output / "membership.json",dict(fold=fold,physical_events=groups,
        training_background_by_view=bg,training_legal_rows=legal,
        normalization_reused_from=relative(norm_path),normalization_sha256=sha(norm_path),
        old_membership=relative(old_membership),old_membership_sha256=sha(old_membership),normalization_fits=0))
    torch.manual_seed(TRAINING["seed"])
    rng = np.random.default_rng(TRAINING["seed"])
    model = model_classes[kind]().cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=TRAINING["learning_rate"],weight_decay=TRAINING["weight_decay"])
    batch_size = TRAINING["event_batch_size"]
    batches = math.ceil(len(groups)/batch_size)
    started,encoded = time.perf_counter(),0
    with (output / "epochs.jsonl").open("x",encoding="utf-8") as log:
        for epoch in range(1,TRAINING["epochs"]+1):
            tick = time.perf_counter()
            order = rng.permutation(len(groups))
            backgrounds = {view:np.array_split(rng.permutation(bg[view]),batches) for view in VIEWS}
            event_sum,background_sum = 0.,0.
            for bi,start in enumerate(range(0,len(groups),batch_size)):
                selected = [groups[i] for i in order[start:start+batch_size]]
                negatives = {view:backgrounds[view][bi].tolist() for view in VIEWS}
                optimizer.zero_grad(set_to_none=True)
                if kind == "dense":
                    loss,event_loss,background_loss,count = dense_batch(model,x,z,windows,selected,negatives,old_loss)
                else:
                    loss,event_loss,background_loss,count = window_batch(model,x,z,windows,supervision,selected,negatives)
                assert torch.isfinite(loss)
                loss.backward()
                optimizer.step()
                encoded += count
                event_sum += float(event_loss.detach())*len(selected)
                background_sum += float(background_loss.detach())
            log.write(json.dumps(dict(epoch=epoch,event_loss=event_sum/len(groups),
                background_batch_mean=background_sum/batches,optimizer_steps=epoch*batches,
                physical_event_visits=len(groups),event_variant_visits=sum(len(g["variants"]) for g in groups),
                epoch_seconds=time.perf_counter()-tick))+"\n")
            log.flush()
            if epoch % 20 == 0:
                print(f"{kind} fold {fold['fold_id']} epoch {epoch}/80, elapsed {time.perf_counter()-started:.1f}s",flush=True)
    fit_seconds = time.perf_counter()-started
    checkpoint = dict(model_state_dict=model.state_dict(),classes=["table_contact","racket_hit","background"],
        architecture=type(model).__name__,seed=TRAINING["seed"],epoch=TRAINING["epochs"],
        held_out_family=fold["held_out_family"],training_views=list(VIEWS))
    if kind == "window": checkpoint["location_classes"] = ["table_contact","racket_hit"]
    torch.save(checkpoint,output / "model.pt")
    tick = time.perf_counter()
    arrays = predict_held(model,kind,x,z,windows,fold,output)
    result = dict(kind=kind,fold_id=fold["fold_id"],held_out_family=fold["held_out_family"],
        physical_training_events=len(groups),event_variants=sum(len(g["variants"]) for g in groups),
        background_by_view={view:len(bg[view]) for view in VIEWS},epochs=TRAINING["epochs"],
        optimizer_steps=TRAINING["epochs"]*batches,fit_seconds=fit_seconds,
        held_out_forward_seconds=time.perf_counter()-tick,parameters=sum(p.numel() for p in model.parameters()),
        encoded_training_rows=encoded,model_sha256=sha(output / "model.pt"),
        normalization_sha256=sha(output / "normalization.json"),held_out_sha256=sha(output / "held_out.npz"))
    save(output / "fit_result.json",result)
    return arrays,result


def check_loss():
    import torch
    torch.set_num_threads(2)
    values = torch.tensor([2.,4.,10.],requires_grad=True)
    by_event = mean_by_owner(values,[0,0,1],2)
    assert torch.equal(by_event,torch.tensor([3.,10.]))
    by_event.mean().backward()
    assert torch.equal(values.grad,torch.tensor([.25,.25,.5]))
    p = torch.tensor([[.1,.2,.7],[.1,.2,.7],[.2,.3,.5]],dtype=torch.float64).log()
    actual = background_mean(p,[[0,1],[2]])
    assert abs(float(actual)-.5*(-math.log(.7)-math.log(.5))) < 1e-12
    models,_,_,old_loss = runtime_modules()
    torch.manual_seed(20261103)
    seq,summary = torch.randn(3,15,23),torch.randn(3,44)
    x,z = torch.cat((seq,seq)),torch.cat((summary,summary))
    windows = [dict(member_rows=[0,1]),dict(member_rows=[0,1]),dict(member_rows=[2]),
               dict(member_rows=[3,4]),dict(member_rows=[3,4]),dict(member_rows=[5])]
    original = dict(indices=[0,1],class_id=0,view="original")
    augmented = dict(indices=[3,4],class_id=0,view="augmented")
    paired = [dict(variants=[original,augmented])]
    single = [dict(variants=[original])]
    with torch.inference_mode():
        dense = models["dense"]().eval()
        old_dense = old_loss.batch_losses(dense,x,z,windows,[original],[2])[0]
        new_dense = dense_batch(dense,x,z,windows,paired,dict(original=[2],augmented=[5]),old_loss)[0]
        single_dense = dense_batch(dense,x,z,windows,single,dict(original=[2],augmented=[]),old_loss)[0]
        assert torch.allclose(old_dense,new_dense,atol=1e-6,rtol=1e-6)
        assert torch.allclose(old_dense,single_dense,atol=1e-6,rtol=1e-6)
        window = models["window"]().eval()
        supervision = dict(positive_windows=[dict(window_index=0,positive_rows=[0],class_id=0),
            dict(window_index=1,positive_rows=[0],class_id=0),
            dict(window_index=3,positive_rows=[3],class_id=0),
            dict(window_index=4,positive_rows=[3],class_id=0)])
        vo = dict(positive_window_indices=[0,1],view="original")
        va = dict(positive_window_indices=[2,3],view="augmented")
        embeddings = window.encode(seq,summary)
        presence = torch.log_softmax(window.window_logits(embeddings,[[0,1],[0,1],[2]]),-1)
        location = window.location(embeddings)[:,0]
        expected = .5*((-presence[:2,0]).mean()+torch.logsumexp(location[:2],0)-location[0]-presence[2,2])
        actual_window = window_batch(window,x,z,windows,supervision,[dict(variants=[vo,va])],
                                     dict(original=[2],augmented=[5]))[0]
        single_window = window_batch(window,x,z,windows,supervision,[dict(variants=[vo])],
                                     dict(original=[2],augmented=[]))[0]
        assert torch.allclose(expected,actual_window,atol=1e-6,rtol=1e-6)
        assert torch.allclose(expected,single_window,atol=1e-6,rtol=1e-6)
    old_loss.check_loss_semantics()
    return dict(passed=True,checks=["two views have one physical-event weight",
        "single available view retains full event weight","background view balance independent of row count",
        "identical-view dense objective equals original single-view objective",
        "identical-view window presence/location objective equals hand-derived single-view objective",
        "single-view dense and window objectives retain full weight",
        "reused dense loss physical-event bag-size and extreme-logit checks"],
        numerical=dict(dense_old=float(old_dense),dense_two_views=float(new_dense),
            window_expected=float(expected),window_two_views=float(actual_window)),
        training_fits=0,synthetic_model_encodes=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path,default=EXP / "runs/dataset_v001")
    parser.add_argument("--kind",choices=("dense","window"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prepared-plan",type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-plan",action="store_true")
    action.add_argument("--execute",action="store_true")
    action.add_argument("--check-loss",action="store_true")
    args = parser.parse_args()
    if args.check_loss:
        assert not args.output.exists()
        result = check_loss(); save(args.output,result); print(json.dumps(result)); return
    if args.kind is None: parser.error("--kind is required")
    plan = prepare_plan(args.dataset,args.kind)
    if args.prepare_plan:
        assert not args.output.exists()
        save(args.output,plan)
        print(json.dumps(dict(prepared=True,kind=args.kind,folds=len(plan["folds"]),output=str(args.output))))
        return
    if args.prepared_plan is None: parser.error("--execute requires --prepared-plan")
    frozen = read(args.prepared_plan)
    for key in ("kind","configuration","dataset","folds","source_sha256"):
        assert frozen[key] == plan[key], "Prepared plan differs: " + key
    args.output.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    status = dict(state="running",pid=os.getpid(),kind=args.kind,started_utc=datetime.now(timezone.utc).isoformat(),
                  command=sys.argv,prepared_plan_sha256=sha(args.prepared_plan),completed_fits=0)
    save(args.output / "run_status.json",status)
    try:
        import torch
        torch.set_num_threads(TRAINING["threads"])
        torch.set_num_interop_threads(TRAINING["interop_threads"])
        _,_,index,windows,supervision,folds,sequence,summary = load_dataset(args.dataset,args.kind)
        runtime = runtime_modules()
        receipts = []
        for fold in folds:
            _,receipt = fit_fold(args.kind,sequence,summary,index,windows,supervision,fold,
                                args.output / f"folds/{fold['fold_id']:02d}",runtime)
            receipts.append(receipt)
            status["completed_fits"] = len(receipts)
            save(args.output / "run_status.json",status)
        save(args.output / "report.json",dict(kind=args.kind,configuration=TRAINING,folds=receipts,
            training_fits=len(receipts),normalization_fits=0,public15_used=False,protected33_used=False,
            evaluation="Raw source-held-out outputs only; downstream event scoring is separate"))
        status.update(state="completed",exit_code=0)
    except Exception:
        status.update(state="failed",exit_code=1,error=traceback.format_exc())
        raise
    finally:
        status.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-started)
        save(args.output / "run_status.json",status)
        (args.output / "exit_code.txt").write_text(str(status["exit_code"])+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
