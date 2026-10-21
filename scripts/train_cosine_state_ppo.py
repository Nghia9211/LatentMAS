#!/usr/bin/env python3
"""Fresh LIVE PPO with h_k, checkpoint10 cosine distance and k/80; no grid training."""
import argparse,fcntl,importlib.util,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch,yaml
from safetensors.torch import load_file
from controller.cosine_state import SCHEMA,CosineObservationNormalizer,select_profile_features,live_adapters
from controller.expanded_dataset import load_expanded_dataset
from controller.profile import file_hash,write_json,write_tensors,atomic_exclusive_bytes
from controller.server_training import current_sources
from scripts.migrate_b_seed_repeats import contained,verify_hashes
from scripts.train_expanded_controller import approved_items
from scripts.run_matched_controller_ppo import completed,new_attempt,commit
def settings(path):
 value=yaml.safe_load(Path(path).read_text())
 keys={"schema","baseline_training","profile","out","job","iterations","validation_every","comparison_iteration","wandb_mode","wandb_project","wandb_entity","wandb_name"}
 if set(value)!=keys or value["schema"]!=SCHEMA:raise ValueError("Unexpected cosine-state settings")
 if value["iterations"]!=64 or value["validation_every"]!=16 or value["comparison_iteration"]!=32:raise ValueError("Require authorized64 iterations and frozen validation schedule")
 if value["wandb_mode"] not in ["disabled","offline","online"]:raise ValueError("Invalid tracking mode")
 return value
def protocol(parent,s):
 cfg=dict(parent["config"])
 if cfg["warmstart_epochs"]!=0 or cfg["tie_weight"]!=0 or cfg["optimization_epochs"]!=4 or cfg["minibatch_questions"]!=8 or cfg["learning_rate"]!=3e-4 or cfg["time_penalty"]!=0 or cfg["entropy_weight"]!=.01:
  raise ValueError("Require the original no-warmstart/no-tie online PPO baseline")
 cfg.update(iterations=s["iterations"],controller_normalization=SCHEMA)
 checkpoints=sorted({0,s["iterations"],s["comparison_iteration"],40}|set(range(s["validation_every"],s["iterations"]+1,s["validation_every"])))
 return cfg,checkpoints
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--config",type=Path,required=True);ap.add_argument("--stage",choices=["prepare","train"],default="prepare")
 ap.add_argument("--dry-run",action="store_true");cli=ap.parse_args()
 config=contained(cli.config,ROOT);s=settings(config)
 profile,donor,out=[contained(ROOT/s[k],ROOT) for k in ["profile","baseline_training","out"]]
 job=contained(s["job"],Path("/data/nghialt"));runner=contained(job/"online_training.py",job)
 if out in [ROOT,profile,donor] or profile in out.parents or donor in out.parents:raise ValueError("Separate new output required")
 parent=json.loads((donor/"manifest.json").read_text());finished=json.loads((donor/"training_finished.json").read_text())
 if finished["status"]!="complete" or finished["iterations"]!=200:raise ValueError("Require complete simple-state200 baseline")
 verify_hashes(donor/"source_snapshot",parent["source_sha256"])
 cfg,checkpoints=protocol(parent,s)
 with (profile/"pipeline.lock").open("r") as lock:
  fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
  raw,dm=load_expanded_dataset(profile,"train");valid,vm=load_expanded_dataset(profile,"validation")
  if dm["file_sha256"]!=parent["train_dataset_sha256"] or vm["file_sha256"]!=parent["validation_dataset_sha256"]:raise ValueError("Baseline data changed")
  plan=json.loads((profile/"plan.json").read_text());p=contained(plan["parent_profile"],ROOT)
  pm=json.loads((p/"manifest.json").read_text())
  items=approved_items(plan,pm,"train");vitems=approved_items(plan,pm,"validation")
  ids=[i["source_id"] for i in items];vids=[i["source_id"] for i in vitems]
  if ids!=parent["train_ids"] or vids!=parent["validation_ids"] or set(ids)&set(vids):raise ValueError("Identity mismatch")
  # Profile outcomes/targets are deliberately omitted: PPO only receives NEW live reward.
  d={"features":select_profile_features(raw["features"])}
  sources=current_sources(ROOT,config)
  job_sources={name:file_hash(contained(job/name,job)) for name in ["online_training.py","runtime_source_binding.json"]}
  manifest=dict(parent,schema="cosine_state_live_ppo_v1",config=cfg,source_sha256=sources,
   validation_checkpoints=checkpoints,expected_training_B_completions=s["iterations"]*32,expected_validation_B_completions=len(checkpoints)*128,
   normalization_fit_rows=1024*7,normalization_fit_split="train",job=str(job),job_source_sha256=job_sources,
   observation=dict(schema=SCHEMA,input_dim=2562,hidden_dim=2560,groups=["h_k","checkpoint10_cosine_distance","k/80"],compression=False),
   baseline=dict(training=str(donor),manifest_sha256=file_hash(donor/"manifest.json"),changed_config_keys=["iterations","controller_normalization"]),
   learning=dict(kind="online_PPO",reward_source="new A rollout and new B generation at policy-selected K only",grid_objective_used=False,
    offline_controller_loaded=False,profile_labels_used_for_updates=False,optimization_epochs_per_rollout=4),
   protocol="Fresh random cosine-state MLP128; no warm-start, no tie auxiliary; live A/B rollout32 then unchanged PPO16 updates",
   warmstart_skipped=True,old_weights_reused=False,PPO_weights_resumed=False,
   wandb={k:s["wandb_"+k] for k in ["mode","project","entity","name"]})
  summary=dict(status="dry_run" if cli.dry_run else "prepared",output=str(out),iterations=s["iterations"],input_dim=2562,
   train_B=s["iterations"]*32,validation_B=len(checkpoints)*128,validation_checkpoints=checkpoints,optimizer_updates=s["iterations"]*16,
   grid_objective_used=False,old_controller_loaded=False,CUDA_initialized=torch.cuda.is_initialized())
  if cli.dry_run:print(json.dumps(summary));return
  if out.exists():
   if json.loads((out/"manifest.json").read_text())!=manifest:raise ValueError("Output immutable protocol differs")
  else:
   out.mkdir();write_json(out/"manifest.json",manifest)
   for name in sources:atomic_exclusive_bytes(out/"source_snapshot"/name,contained(ROOT/name,ROOT).read_bytes())
   for name in job_sources:atomic_exclusive_bytes(out/"source_snapshot/task_runner"/name,(job/name).read_bytes())
  with (out/"pipeline.lock").open("a") as output_lock:
   fcntl.flock(output_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_hashes(ROOT,sources);verify_hashes(job,job_sources)
   norm=CosineObservationNormalizer(2562,cfg["normalization_eps"]);base=out/"normalization";path=completed(base)
   if path is None:
    path=new_attempt(base);norm.fit(d["features"],source_split="train")
    write_tensors(path/"normalizer.safetensors",norm.state_dict())
    write_json(path/"metadata.json",dict(schema=SCHEMA,input_dim=2562,source_split="train",train_questions=1024,observations=7168,
     dataset_sha256=dm["file_sha256"],fitted_without_labels=True,raw_hidden_not_projected=True))
    commit(base,path)
   else:norm.load_state_dict(load_file(str(path/"normalizer.safetensors")))
   if cli.stage=="prepare":print(json.dumps(summary));return
   if s["wandb_mode"]=="online":
    from dotenv import load_dotenv
    load_dotenv(contained(ROOT/".env",ROOT),override=False)
    if not os.environ.get("WANDB_API_KEY"):raise ValueError("Configured online tracking credential unavailable")
   spec=importlib.util.spec_from_file_location("simple_online_training",runner)
   module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   baselines={str(k):dict(accuracy=float(valid["correctness"][:,i].mean())) for i,k in enumerate(range(10,81,10))}
   with live_adapters():module.train(out,manifest,d,norm,dict(summaries=dict(validation=baselines)),train_items=items,validation_items=vitems)
if __name__=="__main__":main()
