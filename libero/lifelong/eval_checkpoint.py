"""
Evaluate a saved multitask checkpoint on all tasks, producing the same
evaluate_loss + evaluate_success output as main.py's post-training eval.

Usage:
    PYTHONPATH=. python libero/lifelong/eval_checkpoint.py \
        --checkpoint experiments/libero_spatial/Multitask/BCRNNPolicy_seed1/run_006/multitask_model_ep50.pth \
        --device_id 0 \
        --save_result
"""
import argparse
import json
import multiprocessing
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import yaml
from easydict import EasyDict

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.algos import get_algo_class
from libero.lifelong.datasets import get_dataset, SequenceVLDataset
from libero.lifelong.metric import evaluate_loss, evaluate_success
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    get_task_embs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a multitask checkpoint")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the .pth checkpoint file",
    )
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--save_result", action="store_true",
        help="Save result.pt next to the checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = f"cuda:{args.device_id}"

    sd, cfg, previous_mask = torch_load_model(args.checkpoint, map_location=device)

    cfg.folder = cfg.get("folder") or get_libero_path("datasets")
    cfg.bddl_folder = cfg.get("bddl_folder") or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.get("init_states_folder") or get_libero_path("init_states")
    cfg.device = device

    control_seed(cfg.seed)

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_tasks = benchmark.n_tasks

    # load datasets
    manip_datasets = []
    descriptions = []
    shape_meta = None
    for i in range(n_tasks):
        try:
            ds, shape_meta = get_dataset(
                dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(i)),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
            )
        except Exception as e:
            print(f"[error] failed to load task {i}: {e}")
            sys.exit(1)
        descriptions.append(benchmark.get_task(i).language)
        manip_datasets.append(ds)

    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    datasets = [
        SequenceVLDataset(ds, emb) for ds, emb in zip(manip_datasets, task_embs)
    ]

    # build algo and load weights
    cfg.shape_meta = shape_meta
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(n_tasks, cfg), cfg.device)
    algo.policy.load_state_dict(sd)
    algo.eval()

    print(f"[info] loaded checkpoint: {args.checkpoint}")
    print(f"[info] benchmark: {cfg.benchmark_name}, {n_tasks} tasks\n")

    # evaluate loss
    print("[info] evaluating loss ...")
    L = evaluate_loss(cfg, algo, benchmark, datasets)

    # evaluate success
    print("[info] evaluating success rate ...")
    S = evaluate_success(
        cfg=cfg,
        algo=algo,
        benchmark=benchmark,
        task_ids=list(range(n_tasks)),
    )

    print()
    print(("[All task loss ] " + " %4.2f |" * n_tasks) % tuple(L))
    print(("[All task succ.] " + " %4.2f |" * n_tasks) % tuple(S))
    print(f"\n[info] mean loss:  {L.mean():.4f}")
    print(f"[info] mean succ.: {S.mean():.4f}")

    if args.save_result:
        result_path = os.path.join(os.path.dirname(args.checkpoint), "eval_result.pt")
        torch.save({"loss": L, "success": S}, result_path)
        print(f"[info] saved results to {result_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
