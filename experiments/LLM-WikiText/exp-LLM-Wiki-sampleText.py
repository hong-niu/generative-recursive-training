"""
Generate qualitative GPT-2 samples from recursive-training checkpoints.

This script loads GPT-2 models saved at selected outer iterations of a
recursive real-synthetic training run and generates continuations from a
user-specified prompt. It is intended for comparing how sample quality,
diversity, repetition, and possible collapse behavior evolve across iterations.

Iterations can be selected with --iters or --iters-str, for example:
    --iters 1 5 10
    --iters-str "1-5,10,20-25"

The script can print samples to stdout, save one file per iteration, or collect
all requested iterations into a single file with --single-file. The --trim-prompt
option saves only the generated continuation rather than the prompt plus
continuation.

Example:
    exp-LLM-Wiki-sampleText.py \
    --run-dir runs/llm_recursive_mix \
    --prompt "The history of science" \
    --iters-str "1-5,10,20,50" \
    --save --single-file --trim-prompt
"""


import argparse
import re
from pathlib import Path
from typing import List, Sequence
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ------------------------------------------------------------
# Iteration parsing
# ------------------------------------------------------------
def parse_iters_arg(iters: Sequence[int], iters_str: str | None) -> List[int]:
    """
    Support either:
      --iters 1 3 10
    or
      --iters-str "1-5,8,10-12"
    """
    out: List[int] = []

    if iters:
        out.extend([int(x) for x in iters])

    if iters_str:
        parts = [p.strip() for p in iters_str.split(",") if p.strip()]
        for p in parts:
            m = re.fullmatch(r"(\d+)-(\d+)", p)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                step = 1 if b >= a else -1
                out.extend(list(range(a, b + step, step)))
            else:
                out.append(int(p))

    # De-duplicate while preserving order
    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            dedup.append(x)
            seen.add(x)

    return dedup


# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------
@torch.no_grad()
def generate_samples(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    n_samples: int,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int | None,
    trim_prompt: bool,
):
    # set seeds
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attn = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # Repeat prompt for N samples
    input_ids = input_ids.repeat(n_samples, 1)
    attn = attn.repeat(n_samples, 1)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        do_sample=True,
        max_new_tokens=max_new_tokens,  # length cap on new tokens
        min_new_tokens=min_new_tokens if min_new_tokens > 0 else None,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=False,
    )

    if trim_prompt:
        prompt_len = input_ids.shape[1]
        out = out[:, prompt_len:]

    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return texts


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    # Required
    ap.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory (e.g. runs/llm_recursive_mix)",
    )
    ap.add_argument("--prompt", type=str, required=True)

    # Iter selection
    ap.add_argument("--iters", type=int, nargs="*", default=[], help="Example: --iters 1 2 10")
    ap.add_argument("--iters-str", type=str, default=None, help='Example: --iters-str "1-5,10,20-22"')

    # Sampling control
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--min-new-tokens", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)

    # Device
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)

    # Output control
    ap.add_argument("--save", action="store_true", help="If set, save samples to disk")
    ap.add_argument("--no-print", action="store_true", help="If set, do not print samples to stdout")
    ap.add_argument(
        "--out-subdir",
        type=str,
        default="samples_across_iters",
        help="Subdirectory name inside run-dir (only used if --save)",
    )

    # NEW: optional trimming
    ap.add_argument("--trim-prompt", action="store_true", help="If set, remove the input prompt from output")

    # NEW: write all iters to ONE file
    ap.add_argument(
        "--single-file",
        action="store_true",
        help="If set (and --save), write ALL iterations into a single text file",
    )
    ap.add_argument(
        "--single-file-name",
        type=str,
        default="all_iters_samples.txt",
        help="Name of the single output file (only used if --single-file and --save)",
    )

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    iters = parse_iters_arg(args.iters, args.iters_str)
    if not iters:
        raise ValueError("No iterations specified. Use --iters or --iters-str.")

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    out_dir = None
    single_fpath = None
    single_fp = None

    # Prepare output directory and (optionally) open a single file once
    if args.save:
        out_dir = run_dir / args.out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.single_file:
            single_fpath = out_dir / args.single_file_name
            single_fp = open(single_fpath, "w", encoding="utf-8")

            # Header once
            single_fp.write("GPT-2 sampling across iterations\n")
            single_fp.write(f"RUN_DIR: {run_dir.resolve()}\n")
            single_fp.write(f"ITERS: {iters}\n")
            single_fp.write("\nPROMPT:\n")
            single_fp.write(args.prompt)
            single_fp.write("\n\n" + "=" * 100 + "\n")
            single_fp.flush()

    try:
        for t in iters:
            iter_model_dir = run_dir / f"iter_{t:03d}" / "model"
            if not iter_model_dir.exists():
                print(f"[skip] iter {t:03d}: model directory missing")
                continue

            print(f"\n=== ITER {t:03d} ===")

            tokenizer = AutoTokenizer.from_pretrained(iter_model_dir, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                iter_model_dir,
                torch_dtype=(torch.float16 if (args.fp16 and device.type == "cuda") else None),
            ).to(device)

            texts = generate_samples(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=args.prompt,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                # don't vary the seed per iteration to see evolution 
                seed=args.seed, #  + t,  # vary seed per iteration
                trim_prompt=args.trim_prompt,
            )

            # -----------------------------
            # Save
            # -----------------------------
            if args.save:
                if args.single_file:
                    assert single_fp is not None
                    single_fp.write(f"\n\n{'=' * 100}\n")
                    single_fp.write(f"ITER {t:03d}\n")
                    single_fp.write(f"{'=' * 100}\n")
                    single_fp.write(
                        f"sampling: n_samples={args.n_samples} max_new_tokens={args.max_new_tokens} "
                        f"min_new_tokens={args.min_new_tokens} temp={args.temperature} top_p={args.top_p} "
                        f"repetition_penalty={args.repetition_penalty} trim_prompt={args.trim_prompt} seed={args.seed + t}\n"
                    )

                    for i, txt in enumerate(texts, 1):
                        single_fp.write(f"\n--- SAMPLE {i:02d} ---\n")
                        single_fp.write(txt)
                        if not txt.endswith("\n"):
                            single_fp.write("\n")

                    single_fp.flush()
                else:
                    # one file per iteration (original behavior)
                    assert out_dir is not None
                    fpath = out_dir / f"iter_{t:03d}_samples.txt"
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(f"ITER={t:03d}\n")
                        f.write(f"PROMPT:\n{args.prompt}\n")
                        f.write("\n" + "=" * 80 + "\n")
                        f.write(
                            f"sampling: n_samples={args.n_samples} max_new_tokens={args.max_new_tokens} "
                            f"min_new_tokens={args.min_new_tokens} temp={args.temperature} top_p={args.top_p} "
                            f"repetition_penalty={args.repetition_penalty} trim_prompt={args.trim_prompt} seed={args.seed + t}\n"
                        )
                        for i, txt in enumerate(texts, 1):
                            f.write(f"\n--- SAMPLE {i:02d} ---\n")
                            f.write(txt)
                            f.write("\n")
                    print(f"[saved] {fpath}")

            # -----------------------------
            # Print
            # -----------------------------
            if not args.no_print:
                for i, txt in enumerate(texts, 1):
                    print(f"\n--- SAMPLE {i:02d} ---\n{txt}\n")

            # Free VRAM between iterations
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    finally:
        if single_fp is not None:
            single_fp.close()

    if args.save:
        if args.single_file:
            assert single_fpath is not None
            print(f"\nAll outputs written to: {single_fpath.resolve()}")
        else:
            assert out_dir is not None
            print(f"\nAll outputs written under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()