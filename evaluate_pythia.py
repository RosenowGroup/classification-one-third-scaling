import argparse
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_CONFIGS = {
    "410m": ("EleutherAI/pythia-410m", "pythia-logit-410m.pt"),
    "2.8b": ("EleutherAI/pythia-2.8b", "pythia-logit-2p8b.pt"),
    "12b": ("EleutherAI/pythia-12b", "pythia-logit-12b.pt"),
}
STEPS = [
    128,
    256,
    512,
    1000,
    2000,
    3000,
    6000,
    12000,
    16000,
    32000,
    48000,
    79000,
    143000,
]
FINEWEB_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
BATCH_SIZE = 12
MAX_LENGTH = 1024
NUM_DOCS = 4800


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Pythia model on FineWeb.")
    parser.add_argument(
        "model",
        type=str.lower,
        choices=MODEL_CONFIGS,
        metavar="{410M|2.8B|12B}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name, output_name = MODEL_CONFIGS[args.model]
    output_path = Path(__file__).resolve().parent / "pythia" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    revisions = [f"step{step}" for step in STEPS]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="default",
        split="train",
        streaming=True,
        revision=FINEWEB_REVISION,
    )

    accumulated_loss = torch.zeros(len(STEPS))
    accumulated_top1_error = torch.zeros(len(STEPS))
    accumulated_top5_error = torch.zeros(len(STEPS))
    accumulated_logit_std = torch.zeros(len(STEPS))
    accumulated_logit_mean = torch.zeros(len(STEPS))
    accumulated_skewness = torch.zeros(len(STEPS))
    accumulated_skewness_extreme = torch.zeros(len(STEPS))
    accumulated_logit_range = torch.zeros(len(STEPS))
    total_tokens = torch.zeros(len(STEPS))

    for rev_idx, revision in enumerate(revisions):
        data_iter = iter(dataset)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            dtype=torch.float16,
            device_map="auto",
        ).eval()
        start_time = time.time()

        for _ in range(NUM_DOCS // BATCH_SIZE):
            batch = [next(data_iter)["text"] for _ in range(BATCH_SIZE)]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
            )
            labels = inputs.input_ids.clone()
            labels[inputs.attention_mask == 0] = -100
            valid_mask = (inputs.attention_mask[:, 1:] > 0) & (
                inputs.attention_mask[:, :-1] > 0
            )

            with torch.no_grad():
                output = model(
                    **inputs.to(model.device),
                    labels=labels.to(model.device),
                )

            valid_mask = valid_mask.to(output.logits.device)
            logits = output.logits[:, :-1][valid_mask].to(torch.float32)
            valid_labels = labels[:, 1:].to(output.logits.device)[valid_mask]
            correct_logits = logits.gather(
                1, valid_labels.unsqueeze(-1)
            ).squeeze(-1)
            top5_ids = logits.topk(5, dim=-1).indices
            logit_means = logits.mean(dim=-1)
            logit_maxes = logits.max(dim=-1).values
            logit_mins = logits.min(dim=-1).values

            num_tokens = logits.shape[0]
            total_tokens[rev_idx] += num_tokens
            accumulated_loss[rev_idx] += output.loss.float().item() * num_tokens
            accumulated_top1_error[rev_idx] += (
                top5_ids[:, 0] != valid_labels
            ).sum().item()
            accumulated_top5_error[rev_idx] += (
                (top5_ids != valid_labels.unsqueeze(-1)).all(dim=-1).sum().item()
            )
            accumulated_logit_std[rev_idx] += (
                logits.std(dim=-1).sum().item() / MAX_LENGTH
            )
            accumulated_logit_mean[rev_idx] += (
                logit_means.sum().item() / MAX_LENGTH
            )
            accumulated_skewness[rev_idx] += (
                (correct_logits - logit_means).sum().item() / MAX_LENGTH
            )
            accumulated_skewness_extreme[rev_idx] += (
                (logit_maxes - logit_means).sum().item() / MAX_LENGTH
            )
            accumulated_logit_range[rev_idx] += (
                (logit_maxes - logit_mins).sum().item() / MAX_LENGTH
            )

        del model
        print(
            f"Revision {revision}, Time for {NUM_DOCS} docs: "
            f"{time.time() - start_time:.2f} seconds",
            flush=True,
        )

    torch.save(
        {
            "model": model_name,
            "steps": torch.tensor(STEPS),
            "completed_steps": list(STEPS),
            "loss": accumulated_loss / total_tokens,
            "top1_error": accumulated_top1_error / total_tokens,
            "top5_error": accumulated_top5_error / total_tokens,
            "logit_std": accumulated_logit_std / total_tokens * MAX_LENGTH,
            "logit_mean": accumulated_logit_mean / total_tokens * MAX_LENGTH,
            "skewness": accumulated_skewness / total_tokens * MAX_LENGTH,
            "skewness_extreme": (
                accumulated_skewness_extreme / total_tokens * MAX_LENGTH
            ),
            "logit_range": accumulated_logit_range / total_tokens * MAX_LENGTH,
            "total_tokens": total_tokens,
        },
        output_path,
    )
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
