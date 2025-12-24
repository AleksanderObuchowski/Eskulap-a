import os

import numpy as np
from config import (
    admed_anoni_dataset,
    admed_human_dataset,
    gemini_dataset,
    prepared_data_dir,
    youtube_dataset,
)
from datasets import load_from_disk


def compute_duration(batch):
    durations = []
    # Check if we should use "path" or "audio" column
    audio_col = "path" if "path" in batch and batch["path"][0] is not None else "audio"

    for audio in batch[audio_col]:
        if audio is None:
            durations.append(0.0)
        # Handle AudioDecoder objects from datasets library
        elif hasattr(audio, "metadata") and hasattr(
            audio.metadata, "duration_seconds_from_header"
        ):
            duration = audio.metadata.duration_seconds_from_header
            durations.append(duration if duration is not None else 0.0)
        # Handle dict-based audio data
        elif isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            durations.append(len(audio["array"]) / audio["sampling_rate"])
        else:
            durations.append(0.0)
    return {"duration": durations}


def get_split_stats(dataset, split_name):
    print(f"Processing split: {split_name} ({len(dataset)} examples)...")

    try:
        ds_with_dur = dataset.map(
            compute_duration,
            batched=True,
            batch_size=100,
            num_proc=8,
            desc=f"Calculating stats for {split_name}",
            remove_columns=dataset.column_names,
        )
    except Exception as e:
        print(f"Error processing {split_name}: {e}")
        return None

    durations = np.array(ds_with_dur["duration"])
    valid_durations = durations[durations > 0]
    count = len(valid_durations)

    if count == 0:
        return None

    total_seconds = np.sum(valid_durations)
    avg_seconds = np.mean(valid_durations)
    min_seconds = np.min(valid_durations)
    max_seconds = np.max(valid_durations)

    return {
        "split": split_name,
        "count": count,
        "total_hours": total_seconds / 3600,
        "avg_sec": avg_seconds,
        "min_sec": min_seconds,
        "max_sec": max_seconds,
    }


def get_dataset_durations(dataset, source_name):
    """Calculate total duration for a source dataset."""
    print(f"  Computing durations for {source_name}...")
    try:
        ds_with_dur = dataset.map(
            compute_duration,
            batched=True,
            batch_size=100,
            num_proc=8,
            desc=f"Duration for {source_name}",
        )
        durations = np.array(ds_with_dur["duration"])
        valid_durations = durations[durations > 0]
        return np.sum(valid_durations) / 3600  # hours
    except Exception as e:
        print(f"    Error: {e}")
        return 0.0


def print_contribution_table(title, contributions, show_estimate=False):
    """Print a pretty table showing dataset contributions."""
    print(f"\n{title}")
    header = "╔═══════════════════╦═══════════════╦══════════════╦═══════════════╦══════════════╗"
    separator = "╠═══════════════════╬═══════════════╬══════════════╬═══════════════╬══════════════╣"
    footer = "╚═══════════════════╩═══════════════╩══════════════╩═══════════════╩══════════════╝"

    estimate_marker = "~" if show_estimate else ""

    print(header)
    print(
        f"║ {'Dataset':<17} ║ {'Examples':>13} ║ {'%':>12} ║ {'Hours':>13} ║ {'%':>12} ║"
    )
    print(separator)

    total_examples = sum(c["examples"] for c in contributions)
    total_hours = sum(c["hours"] for c in contributions)

    for contrib in contributions:
        examples = contrib["examples"]
        hours = contrib["hours"]
        ex_pct = (examples / total_examples * 100) if total_examples > 0 else 0
        hr_pct = (hours / total_hours * 100) if total_hours > 0 else 0

        print(
            f"║ {contrib['name']:<17} ║ {estimate_marker}{examples:>12,} ║ {ex_pct:>11.1f}% ║ {estimate_marker}{hours:>12.1f} ║ {hr_pct:>11.1f}% ║"
        )

    print(separator)
    print(
        f"║ {'TOTAL':<17} ║ {estimate_marker}{total_examples:>12,} ║ {100.0:>11.1f}% ║ {estimate_marker}{total_hours:>12.1f} ║ {100.0:>11.1f}% ║"
    )
    print(footer)


def get_source_contributions():
    """Analyze how each source dataset contributes to the final prepared dataset."""
    print("\n" + "=" * 100)
    print("DATASET CONTRIBUTION ANALYSIS")
    print("=" * 100)

    # Get original dataset sizes and durations
    print("\nCalculating source dataset statistics...")
    sources = {
        "youtube": {"examples": len(youtube_dataset), "dataset": youtube_dataset},
        "gemini": {"examples": len(gemini_dataset), "dataset": gemini_dataset},
        "admed_anoni": {
            "examples": len(admed_anoni_dataset),
            "dataset": admed_anoni_dataset,
        },
        "admed_human": {
            "examples": len(admed_human_dataset),
            "dataset": admed_human_dataset,
        },
    }

    # Calculate durations for original datasets
    for name, info in sources.items():
        info["hours"] = get_dataset_durations(info["dataset"], name)

    # Show original dataset sizes
    original_contribs = [
        {"name": name, "examples": info["examples"], "hours": info["hours"]}
        for name, info in sources.items()
    ]
    print_contribution_table("ORIGINAL SOURCE DATASETS", original_contribs)

    # Calculate contributions based on prepare_data.py logic
    # Test split: 3k from admed_anoni + 2k from admed_human = 5k total
    test_ex = {"admed_anoni": 3000, "admed_human": 2000}

    # Estimate test hours proportionally
    test_hrs = {
        "admed_anoni": sources["admed_anoni"]["hours"]
        * (3000 / sources["admed_anoni"]["examples"]),
        "admed_human": sources["admed_human"]["hours"]
        * (2000 / sources["admed_human"]["examples"]),
    }

    # Remaining for training pool
    train_pool_ex = {
        "youtube": sources["youtube"]["examples"],
        "gemini": sources["gemini"]["examples"],
        "admed_anoni": sources["admed_anoni"]["examples"] - 3000,
        "admed_human": sources["admed_human"]["examples"] - 2000,
    }

    train_pool_hrs = {
        "youtube": sources["youtube"]["hours"],
        "gemini": sources["gemini"]["hours"],
        "admed_anoni": sources["admed_anoni"]["hours"] - test_hrs["admed_anoni"],
        "admed_human": sources["admed_human"]["hours"] - test_hrs["admed_human"],
    }

    # Show training pool contributions
    train_pool_contribs = [
        {"name": name, "examples": train_pool_ex[name], "hours": train_pool_hrs[name]}
        for name in ["youtube", "gemini", "admed_anoni", "admed_human"]
    ]
    print_contribution_table(
        "TRAINING POOL (before filtering & train/dev split)",
        train_pool_contribs,
        show_estimate=True,
    )

    # Show test pool contributions
    test_pool_contribs = [
        {"name": name, "examples": test_ex[name], "hours": test_hrs[name]}
        for name in ["admed_anoni", "admed_human"]
    ]
    print_contribution_table(
        "TEST POOL (before filtering)", test_pool_contribs, show_estimate=True
    )

    # Load actual prepared dataset to get real numbers after filtering
    if os.path.exists(prepared_data_dir):
        print("\n" + "=" * 100)
        print("FINAL PREPARED DATASET (after filtering & splitting)")
        print("=" * 100)

        dataset_dict = load_from_disk(prepared_data_dir)

        # Calculate actual durations for each split
        print("\nCalculating actual split durations...")
        train_hours = get_dataset_durations(dataset_dict["train"], "train split")
        dev_hours = get_dataset_durations(dataset_dict["dev"], "dev split")
        test_hours = get_dataset_durations(dataset_dict["test"], "test split")

        train_size = len(dataset_dict["train"])
        dev_size = len(dataset_dict["dev"])
        test_size = len(dataset_dict["test"])
        total_after_filter = train_size + dev_size

        total_train_pool = sum(train_pool_ex.values())
        total_train_pool_hrs = sum(train_pool_hrs.values())

        # Estimate contributions to train split (proportional to original contributions)
        train_contribs = []
        for name in ["youtube", "gemini", "admed_anoni", "admed_human"]:
            est_examples = int(
                train_pool_ex[name] / total_train_pool * total_after_filter * 0.9
            )
            est_hours = (
                train_pool_hrs[name]
                / total_train_pool_hrs
                * (train_hours + dev_hours)
                * 0.9
            )
            train_contribs.append(
                {"name": name, "examples": est_examples, "hours": est_hours}
            )

        print_contribution_table(
            f"TRAIN SPLIT (90% of pool, {train_size:,} examples total)",
            train_contribs,
            show_estimate=True,
        )

        # Estimate contributions to dev split
        dev_contribs = []
        for name in ["youtube", "gemini", "admed_anoni", "admed_human"]:
            est_examples = int(
                train_pool_ex[name] / total_train_pool * total_after_filter * 0.1
            )
            est_hours = (
                train_pool_hrs[name]
                / total_train_pool_hrs
                * (train_hours + dev_hours)
                * 0.1
            )
            dev_contribs.append(
                {"name": name, "examples": est_examples, "hours": est_hours}
            )

        print_contribution_table(
            f"DEV SPLIT (10% of pool, {dev_size:,} examples total)",
            dev_contribs,
            show_estimate=True,
        )

        # Test contributions (actual after filtering)
        test_retention_ex = test_size / 5000
        test_contribs = [
            {
                "name": "admed_anoni",
                "examples": int(3000 * test_retention_ex),
                "hours": test_hrs["admed_anoni"] * test_retention_ex,
            },
            {
                "name": "admed_human",
                "examples": int(2000 * test_retention_ex),
                "hours": test_hrs["admed_human"] * test_retention_ex,
            },
        ]
        print_contribution_table(
            f"TEST SPLIT ({test_size:,} examples total, {test_size/5000*100:.1f}% retention)",
            test_contribs,
            show_estimate=True,
        )

        # Summary statistics
        print("\n" + "=" * 100)
        print("FILTERING STATISTICS")
        print("=" * 100)
        print(
            f"\nTraining Pool Retention: {total_after_filter:,} / {total_train_pool:,} examples ({total_after_filter/total_train_pool*100:.1f}%)"
        )
        print(
            f"Test Pool Retention:     {test_size:,} / 5,000 examples ({test_size/5000*100:.1f}%)"
        )

    print("\n" + "=" * 100 + "\n")


def print_table(stats_list):
    if not stats_list:
        print("No statistics to display.")
        return

    # Define headers and format string
    headers = ["Split", "Examples", "Total Hours", "Avg (s)", "Min (s)", "Max (s)"]
    # Adjust widths as needed
    row_fmt = "{:<10} | {:>10} | {:>12} | {:>10} | {:>10} | {:>10}"

    print("\n" + "=" * 75)
    print(row_fmt.format(*headers))
    print("-" * 75)

    for row in stats_list:
        print(
            row_fmt.format(
                row["split"],
                f"{row['count']:,}",
                f"{row['total_hours']:.2f}",
                f"{row['avg_sec']:.2f}",
                f"{row['min_sec']:.2f}",
                f"{row['max_sec']:.2f}",
            )
        )
    print("=" * 75 + "\n")


def main():
    # First, show dataset contributions analysis
    try:
        get_source_contributions()
    except Exception as e:
        print(f"Error analyzing contributions: {e}\n")

    if not os.path.exists(prepared_data_dir):
        print(f"Error: Dataset directory '{prepared_data_dir}' does not exist.")
        print("Please run 'prepare_data.py' first to generate the dataset.")
        return

    print(f"Loading dataset from: {prepared_data_dir}")
    try:
        dataset_dict = load_from_disk(prepared_data_dir)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    stats_results = []
    # Process splits in a specific order if present, else just keys
    order = ["train", "dev", "test"]
    sorted_keys = sorted(
        dataset_dict.keys(), key=lambda k: order.index(k) if k in order else 999
    )

    for split in sorted_keys:
        stats = get_split_stats(dataset_dict[split], split)
        if stats:
            stats_results.append(stats)

    print_table(stats_results)


if __name__ == "__main__":
    main()
