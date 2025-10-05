"""
Script to generate adversarial examples and extract representations.

Usage:
    python scripts/generate_adversarial.py --folder /path/to/model --recreate
"""

import argparse
import os

from dotenv import load_dotenv

from nadf.data.adversarial import load_or_create_dataset

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial dataset")
    parser.add_argument(
        "--model-folder", type=str, default=os.getenv("RESNET_MODEL_FOLDER"), help="Path to pre-trained model folder"
    )
    parser.add_argument("--target-class", type=int, default=-1, help="Target class for filtering (-1 for all classes)")
    parser.add_argument(
        "--attacks",
        type=str,
        default="4,0.25;2,0.5;3,1;1,2",
        help="Attack configs as 'num_attacks,eps_coef' separated by ';' (e.g., '4,0.25;2,0.5')",
    )
    parser.add_argument(
        "--splits", type=str, default="train,val,test", help="Dataset splits to process (comma-separated)"
    )
    parser.add_argument("--recreate", action="store_true", help="Force recreation even if cached version exists")

    args = parser.parse_args()

    # Parse attack configurations
    num_attacks_eps_coef = []
    for attack_str in args.attacks.split(";"):
        num, coef = attack_str.split(",")
        num_attacks_eps_coef.append((int(num), float(coef)))

    # Parse splits
    splits = args.splits.split(",")

    print("=" * 60)
    print("Adversarial Dataset Generation")
    print("=" * 60)
    print(f"Model folder: {args.model_folder}")
    print(f"Target class: {args.target_class}")
    print(f"Attack configs: {num_attacks_eps_coef}")
    print(f"Splits: {splits}")
    print(f"Recreate: {args.recreate}")
    print("=" * 60)

    # Generate dataset
    dataset = load_or_create_dataset(
        folder=args.model_folder,
        target_class=args.target_class,
        num_attacks_eps_coef=num_attacks_eps_coef,
        splits=splits,
        recreate=args.recreate,
    )

    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    for split in splits:
        print(f"\n{split.upper()}:")
        print(f"  Clean examples: {len(dataset['z_clean'][split])}")
        print(f"  Adversarial examples: {len(dataset['z_adv'][split])}")
        print(f"  Total examples: {len(dataset['z'][split])}")
        print(f"  Representation dim: {dataset['z'][split].shape[1] if len(dataset['z'][split]) > 0 else 'N/A'}")


if __name__ == "__main__":
    main()
