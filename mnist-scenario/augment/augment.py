import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="input")
    ap.add_argument("--out-dir", type=str, default="output")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-samples", type=int, default=64, help="How many base MNIST samples to visualize")
    ap.add_argument("--num-aug", type=int, default=4, help="How many augmented versions per sample")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Augmentation ONLY (no model/training).
    # Keep it mild so digits remain recognizable.
    aug_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
            ),
            transforms.ToTensor(),
        ]
    )

    base_transform = transforms.Compose([transforms.ToTensor()])

    # Load MNIST twice: once without aug (for "original"), once with aug pipeline
    base_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=base_transform)
    aug_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=aug_transform)

    n = min(args.num_samples, len(base_ds))
    idxs = list(range(n))
    base_subset = Subset(base_ds, idxs)
    aug_subset = Subset(aug_ds, idxs)

    base_loader = DataLoader(base_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    aug_loader = DataLoader(aug_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Save a grid of originals
    originals = []
    labels = []
    for x, y in base_loader:
        originals.append(x)
        labels.append(y)
    originals = torch.cat(originals, dim=0)[:n]
    labels = torch.cat(labels, dim=0)[:n]

    grid_orig = make_grid(originals, nrow=8, padding=2)  # 8x8 for 64 samples
    save_image(grid_orig, out_dir / "mnist_originals.png")

    # Save multiple grids of augmented versions
    for k in range(1, args.num_aug + 1):
        augmented = []
        for x, _ in tqdm(aug_loader, desc=f"augment pass {k}", leave=False):
            augmented.append(x)
        augmented = torch.cat(augmented, dim=0)[:n]

        grid_aug = make_grid(augmented, nrow=8, padding=2)
        save_image(grid_aug, out_dir / f"mnist_augmented_{k}.png")

    # Optional: save a single "comparison" grid: original + first aug per sample (2 rows per sample)
    # This creates a tall image: for each sample, [original, aug1] stacked; then arranged in columns.
    # Keep it simple by just saving the first 16 comparisons.
    comp_n = min(16, n)
    aug1 = []
    # regenerate one aug pass for comparison (ensures it's consistent with transform)
    for x, _ in DataLoader(Subset(aug_ds, list(range(comp_n))), batch_size=comp_n, shuffle=False, num_workers=2):
        aug1 = x
        break

    comp = torch.cat([originals[:comp_n], aug1], dim=0)  # 32 images: first 16 originals then 16 aug
    grid_comp = make_grid(comp, nrow=16, padding=2)
    save_image(grid_comp, out_dir / "mnist_compare_original_vs_aug1.png")

    print(f"Wrote outputs to: {out_dir.resolve()}")
    print("Files:")
    print("- mnist_originals.png")
    for k in range(1, args.num_aug + 1):
        print(f"- mnist_augmented_{k}.png")
    print("- mnist_compare_original_vs_aug1.png")


if __name__ == "__main__":
    from torchvision import datasets
    datasets.MNIST(root="./data", train=True, download=True)
    datasets.MNIST(root="./data", train=False, download=True)
    # main()