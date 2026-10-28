#!/usr/bin/env python3
"""Evaluate six ImageNet JEM checkpoints on OSCAR; export a static Next.js dataset.

Run in JEM/imagenet/ where wideresnet_vae.py, vae_kl.py and imagenet_dataset.py
are importable. This script does NOT run on Vercel.

The frozen VAE encodes each validation image exactly once; every JEM receives
identical VAE latents. Default: alpha 0.0...0.5, seed 1, best_valid_ckpt.pt.

For public deployment, omit thumbnails unless you have distribution permission
for the ImageNet images. Use --thumbnail-count N only when authorized.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Subset

import wideresnet_vae
from vae_kl import FrozenVAEWrapper
from imagenet_dataset import ImageNetValDataset

DEFAULT_ALPHAS = [f"{i / 10:.1f}" for i in range(6)]


class JEM(nn.Module):
    """Matches the F module from the user's evaluation code, without torch.compile."""

    def __init__(self, *, depth: int, width: int, norm: str | None, latent_dim: int, in_proj: int):
        super().__init__()
        self.f = wideresnet_vae.Wide_ResNet(
            depth, width, norm=norm, dropout_rate=0.0,
            latent_dim=latent_dim, in_proj=in_proj,
        )
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, 1000)
        self.class_dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.class_output(self.f(x))


def get_checkpoint(root: Path, alpha: str, seed: int) -> Path:
    path = root / alpha / str(seed) / ("last_ckpt.pt" if alpha == "1.0" else "best_valid_ckpt.pt")
    if not path.is_file() or not (path.parent / "params.txt").is_file():
        raise FileNotFoundError(f"Missing checkpoint or params.txt for alpha={alpha}: {path}")
    return path


def load_model(path: Path, device: torch.device) -> JEM:
    with (path.parent / "params.txt").open() as file:
        params = json.load(file)
    depth = int(params.get("depth", 28))
    width = int(params.get("width", 2))
    latent_dim = int(params.get("latent_dim", 512))
    in_proj = 31360 if width == 10 else 25088  # Same rule as provided eval.py
    model = JEM(
        depth=depth, width=width,
        norm=params.get("norm"), latent_dim=latent_dim, in_proj=in_proj,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    cleaned = {}
    for name, value in state.items():
        while name.startswith("_orig_mod.") or name.startswith("module."):
            name = name.split(".", 1)[1]
        cleaned[name] = value
    model.load_state_dict(cleaned, strict=True)
    print(f"  Loaded {path} (WideResNet-{depth}, width={width}, head={latent_dim})", flush=True)
    return model.to(device).eval()


def class_names() -> list[str]:
    # Only reads torchvision's package metadata; does not download ResNet weights.
    try:
        from torchvision.models import ResNet50_Weights
        names = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
        if len(names) == 1000:
            return [str(name) for name in names]
    except Exception as exc:
        print(f"  Could not load ImageNet category names: {exc}", flush=True)
    return [f"ImageNet class {i}" for i in range(1000)]


def image_paths(dataset, val_dir: Path) -> list[Path]:
    if not hasattr(dataset, "samples"):
        raise RuntimeError("ImageNetValDataset must expose .samples of (filename, label) pairs")
    base = val_dir.resolve()
    paths = []
    for entry in dataset.samples:
        filename = str(entry[0])
        absolute = (base / filename).resolve()
        if not absolute.is_relative_to(base):
            raise ValueError(f"Image path outside validation directory: {filename}")
        paths.append(absolute)
    return paths


def collect_latents(dataset, device: torch.device, args) -> tuple[torch.Tensor, np.ndarray]:
    loader = DataLoader(
        dataset, batch_size=args.encode_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    vae = FrozenVAEWrapper(
        device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32,
        use_tiling=False,
    )
    vae.eval()
    latents = []
    labels = []
    with torch.inference_mode():
        for step, (images, targets) in enumerate(loader, 1):
            z = vae.encode(images.to(device, non_blocking=True)).float()
            if z.ndim != 4 or z.shape[1] != 4:
                raise RuntimeError(f"Unexpected frozen VAE latent shape: {tuple(z.shape)}")
            latents.append(z.cpu())
            labels.append(targets.numpy().astype(np.int64))
            if step % 50 == 0:
                print(f"  VAE encoded {min(step * args.encode_batch_size, len(dataset)):,} / {len(dataset):,}", flush=True)
    z_all = torch.cat(latents).contiguous()
    y = np.concatenate(labels)
    del vae, latents, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not (np.all(y >= 0) and np.all(y < 1000)):
        raise ValueError("ImageNet labels must be zero-based class indices in [0, 999]")
    print(f"  Frozen VAE latents: {tuple(z_all.shape)}, dtype={z_all.dtype}", flush=True)
    return z_all, y


def evaluate_model(model: JEM, z_all: torch.Tensor, y: np.ndarray, device: torch.device, batch_size: int):
    n = len(y)
    predictions = np.empty(n, dtype=np.int16)
    confidence = np.empty(n, dtype=np.float32)
    rank = np.empty(n, dtype=np.int16)
    p_true = np.empty(n, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            images = z_all[start:end].to(device, non_blocking=True)
            logits = model(images).float()
            if logits.shape != (end - start, 1000):
                raise RuntimeError(f"Expected logits [batch,1000], got {tuple(logits.shape)}. Check the unprojected WideResNet feature dimension.")
            target = torch.as_tensor(y[start:end], device=device, dtype=torch.long)
            pred = logits.argmax(dim=1)
            probabilities = torch.softmax(logits, dim=1)
            row_idx = torch.arange(len(target), device=device)
            gt_logit = logits[row_idx, target].unsqueeze(1)
            # Rank based on descending logits; ties are rare, deterministic by lower class index.
            class_idx = torch.arange(1000, device=device).unsqueeze(0)
            ranks = 1 + (logits > gt_logit).sum(dim=1) + ((logits == gt_logit) & (class_idx < target[:, None])).sum(dim=1)
            predictions[start:end] = pred.cpu().numpy()
            confidence[start:end] = probabilities[row_idx, pred].cpu().numpy()
            rank[start:end] = ranks.cpu().numpy()
            p_true[start:end] = probabilities[row_idx, target].cpu().numpy()
            if end % 10000 < batch_size or end == n:
                print(f"    Inferred {end:,} / {n:,}", flush=True)
    return predictions, confidence, rank, p_true


def select_thumbnail_indices(preds: np.ndarray, y: np.ndarray, count: int, seed: int) -> list[int]:
    """Deterministic, balanced preview: consensus and individual rescues/losses."""
    if count <= 0:
        return []
    rng = random.Random(seed)
    n, n_models = preds.shape
    base_correct = preds[:, 0] == y
    hybrid_correct = preds[:, 1:] == y[:, None]
    multi_rescued = np.where(~base_correct & (hybrid_correct.sum(axis=1) >= 3))[0].tolist()
    pools = [multi_rescued]
    for j in range(1, n_models):
        pools.append(np.where(~base_correct & hybrid_correct[:, j-1])[0].tolist())
        pools.append(np.where(base_correct & ~hybrid_correct[:, j-1])[0].tolist())
    for pool in pools:
        rng.shuffle(pool)
    selected = []
    seen = set()
    while len(selected) < count:
        progress = False
        for pool in pools:
            while pool and pool[-1] in seen:
                pool.pop()
            if pool:
                index = pool.pop()
                selected.append(index)
                seen.add(index)
                progress = True
                if len(selected) >= count:
                    break
        if not progress:
            break
    if len(selected) < count:
        remainder = [i for i in range(n) if i not in seen]
        rng.shuffle(remainder)
        selected.extend(remainder[:count-len(selected)])
    return sorted(selected)


def export_thumbnails(paths: list[Path], dest: Path, indices: list[int], size: int) -> dict[int, str]:
    if not indices:
        return {}
    thumb_dir = dest / "thumbs"
    thumb_dir.mkdir(exist_ok=True, parents=True)
    names = {}
    for n, index in enumerate(indices, 1):
        file_name = f"{index:05d}.webp"
        try:
            with Image.open(paths[index]) as original:
                img = ImageOps.exif_transpose(original).convert("RGB")
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                img.save(thumb_dir / file_name, format="WEBP", quality=78, method=5)
            names[index] = f"thumbs/{file_name}"
        except Exception as exc:
            print(f"  WARNING thumbnail {paths[index]} failed: {exc}", flush=True)
        if n % 200 == 0:
            print(f"  Thumbnails exported {n:,} / {len(indices):,}", flush=True)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("../bl_imagenet28"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alphas", nargs="+", default=DEFAULT_ALPHAS, help="Alpha directories, baseline must be first")
    parser.add_argument("--data-root", type=Path, default=Path("/oscar/scratch/jchang88/imagenet2/"))
    parser.add_argument("--val-map-path", type=Path, default=Path("/gpfs/data/shared/imagenet/ILSVRC2012/val/val_map.txt"))
    parser.add_argument("--out-dir", type=Path, required=True, help="Next.js public/jem-analysis directory")
    parser.add_argument("--n-images", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--thumbnail-count", type=int, default=0, help="Export thumbnails ONLY if ImageNet redistribution rights allow it")
    parser.add_argument("--thumbnail-size", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.alphas[0] != "0.0" or len(args.alphas) != 6:
        raise ValueError("This dashboard expects six ordered models: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5")
    if args.alphas != DEFAULT_ALPHAS:
        raise ValueError(f"Use the configured alpha order: {DEFAULT_ALPHAS}")
    checkpoints = [get_checkpoint(args.root, alpha, args.seed) for alpha in args.alphas]
    print("Selected checkpoints:")
    for alpha, checkpoint in zip(args.alphas, checkpoints):
        print(f"  alpha {alpha}: {checkpoint}")
    if args.dry_run:
        return
    if args.n_images <= 0 or args.thumbnail_count < 0:
        raise ValueError("n-images must be positive; thumbnail-count cannot be negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Run on an allocated OSCAR GPU node.")
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_dir = args.data_root / "val"
    dataset = ImageNetValDataset(str(val_dir), str(args.val_map_path), transform=transform)
    if args.n_images > len(dataset):
        raise ValueError(f"Requested {args.n_images} images, dataset has {len(dataset)}")
    paths = image_paths(dataset, val_dir)[:args.n_images]
    if args.n_images < len(dataset):
        dataset = Subset(dataset, range(args.n_images))
    print(f"ImageNet validation count: {len(dataset):,} | device={device}", flush=True)
    z, labels = collect_latents(dataset, device, args)
    n, m = len(labels), len(checkpoints)
    preds = np.empty((n, m), dtype=np.int16)
    conf = np.empty((n, m), dtype=np.float32)
    ranks = np.empty((n, m), dtype=np.int16)
    probs = np.empty((n, m), dtype=np.float32)
    for col, (alpha, checkpoint) in enumerate(zip(args.alphas, checkpoints)):
        print(f"Evaluating alpha={alpha} ({col+1}/{m})", flush=True)
        model = load_model(checkpoint, device)
        preds[:, col], conf[:, col], ranks[:, col], probs[:, col] = evaluate_model(model, z, labels, device, args.batch_size)
        accuracy = float((preds[:, col] == labels).mean())
        print(f"  alpha {alpha}: top-1={accuracy:.4%}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out = args.out_dir
    out.mkdir(exist_ok=True, parents=True)
    if args.thumbnail_count:
        print("WARNING: Exporting ImageNet thumbnails. Ensure you have permission to redistribute these images.")
    chosen = select_thumbnail_indices(preds, labels, min(n, args.thumbnail_count), args.random_seed)
    thumb_names = export_thumbnails(paths, out, chosen, args.thumbnail_size)
    doc = {
        "schemaVersion": 1,
        "meta": {
            "dataset": "ImageNet ILSVRC2012 validation",
            "seed": args.seed,
            "alphas": [float(a) for a in args.alphas],
            "nImages": n,
            "thumbnailCount": len(thumb_names),
            "classNames": class_names(),
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "checkpoints": {alpha: checkpoint.name for alpha, checkpoint in zip(args.alphas, checkpoints)},
        },
        "rows": [],
    }
    for i, image_path in enumerate(paths):
        row = {
            "id": image_path.name,
            "y": int(labels[i]),
            "p": [int(v) for v in preds[i]],
            "c": [round(float(v), 6) for v in conf[i]],
            "r": [int(v) for v in ranks[i]],
            "q": [round(float(v), 7) for v in probs[i]],
        }
        if i in thumb_names:
            row["thumb"] = thumb_names[i]
        doc["rows"].append(row)
    output_path = out / "results.json"
    with output_path.open("w") as file:
        json.dump(doc, file, separators=(",", ":"), ensure_ascii=False)
    print(f"Wrote {output_path} ({output_path.stat().st_size/1e6:.1f} MB); thumbnails={len(thumb_names)}")
    print("Copy public/jem-analysis to your Vercel Next.js repo and deploy. No GPU needed on Vercel.")


if __name__ == "__main__":
    main()
