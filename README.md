# JEM ImageNet Classification Explorer — Next.js + Vercel

This adds a **new `/jem-rescues` route** to https://github.com/chang-1/nextjs-boilerplate, preserving its existing Plotly gloss page at `/`.

## Files

```text
app/jem-rescues/page.tsx         Next.js App Router page
app/jem-rescues/Explorer.tsx     Interactive dashboard client component
app/jem-rescues/explorer.module.css     Dark, responsive dashboard styles
scripts/export_jem_classification.py   Offline inference/export on OSCAR
public/jem-analysis/README.md   Placeholder; results.json is generated, not included
```

**Important:** The repo is public. ImageNet image content may be subject to dataset licensing and third-party copyright restrictions. The exporter defaults to *no thumbnails*. Only pass `--thumbnail-count` if you have permission to publish those images, or use an authorized/private hosted image source.

## 1. Install into the existing Next.js repository

Copy the folders `app`, `scripts`, `public` from this ZIP into the root of your `nextjs-boilerplate` checkout, merging directories rather than overwriting existing `app/page.tsx`.

```bash
cd nextjs-boilerplate
npm install
npm run dev
```

Open http://localhost:3000/jem-rescues . Until you export the data, the app shows an intentional 'Results not found' message. The original plot page remains unchanged.

Optional: add the navigation link below near the top of the existing `app/page.tsx`:

```tsx
<a href="/jem-rescues">Explore JEM ImageNet classification rescues →</a>
```

## 2. Run model evaluation on OSCAR

Place `scripts/export_jem_classification.py` in the JEM repository (e.g. `JEM/imagenet/` or `JEM/imagenet/scripts/`) and **execute from `JEM/imagenet/`**, where the imports `wideresnet_vae`, `vae_kl`, `imagenet_dataset` are available.

```bash
cd /oscar/data/tserre/jchang88/JEM/imagenet
conda activate env_cka

python export_jem_classification.py \
  --root ../bl_imagenet28 \
  --seed 1 \
  --n-images 50000 \
  --data-root /oscar/scratch/jchang88/imagenet2 \
  --val-map-path /gpfs/data/shared/imagenet/ILSVRC2012/val/val_map.txt \
  --batch-size 64 \
  --encode-batch-size 64 \
  --out-dir ./jem_classification_export
```

If the file is under `JEM/imagenet/scripts/`, use `python scripts/export_jem_classification.py` (running it from `imagenet/` ensures local imports; when executed as a script from a subdirectory, set `PYTHONPATH=.` if needed).

For optional thumbnails, **only if authorized to redistribute ImageNet images**, add `--thumbnail-count 800 --thumbnail-size 256`. These are selected in a balanced way across multi-hybrid rescues, individual rescues and losses. The remaining images have metadata but no web thumbnail. Exporting all 50,000 images to a public GitHub/Vercel deployment is not recommended.

For a quick path verification before starting:

```bash
python export_jem_classification.py --root ../bl_imagenet28 --seed 1 --out-dir ./jem_classification_export --dry-run
```

Model evaluation uses `best_valid_ckpt.pt` for alpha 0.0–0.5 and the provided evaluation convention (but alpha=1.0 is not part of this experiment). No CKA is required. Files generated:

```text
jem_classification_export/
  results.json
  thumbs/                optional WebP thumbnails
```

## 3. Transfer the export to your Next.js repo

Copy `results.json` and any authorized thumbnails into the web repo's `public/jem-analysis/`. If OSCAR and local laptop are separate machines, use `scp` or your normal cluster transfer method. Do **not** commit the ImageNet images to a public repository without permission.

```bash
# From the local Next.js checkout, if your export is already local:
cp /path/to/jem_classification_export/results.json public/jem-analysis/results.json
# Optional, if authorized:
cp -r /path/to/jem_classification_export/thumbs public/jem-analysis/

npm run dev
# then open http://localhost:3000/jem-rescues
```

Use `npm run build` to validate before pushing to GitHub. Connect this GitHub repo to Vercel or push to an existing Vercel-linked branch; the static JSON is served without Python or GPU.

## Data format

`results.json` is a single static document:

```json
{
  "schemaVersion": 1,
  "meta": { "dataset": "ImageNet ILSVRC2012 validation", "seed": 1, "alphas": [0, 0.1, 0.2, 0.3, 0.4, 0.5], "nImages": 50000, "thumbnailCount": 0, "classNames": ["tench", "goldfish"] },
  "rows": [
    {
      "id": "ILSVRC2012_val_00000001.JPEG",
      "y": 65,
      "p": [65, 65, 65, 65, 65, 65],
      "c": [0.55, 0.6, 0.7, 0.8, 0.85, 0.86],
      "r": [1, 1, 1, 1, 1, 1],
      "q": [0.55, 0.6, 0.7, 0.8, 0.85, 0.86]
    }
  ]
}
```

The above row only documents the schema; **it is not a measured JEM result**. The `p`, `c`, `r`, `q` arrays are in the exact order of `meta.alphas`; `p` and `y` are zero-based ImageNet class IDs.

The dashboard measures top-1 correctness as `p[i] === y` and top-5 inclusion as `r[i] <= 5`. 'Ground-truth probability gain' means hybrid `q[i] - q[0]`. Softmax probabilities are not necessarily calibrated. In collective 'any hybrid' mode, rescue/loss sets are unions, not performance of one ensemble model.
