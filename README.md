# Port the EXISTING JEM Streamlit rescue browser to Vercel

This reproduces the Streamlit rescue/regression browser for **alpha 0.0 vs 0.1**, seed 1, using the data already produced on OSCAR. It does not evaluate the JEM or VAE.

## Files

- `app/jem-rescues/page.tsx`, `Explorer.tsx`, `explorer.module.css`: replace the old files in your Vercel-linked Next.js repository.
- `scripts/pack_existing_streamlit_data.py`: run once on OSCAR to package its **existing** Streamlit CSV + evaluation crops into a web-ready folder.
- `public/jem-analysis/`: target directory in your GitHub Next.js repository (only generated files go here).

## On OSCAR

From `JEM/imagenet/`, copy the `pack_existing_streamlit_data.py` file here and run:

```bash
python pack_existing_streamlit_data.py \
  --results-dir ../bl_imagenet28/hybrid_rescue_seed1 \
  --image-root /oscar/scratch/jchang88/imagenet2/val \
  --output-dir ./jem_vercel_upload
```

The script reads `all_image_predictions.csv`, optional `summary.json`, and optional `class_grouped/classified_examples.csv`, and copies existing JPEGs from `class_grouped/{rescued,regressed}/*/evaluation_crops/`. If a crop is missing and the source JPEG is accessible, it recreates the *evaluation crop only* (Resize shortest side to 256, CenterCrop 224).

Output:

```
JEM/imagenet/jem_vercel_upload/
  results.json
  crops/
    00001.jpg
    ...
```

It exports all rescued and regressed crops by default. For a preliminary reduced public gallery, pass `--max-crops 200`; counts remain based on all changed images.

## From OSCAR to GitHub

Copy the output folder to your computer via OSCAR login host, then place **the contents** of `jem_vercel_upload/` in this path in your GitHub project:

```
nextjs-boilerplate/
  app/
    jem-rescues/
      page.tsx
      Explorer.tsx
      explorer.module.css
  public/
    jem-analysis/
      results.json
      crops/
        00001.jpg
        ...
```

Once the Vercel deployment finishes, visit `/jem-rescues`.

The JSON file is a lightweight **conversion of your existing Streamlit CSV**, not a new model result. The existing Streamlit application used the CSV directly; the Next.js page uses this JSON to avoid adding a CSV parsing dependency.

## What is preserved

- Rescued, regressed, net gain, and alpha summary cards.
- Gallery: status, true class, search, sort, pages of 12/24/36/48, per-image prediction/confidence/name, filtered CSV download.
- Class counts: top-30 bars, class table, CSV download.
- Confusions: most common wrong predictions, CSV download.
- 224×224 evaluation crops only. No original JPEG gallery or re-running inference.

The layout is Streamlit-like, implemented in React/Next.js (Vercel does **not** run the Streamlit Python app unchanged).

### Publishing ImageNet crops

ImageNet access does not grant blanket redistribution rights for individual copyrighted images. Before uploading crops to a **public** GitHub/Vercel deployment, obtain appropriate rights or restrict the image viewer to an access-controlled research audience under applicable permissions.
