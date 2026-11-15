# Research website and JEM explorer

The Next.js website is deployed on Vercel. `/jem-rescues` compares baseline α=0 and hybrid α=0.1 on all 50,000 ImageNet validation images (seed 1, batch size 4).

## Development

Use Node.js 22, then run `npm ci` and `npm run dev`. Run `npm run build` before publishing.

## JEM data in Cloudflare R2

Dataset root: `https://assets.jorgechang.xyz/seed1-batch4/`

- `data/manifest.json`: class names, totals, analysis configuration, detail chunk paths.
- `data/catalog.json`: compact prediction and confidence rows for filtering and CSV export.
- `data/details/000.json` through `099.json`: chunks of 500 images with map paths and statistics, fetched on demand.
- `crops/`: evaluation crops.
- `maps/`: rendered decoder-Jacobian projections.

`next.config.ts` forwards `/jem-data/*` to R2 metadata. This allows local development and Vercel preview URLs to load JSON without expanding the R2 browser CORS allowlist. Images load directly from R2; no credentials are required for the public assets. Model inference never runs on Vercel.

The original `public/jem-analysis/` dataset has been removed from the current branch and is ignored. Previous commits still contain it; Git history has not been rewritten. Keep numeric arrays in Oscar's `raw_do_not_upload/` directory. Do not commit datasets or upload raw arrays for this viewer.

## Explorer

The existing sidebar, gallery, class counts, confusions, pagination, search, confidence sorting and CSV downloads are retained. The default shows rescued/regressed images; additional filters show both-correct, both-wrong, or all images.

Choose **View Jacobians** on a card to compare the crop, baseline, hybrid, difference magnitude and signed channel-mean difference. Ground-truth logit maps exist for every image. Margin maps exist only for rescued/regressed examples, using the same ground-truth-minus-competitor target for both models. The detail panel reports cosine similarity and gradient norms. Overlays use shared per-image 99th-percentile scaling; absolute strengths should not be compared across images using color alone.

Confidence is the softmax probability of the predicted class. The export verified all saved predictions before adding confidence values.

## Updating the data

Upload the matching crops, maps and metadata together to a new versioned R2 prefix. Update both the rewrite destination in `next.config.ts` and the validated asset root in `app/jem-rescues/data.ts`. Verify counts and representative images/maps before deploying. The legacy scripts in `scripts/` are retained for reference; they are not the input pipeline for this viewer.
