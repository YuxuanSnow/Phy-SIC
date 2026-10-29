# PhySIC evaluation

Reproduces the paper's quantitative results (Table 2) on **PROX-quantitative**
(178 frames) and **RICH** (100 frames) from our released per-frame predictions.
No reconstruction pipeline is needed, and no dataset imagery is redistributed.

| | PA-MPJPE | MPJPE | MPVPE | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| PROX-quantitative | 41.99 | 230.26 | 227.19 | 0.508 | 0.514 | 0.511 |
| RICH | 46.50 | 616.27 | 617.34 | 0.310 | 0.689 | 0.428 |

Pose errors in mm; contact scored per body vertex against the dataset's scanned
scene (in contact = within 2 cm, or behind the surface and within 15 cm). The
paper prints RICH MPVPE as 617.33; the underlying value is 617.338.

## Setup

```bash
pip install -r requirements.txt     # pinned; install torch for your CUDA first
pip install --no-deps "smplfitter @ git+https://github.com/isarandi/smplfitter.git@13180c45a9201c8113690ad5158fad20b94be36b"
```

smplfitter goes in without its dependencies because it declares `chumpy`, which
no longer builds and is never imported for SMPL-X. The evaluation also needs
two body-model files under the repo root: `data/body_models/smplx/SMPLX_NEUTRAL.npz`
(SMPL-X) and `data/body_models/smplx/kid_template.npy` (AGORA). `bash fetch_data.sh`,
run from the repo root, downloads them; of the accounts it asks for, only the
SMPL-X and AGORA ones are needed here.

Our predictions are downloaded from Hugging Face
([`ympradyumna/physic-eval-release`](https://huggingface.co/datasets/ympradyumna/physic-eval-release))
on first run. Ground truth comes from your own copies of the datasets:

* **PROX** — the *quantitative* release from https://prox.is.tue.mpg.de, used as
  downloaded. Pass its root as `--prox-root`.
* **RICH** — build the 100-frame sample with the builder from
  [GRAFT](https://github.com/pradyumnaym/graft) (same frames, same layout; see its
  [`docs/EVALUATION.md`](https://github.com/pradyumnaym/graft/blob/main/docs/EVALUATION.md)).
  Only `test_body.zip` and `scan_calibration.zip` from https://rich.is.tue.mpg.de
  are needed, since this evaluation reads no images; below, `<RICH>` is the
  folder holding the extracted `test_body/` and `scan_calibration/`. Our numbers
  use RICH's full-resolution scene scans, so copy those in afterwards:

  ```bash
  python scripts/build_rich_sample100.py --rich-root <RICH> --out <rich-sample100> \
      --skip-images --skip-meshes          # run inside the GRAFT repo
  for s in Gym/scan_camcoord Gym/scan_table_camcoord LectureHall/scan_yoga_scene_camcoord; do
    mkdir -p <rich-sample100>/scene/scan_calibration/$(dirname $s)
    cp <RICH>/scan_calibration/$s.ply <rich-sample100>/scene/scan_calibration/$s.ply
  done
  ```

  The builder warns that the result is incomplete without frames and scans; that
  applies to GRAFT's evaluation, not this one. Pass `<rich-sample100>` as
  `--rich-root`. GRAFT's own decimated scans (from a build without
  `--skip-meshes`) also work with `--scan-suffix _simp`: pose is unchanged and
  contact F1 drops by 0.004 (`expected_results/rich_simp.json`).

## Run

```bash
python evaluate.py --prox-root <PROX>/quantitative --rich-root <rich-sample100>
python evaluate.py --benchmark prox --prox-root ... --limit 10     # quick check
```

It prints the tables above (plus the initial single-image estimate, DECO's
contacts, PROX's visible/occluded split and a vertex-to-scene vector metric),
writes `results/<benchmark>.json` (`rich_full.json`, `rich_simp.json` for the
other configurations), and compares every full run against the matching file in
`expected_results/`. `--device` picks `auto`/`cuda`/`cpu`; the roots can also be
set as `$PHYSIC_PROX_ROOT` / `$PHYSIC_RICH_ROOT`.

The ground-truth bodies are posed on the GPU, so a different GPU model moves a
few ground-truth contact vertices across the 2 cm boundary. Runs on other
hardware agree with `expected_results/` to about 1e-4 (relative), identical at
the paper's precision; the check passes within `--tolerance 1e-3`.

## Released data

One `.npz` per frame: our SMPL-X parameters and camera before (`init_*`) and
after (`final_*`) the scene-aware optimisation, the predicted intrinsics `K`,
the DECO contacts the optimiser was given, the reconstructed scene
(`scene_points`, `scene_normals`), the visible-vertex mask (PROX only), and the
pipeline's own contact mask, which the evaluation re-derives and checks frame by
frame. No dataset images, depth maps, meshes or ground truth are included.

Two tiers give identical numbers: `--tier compact` (default, 25 MB) keeps the
scene points contact can reach; `--tier full` (13 GB) is the complete
reconstructed cloud. `prepare/` documents how both were made from the raw
pipeline outputs; it needs the internal pipeline and is for reference only.

The predictions are released for non-commercial research use, under the same
terms as PROX and RICH.

**Ground-truth details.** PROX's MoSh fits store shape under `beta`, not
`betas`, so SMPL-X falls back to its mean shape; the published numbers were
computed that way and are reproduced as-is. RICH's ground truth is a gendered
SMPL-X body, so the neutral model is refitted to its mesh with `smplfitter`
(3 iterations, beta regulariser 1).
