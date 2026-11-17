# How it works: mapping forest loss from space with a U-Net

A lesson for a software engineer moving into AI engineering. Concepts first, then a walk
through the code in the order the data flows.

---

## Part 1: Concepts

### 1.1 What a satellite image really is

A phone photo has 3 numbers per pixel (red, green, blue). **Sentinel-2**, two European
satellites that photograph all land every 5 days, records **13 bands**, including infrared
light our eyes can't see. Each pixel value is **reflectance**: the fraction of sunlight
the ground reflected in that band, from 0 to 1. We use 6 bands at 10–20 m per pixel:

| band | wavelength | what it tells us |
|---|---|---|
| blue, green, red | visible | what you'd see; forest is dark, bare soil is bright |
| **NIR** (near-infrared) | 842 nm | **leaves glow** here: healthy canopy reflects ~30–40% |
| SWIR1, SWIR2 (short-wave infrared) | 1.6, 2.2 µm | water content; bare and burnt ground is bright |

So an image is not `H × W × 3`, it's `6 × H × W` (bands first, the PyTorch convention).

### 1.2 Vegetation indices: the 1970s baseline

**NDVI = (NIR − red) / (NIR + red).** Leaves absorb red light (photosynthesis) and reflect
NIR, so dense forest scores ~0.8, pasture 0.3–0.6, bare soil ~0.1. A single threshold on NDVI
is the classic forest map, and it's our **baseline**. Always build the dumb baseline first:
here it gets IoU 0.87, and the U-Net only adds 3 points on top. Without the baseline, "0.90"
would sound more impressive than it is.

**NBR** (normalized burn ratio) is the same idea with SWIR2, and is sensitive to burnt ground.

### 1.3 Semantic segmentation

**Classification** gives one label per image ("this photo contains forest").
**Segmentation** gives one label **per pixel**. The output is a mask the same size as the input.

### 1.4 The U-Net

```
input 8×128×128
 ├─ enc1: conv,conv → 24×128×128 ───────────────────────────┐ skip
 │   pool → 24×64×64                                        │
 ├─ enc2: conv,conv → 48×64×64 ─────────────────────┐ skip  │
 │   pool → 48×32×32                                │       │
 ├─ enc3: conv,conv → 96×32×32 ─────────┐ skip      │       │
 │   pool → 96×16×16                    │           │       │
 ├─ bottleneck: conv,conv → 192×16×16   │           │       │
 │   up → 96×32×32 ⊕ enc3 → dec3 → 96×32×32         │       │
 │   up → 48×64×64 ⊕ enc2 → dec2 → 48×64×64 ────────┘       │
 │   up → 24×128×128 ⊕ enc1 → dec1 → 24×128×128 ────────────┘
 └─ 1×1 conv → 1×128×128   (a "forest?" score per pixel)
```

- **Encoder (left side, going down):** each level halves the size and doubles the channels.
  Deeper layers see a wider area ("this is a big uniform dark region bordered by a road"),
  but lose exact positions.
- **Decoder (right side, going up):** upsamples back to full size.
- **Skip connections (⊕):** the decoder gets the encoder's full-resolution features from the
  same level. Without them, edges come out blurry. They're why it's shaped like a "U".

> Software analogy: the encoder is summarising a log into "what happened"; the skips are keeping
> the original line numbers so you can still point at exactly where.

### 1.5 Where the labels come from

Nobody hand-drew forest outlines for us. We use **ESA WorldCover 2021**: a global 10 m
land-cover map that ESA made with its own model. Our U-Net learns to reproduce its "tree
cover" class. This is common in remote sensing and you should always say so: our "accuracy"
is agreement with another model, which is itself only about 77% accurate globally.

### 1.6 Spatial leakage: why we split by block

Neighbouring pixels are nearly identical. If you put pixel (100, 100) in training and
(100, 101) in test, the test is a memory exercise. This is the same lesson as splitting the
ECG data by patient. So the area is cut into 16 blocks of 5.5 km; whole blocks go to train
(11), validation (2), or test (3), and training patches stay 16 px away from block edges.

### 1.7 Change detection

Run the same model on 2019, 2021 and 2024 images. A pixel is "newly cleared" if it was forest
before and isn't after. Two practical details:

- **Use the same season every year.** Pasture is green in the wet season and brown in the dry
  season. All our images are June–August.
- **Drop tiny patches** (< 0.5 ha). A single flipped pixel is usually noise, not a chainsaw.

### 1.8 Checking change with an independent dataset

**Hansen Global Forest Change** (University of Maryland) gives, for every 30 m pixel on
Earth, the year it lost tree cover (2001–2024), from Landsat. It's independent of our
pipeline: a different satellite, team and method. We never train on it.

The lesson from comparing them: **two datasets that both say "forest loss" can mean different
things.**

| | our U-Net | Hansen |
|---|---|---|
| "forest" means | WorldCover "tree cover" (includes regrowth) | ≥30% canopy in 2000, not yet lost |
| clearing of regrowth | detected | invisible (only the first loss year is kept) |
| fire-damaged but standing forest | still "forest" | "loss" |
| timing | July to July | calendar year |

So before computing agreement, restrict to land where both can answer the question: land
Hansen still considered intact forest in 2019. There, 94% of what we flag is confirmed.

### 1.9 Bugs that would have silently produced fake results

Real-world geospatial work is mostly about data plumbing. Two bugs from this project are
worth remembering:

1. **Reflectance offsets.** Since 2022 ESA adds +1000 to Sentinel-2 values. The data provider
   had already removed it but left the metadata saying "subtract 0.1". Doing it twice made
   newer images 0.1 darker, which is a fake change. **Sanity-check physical ranges:**
   reflectance can't be negative.
2. **No-data values.** Telling GDAL "0 means no data" on a file where 0 means "no loss" made it
   quietly change every real 0 into 1 ("lost in 2001"). **Check value histograms after every
   resampling step.**

Both were caught because a summary number looked physically impossible: negative reflectance,
and Hansen saying half the area was cleared in 2001. Print summary statistics early and often.

---

## Part 2: Code walkthrough

### 2.1 `src/forest/grid.py`

The area in lon/lat, and `grid()`: the same area in UTM zone 21S (metres, the satellite tiles'
own projection), snapped to 10 m. **Every dataset is resampled onto this one grid**, so pixel
`[i, j]` means the same piece of ground in every array.

### 2.2 `scripts/download_data.py`

- **STAC search** (`pystac_client`): a standard JSON API for "which images cover this box,
  between these dates, with < 10% cloud?". It returns items, each with links (assets) to
  one file per band.
- **Cloud-Optimized GeoTIFF + `WarpedVRT`**: `read_on_grid()` opens a 100 km image over HTTP,
  but GDAL fetches only the internal tiles that overlap our 22 km box (HTTP range requests),
  and reprojects them onto our grid on the fly.
- **`scene_dn`** caches each scene's raw values in `data/raw/scenes/`, so changing the
  processing never needs a re-download (learned the hard way).
- **`reflectance_offset`**: the offset rule from 1.9, with a test.
- **Cloud mask + median composite**: the scene classification layer (SCL) marks clouds and
  shadows; those pixels become NaN; `np.nanmedian` over 4 scenes picks a typical clear value
  per pixel.

### 2.3 `src/forest/data.py`

`features()` adds NDVI and NBR to the 6 bands (8 channels). `split_masks()` and
`block_bounds()` implement the block split.

### 2.4 `src/forest/unet.py`

Exactly the diagram above. `conv_block` is two 3×3 convolutions with batch norm and ReLU.
`ConvTranspose2d(..., stride=2)` is a learned upsampling. `torch.cat([up(x), skip], dim=1)`
is the skip connection: it stacks channels.

### 2.5 `scripts/train.py`

- **Normalisation**: each channel is shifted and scaled to mean 0, std 1, using statistics
  from 2021. The **same** statistics are reused for 2019 and 2024; recomputing them per year
  would hide real change.
- **`sample()`**: random 128 px patches from training blocks, random 90° rotations and flips
  (forests have no "up"), and ±10% brightness jitter so slightly hazier years don't confuse it.
- **Loss = BCE + Dice.** Binary cross-entropy scores each pixel. **Dice** scores the overlap of
  the whole predicted mask with the label, which helps with thin features like riverside forest.
  Both are multiplied by a mask so unlabelled pixels count for nothing.
- **`predict_full()`**: predicts a big image in 256 px tiles that overlap by 32 px, keeping only
  each tile's centre. Edges of a tile have less context and are less reliable.

### 2.6 `scripts/evaluate.py`

- `seg_scores`: IoU = overlap / union of the predicted and true forest masks.
- `clean`: `scipy.ndimage.label` finds connected patches; small ones are dropped.
- `hansen_reference`: "lost between y1 and y2" and "forest at y1" from loss-year codes.
- `change_scores`: precision and recall with a 1-pixel tolerance (`binary_dilation`), because
  30 m and 10 m pixel edges never line up exactly.
- The breakdowns of disagreement (`unet_only_share_on_previously_cleared_land`,
  `hansen_only_share_by_loss_year`) turn "the numbers are low" into "here is why".

### 2.7 `scripts/make_map.py`

Leaflet (via Folium) works in longitude/latitude, so rasters are reprojected to EPSG:4326 and
embedded as PNGs. Clearings are turned into polygons (`rasterio.features.shapes`, the raster to
vector step) with their area in hectares, and saved as GeoJSON layers you can click.

---

## Part 3: Things to try next

1. **Train on change directly**: input both years' images (12 channels), label = Hansen loss
   in between. Then fire damage becomes learnable. You'd also need a new independent check.
2. **Monthly images** instead of yearly, to catch clearing within weeks and to separate fire
   from clear-cutting by how the pixels change over time.
3. **Another region** (e.g. the Congo basin) to test whether the model transfers.
4. **Sentinel-1 radar**, which sees through clouds: essential in the wet season.

## Interview questions you should now be able to answer

- Why is NIR so useful for vegetation, and what is NDVI?
- Why do U-Nets have skip connections?
- Why split by spatial block instead of by pixel?
- Your change detector's precision against a reference dataset is 25%. What would you check
  before concluding the model is bad?
- Why must the normalisation statistics be the same for every year?
