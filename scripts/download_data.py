"""Download everything for the study area, cropped to it, onto one 10 m grid.

Usage:  uv run python scripts/download_data.py

Only the pixels inside the ~22 x 22 km area are fetched: Cloud-Optimized GeoTIFFs let
rasterio read a window over HTTP instead of the whole 100 km tile. About 250 MB of traffic.

  * Sentinel-2 L2A (AWS Earth Search STAC, no account): for each year in YEARS, the 4 clearest
    dry-season (Jun-Sep) scenes, cloud-masked with the scene classification layer, combined
    into a median composite of 6 bands (blue, green, red, NIR, SWIR1, SWIR2), as reflectance.
  * ESA WorldCover 2021 v200 (AWS open data): land-cover map; class 10 = "tree cover".
    These are our training labels for "forest".
  * Hansen Global Forest Change v1.12 (Google Cloud public bucket): tree cover in 2000 and the
    year of loss 2001-2024. Used ONLY to check the change detection, never for training.

Writes data/raw/s2_<year>.tif, worldcover_2021.tif, hansen_lossyear.tif, hansen_treecover2000.tif
and data/raw/scenes.json (which scenes went into each composite).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from forest.grid import AOI_LONLAT, BANDS, CRS_UTM, YEARS, grid  # noqa: E402

RAW = ROOT / "data" / "raw"
STAC = "https://earth-search.aws.element84.com/v1"
WORLDCOVER = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_S09W057_Map.tif"
HANSEN = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/Hansen_GFC-2024-v1.12_{layer}_00N_060W.tif"
CLOUDY_SCL = [0, 1, 3, 8, 9, 10]  # no data, saturated, cloud shadow, cloud (med/high), cirrus
SCENES_PER_YEAR = 4

rasterio_env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                            GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2")


def read_on_grid(url: str, resampling=Resampling.bilinear, dtype="float32", tries: int = 5) -> np.ndarray:
    """Read one remote band, warped onto our grid. Retries: a dropped connection mid-read is common."""
    transform, width, height = grid()
    for attempt in range(tries):
        try:
            with rasterio.open(url) as src, WarpedVRT(src, crs=CRS_UTM, transform=transform, width=width,
                                                      height=height, resampling=resampling,
                                                      src_nodata=src.nodata, nodata=0 if src.nodata is not None else None) as vrt:
                # Only declare a no-data value if the source has one. Hansen files have none, and
                # 0 is a real value there ("no loss"); with nodata=0 GDAL's warper silently turns
                # real zeros into 1 ("lost in 2001") to keep them apart from no-data.
                return vrt.read(1).astype(dtype)
        except rasterio.errors.RasterioIOError as e:
            if attempt == tries - 1:
                raise
            print(f"  retry {attempt + 1}/{tries - 1} reading {url.rsplit('/', 1)[-1]}: {e}", flush=True)
            time.sleep(5 * 2 ** attempt)


def save(path: Path, arr: np.ndarray, dtype: str, descriptions: list[str] | None = None) -> None:
    transform, width, height = grid()
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=arr.shape[0], dtype=dtype,
                       crs=CRS_UTM, transform=transform, compress="deflate", tiled=True) as dst:
        dst.write(arr.astype(dtype))
        for i, d in enumerate(descriptions or []):
            dst.set_band_description(i + 1, d)


def reflectance_offset(item) -> float:
    """How much to subtract after scaling DN by 1e-4 to get surface reflectance.

    Since processing baseline 04.00 (Jan 2022, and ESA's reprocessed "Collection 1" scenes)
    ESA adds +1000 to every DN, i.e. 0.1 reflectance. Ignoring that makes newer scenes look
    brighter and fakes a change. BUT Earth Search removes it for us on many scenes and marks
    them `earthsearch:boa_offset_applied: true`, while still listing offset -0.1 in the band
    metadata. Applying that offset again gives negative reflectance (a bug I hit, see README).
    """
    if item.properties.get("earthsearch:boa_offset_applied"):
        return 0.0
    return 0.1 if float(item.properties.get("s2:processing_baseline", "0")) >= 4.0 else 0.0


def scene_dn(item) -> tuple[np.ndarray, np.ndarray]:
    """Raw DNs of one scene on our grid, cached in data/raw/scenes/ so re-processing never re-downloads."""
    cache = RAW / "scenes" / f"{item.id}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["dn"], z["scl"]
    scl = read_on_grid(item.assets["scl"].href, Resampling.nearest, dtype="uint8")
    dn = np.stack([read_on_grid(item.assets[b].href, dtype="uint16") for b in BANDS])
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, dn=dn, scl=scl)
    return dn, scl


def composite(year: int, client: Client) -> tuple[np.ndarray, list[str]]:
    items = list(client.search(collections=["sentinel-2-l2a"], bbox=AOI_LONLAT,
                               datetime=f"{year}-06-01/{year}-09-30",
                               query={"eo:cloud_cover": {"lt": 10}}).items())
    # One processing per (date, tile): Earth Search sometimes lists a scene twice.
    seen, picked = set(), []
    for it in sorted(items, key=lambda i: (i.properties["eo:cloud_cover"], i.id)):
        key = (it.datetime.date(), it.properties.get("grid:code"))
        if key not in seen:
            seen.add(key)
            picked.append(it)
    # The area straddles two tiles, so take the clearest dates and BOTH tiles for each date.
    dates = list(dict.fromkeys(it.datetime.date() for it in picked))[:SCENES_PER_YEAR // 2]
    picked = [it for it in picked if it.datetime.date() in dates]

    stack = []
    for it in picked:
        dn, scl = scene_dn(it)
        refl = dn * 1e-4 - reflectance_offset(it)
        refl[:, (dn[0] == 0) | np.isin(scl, CLOUDY_SCL)] = np.nan
        stack.append(refl)
        print(f"  {it.id}: {np.isfinite(refl[0]).mean():.0%} of the area covered and cloud-free", flush=True)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # pixels outside every scene are all-NaN
        comp = np.nanmedian(np.stack(stack), axis=0)
    print(f"  {year}: {len(picked)} scenes on {[str(d) for d in dates]}, "
          f"{np.isnan(comp[0]).mean():.2%} of pixels still empty")
    return np.nan_to_num(comp, nan=0.0), [it.id for it in picked]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    client = Client.open(STAC)
    scenes = {}
    with rasterio_env:
        for year in YEARS:
            out = RAW / f"s2_{year}.tif"
            if out.exists():
                continue
            print(f"Sentinel-2 {year}")
            comp, ids = composite(year, client)
            save(out, comp, "float32", BANDS)
            scenes[year] = ids
        if scenes:
            prev = json.loads((RAW / "scenes.json").read_text()) if (RAW / "scenes.json").exists() else {}
            (RAW / "scenes.json").write_text(json.dumps({**prev, **{str(k): v for k, v in scenes.items()}}, indent=1))

        if not (RAW / "worldcover_2021.tif").exists():
            print("ESA WorldCover 2021")
            save(RAW / "worldcover_2021.tif", read_on_grid(WORLDCOVER, Resampling.nearest), "uint8")
        for layer in ("lossyear", "treecover2000"):
            if not (RAW / f"hansen_{layer}.tif").exists():
                print(f"Hansen GFC {layer}")
                save(RAW / f"hansen_{layer}.tif", read_on_grid(HANSEN.format(layer=layer), Resampling.nearest), "uint8")
    print("done")


if __name__ == "__main__":
    main()
