"""Build the interactive map: results/deforestation_map.html (open it in any browser).

Usage:  uv run python scripts/make_map.py      (after evaluate.py)

Layers you can toggle (top right):
  * Sentinel-2 true colour for each year
  * U-Net forest map for the latest year
  * Newly cleared patches per interval (click one for its area)
  * Hansen GFC loss 2020-2024 (the independent reference), off by default
  * The held-out test blocks
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import folium
import numpy as np
import rasterio
from PIL import Image
from rasterio import features
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_geom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from forest.data import TEST_BLOCKS, block_bounds, read  # noqa: E402
from forest.grid import AOI_LONLAT, CRS_UTM, YEARS, grid  # noqa: E402

PROC = ROOT / "data" / "processed"
INTERVAL_STYLE = {(2019, 2021): ("#ff9f1c", "Cleared 2019→2021"), (2021, 2024): ("#e63946", "Cleared 2021→2024")}


def to_lonlat(arr: np.ndarray, resampling=Resampling.nearest) -> tuple[np.ndarray, list]:
    """Reproject a (bands, H, W) array from the UTM grid to lon/lat so Leaflet can overlay it."""
    transform, width, height = grid()
    left, top = transform.c, transform.f
    right, bottom = left + width * transform.a, top + height * transform.e
    dst_t, dw, dh = calculate_default_transform(CRS_UTM, "EPSG:4326", width, height, left, bottom, right, top)
    out = np.zeros((arr.shape[0], dh, dw), dtype=arr.dtype)
    for i in range(arr.shape[0]):
        reproject(arr[i], out[i], src_transform=transform, src_crs=CRS_UTM, dst_transform=dst_t,
                  dst_crs="EPSG:4326", resampling=resampling)
    w, n = dst_t.c, dst_t.f
    e, s = w + dw * dst_t.a, n + dh * dst_t.e
    return out, [[s, w], [n, e]]


def png_url(rgba: np.ndarray, max_side: int = 1000) -> str:
    img = Image.fromarray(rgba)
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS if rgba.shape[-1] == 3 else Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def rgb_layer(year: int) -> tuple[str, list]:
    s2 = read(f"s2_{year}.tif")
    arr, bounds = to_lonlat(np.stack([s2[2], s2[1], s2[0]]), Resampling.bilinear)
    img = (np.clip(arr.transpose(1, 2, 0) * 3.5, 0, 1) * 255).astype(np.uint8)
    return png_url(img), bounds


def mask_layer(mask: np.ndarray, color: tuple[int, int, int], alpha: int = 170) -> tuple[str, list]:
    arr, bounds = to_lonlat(mask[None].astype(np.uint8))
    rgba = np.zeros((*arr.shape[1:], 4), np.uint8)
    rgba[arr[0] > 0] = [*color, alpha]
    return png_url(rgba), bounds


def polygons(mask: np.ndarray) -> list[dict]:
    transform, _, _ = grid()
    feats = []
    for geom, _ in features.shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        rings = geom["coordinates"]
        area_ha = (abs(_ring_area(rings[0])) - sum(abs(_ring_area(r)) for r in rings[1:])) / 1e4  # minus holes
        g = transform_geom(CRS_UTM, "EPSG:4326", geom, precision=5)
        feats.append({"type": "Feature", "geometry": g, "properties": {"area_ha": round(area_ha, 2)}})
    return feats


def _ring_area(ring) -> float:
    xs, ys = np.array(ring).T
    return 0.5 * float(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))


def main() -> None:
    w, s, e, n = AOI_LONLAT
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=12, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery (today)", name="Esri satellite basemap (current)").add_to(m)

    for i, y in enumerate(YEARS):
        url, b = rgb_layer(y)
        folium.raster_layers.ImageOverlay(url, bounds=b, name=f"Sentinel-2 {y} (dry season)",
                                          show=(i == len(YEARS) - 1)).add_to(m)

    latest = np.load(PROC / f"forest_prob_{YEARS[-1]}.npy") > 0.5
    url, b = mask_layer(latest, (30, 140, 60), 110)
    folium.raster_layers.ImageOverlay(url, bounds=b, name=f"U-Net forest {YEARS[-1]}", show=False).add_to(m)

    lossyear, tc2000 = read("hansen_lossyear.tif")[0], read("hansen_treecover2000.tif")[0]
    url, b = mask_layer((lossyear >= 20) & (lossyear <= 24) & (tc2000 >= 30), (40, 120, 255))
    folium.raster_layers.ImageOverlay(url, bounds=b, name="Hansen GFC loss 2020-2024 (reference)", show=False).add_to(m)

    totals = {}
    for (y1, y2), (color, name) in INTERVAL_STYLE.items():
        cleared = np.load(PROC / f"cleared_{y1}_{y2}.npy")
        feats = polygons(cleared)
        totals[name] = (len(feats), sum(f["properties"]["area_ha"] for f in feats))
        folium.GeoJson({"type": "FeatureCollection", "features": feats}, name=name,
                       style_function=lambda _, c=color: {"color": c, "weight": 1, "fillColor": c, "fillOpacity": 0.55},
                       tooltip=folium.GeoJsonTooltip(fields=["area_ha"], aliases=[f"{name}, hectares:"])).add_to(m)

    transform, width, height = grid()
    tb = folium.FeatureGroup(name="Held-out test blocks", show=False)
    for (r, c), (y0, y1, x0, x1) in block_bounds(height, width).items():
        if (r, c) in TEST_BLOCKS:
            ring = [transform * (x, y) for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]
            g = transform_geom(CRS_UTM, "EPSG:4326", {"type": "Polygon", "coordinates": [ring]})
            folium.GeoJson(g, style_function=lambda _: {"color": "white", "weight": 2, "dashArray": "6",
                                                        "fillOpacity": 0}).add_to(tb)
    tb.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    ch = metrics["change_vs_hansen"]["2019-2024"]
    c = ch["unet_on_hansen_intact_forest"]
    rows = "".join(f"<div><span style='background:{col};width:12px;height:12px;display:inline-block'></span> "
                   f"{nm}: {totals[nm][0]} patches, {totals[nm][1]:,.0f} ha</div>"
                   for col, nm in INTERVAL_STYLE.values())
    legend = (f"<div style='position:fixed;bottom:24px;left:12px;z-index:9999;background:white;padding:10px 12px;"
              f"border-radius:6px;font:13px sans-serif;box-shadow:0 1px 4px #0004;max-width:320px'>"
              f"<b>Newly cleared forest, Novo Progresso (Pará, Brazil)</b>{rows}"
              f"<div style='margin-top:6px;color:#444'>Checked against Hansen GFC on land it counted as intact "
              f"forest in 2019: {c['precision']:.0%} of flagged clearing confirmed; {c['recall']:.0%} of its 2020-24 "
              f"loss found (fire damage and late-2024 loss are mostly missed). "
              f"{ch['unet_only_share_on_previously_cleared_land']:.0%} of the remaining flags are re-clearing of land "
              f"first cleared before 2019, which Hansen cannot record.</div></div>")
    m.get_root().html.add_child(folium.Element(legend))
    out = ROOT / "results" / "deforestation_map.html"
    m.save(out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", totals)


if __name__ == "__main__":
    main()
