"""The study area and the fixed 10 m pixel grid everything is resampled onto."""
from __future__ import annotations

from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

# Novo Progresso, Pará, Brazil: about 22 x 22 km along the BR-163 highway, one of the most
# active deforestation fronts in the Amazon. Hansen GFC shows ~14% tree-cover loss here in 2020-24.
AOI_LONLAT = (-55.55, -7.35, -55.35, -7.15)  # west, south, east, north
CRS_UTM = CRS.from_epsg(32721)  # UTM zone 21S, the Sentinel-2 tiles' own projection
RES = 10.0

BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]  # Earth Search asset names
YEARS = [2019, 2021, 2024]
TRAIN_YEAR = 2021  # ESA WorldCover 2021 labels are matched with the 2021 imagery


def grid() -> tuple[object, int, int]:
    """Affine transform, width, height of the AOI grid in UTM, snapped to 10 m."""
    w, s, e, n = transform_bounds("EPSG:4326", CRS_UTM, *AOI_LONLAT)
    w, n = (w // RES) * RES, -(-n // RES) * RES
    width, height = int(-(-(e - w) // RES)), int(-(-(n - s) // RES))
    return from_origin(w, n, RES, RES), width, height
