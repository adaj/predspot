"""
Synthetic Data Module
=====================

Generates realistic-looking synthetic crime events inside a study area, so
that Predspot can be tried, demonstrated and tested without real data.

The generator is a simple inhomogeneous space-time point process:

* **Space** — a mixture of ``n_hotspots`` Gaussian hotspots (centres drawn
  uniformly inside the study area) plus a uniform background. The fraction
  of events that belong to hotspots is ``hotspot_share``.
* **Time** — an intensity built from a linear trend, an annual cycle, a
  day-of-week profile and an hour-of-day profile. Timestamps are drawn by
  thinning uniform candidates, which is exact for a fixed number of events.

Example::

    from predspot.synthetic import generate_crimes
    crimes = generate_crimes(study_area, n_events=5000, seed=0)
    dataset = Dataset(crimes, study_area)
"""

__author__ = "Adelson Araujo"

import logging
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from predspot.crime_mapping import KM_PER_DEG_LAT, KM_PER_DEG_LON, WGS84, _check_bbox

logger = logging.getLogger(__name__)

DEFAULT_TAGS = {"burglary": 0.45, "robbery": 0.30, "assault": 0.20, "homicide": 0.05}

# Relative intensity Monday..Sunday (normalised to mean 1 internally).
DEFAULT_WEEKLY_PROFILE = (0.90, 0.85, 0.90, 0.95, 1.10, 1.30, 1.00)

# Relative intensity by hour of day, 0..23: quiet early morning, busy evening.
DEFAULT_HOURLY_PROFILE = (
    0.7, 0.5, 0.4, 0.3, 0.25, 0.3, 0.45, 0.7, 0.9, 1.0, 1.05, 1.1,
    1.1, 1.05, 1.1, 1.15, 1.25, 1.4, 1.55, 1.7, 1.75, 1.6, 1.3, 1.0,
)  # fmt: skip


def _study_polygon(study_area):
    """Return the union of the study area geometries as a WGS84 shapely geometry."""
    _check_bbox(study_area)
    return study_area.to_crs(WGS84).geometry.union_all()


def sample_points_in_polygon(polygon, n, rng, max_iterations=1000):
    """
    Draw ``n`` points uniformly inside a polygon by rejection sampling.

    Args:
        polygon (shapely.Geometry): Polygon in WGS84.
        n (int): Number of points.
        rng (numpy.random.Generator): Random generator.
        max_iterations (int): Safety cap on rejection rounds.

    Returns:
        tuple: ``(lon, lat)`` arrays of length ``n``.
    """
    minx, miny, maxx, maxy = polygon.bounds
    lon = np.empty(0)
    lat = np.empty(0)
    fill = polygon.area / ((maxx - minx) * (maxy - miny)) if polygon.area > 0 else 1.0
    batch = int(max(n / max(fill, 1e-3) * 1.2, 64))
    for _ in range(max_iterations):
        cx = rng.uniform(minx, maxx, batch)
        cy = rng.uniform(miny, maxy, batch)
        inside = shapely.contains_xy(polygon, cx, cy)
        lon = np.concatenate([lon, cx[inside]])
        lat = np.concatenate([lat, cy[inside]])
        if len(lon) >= n:
            return lon[:n], lat[:n]
    raise RuntimeError("Could not sample enough points inside the study area.")


def sample_points_around(polygon, centers, sd_km, n_per_center, rng, max_iterations=1000):
    """
    Draw points from Gaussian clouds around hotspot centres, kept inside the polygon.

    Args:
        polygon (shapely.Geometry): Study area in WGS84.
        centers (numpy.ndarray): ``(k, 2)`` array of ``(lon, lat)`` centres.
        sd_km (float or array): Standard deviation of each cloud in km.
        n_per_center (array): Number of points to draw per centre.
        rng (numpy.random.Generator): Random generator.
        max_iterations (int): Safety cap on rejection rounds.

    Returns:
        tuple: ``(lon, lat)`` arrays.
    """
    sd_km = np.broadcast_to(np.asarray(sd_km, dtype=float), (len(centers),))
    lons, lats = [], []
    for (c_lon, c_lat), sd, n in zip(centers, sd_km, n_per_center, strict=True):
        if n == 0:
            continue
        sd_lat = sd / KM_PER_DEG_LAT
        sd_lon = sd / (KM_PER_DEG_LON * max(math.cos(math.radians(c_lat)), 0.05))
        lon = np.empty(0)
        lat = np.empty(0)
        remaining = int(n)
        for _ in range(max_iterations):
            cx = rng.normal(c_lon, sd_lon, remaining * 2)
            cy = rng.normal(c_lat, sd_lat, remaining * 2)
            inside = shapely.contains_xy(polygon, cx, cy)
            lon = np.concatenate([lon, cx[inside]])
            lat = np.concatenate([lat, cy[inside]])
            if len(lon) >= n:
                break
            remaining = int(n - len(lon))
        else:
            raise RuntimeError("Could not sample enough hotspot points inside the study area.")
        lons.append(lon[:n])
        lats.append(lat[:n])
    if not lons:
        return np.empty(0), np.empty(0)
    return np.concatenate(lons), np.concatenate(lats)


def temporal_intensity(
    timestamps,
    start,
    end,
    trend=0.0,
    annual_amplitude=0.0,
    annual_peak_month=1,
    weekly_profile=None,
    hourly_profile=None,
):
    """
    Relative event intensity at each timestamp (mean around 1).

    Args:
        timestamps (DatetimeIndex): Times to evaluate.
        start, end (Timestamp): Bounds of the simulation, used for the trend.
        trend (float): Relative change of the intensity from ``start`` to
            ``end`` (``0.5`` means +50% at the end, ``-0.3`` means -30%).
        annual_amplitude (float): Amplitude of the annual cosine (0-1).
        annual_peak_month (int): Month (1-12) where the annual cycle peaks.
        weekly_profile (sequence): 7 relative weights, Monday to Sunday.
        hourly_profile (sequence): 24 relative weights, hour 0 to 23.

    Returns:
        numpy.ndarray: Intensity values, one per timestamp.
    """
    timestamps = pd.DatetimeIndex(timestamps)
    span = max((end - start).total_seconds(), 1.0)
    frac = (timestamps - start).total_seconds() / span
    lam = np.clip(1.0 + trend * frac, 0.0, None)
    if annual_amplitude:
        peak_doy = (annual_peak_month - 1) * 365.25 / 12 + 15
        lam = lam * (
            1 + annual_amplitude * np.cos(2 * np.pi * (timestamps.dayofyear - peak_doy) / 365.25)
        )
    if weekly_profile is not None:
        weekly = np.asarray(weekly_profile, dtype=float)
        if weekly.shape != (7,):
            raise ValueError("weekly_profile must have 7 values (Monday..Sunday).")
        lam = lam * (weekly / weekly.mean())[timestamps.dayofweek]
    if hourly_profile is not None:
        hourly = np.asarray(hourly_profile, dtype=float)
        if hourly.shape != (24,):
            raise ValueError("hourly_profile must have 24 values (hour 0..23).")
        lam = lam * (hourly / hourly.mean())[timestamps.hour]
    return np.asarray(lam)


def sample_timestamps(n, start, end, rng, max_iterations=1000, **intensity_kwargs):
    """
    Draw ``n`` timestamps from an inhomogeneous process by thinning.

    Args:
        n (int): Number of timestamps.
        start, end (Timestamp): Simulation bounds.
        rng (numpy.random.Generator): Random generator.
        max_iterations (int): Safety cap on thinning rounds.
        **intensity_kwargs: Forwarded to :func:`temporal_intensity`.

    Returns:
        pandas.DatetimeIndex: ``n`` timestamps, unsorted.
    """
    span = (end - start).total_seconds()
    if span <= 0:
        raise ValueError("end must be after start.")
    # Upper bound of the intensity: product of the maxima of each factor.
    lam_max = max(1.0, 1.0 + intensity_kwargs.get("trend", 0.0))
    lam_max *= 1 + abs(intensity_kwargs.get("annual_amplitude", 0.0))
    for key, size in (("weekly_profile", 7), ("hourly_profile", 24)):
        profile = intensity_kwargs.get(key)
        if profile is not None:
            profile = np.asarray(profile, dtype=float)
            if profile.shape != (size,):
                raise ValueError(f"{key} must have {size} values.")
            lam_max *= profile.max() / profile.mean()
    accepted = []
    total = 0
    batch = int(max(n * lam_max * 1.2, 64))
    for _ in range(max_iterations):
        candidates = start + pd.to_timedelta(rng.uniform(0, span, batch), unit="s")
        lam = temporal_intensity(candidates, start, end, **intensity_kwargs)
        keep = rng.uniform(0, lam_max, batch) < lam
        accepted.append(candidates[keep])
        total += int(keep.sum())
        if total >= n:
            break
    else:
        raise RuntimeError("Could not sample enough timestamps.")
    stamps = accepted[0].append(accepted[1:]) if len(accepted) > 1 else accepted[0]
    return stamps[:n].floor("s")


def generate_crimes(
    study_area,
    n_events=5000,
    start="2019-01-01",
    end="2020-12-31",
    n_hotspots=3,
    hotspot_share=0.7,
    hotspot_sd_km=0.5,
    tags=None,
    trend=0.0,
    annual_amplitude=0.2,
    annual_peak_month=1,
    weekly_profile=DEFAULT_WEEKLY_PROFILE,
    hourly_profile=DEFAULT_HOURLY_PROFILE,
    seed=None,
    return_hotspots=False,
):
    """
    Generate synthetic crime events inside a study area.

    Args:
        study_area (GeoDataFrame): Boundary of the study area (any CRS). See
            :func:`predspot.crime_mapping.load_study_area` to fetch one from
            OpenStreetMap.
        n_events (int): Number of events to generate.
        start (str or Timestamp): First possible timestamp.
        end (str or Timestamp): Last possible timestamp.
        n_hotspots (int): Number of Gaussian hotspots (0 for a uniform map).
        hotspot_share (float): Fraction of events that belong to hotspots;
            the rest is uniform background (0-1).
        hotspot_sd_km (float or sequence): Standard deviation of the hotspot
            clouds in km (one value or one per hotspot).
        tags (dict or sequence): Crime types. A dict maps type to relative
            weight; a sequence gives equal weights. Defaults to
            :data:`DEFAULT_TAGS`.
        trend (float): Relative change of the event rate from ``start`` to
            ``end`` (``0.5`` = +50%).
        annual_amplitude (float): Amplitude of the annual cycle (0 disables).
        annual_peak_month (int): Month (1-12) where the annual cycle peaks.
        weekly_profile (sequence or None): 7 weights Monday..Sunday
            (``None`` disables the weekly pattern).
        hourly_profile (sequence or None): 24 weights, hour 0..23
            (``None`` disables the hour-of-day pattern).
        seed (int, optional): Seed for reproducibility.
        return_hotspots (bool): Also return the hotspot centres.

    Returns:
        pandas.DataFrame: Events with ``tag``, ``t``, ``lon``, ``lat`` columns
        sorted by time, ready for :class:`predspot.Dataset`. If
        ``return_hotspots`` is True, a tuple ``(crimes, hotspots)`` where
        ``hotspots`` is a GeoDataFrame with the centre, ``sd_km`` and
        ``share`` of each hotspot.
    """
    if n_events <= 0:
        raise ValueError("n_events must be positive.")
    if not 0 <= hotspot_share <= 1:
        raise ValueError("hotspot_share must be between 0 and 1.")
    if n_hotspots < 0:
        raise ValueError("n_hotspots must be >= 0.")
    rng = np.random.default_rng(seed)
    polygon = _study_polygon(study_area)
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    # --- tags ---------------------------------------------------------------
    if tags is None:
        tags = DEFAULT_TAGS
    if isinstance(tags, dict):
        names, weights = list(tags.keys()), np.asarray(list(tags.values()), dtype=float)
    else:
        names, weights = list(tags), np.ones(len(tags))
    if len(names) == 0 or (weights < 0).any() or weights.sum() == 0:
        raise ValueError("tags must contain at least one type with a positive weight.")
    tag_values = rng.choice(names, size=n_events, p=weights / weights.sum())

    # --- space --------------------------------------------------------------
    if n_hotspots == 0:
        n_hot = 0
    else:
        n_hot = int(round(n_events * hotspot_share))
    n_bg = n_events - n_hot
    hot_lon, hot_lat = sample_points_in_polygon(polygon, n_hotspots, rng)
    centers = np.column_stack([hot_lon, hot_lat]) if n_hotspots else np.empty((0, 2))
    shares = rng.dirichlet(np.full(n_hotspots, 2.0)) if n_hotspots else np.empty(0)
    per_center = rng.multinomial(n_hot, shares) if n_hot else np.zeros(n_hotspots, dtype=int)
    sd_km = np.broadcast_to(np.asarray(hotspot_sd_km, dtype=float), (n_hotspots,))
    lon_h, lat_h = sample_points_around(polygon, centers, sd_km, per_center, rng)
    lon_b, lat_b = (
        sample_points_in_polygon(polygon, n_bg, rng) if n_bg else (np.empty(0), np.empty(0))
    )
    lon = np.concatenate([lon_h, lon_b])
    lat = np.concatenate([lat_h, lat_b])
    order = rng.permutation(n_events)
    lon, lat = lon[order], lat[order]

    # --- time ---------------------------------------------------------------
    timestamps = sample_timestamps(
        n_events,
        start,
        end,
        rng,
        trend=trend,
        annual_amplitude=annual_amplitude,
        annual_peak_month=annual_peak_month,
        weekly_profile=weekly_profile,
        hourly_profile=hourly_profile,
    )

    crimes = (
        pd.DataFrame({"tag": tag_values, "t": timestamps, "lon": lon, "lat": lat})
        .sort_values("t")
        .reset_index(drop=True)
    )
    logger.debug("Generated %d synthetic events with %d hotspots", n_events, n_hotspots)
    if not return_hotspots:
        return crimes
    hotspots = gpd.GeoDataFrame(
        {"sd_km": sd_km, "share": shares * hotspot_share},
        geometry=gpd.points_from_xy(centers[:, 0], centers[:, 1]),
        crs=WGS84,
    )
    return crimes, hotspots
