"""
================================================================================
Distribute Output Matrix Loading, NoHAM Sector Translation and Comparison
================================================================================

PURPOSE:
    Loads OD demand matrices produced by the distribution process, translates
    zone indices from NoRMITS zones to NoHAM sectors, and compares translated
    matrices against post-ME target matrices using SQV, GEH and PerDiff metrics.

FOLDER STRUCTURE:
    Distribute outputs:
        ROOT_DIR / p{purpose} / {mode}_{ts}_p{purpose}_{fr|to} /
            {mode}_{ts}_p{purpose}_{fr|to}_matrix.csv

    Translated outputs (saved alongside source):
        ROOT_DIR / p{purpose} / {mode}_{ts}_p{purpose}_{fr|to} /
            {mode}_{ts}_p{purpose}_{fr|to}_sector_matrix.csv

    Target matrices (flat folder, .csv.bz2):
        TARGET_DIR / OD_{mode}_{ts}_p{purpose}_{fr|to}.csv.bz2   (HB)
        TARGET_DIR / OD_{mode}_{ts}_p{purpose}_nhb.csv.bz2       (NHB)

METRICS:
    SQV     = 1 / ( 1 + sqrt( (M-C)^2 / (f*C) ) )
    GEH     = sqrt( 2*(M-C)^2 / (M+C) )
    PerDiff = 100 * (M-C) / max(C, floor)

CLASS STRUCTURE:
    PipelineConfig            -- all user-configurable settings (dataclass)
    MatrixKey                 -- immutable (purpose, period, direction) identifier
    MatrixStore               -- typed container with iteration and bulk apply()
    NohamSectorZoning         -- resolves sector names to integer zone IDs
    ZoneTranslator            -- loads correspondence and applies caf.toolkit translation
    MatrixLoader              -- reads CSVs from disk into MatrixStores
    MetricCalculator          -- floor application, SQV / GEH / PerDiff computation
    Reporter                  -- log tables, statistics, sector summaries, CSV exports
    MatrixComparisonPipeline  -- orchestrates all steps with per-step opt-in/out
================================================================================
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import caf.toolkit as ctk


# =============================================================================
# CONSTANTS — pipeline step names
# =============================================================================

@dataclass
class StepConfig:
    """
    Boolean switches controlling which pipeline steps are executed.

    Set a field to False to skip that step entirely.  Steps that depend on
    the output of a skipped step will log a warning and also be skipped
    automatically if their prerequisite data is unavailable.

    Attributes
    ----------
    translate : bool
        Step 3 — Load raw distribute-output CSVs, translate NoRMITS zones to
        NoHAM sector IDs, and save *_sector_matrix.csv alongside each source
        file.  Disable when translated CSVs already exist on disk.
    load_translated : bool
        Step 4 — Load the saved *_sector_matrix.csv files into memory.
    load_targets : bool
        Step 5 — Load post-ME target matrices (.csv.bz2) from TARGET_DIR.
    floor : bool
        Step 6 — Apply the demand floor to both translated and target stores.
    metrics : bool
        Step 7 — Compute cell-wise SQV, GEH and PerDiff metric matrices.
    export_metrics : bool
        Step 8 — Write one CSV per (key, metric) to OUTPUT_DIR.
    stats : bool
        Step 9 — Compute per-matrix metric distribution statistics CSV.
    totals : bool
        Step 10 — Grand-total comparison table (processed vs target) CSV.
    origin_summary : bool
        Step 11 — Origin-sector demand summary (row-sum totals) CSV.
    dest_summary : bool
        Step 12 — Destination-sector demand summary (col-sum totals) CSV.
    """
    translate:       bool = True
    load_translated: bool = True
    load_targets:    bool = True
    floor:           bool = True
    metrics:         bool = True
    export_metrics:  bool = True
    stats:           bool = True
    totals:          bool = True
    origin_summary:  bool = True
    dest_summary:    bool = True

@dataclass
class MetricConfig:
    """
    Boolean switches controlling which cell-level metrics are computed.

    Used by MetricCalculator.compute_all() and Reporter.compute_stats().
    Disabling a metric skips both its calculation and all downstream outputs
    (per-cell CSV export, statistics rows, log table section).

    Attributes
    ----------
    sqv : bool
        Compute SQV (Standardised Quality Value).
        SQV = 1 / (1 + sqrt((M-C)^2 / (f*C))).  Bounded (0, 1].
    geh : bool
        Compute GEH statistic.
        GEH = sqrt(2*(M-C)^2 / (M+C)).  Transport-standard fit test.
    perdiff : bool
        Compute percentage difference.
        PerDiff = 100*(M-C) / max(C, floor).  Signed, unbounded.
    """
    sqv:     bool = False
    geh:     bool = True
    perdiff: bool = False



# =============================================================================
# LOGGING HELPER
# =============================================================================

def _build_logger(log_dir: Path, name: str = "matrix_comparison") -> logging.Logger:
    """Create (or reconfigure) a named logger writing to console and a file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = log_dir / f"log_comp_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger


log: logging.Logger = logging.getLogger("noham_translation")


# =============================================================================
# PipelineConfig
# =============================================================================

@dataclass
class PipelineConfig:
    """
    All user-configurable settings for the translation and comparison pipeline.

    Required parameters (no defaults — must be supplied explicitly):
        root_dir, target_dir, output_dir, translation_path,
        noham_sector_zoning_path

    Optional parameters with defaults:
        mode, time_periods, hb_purposes, nhb_purposes, directions,
        nhb_direction,
        trans_csv_columns,
        trans_from_col, trans_to_col, trans_factors_col,
        demand_floor, sqv_factor, metric_config, log_dir

    Attributes
    ----------
    root_dir : Path
        Root folder containing the distribute-output subfolders (p1-p8, p11-p18).
    target_dir : Path
        Flat folder containing post-ME target matrices (.csv.bz2).
    output_dir : Path
        Root folder for exported comparison CSVs and statistics.
    translation_path : Path
        Path to the NoRMITS → NoHAM sector spatial correspondence CSV.
    noham_sector_zoning_path : Path
        Path to the NoHAM sector zoning CSV (columns: zone_id, zone_name).
        zone_name must match the noham_sector_id strings in translation_path.
    mode : str
        Mode prefix in folder/file names, e.g. "m3".
    time_periods : list[str]
        Time period codes, e.g. ["ts1", "ts2", "ts3"].
    hb_purposes : list[int]
        Home-Based purpose IDs.
    nhb_purposes : list[int]
        Non-Home-Based purpose IDs.
    directions : list[str]
        Direction suffixes for HB, e.g. ["fr", "to"].
    nhb_direction : str
        Fixed direction suffix for NHB files, e.g. "nhb".
    trans_csv_columns : list[str]
        The four column names assigned in order to the columns of
        translation_path when loaded::

            [col0_sector_name, col1_from_zone,
             col2_sector_to_from, col3_from_to_sector]

        col0 holds NoHAM sector name strings (e.g. "Barnsley"), resolved
        to integer zone IDs via the zoning file.  col3 is the zone-to-sector
        weight used by caf.toolkit.
    trans_from_col : str
        Internal name for the "from zone" column
        (must equal trans_csv_columns[1]).
    trans_to_col : str
        Internal name for the resolved integer sector ID column.
    trans_factors_col : str
        Internal name for the translation weight column
        (must equal trans_csv_columns[3]).
    demand_floor : float
        Minimum cell value applied before metric calculation (avoids ÷0).
    sqv_factor : float
        Scaling factor f in the SQV denominator.
    metric_config : MetricConfig
        Boolean switches selecting which metrics (SQV, GEH, PerDiff) to
        compute. Defaults to MetricConfig() which enables all three.
    log_dir : Path or None
        Log file directory. Defaults to output_dir when None.
    """

    # ── Required paths ────────────────────────────────────────────────────
    root_dir:                 Path
    target_dir:               Path
    output_dir:               Path
    translation_path:         Path
    noham_sector_zoning_path: Path

    # ── Model dimensions ─────────────────────────────────────────────────
    mode:          str
    time_periods:  list
    hb_purposes:   list
    nhb_purposes:  list
    directions:    list
    nhb_direction: str

    # ── Raw correspondence CSV column names (col 0-3 in the CSV) ──────────────
    #
    #   Assigned to the four positional columns when translation_path is
    #   loaded.  To adapt to a different CSV layout, change only these four
    #   values in main() — no other code needs to change.
    #
    trans_csv_columns: list
    # ── Internal (caf.toolkit) column names ──────────────────────────────────
    #
    #   Used by caf.toolkit for "from zone", "to zone" and "factor".
    #   trans_from_col must equal trans_csv_columns[1];
    #   trans_factors_col must equal trans_csv_columns[3].
    #
    trans_from_col:    str
    trans_to_col:      str
    trans_factors_col: str
    # ── Metric settings ───────────────────────────────────────────────────
    demand_floor:  float
    sqv_factor:    float
    metric_config: MetricConfig
    log_dir:       Optional[Path] = None

    # ── Derived helpers ───────────────────────────────────────────────────

    @property
    def all_purposes(self) -> list:
        return self.hb_purposes + self.nhb_purposes

    @property
    def effective_log_dir(self) -> Path:
        return Path(self.log_dir) if self.log_dir else self.output_dir

    def is_nhb(self, purpose: int) -> bool:
        return purpose in self.nhb_purposes

    def purpose_type(self, purpose: int) -> str:
        return "NHB" if self.is_nhb(purpose) else "HB"

    def effective_direction(self, purpose: int, direction: str) -> str:
        """Return the actual file/folder suffix for a given purpose."""
        return self.nhb_direction if self.is_nhb(purpose) else direction

    # ── Path builders ─────────────────────────────────────────────────────

    def _stem(self, purpose: int, period: str, direction: str) -> str:
        """Shared stem: {mode}_{period}_p{purpose}_{eff_dir}."""
        eff = self.effective_direction(purpose, direction)
        return f"{self.mode}_{period}_p{purpose}_{eff}"

    def distribute_matrix_path(self, purpose: int, period: str, direction: str) -> Path:
        stem = self._stem(purpose, period, direction)
        return self.root_dir / f"p{purpose}" / stem / f"{stem}_matrix.csv"

    def translated_matrix_path(self, purpose: int, period: str, direction: str) -> Path:
        stem = self._stem(purpose, period, direction)
        return self.root_dir / f"p{purpose}" / stem / f"{stem}_sector_matrix.csv"

    def target_matrix_path(self, purpose: int, period: str, direction: str) -> Path:
        stem  = self._stem(purpose, period, direction)
        fname = f"OD_{stem}.csv.bz2"
        return self.target_dir / fname

    def metric_export_path(self, purpose: int, period: str, direction: str, metric: str) -> Path:
        stem = self._stem(purpose, period, direction)
        return self.output_dir / f"p{purpose}" / stem / f"{stem}_{metric}.csv"


# =============================================================================
# MatrixKey
# =============================================================================

@dataclass(frozen=True, order=True)
class MatrixKey:
    """Immutable identifier for a single OD matrix."""
    purpose:   int
    period:    str
    direction: str

    def __str__(self) -> str:
        return f"p{self.purpose} {self.period} {self.direction}"


# =============================================================================
# MatrixStore
# =============================================================================

class MatrixStore:
    """
    Typed container for a collection of wide OD demand matrices.

    Internally a flat dict keyed by MatrixKey — O(1) lookup, no repeated
    purposes/periods/directions argument threading.
    """

    def __init__(self, name: str = "") -> None:
        self.name  = name
        self._data: Dict[MatrixKey, Optional[pd.DataFrame]] = {}

    def set(self, key: MatrixKey, df: Optional[pd.DataFrame]) -> None:
        self._data[key] = df

    def get(self, key: MatrixKey) -> Optional[pd.DataFrame]:
        return self._data.get(key)

    def __contains__(self, key: MatrixKey) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"MatrixStore({self.name!r}, loaded={self.n_loaded}, missing={self.n_missing})"

    def items(self) -> Iterator[Tuple[MatrixKey, Optional[pd.DataFrame]]]:
        return iter(self._data.items())

    def valid_items(self) -> Iterator[Tuple[MatrixKey, pd.DataFrame]]:
        for k, df in self._data.items():
            if df is not None:
                yield k, df

    def keys(self) -> Iterator[MatrixKey]:
        return iter(self._data.keys())

    @property
    def n_loaded(self) -> int:
        return sum(1 for df in self._data.values() if df is not None)

    @property
    def n_missing(self) -> int:
        return len(self._data) - self.n_loaded

    def missing_keys(self) -> list:
        return [k for k, df in self._data.items() if df is None]

    def apply(self, func, name: str = "") -> "MatrixStore":
        """Return a new store with func applied to every non-None matrix."""
        out = MatrixStore(name=name or self.name)
        for k, df in self._data.items():
            out.set(k, func(df) if df is not None else None)
        return out


# =============================================================================
# NohamSectorZoning
# =============================================================================

class NohamSectorZoning:
    """
    Resolves NoHAM sector names to integer zone IDs via zoning.csv.

    The correspondence CSV contains sector *name* strings in its to-column.
    This class loads the canonical zoning file (zone_id int, zone_name str)
    and builds lookup dicts so ZoneTranslator can substitute integer IDs
    before calling caf.toolkit, ensuring output matrices share the same
    integer index as the post-ME target matrices.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config     = config
        self.name_to_id: dict = {}
        self.id_to_name: dict = {}
        self.zone_ids:   list = []

    def load(self) -> "NohamSectorZoning":
        df = pd.read_csv(
            self.config.noham_sector_zoning_path,
            usecols=["zone_id", "zone_name"],
        )
        df["zone_id"]   = df["zone_id"].astype(int)
        df["zone_name"] = df["zone_name"].astype(str).str.strip()

        self.name_to_id = dict(zip(df["zone_name"], df["zone_id"]))
        self.id_to_name = dict(zip(df["zone_id"],   df["zone_name"]))
        self.zone_ids   = sorted(df["zone_id"].tolist())

        log.info(
            f"  NohamSectorZoning: {len(df):,} sectors  "
            f"(IDs {df['zone_id'].min()}-{df['zone_id'].max()})"
        )
        return self


# =============================================================================
# ZoneTranslator
# =============================================================================

class ZoneTranslator:
    """
    Loads the NoRMITS → NoHAM sector correspondence and translates matrices.

    On load(), the correspondence CSV's name-based to-column is replaced with
    integer zone IDs (via NohamSectorZoning), so caf.toolkit produces matrices
    directly indexed by the same integer IDs used in the target matrices.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config:          PipelineConfig          = config
        self.correspondence:  Optional[pd.DataFrame]  = None
        self.zoning:          Optional[NohamSectorZoning] = None

    def load(self) -> "ZoneTranslator":
        """Load correspondence CSV and resolve sector names to integer zone IDs."""
        self.zoning = NohamSectorZoning(self.config).load()

        df = pd.read_csv(self.config.translation_path)
        cols = self.config.trans_csv_columns
        df.columns = cols
        sector_name_col = cols[0]  # col 0: NoHAM sector name string

        df["_zone_id"] = (
            df[sector_name_col]
            .astype(str).str.strip().map(self.zoning.name_to_id)
        )
        n_bad = df["_zone_id"].isna().sum()
        if n_bad:
            bad = df.loc[
                df["_zone_id"].isna(), sector_name_col
            ].unique().tolist()
            raise ValueError(
                f"ZoneTranslator: {n_bad} rows unmapped. "
                f"Unrecognised names (first 10): {bad[:10]}"
            )
        df["_zone_id"] = df["_zone_id"].astype(int)

        corr = df[[
            self.config.trans_from_col,
            "_zone_id",
            self.config.trans_factors_col,
        ]].rename(columns={"_zone_id": self.config.trans_to_col}).copy()
        corr[self.config.trans_from_col] = corr[self.config.trans_from_col].astype(int)

        self.correspondence = corr
        log.info(
            f"  ZoneTranslator: {len(corr):,} rows  "
            f"({corr[self.config.trans_from_col].nunique():,} NoRMITS zones → "
            f"{corr[self.config.trans_to_col].nunique():,} NoHAM sector IDs  "
            f"range {corr[self.config.trans_to_col].min()}-"
            f"{corr[self.config.trans_to_col].max()})"
        )
        return self

    def translate(self, matrix: pd.DataFrame, label: str = "") -> pd.DataFrame:
        """Translate one wide OD matrix from NoRMITS zones to NoHAM sector IDs."""
        if self.correspondence is None:
            raise RuntimeError("Call ZoneTranslator.load() before translate().")

        in_total   = matrix.values.sum()
        translated = ctk.translation.pandas_matrix_zone_translation(
            matrix                  = matrix,
            zone_correspondence     = self.correspondence,
            translation_from_col    = self.config.trans_from_col,
            translation_to_col      = self.config.trans_to_col,
            translation_factors_col = self.config.trans_factors_col,
            check_totals            = True,
        )
        out_total = translated.values.sum()
        pct       = abs(out_total - in_total) / in_total * 100 if in_total else 0.0
        log.info(
            f"    [{label}]  "
            f"{matrix.shape[0]} zones → {translated.shape[0]} sectors  "
            f"in={in_total:,.4f}  out={out_total:,.4f}  ({pct:.4f}% diff)"
        )
        return translated


# =============================================================================
# MatrixLoader
# =============================================================================

class MatrixLoader:
    """Reads distribute-output, translated sector, and target CSVs from disk."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        """Read a wide OD CSV (plain or .bz2), cast index and columns to int."""
        df = pd.read_csv(path, index_col=0)
        df.index      = df.index.astype(int)
        df.index.name = "o_zon"
        df.columns    = df.columns.astype(int)
        log.info(
            f"    Loaded: {path.name}  "
            f"({df.shape[0]}×{df.shape[1]}, total={df.values.sum():,.4f})"
        )
        return df

    def _load_nhb_aware(
        self,
        path_fn,           # callable(purpose, period, direction) -> Path
        store_name: str,
        entity: str,       # "distribute" / "translated" / "target" for logging
    ) -> MatrixStore:
        """
        Generic loader that handles the HB (fr/to) vs NHB (nhb) split.

        For NHB purposes, the single file is loaded once per (purpose, period)
        and stored under both "fr" and "to" keys so the MatrixStore has a
        uniform key structure throughout the pipeline.

        Parameters
        ----------
        path_fn : callable
            Maps (purpose, period, direction) → Path of the CSV to load.
        store_name : str
            Name for the created MatrixStore.
        entity : str
            Label used in log messages.
        """
        store     = MatrixStore(name=store_name)
        nhb_cache: Dict[Tuple[int, str], Optional[pd.DataFrame]] = {}

        for p in self.config.all_purposes:
            for ts in self.config.time_periods:

                if self.config.is_nhb(p):
                    ck = (p, ts)
                    if ck not in nhb_cache:
                        path = path_fn(p, ts, "fr")   # effective_direction → "nhb"
                        if not path.exists():
                            log.warning(f"  MISSING {entity} (NHB): {path}")
                            nhb_cache[ck] = None
                        else:
                            log.info(f"  Loading {entity} [p{p} {ts} nhb]")
                            try:
                                nhb_cache[ck] = self._read_csv(path)
                            except Exception as exc:
                                log.error(f"  ERROR loading {entity} [p{p} {ts} nhb]: {exc}")
                                nhb_cache[ck] = None
                    else:
                        log.info(f"  Reusing cached NHB {entity} for [p{p} {ts}]")
                    for d in self.config.directions:
                        store.set(MatrixKey(p, ts, d), nhb_cache[ck])

                else:
                    for d in self.config.directions:
                        key  = MatrixKey(p, ts, d)
                        path = path_fn(p, ts, d)
                        if not path.exists():
                            log.warning(f"  MISSING {entity}: {path}")
                            store.set(key, None)
                        else:
                            log.info(f"  Loading {entity} [{key}]")
                            try:
                                store.set(key, self._read_csv(path))
                            except Exception as exc:
                                log.error(f"  ERROR loading {entity} [{key}]: {exc}")
                                store.set(key, None)

        log.info(f"  {store_name}: {store.n_loaded} loaded, {store.n_missing} missing.")
        return store

    def load_distribute_outputs(self) -> MatrixStore:
        return self._load_nhb_aware(
            self.config.distribute_matrix_path, "distribute_outputs", "distribute"
        )

    def load_translated_matrices(self) -> MatrixStore:
        return self._load_nhb_aware(
            self.config.translated_matrix_path, "translated", "translated"
        )

    def load_target_matrices(self) -> MatrixStore:
        return self._load_nhb_aware(
            self.config.target_matrix_path, "targets", "target"
        )


# =============================================================================
# MetricCalculator
# =============================================================================

class MetricCalculator:
    """Applies demand floor and computes SQV, GEH and PerDiff metrics."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def apply_floor(self, matrix: pd.DataFrame) -> pd.DataFrame:
        return matrix.clip(lower=self.config.demand_floor)

    def apply_floor_store(self, store: MatrixStore) -> MatrixStore:
        return store.apply(self.apply_floor, name=f"{store.name}_floored")

    def sqv(self, M: np.ndarray, C: np.ndarray) -> np.ndarray:
        """SQV = 1 / (1 + sqrt((M-C)^2 / (f*C)))"""
        return 1.0 / (1.0 + np.sqrt((M - C) ** 2 / (self.config.sqv_factor * C)))

    def geh(self, M: np.ndarray, C: np.ndarray) -> np.ndarray:
        """GEH = sqrt(2*(M-C)^2 / (M+C))"""
        return np.sqrt(2.0 * (M - C) ** 2 / (M + C))

    def perdiff(self, M: np.ndarray, C: np.ndarray) -> np.ndarray:
        """PerDiff = 100 * (M-C) / max(C, floor)"""
        return 100.0 * (M - C) / np.maximum(C, self.config.demand_floor)

    def _align(
        self, mod: pd.DataFrame, tgt: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.Index, pd.Index]:
        """Align two DataFrames on outer join, return numpy arrays + axes."""
        m_al, c_al = mod.align(tgt, join="outer", fill_value=self.config.demand_floor)
        return m_al.values.astype(float), c_al.values.astype(float), m_al.index, m_al.columns

    def compute_all(
        self,
        translated_floored: MatrixStore,
        targets_floored:    MatrixStore,
    ) -> Dict[MatrixKey, Optional[Dict[str, pd.DataFrame]]]:
        """
        Compute enabled metrics for every key present in both stores.

        Which metrics are computed is controlled by
        config.metric_config (a MetricConfig instance).
        """
        mc       = self.config.metric_config
        enabled  = [n for n, flag in [("sqv", mc.sqv), ("geh", mc.geh), ("perdiff", mc.perdiff)] if flag]
        if not enabled:
            log.warning("  compute_all: no metrics enabled in MetricConfig — skipping.")
            return {}
        log.info(f"  Enabled metrics: {enabled}")

        results  = {}
        all_keys = sorted(set(translated_floored.keys()) | set(targets_floored.keys()))

        for key in all_keys:
            mod_df = translated_floored.get(key)
            tgt_df = targets_floored.get(key)

            if mod_df is None or tgt_df is None:
                log.warning(
                    f"  Skipping metrics [{key}] — "
                    f"modelled={'ok' if mod_df is not None else 'MISSING'}  "
                    f"target={'ok' if tgt_df is not None else 'MISSING'}"
                )
                results[key] = None
                continue

            try:
                M, C, idx, cols = self._align(mod_df, tgt_df)
                result = {}

                if mc.sqv:
                    sqv_arr        = self.sqv(M, C)
                    result["sqv"]  = pd.DataFrame(sqv_arr, index=idx, columns=cols)
                if mc.geh:
                    geh_arr        = self.geh(M, C)
                    result["geh"]  = pd.DataFrame(geh_arr, index=idx, columns=cols)
                if mc.perdiff:
                    pct_arr           = self.perdiff(M, C)
                    result["perdiff"] = pd.DataFrame(pct_arr, index=idx, columns=cols)

                results[key] = result

                parts = []
                if mc.sqv:
                    parts.append(f"SQV mean={sqv_arr.mean():.4f}  SQV>0.8={(sqv_arr > 0.8).mean()*100:.1f}%")
                if mc.geh:
                    parts.append(f"GEH mean={geh_arr.mean():.4f}  GEH<5={(geh_arr < 5).mean()*100:.1f}%")
                if mc.perdiff:
                    parts.append(f"PerDiff mean={pct_arr.mean():.2f}%  |<10%|={(np.abs(pct_arr) < 10).mean()*100:.1f}%")
                log.info(f"  Metrics [{key}]  " + "  ".join(parts))

            except Exception as exc:
                log.error(f"  ERROR computing metrics [{key}]: {exc}")
                results[key] = None

        return results


# =============================================================================
# Reporter
# =============================================================================

class Reporter:
    """Produces log tables, statistics, sector summaries and CSV exports."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _diff_cols(
        proc_col: str,
        tgt_col: str,
        rows: list,
        trans_col: str,
        tgt_col_name: str,
    ) -> None:
        """Append abs_diff / pct_diff to each row dict in-place."""
        for r in rows:
            p, t = r.get(proc_col, np.nan), r.get(tgt_col, np.nan)
            has  = not (
                (isinstance(p, float) and np.isnan(p)) or
                (isinstance(t, float) and np.isnan(t))
            )
            r["abs_diff"] = abs(p - t) if has else np.nan
            r["pct_diff"] = abs(p - t) / t * 100 if (has and t > 0) else np.nan

    def _base_row(self, key: MatrixKey) -> dict:
        """Return the common prefix columns for a summary row."""
        is_nhb = self.config.is_nhb(key.purpose)
        return {
            "purpose":      f"p{key.purpose}",
            "period":       key.period,
            "direction":    self.config.nhb_direction if is_nhb else key.direction,
            "purpose_type": self.config.purpose_type(key.purpose),
        }

    def _unique_keys(self, *stores: MatrixStore) -> list:
        """
        Return sorted unique MatrixKeys, deduplicating NHB (which stores the
        same matrix under both 'fr' and 'to' direction keys).
        """
        seen_nhb: Set[Tuple[int, str]] = set()
        out = []
        for key in sorted(set().union(*(set(s.keys()) for s in stores))):
            if self.config.is_nhb(key.purpose):
                ck = (key.purpose, key.period)
                if ck in seen_nhb:
                    continue
                seen_nhb.add(ck)
            out.append(key)
        return out

    # ── Export metric matrices ────────────────────────────────────────────

    def export_metric_matrices(
        self,
        metrics: Dict[MatrixKey, Optional[Dict[str, pd.DataFrame]]],
    ) -> None:
        """Write per-cell SQV / GEH / PerDiff matrices to CSV."""
        exported = 0
        for key, result in metrics.items():
            if result is None:
                continue
            for metric_name, df in result.items():
                fpath = self.config.metric_export_path(
                    key.purpose, key.period, key.direction, metric_name
                )
                fpath.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(fpath)
                exported += 1
        log.info(f"  Metric files exported: {exported}")

    # ── Distribution statistics ───────────────────────────────────────────

    def compute_stats(
        self,
        metrics: Dict[MatrixKey, Optional[Dict[str, pd.DataFrame]]],
    ) -> pd.DataFrame:
        """
        Per-(key, metric) distribution statistics with quality-band percentages.

        Quality bands:
            GEH    : <5 / 5-10 / >10
            SQV    : >0.95 / 0.80-0.95 / <0.80
            PerDiff: |x|<5% / 5-10% / >10%
        """
        n_total = len(metrics)
        n_none  = sum(1 for v in metrics.values() if v is None)
        log.info(
            f"  compute_stats: {n_total} keys, "
            f"{n_total - n_none} with results, {n_none} skipped."
        )
        if n_none:
            log.warning(
                f"  Skipped keys (first 10): "
                f"{[str(k) for k, v in sorted(metrics.items()) if v is None][:10]}"
            )

        rows = []
        for key, result in sorted(metrics.items()):
            if result is None:
                continue
            for metric_name, df in result.items():
                vals = df.values.flatten()
                n    = len(vals)
                row  = {
                    **self._base_row(key),
                    "metric":   metric_name,
                    "n_cells":  n,
                    "mean":     np.mean(vals),
                    "std":      np.std(vals),
                    "min":      np.min(vals),
                    "p5":       np.percentile(vals, 5),
                    "p25":      np.percentile(vals, 25),
                    "median":   np.median(vals),
                    "p75":      np.percentile(vals, 75),
                    "p95":      np.percentile(vals, 95),
                    "max":      np.max(vals),
                }
                if metric_name == "geh":
                    row["band1_pct"]  = (vals < 5).sum()  / n * 100
                    row["band2_pct"]  = ((vals >= 5) & (vals < 10)).sum() / n * 100
                    row["band3_pct"]  = (vals >= 10).sum() / n * 100
                    row["band_label"] = "GEH<5 / 5-10 / >10"
                elif metric_name == "sqv":
                    row["band1_pct"]  = (vals > 0.95).sum() / n * 100
                    row["band2_pct"]  = ((vals >= 0.80) & (vals <= 0.95)).sum() / n * 100
                    row["band3_pct"]  = (vals < 0.80).sum() / n * 100
                    row["band_label"] = "SQV>0.95 / 0.80-0.95 / <0.80"
                else:  # perdiff
                    av = np.abs(vals)
                    row["band1_pct"]  = (av < 5).sum()  / n * 100
                    row["band2_pct"]  = ((av >= 5) & (av < 10)).sum() / n * 100
                    row["band3_pct"]  = (av >= 10).sum() / n * 100
                    row["band_label"] = "|PerDiff|<5% / 5-10% / >10%"
                rows.append(row)

        if not rows:
            log.warning(
                "  compute_stats: no rows — check translated/target matrices loaded."
            )
        return pd.DataFrame(rows)

    def log_stats_table(self, stats: pd.DataFrame) -> None:
        """Print statistics to the log, grouped by metric."""
        log.info("\n  METRIC DISTRIBUTION STATISTICS")

        if stats.empty or "metric" not in stats.columns:
            log.warning("  No statistics to display — stats DataFrame is empty.")
            return

        for metric in ["sqv", "geh", "perdiff"]:
            sub = stats[stats["metric"] == metric].copy()
            if sub.empty:
                continue
            log.info(f"\n  --- {metric.upper()} ---  {sub['band_label'].iloc[0]}")
            w = [8, 6, 5, 5, 7, 7, 7, 7, 7, 7, 9, 9, 9]
            sep = "  " + "-" * (sum(w) + len(w))
            log.info(sep)
            log.info(
                f"  {'Purpose':<{w[0]}} {'Period':<{w[1]}} {'Dir':<{w[2]}}"
                f" {'Type':<{w[3]}} {'Mean':>{w[4]}} {'Std':>{w[5]}}"
                f" {'P5':>{w[6]}} {'Med':>{w[7]}} {'P95':>{w[8]}}"
                f" {'Max':>{w[9]}} {'Band1%':>{w[10]}} {'Band2%':>{w[11]}} {'Band3%':>{w[12]}}"
            )
            log.info(sep)
            for _, r in sub.iterrows():
                log.info(
                    f"  {r.purpose:<{w[0]}} {r.period:<{w[1]}} {r.direction:<{w[2]}}"
                    f" {r.purpose_type:<{w[3]}}"
                    f" {r['mean']:>{w[4]}.3f} {r['std']:>{w[5]}.3f}"
                    f" {r['p5']:>{w[6]}.3f} {r['median']:>{w[7]}.3f}"
                    f" {r['p95']:>{w[8]}.3f} {r['max']:>{w[9]}.3f}"
                    f" {r['band1_pct']:>{w[10]}.1f}%"
                    f" {r['band2_pct']:>{w[11]}.1f}%"
                    f" {r['band3_pct']:>{w[12]}.1f}%"
                )
            log.info(sep)

    # ── Grand-total comparison ────────────────────────────────────────────

    def compute_totals_summary(
        self,
        translated_store: MatrixStore,
        target_store:     MatrixStore,
    ) -> pd.DataFrame:
        """Grand-total per matrix: translated vs target with abs/pct diff."""
        rows = []
        for key in self._unique_keys(translated_store, target_store):
            trans_df = translated_store.get(key)
            tgt_df   = target_store.get(key)
            row = {
                **self._base_row(key),
                "processed_total": trans_df.values.sum() if trans_df is not None else np.nan,
                "target_total":    tgt_df.values.sum()   if tgt_df   is not None else np.nan,
            }
            rows.append(row)
        self._diff_cols("processed_total", "target_total", rows, "", "")
        df = pd.DataFrame(rows)
        log.info(
            f"  Totals summary: {len(df)} rows  "
            f"(processed={df['processed_total'].sum():,.4f}  "
            f"target={df['target_total'].sum():,.4f})"
        )
        return df

    def log_totals_table(self, summary: pd.DataFrame) -> None:
        w = [8, 6, 5, 5, 16, 16, 14, 10]
        sep = "  " + "-" * (sum(w) + len(w) - 1)
        log.info("\n  PROCESSED vs TARGET TOTAL SUMMARY")
        log.info(sep)
        log.info(
            f"  {'Purpose':<{w[0]}} {'Period':<{w[1]}} {'Dir':<{w[2]}}"
            f" {'Type':<{w[3]}} {'Processed':>{w[4]}} {'Target':>{w[5]}}"
            f" {'Abs diff':>{w[6]}} {'% diff':>{w[7]}}"
        )
        log.info(sep)

        def _f(v, fmt=",.2f"):
            return f"{v:{fmt}}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"

        for _, r in summary.iterrows():
            log.info(
                f"  {r.purpose:<{w[0]}} {r.period:<{w[1]}} {r.direction:<{w[2]}}"
                f" {r.purpose_type:<{w[3]}} {_f(r.processed_total):>{w[4]}}"
                f" {_f(r.target_total):>{w[5]}}"
                f" {_f(r.abs_diff,',.4f'):>{w[6]}} {_f(r.pct_diff,'.4f'):>{w[7]}}"
            )
        log.info(sep)
        log.info(f"  Grand total (processed): {summary['processed_total'].sum():,.4f}")
        log.info(f"  Grand total (target):    {summary['target_total'].sum():,.4f}")

    # ── Sector-marginal summaries (Checks 2 & 3) ─────────────────────────

    def _sector_summary(
        self,
        translated_store: MatrixStore,
        target_store:     MatrixStore,
        axis: int,
        axis_label: str,
    ) -> pd.DataFrame:
        """
        Aggregate demand along one matrix axis and compare processed vs target.

        Parameters
        ----------
        axis : int
            1 = row sums → origin sector totals.
            0 = column sums → destination sector totals.
        axis_label : str
            Column name for the sector ID, e.g. "origin_sector_id".
        """
        rows = []
        for key in self._unique_keys(translated_store, target_store):
            trans_df = translated_store.get(key)
            tgt_df   = target_store.get(key)
            base     = self._base_row(key)

            proc_margin = trans_df.sum(axis=axis) if trans_df is not None else pd.Series(dtype=float)
            tgt_margin  = tgt_df.sum(axis=axis)   if tgt_df  is not None else pd.Series(dtype=float)

            for sector_id in sorted(proc_margin.index.union(tgt_margin.index)):
                row = {
                    **base,
                    axis_label:          sector_id,
                    "processed_total":   proc_margin.get(sector_id, np.nan),
                    "target_total":      tgt_margin.get(sector_id,  np.nan),
                }
                rows.append(row)

        self._diff_cols("processed_total", "target_total", rows, "", "")
        return pd.DataFrame(rows)

    def compute_origin_sector_summary(
        self,
        translated_store: MatrixStore,
        target_store:     MatrixStore,
    ) -> pd.DataFrame:
        """Row-sum (origin) sector totals: processed vs target."""
        df = self._sector_summary(translated_store, target_store, axis=1, axis_label="origin_sector_id")
        log.info(
            f"  Origin sector summary: {len(df)} rows  "
            f"({df['origin_sector_id'].nunique() if not df.empty else 0} sectors)"
        )
        return df

    def compute_destination_sector_summary(
        self,
        translated_store: MatrixStore,
        target_store:     MatrixStore,
    ) -> pd.DataFrame:
        """Column-sum (destination) sector totals: processed vs target."""
        df = self._sector_summary(translated_store, target_store, axis=0, axis_label="destination_sector_id")
        log.info(
            f"  Destination sector summary: {len(df)} rows  "
            f"({df['destination_sector_id'].nunique() if not df.empty else 0} sectors)"
        )
        return df

    # ── CSV save helper ───────────────────────────────────────────────────

    def save_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Save a DataFrame to output_dir / filename."""
        path = self.config.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        log.info(f"  Saved: {path}")


# =============================================================================
# MatrixComparisonPipeline
# =============================================================================

class MatrixComparisonPipeline:
    """
    Orchestrates the full translation and comparison workflow.

    Each pipeline step is named and can be individually enabled or disabled
    via the ``steps`` parameter of run().

    Each step is controlled by a boolean field in StepConfig:
        "translate"       — load raw CSVs, translate, save sector CSVs
        "load_translated" — load saved sector CSVs
        "load_targets"    — load post-ME target matrices
        "floor"           — apply demand floor to both stores
        "metrics"         — compute SQV / GEH / PerDiff
        "export_metrics"  — write per-cell metric CSVs
        "stats"           — metric distribution statistics CSV
        "totals"          — grand-total comparison CSV
        "origin_summary"  — origin-sector demand summary CSV
        "dest_summary"    — destination-sector demand summary CSV

    Example
    -------
    >>> cfg      = PipelineConfig(...)
    >>> pipeline = MatrixComparisonPipeline(cfg)

    # Run everything
    >>> results = pipeline.run()

    # Skip translation (sector CSVs already exist) and metric export

    # Run comparison checks only (sector CSVs and targets already loaded)
    >>> results = pipeline.run(steps={"load_translated", "load_targets",
    ...                                "totals", "origin_summary", "dest_summary"})
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config     = config
        self.loader     = MatrixLoader(config)
        self.translator = ZoneTranslator(config)
        self.calculator = MetricCalculator(config)
        self.reporter   = Reporter(config)

    # ── Internal step runner ──────────────────────────────────────────────

    def _run_step(self, name: str, active_steps: Set[str], func, *args):
        """Execute func(*args) only if name is in active_steps, else skip."""
        if name in active_steps:
            return func(*args)
        log.info(f"  Skipping step: {name}")
        return None

    def run(self, steps: Optional[StepConfig] = None) -> dict:
        """
        Execute the pipeline, honouring the boolean switches in ``steps``.

        Parameters
        ----------
        steps : StepConfig, optional
            Boolean switches for each step.  Defaults to StepConfig() which
            enables every step.  Pass a customised instance to skip steps:

                pipeline.run(StepConfig(translate=False, export_metrics=False))

        Returns
        -------
        dict
            Keys: config, translated, targets, translated_floored,
                  targets_floored, metrics, stats, summary,
                  origin_summary, dest_summary.
            Values are None for any skipped step.
        """
        global log
        active = steps if steps is not None else StepConfig()

        # ── step 1: initiate─────────────────────────────────────────────
        log = _build_logger(self.config.effective_log_dir)
        log.info("=" * 70)
        log.info("NoHAM Sector Translation & Comparison Pipeline")
        log.info("=" * 70)
        log.info("  Step switches:")
        for _field in active.__dataclass_fields__:
            _val = getattr(active, _field)
            _mark = "RUN " if _val else "SKIP"
            log.info(f"    [{_mark}]  {_field}")
        for attr in (
            "demand_floor sqv_factor mode time_periods "
            "hb_purposes nhb_purposes"
        ).split():
            log.info(f"  {attr} = {getattr(self.config, attr)}")
        mc = self.config.metric_config
        log.info(f"  metrics: sqv={mc.sqv}  geh={mc.geh}  perdiff={mc.perdiff}")

        # ── Step 2: zone correspondence ───────────────────────────────────
        if active.translate:
            log.info("\nSTEP 2: Load zone correspondence")
            self.translator.load()

        # ── Step 3: translate ─────────────────────────────────────────────
        if active.translate:
            log.info("\nSTEP 3: Translate distribute outputs → sector CSVs")
            seen_nhb: Set[Tuple[int, str]] = set()
            for p in self.config.all_purposes:
                for ts in self.config.time_periods:
                    dirs = ["fr"] if self.config.is_nhb(p) else self.config.directions
                    for d in dirs:
                        if self.config.is_nhb(p):
                            ck = (p, ts)
                            if ck in seen_nhb:
                                continue
                            seen_nhb.add(ck)
                            label = f"p{p} {ts} nhb"
                        else:
                            label = str(MatrixKey(p, ts, d))

                        raw_path = self.config.distribute_matrix_path(p, ts, d)
                        if not raw_path.exists():
                            log.warning(f"  MISSING: {raw_path}")
                            continue
                        log.info(f"  Processing [{label}]")
                        try:
                            df     = MatrixLoader._read_csv(raw_path)
                            out_df = self.translator.translate(df, label=label)
                            out_path = self.config.translated_matrix_path(p, ts, d)
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            out_df.to_csv(out_path)
                            log.info(f"    → {out_path.name}")
                        except Exception as exc:
                            log.error(f"  ERROR [{label}]: {exc}")

        # ── Step 4: load translated ───────────────────────────────────────
        translated = None
        if active.load_translated:
            log.info("\nSTEP 4: Load translated sector matrices")
            translated = self.loader.load_translated_matrices()

        # ── Step 5: load targets ──────────────────────────────────────────
        targets = None
        if active.load_targets:
            log.info("\nSTEP 5: Load post-ME target matrices")
            targets = self.loader.load_target_matrices()

        # ── Step 6: floor ─────────────────────────────────────────────────
        translated_floored = targets_floored = None
        if active.floor:
            if translated is None or targets is None:
                log.warning("  floor skipped — translated or targets store is None.")
            else:
                log.info(f"\nSTEP 6: Apply demand floor ({self.config.demand_floor})")
                translated_floored = self.calculator.apply_floor_store(translated)
                targets_floored    = self.calculator.apply_floor_store(targets)

        # ── Step 7: metrics ───────────────────────────────────────────────
        metrics = None
        if active.metrics:
            if translated_floored is None or targets_floored is None:
                log.warning("  metrics skipped — floored stores not available.")
            else:
                mc = self.config.metric_config
                enabled = [n for n, f in [("sqv", mc.sqv), ("geh", mc.geh), ("perdiff", mc.perdiff)] if f]
                log.info(f"\nSTEP 7: Compute metrics ({', '.join(enabled) or 'NONE'})")
                metrics = self.calculator.compute_all(translated_floored, targets_floored)

        # ── Step 8: export metric matrices ────────────────────────────────
        if active.export_metrics:
            if metrics is None:
                log.warning("  export_metrics skipped — metrics not computed.")
            else:
                log.info(f"\nSTEP 8: Export metric matrices → {self.config.output_dir}")
                self.reporter.export_metric_matrices(metrics)

        # ── Step 9: statistics ────────────────────────────────────────────
        stats = None
        if active.stats:
            if metrics is None:
                log.warning("  stats skipped — metrics not computed.")
            else:
                log.info("\nSTEP 9: Distribution statistics")
                stats = self.reporter.compute_stats(metrics)
                self.reporter.log_stats_table(stats)
                self.reporter.save_csv(stats, "metric_statistics.csv")

        # ── Step 10: grand-total summary ──────────────────────────────────
        summary = None
        if active.totals:
            if translated is None or targets is None:
                log.warning("  totals skipped — translated or targets store is None.")
            else:
                log.info("\nSTEP 10: Grand-total comparison")
                summary = self.reporter.compute_totals_summary(translated, targets)
                self.reporter.log_totals_table(summary)
                self.reporter.save_csv(summary, "totals_summary.csv")

        # ── Step 11: origin sector summary ───────────────────────────────
        origin_summary = None
        if active.origin_summary:
            if translated is None or targets is None:
                log.warning("  origin_summary skipped — translated or targets store is None.")
            else:
                log.info("\nSTEP 11: Origin-sector demand summary")
                origin_summary = self.reporter.compute_origin_sector_summary(
                    translated, targets
                )
                self.reporter.save_csv(origin_summary, "origin_sector_summary.csv")

        # ── Step 12: destination sector summary ──────────────────────────
        dest_summary = None
        if active.dest_summary:
            if translated is None or targets is None:
                log.warning("  dest_summary skipped — translated or targets store is None.")
            else:
                log.info("\nSTEP 12: Destination-sector demand summary")
                dest_summary = self.reporter.compute_destination_sector_summary(
                    translated, targets
                )
                self.reporter.save_csv(dest_summary, "destination_sector_summary.csv")

        log.info("\nPipeline complete.")

        return {
            "config":              self.config,
            "translated":          translated,
            "targets":             targets,
            "translated_floored":  translated_floored,
            "targets_floored":     targets_floored,
            "metrics":             metrics,
            "stats":               stats,
            "summary":             summary,
            "origin_summary":      origin_summary,
            "dest_summary":        dest_summary,
        }


# =============================================================================
# ENTRY POINT
# =============================================================================

def main(steps: Optional[StepConfig] = None) -> dict:
    """
    Run the pipeline with explicit path configuration.

    Parameters
    ----------
    steps : StepConfig, optional
        Boolean step switches. Defaults to StepConfig() (all steps enabled).

        Examples
        --------
        # Skip translation — translated sector CSVs already on disk:
        main(steps=StepConfig(translate=False))

        # Comparison checks only (no re-translation, no metric export):
        main(steps=StepConfig(
            translate=False, export_metrics=False,
        ))
    """
    cfg = PipelineConfig(
        # ── Required paths ────────────────────────────────────────────────
        root_dir = Path(
            r"I:\Prior adjustment\distribution\distribute_outputs\full_run_27_02_v1"
        ),
        target_dir = Path(
            r"I:\Prior adjustment\distribution\postme_p"
        ),
        output_dir = Path(
            r"I:\Prior adjustment\distribution\comparison_outputs_v2"
        ),
        translation_path = Path(
            r"I:\Data\Zone Translations\cache\noham_sector_normits"
            r"\noham_sector_to_normits_spatial.csv"
        ),
        noham_sector_zoning_path = Path(
            r"I:\Data\Zoning Systems\core_zoning\noham_sector\zoning.csv"
        ),
        # ── Model dimensions (edit if run changes) ────────────────────────
        mode         = "m3",
        time_periods = ["ts1", "ts2", "ts3"],
        hb_purposes  = list(range(1, 9)),
        nhb_purposes = [11, 12, 13, 14, 15, 16, 17, 18],
        directions   = ["fr", "to"],
        nhb_direction = "nhb",
        # ── Raw correspondence CSV column names (col 0-3 of the CSV) ────────
        trans_csv_columns = [
            "noham_sector_id",         # col 0: NoHAM sector name string
            "normits_id",              # col 1: NoRMITS zone ID
            "noham_sector_to_normits", # col 2: sector→zone weight
            "normits_to_noham_sector", # col 3: zone→sector weight (used by caf.toolkit)
        ],
        # ── Internal translation column names (used by caf.toolkit) ──────
        trans_from_col    = "normits_id",
        trans_to_col      = "noham_sector_id",
        trans_factors_col = "normits_to_noham_sector",
        # ── Metric settings ───────────────────────────────────────────────
        demand_floor  = 1e-6,
        sqv_factor    = 1.0,
        metric_config = MetricConfig(
            sqv     = False,  # SQV is less interpretable for very small flows → disable by default
            geh     = True,
            perdiff = False,  # PerDiff can be very volatile for small flows → disable by default
        ),
        log_dir = None,   # defaults to output_dir
    )

    pipeline = MatrixComparisonPipeline(cfg)
    return pipeline.run(steps)


if __name__ == "__main__":
    main()
    # main(steps=StepConfig(
    #     translate=False,       # skip translation (sector CSVs already exist)
    #     export_metrics=False, # skip metric export (already done in a previous run)
    # ))
