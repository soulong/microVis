from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microVis.log_utils import get_logger

_log = get_logger("microVis.data_module")


def _infer_plate_dims(wells: list[str]) -> tuple[int, int]:
    """Infer plate dimensions from well names (e.g. B2, D2 → 4 rows, 2 cols)."""
    max_row = 0
    max_col = 0
    for w in wells:
        m = re.match(r"([A-Z]+)(\d+)", w)
        if m:
            max_row = max(max_row, ord(m.group(1)[-1]) - ord("A") + 1)
            max_col = max(max_col, int(m.group(2)))
    return max_row, max_col


class DataModule:
    """Data access layer wrapping microProfiler ImageDataset + results.db."""

    def __init__(self, measurement_dir: str):
        self._root_dir = Path(measurement_dir)
        self._image_dir: Path | None = None
        self._dataset: Any = None
        self._db_conn: sqlite3.Connection | None = None
        self._db_tables: dict[str, dict[str, str]] = {}
        self._metadata: pd.DataFrame | None = None
        self._img_dtype_cache: str | None = None
        self._wells_cache: list[str] | None = None
        self._ready = False

        self._init_dataset()
        self._init_db()
        self._ready = True

    # ── Initialization ─────────────────────────────────────────────

    def _init_dataset(self):
        image_dir = self._root_dir / "image"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"image/ directory not found in {self._root_dir}")

        from microProfiler import ImageDataset

        self._dataset = ImageDataset(str(image_dir))
        self._metadata = self._dataset.metadata
        self._image_dir = image_dir
        _log.info("ImageDataset loaded: %d rows, channels=%s, masks=%s",
                   len(self._metadata), self.channels, self.mask_names)

    def _init_db(self):
        db_path = self._root_dir / "results.db"
        if not db_path.exists():
            _log.info("No results.db found — profiling unavailable")
            return

        self._db_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db_conn.execute("PRAGMA journal_mode=WAL")
        self._db_conn.row_factory = sqlite3.Row

        cursor = self._db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        for (tname,) in cursor.fetchall():
            if tname.startswith("sqlite_"):
                continue
            cur = self._db_conn.execute(f'PRAGMA table_info("{tname}")')
            self._db_tables[tname] = {row[1]: row[2] for row in cur.fetchall()}
        _log.info("DB loaded: tables=%s", list(self._db_tables.keys()))

    # ── Properties ─────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def root_dir(self) -> str:
        return str(self._root_dir)

    @property
    def channels(self) -> list[str]:
        return list(self._dataset.intensity_colnames)

    @property
    def mask_names(self) -> list[str]:
        return [m.replace("mask_", "", 1) for m in self._dataset.mask_colnames]

    @property
    def img_shape(self) -> tuple[int, int]:
        return self._dataset.img_shape

    @property
    def img_dtype(self) -> str:
        if self._img_dtype_cache is not None:
            return self._img_dtype_cache
        try:
            row = self._metadata.iloc[0]
            img, _ = self._dataset.get_imageset(row.name)
            self._img_dtype_cache = str(img.dtype)
        except Exception:
            self._img_dtype_cache = "uint16"
        return self._img_dtype_cache

    def get_wells(self) -> list[str]:
        if self._wells_cache is None:
            self._wells_cache = sorted(self._metadata["well"].unique())
        return self._wells_cache

    def get_fields(self) -> list[int]:
        return sorted(int(f) for f in self._metadata["field"].unique())

    def get_stacks(self) -> list[int]:
        return sorted(int(s) for s in self._metadata["stack"].unique())

    def get_timepoints(self) -> list[int]:
        return sorted(int(t) for t in self._metadata["timepoint"].unique())

    def get_plate_dims(self) -> tuple[int, int]:
        return _infer_plate_dims(self.get_wells())

    # ── DB access ──────────────────────────────────────────────────

    def has_db(self) -> bool:
        return self._db_conn is not None

    def get_profiling_tables(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        exclude = {"directory", "index"}
        for tname, cols in self._db_tables.items():
            profiling = [c for c in cols if c not in exclude]
            if profiling:
                result[tname] = profiling
        return result

    def get_profiling_columns(self, table: str) -> list[tuple[str, str, bool]]:
        cols = self._db_tables.get(table, {})
        exclude = {"directory", "index"}
        result = []
        for cname, ctype in cols.items():
            if cname not in exclude:
                is_num = any(k in ctype.upper() for k in ("INT", "REAL", "FLOAT", "NUM"))
                result.append((cname, ctype, is_num))
        return result

    def get_table_df(self, table: str) -> pd.DataFrame | None:
        if self._db_conn is None:
            return None
        try:
            return pd.read_sql(f'SELECT * FROM "{table}"', self._db_conn)
        except Exception as e:
            _log.warning("get_table_df(%s) failed: %s", table, e)
            return None

    def aggregate(self, table: str, column: str, method: str) -> dict:
        df = self.get_table_df(table)
        if df is None:
            _log.warning("aggregate: get_table_df(%s) returned None", table)
            return {}
        if column not in df.columns:
            _log.warning("aggregate: column %s not in table %s, cols=%s",
                         column, table, list(df.columns[:10]))
            return {}
        if "well" not in df.columns:
            _log.warning("aggregate: no 'well' column in table %s, cols=%s",
                         table, list(df.columns))
            return {}

        col_type = self._db_tables.get(table, {}).get(column, "")
        is_num = any(k in col_type.upper() for k in ("INT", "REAL", "FLOAT", "NUM"))

        _log.debug("aggregate: table=%s col=%s is_num=%s method=%s rows=%d",
                   table, column, is_num, method, len(df))

        if not is_num:
            grouped = df.groupby("well")[column].first()
        elif method == "std":
            grouped = df.groupby("well")[column].std()
        elif method == "sum":
            grouped = df.groupby("well")[column].sum()
        else:
            grouped = df.groupby("well")[column].mean()

        result = grouped.dropna().to_dict()
        _log.debug("aggregate: result has %d wells", len(result))
        return result

    # ── Image access ───────────────────────────────────────────────

    def lookup_row_indices(
        self, wells: list[str], fields: list[int],
        stacks: list[int] | None = None,
        timepoints: list[int] | None = None,
    ) -> list[tuple[int, str, int, int, int]]:
        meta = self._metadata
        if stacks is None:
            stacks = self.get_stacks()
        if timepoints is None:
            timepoints = self.get_timepoints()
        results: list[tuple[int, str, int, int, int]] = []
        for well in wells:
            for field in fields:
                for stack in stacks:
                    for timepoint in timepoints:
                        mask = (
                            (meta["well"] == well)
                            & (meta["field"] == field)
                            & (meta["stack"] == stack)
                            & (meta["timepoint"] == timepoint)
                        )
                        matching = meta[mask]
                        for idx in matching.index:
                            results.append((int(idx), well, field, stack, timepoint))
        return results

    def get_imageset(self, row_idx: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return self._dataset.get_imageset(row_idx)

    def get_imageset_with_masks(
        self, row_idx: int, channels: list[str] | None = None,
        masks: list[str] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return self._dataset.get_imageset(row_idx, channels=channels, masks=masks)

    # ── Cleanup ────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._db_conn is not None:
            self._db_conn.close()
            self._db_conn = None

    # ── Display info ───────────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        n_rows, n_cols = self.get_plate_dims()
        return {
            "measurement_dir": str(self._root_dir),
            "image_dir": str(self._image_dir) if self._image_dir else None,
            "metadata_rows": len(self._metadata),
            "wells": len(self.get_wells()),
            "fields": len(self.get_fields()),
            "stacks": self.get_stacks(),
            "timepoints": self.get_timepoints(),
            "channels": self.channels,
            "masks": self.mask_names,
            "plate_rows": n_rows,
            "plate_cols": n_cols,
            "img_shape": self.img_shape,
            "img_dtype": self.img_dtype,
            "db_connected": self.has_db(),
            "profiling_tables": list(self.get_profiling_tables().keys()),
        }


def parse_plate_metadata(path: str) -> pd.DataFrame:
    """Parse a plate-shaped Excel metadata file into a long DataFrame.

    Each sheet is a plate layout:
      - Row 1: column numbers [None, 1, 2, ...]
      - Col A: row IDs [A, B, C, ...]
      - Data starts at B2 (well A1)

    Returns a DataFrame with 'well' column + one column per sheet name.
    """
    xls = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []

    for sheet_name in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        if raw.empty or raw.shape[0] < 2 or raw.shape[1] < 2:
            continue

        # Column numbers from row 0 (skip first cell which is None)
        col_numbers = raw.iloc[0, 1:].tolist()
        # Row IDs from column 0 (skip header row)
        row_ids = raw.iloc[1:, 0].tolist()

        records = []
        for r_idx, row_id in enumerate(row_ids):
            if pd.isna(row_id):
                continue
            row_id = str(row_id).strip()
            for c_idx, col_num in enumerate(col_numbers):
                if pd.isna(col_num):
                    continue
                well = f"{row_id}{int(col_num)}"
                val = raw.iloc[r_idx + 1, c_idx + 1]
                records.append({"well": well, sheet_name: val})

        if records:
            frames.append(pd.DataFrame(records))

    if not frames:
        return pd.DataFrame(columns=["well"])

    # Merge all sheets on 'well'
    result = frames[0]
    for df in frames[1:]:
        result = result.merge(df, on="well", how="outer")

    # Auto-detect column types: try numeric, fall back to string
    for col in result.columns:
        if col == "well":
            continue
        converted = pd.to_numeric(result[col], errors="coerce")
        if converted.notna().sum() > 0:
            result[col] = converted

    return result
