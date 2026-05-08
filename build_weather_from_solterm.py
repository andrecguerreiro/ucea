#!/usr/bin/env python
"""
Build an EPW weather file from SOLTERM simplified data.

What this script does:
1. Reads a base EPW (for header + non-solar columns).
2. Reads SOLTERM text rows: month day hour GHI DHI.
3. Replaces EPW GHI/DHI with SOLTERM values.
4. Computes geometry-based DNI using:
      BHI = max(GHI - DHI, 0)
      DNI = BHI / cos(zenith)  (when sun is above horizon threshold)
5. Writes a new EPW and prints validation metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


EPW_HEADER_LINES = 8
EPW_COLUMNS = 35
HOURS_PER_YEAR = 8760

# EPW data columns (0-based)
COL_MONTH = 1
COL_DAY = 2
COL_HOUR = 3
COL_GHI = 13
COL_DNI = 14
COL_DHI = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create EPW from SOLTERM GHI/DHI and geometry-based DNI."
    )
    parser.add_argument(
        "--base-epw",
        type=Path,
        default=Path("weather.epw"),
        help="Base EPW path (header + non-solar columns). Default: weather.epw",
    )
    parser.add_argument(
        "--solterm-txt",
        type=Path,
        default=Path("solterm_simplificado.txt"),
        help="SOLTERM simplified text file. Default: solterm_simplificado.txt",
    )
    parser.add_argument(
        "--output-epw",
        type=Path,
        default=Path("weather_from_solterm.epw"),
        help="Output EPW path. Default: weather_from_solterm.epw",
    )
    parser.add_argument(
        "--cosz-min",
        type=float,
        default=0.065,
        help="Minimum cos(zenith) threshold to compute DNI. Default: 0.065",
    )
    parser.add_argument(
        "--dni-max",
        type=float,
        default=1200.0,
        help="Upper clamp for DNI [W/m2]. Default: 1200",
    )
    return parser.parse_args()


def read_epw(epw_path: Path) -> Tuple[List[str], List[List[str]]]:
    with epw_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        lines = f.readlines()

    if len(lines) < EPW_HEADER_LINES:
        raise ValueError(f"Invalid EPW (too few lines): {epw_path}")

    header = lines[:EPW_HEADER_LINES]
    raw_rows = list(csv.reader(lines[EPW_HEADER_LINES:]))

    # Keep only valid-looking data rows.
    rows = [r[:EPW_COLUMNS] for r in raw_rows if len(r) >= EPW_COLUMNS and any(c.strip() for c in r)]
    if len(rows) != HOURS_PER_YEAR:
        raise ValueError(
            f"Expected {HOURS_PER_YEAR} EPW rows, got {len(rows)} (raw parsed={len(raw_rows)})"
        )

    return header, rows


def read_solterm(solterm_path: Path) -> List[Tuple[int, int, int, float, float]]:
    rows: List[Tuple[int, int, int, float, float]] = []
    with solterm_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            month = int(parts[0])
            day = int(parts[1])
            hour = int(parts[2])
            ghi = float(parts[3])
            dhi = float(parts[4])
            rows.append((month, day, hour, ghi, dhi))

    if len(rows) != HOURS_PER_YEAR:
        raise ValueError(f"Expected {HOURS_PER_YEAR} SOLTERM rows, got {len(rows)}")

    return rows


def get_location_from_epw_header(header: Sequence[str]) -> Tuple[float, float, float]:
    loc = header[0].strip().split(",")
    if len(loc) < 10 or loc[0].upper() != "LOCATION":
        raise ValueError("EPW LOCATION line is missing or malformed.")

    latitude = float(loc[6])
    longitude = float(loc[7])
    utc_offset = float(loc[8])
    return latitude, longitude, utc_offset


def day_of_year(month: int, day: int) -> int:
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return sum(month_days[: month - 1]) + day


def solar_cos_zenith(
    month: int, day: int, hour_1_to_24: int, latitude_deg: float, longitude_deg: float, utc_offset: float
) -> float:
    """
    Approximate solar geometry (hourly).
    Uses mid-hour local standard time.
    """
    latitude = math.radians(latitude_deg)
    lstm_deg = 15.0 * utc_offset
    n = day_of_year(month, day)

    # Midpoint of EPW hour (EPW hour is end-of-hour convention).
    local_clock_hour = (hour_1_to_24 - 1) + 0.5

    b = math.radians((360.0 / 365.0) * (n - 81))
    equation_of_time_min = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
    time_correction_min = 4.0 * (longitude_deg - lstm_deg) + equation_of_time_min
    local_solar_time_h = local_clock_hour + time_correction_min / 60.0

    hour_angle = math.radians(15.0 * (local_solar_time_h - 12.0))
    declination = math.radians(23.45 * math.sin(math.radians((360.0 / 365.0) * (284 + n))))

    cosz = (
        math.sin(latitude) * math.sin(declination)
        + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle)
    )
    return max(-1.0, min(1.0, cosz))


def derive_dni(ghi: float, dhi: float, cosz: float, cosz_min: float, dni_max: float) -> float:
    bhi = max(ghi - dhi, 0.0)
    if cosz > cosz_min and bhi > 0.0:
        dni = bhi / cosz
    else:
        dni = 0.0
    return max(0.0, min(dni, dni_max))


def update_rows(
    epw_rows: List[List[str]],
    sol_rows: Sequence[Tuple[int, int, int, float, float]],
    latitude_deg: float,
    longitude_deg: float,
    utc_offset: float,
    cosz_min: float,
    dni_max: float,
) -> Tuple[List[List[str]], List[float], int, int]:
    out_rows: List[List[str]] = []
    abs_ghi_recon_errors: List[float] = []
    ghi_mismatch = 0
    dhi_mismatch = 0

    for i, row in enumerate(epw_rows):
        month, day, hour, ghi, dhi = sol_rows[i]

        em = int(float(row[COL_MONTH]))
        ed = int(float(row[COL_DAY]))
        eh = int(float(row[COL_HOUR]))
        if (em, ed, eh) != (month, day, hour):
            raise ValueError(
                f"Timestamp mismatch at row {i + 1}: EPW {(em, ed, eh)} vs SOLTERM {(month, day, hour)}"
            )

        cosz = solar_cos_zenith(month, day, hour, latitude_deg, longitude_deg, utc_offset)
        dni = derive_dni(ghi, dhi, cosz, cosz_min, dni_max)

        new_row = row.copy()
        new_row[COL_GHI] = str(int(round(ghi)))
        new_row[COL_DHI] = str(int(round(dhi)))
        new_row[COL_DNI] = str(int(round(dni)))
        out_rows.append(new_row)

        if int(round(float(new_row[COL_GHI]))) != int(round(ghi)):
            ghi_mismatch += 1
        if int(round(float(new_row[COL_DHI]))) != int(round(dhi)):
            dhi_mismatch += 1

        ghi_recon = dhi + dni * max(cosz, 0.0)
        abs_ghi_recon_errors.append(abs(ghi_recon - ghi))

    return out_rows, abs_ghi_recon_errors, ghi_mismatch, dhi_mismatch


def write_epw(output_path: Path, header: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        f.writelines(header)
        for row in rows:
            f.write(",".join(row) + "\n")


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = int(q * (len(s) - 1))
    return s[idx]


def main() -> None:
    args = parse_args()
    header, epw_rows = read_epw(args.base_epw)
    sol_rows = read_solterm(args.solterm_txt)
    latitude_deg, longitude_deg, utc_offset = get_location_from_epw_header(header)

    out_rows, abs_errors, ghi_mismatch, dhi_mismatch = update_rows(
        epw_rows=epw_rows,
        sol_rows=sol_rows,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        utc_offset=utc_offset,
        cosz_min=args.cosz_min,
        dni_max=args.dni_max,
    )

    write_epw(args.output_epw, header, out_rows)

    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(e * e for e in abs_errors) / len(abs_errors))
    p90 = percentile(abs_errors, 0.90)
    maxe = max(abs_errors)

    print(f"Base EPW:   {args.base_epw.resolve()}")
    print(f"SOLTERM:    {args.solterm_txt.resolve()}")
    print(f"Output EPW: {args.output_epw.resolve()}")
    print(f"Location used (header): lat={latitude_deg}, lon={longitude_deg}, utc={utc_offset}")
    print(f"Rows written: {len(out_rows)}")
    print(f"GHI mismatches vs SOLTERM: {ghi_mismatch}")
    print(f"DHI mismatches vs SOLTERM: {dhi_mismatch}")
    print(
        "Reconstruction |GHI - (DHI + DNI*cosz)| "
        f"MAE={mae:.2f}, RMSE={rmse:.2f}, P90={p90:.2f}, MAX={maxe:.2f} W/m2"
    )


if __name__ == "__main__":
    main()
