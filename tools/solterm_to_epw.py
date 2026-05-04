#!/usr/bin/env python3
"""
Convert a simplified SOLTERM-style hourly text file to an EPW file for CEA.

Expected SOLTERM format (whitespace or tab separated, no header, 8760 rows):
    month day hour col4 col5

By default this script assumes:
    col4 = GHI (global horizontal irradiance, Wh/m2)
    col5 = DHI (diffuse horizontal irradiance, Wh/m2)

It uses a base EPW file to preserve non-solar weather variables, and replaces:
    - GHI  (EPW column 14) -> glohorrad_Whm2
    - DNI  (EPW column 15) -> dirnorrad_Whm2
    - DHI  (EPW column 16) -> difhorrad_Whm2

Typical usage:
python tools/solterm_to_epw.py ^
  --solterm "C:\\Users\\Andre\\Downloads\\solterm_simplificado.txt" ^
  --base-epw "C:\\path\\to\\base.epw" ^
  --output-epw "C:\\path\\to\\weather_from_solterm.epw" ^
  --lat 38.72 --lon -9.14 --utc 0
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import re
from dataclasses import dataclass
from pathlib import Path


HOURS_IN_YEAR = 8760
EPW_HEADER_LINES = 8
EPW_MIN_DATA_COLUMNS = 16


@dataclass(frozen=True)
class SoltermRow:
    month: int
    day: int
    hour: int
    col4: float
    col5: float


@dataclass(frozen=True)
class RadiationRow:
    ghi: float
    dni: float
    dhi: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert simplified SOLTERM data to EPW for CEA.")
    parser.add_argument("--solterm", required=True, help="Path to input SOLTERM txt file.")
    parser.add_argument("--base-epw", required=True, help="Path to base EPW file.")
    parser.add_argument("--output-epw", required=True, help="Path to output EPW file.")
    parser.add_argument(
        "--column-order",
        choices=("ghi_dhi", "dni_dhi"),
        default="ghi_dhi",
        help=(
            "Interpretation of SOLTERM columns 4 and 5. "
            "Default: ghi_dhi (col4=GHI, col5=DHI)."
        ),
    )
    parser.add_argument("--lat", type=float, default=None, help="Latitude in degrees. Defaults to base EPW location.")
    parser.add_argument("--lon", type=float, default=None, help="Longitude in degrees. Defaults to base EPW location.")
    parser.add_argument(
        "--utc",
        type=float,
        default=None,
        help="UTC offset in hours for standard time. Defaults to base EPW location.",
    )
    parser.add_argument("--year", type=int, default=None, help="Optional override of EPW year column.")
    parser.add_argument(
        "--max-dni",
        type=float,
        default=1200.0,
        help="Upper clipping limit for computed DNI (Wh/m2). Default 1200.",
    )
    parser.add_argument(
        "--cosz-min",
        type=float,
        default=0.065,
        help="Minimum cosine(zenith) used to infer DNI from beam horizontal. Default 0.065.",
    )
    return parser.parse_args()


def read_solterm(path: Path) -> list[SoltermRow]:
    rows: list[SoltermRow] = []
    splitter = re.compile(r"\s+")

    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = [p for p in splitter.split(line) if p]
            if len(parts) != 5:
                raise ValueError(f"Invalid row at line {line_number}: expected 5 columns, got {len(parts)}.")
            month, day, hour = map(int, parts[:3])
            col4, col5 = map(float, parts[3:])
            rows.append(SoltermRow(month=month, day=day, hour=hour, col4=col4, col5=col5))

    if len(rows) != HOURS_IN_YEAR:
        raise ValueError(f"Invalid number of rows in {path}: expected {HOURS_IN_YEAR}, got {len(rows)}.")

    # Validate date/hour values and uniqueness of month/day/hour keys.
    seen_keys: set[tuple[int, int, int]] = set()
    for r in rows:
        try:
            dt.date(2021, r.month, r.day)  # non-leap validation
        except ValueError as e:
            raise ValueError(f"Invalid month/day found in SOLTERM data: {(r.month, r.day)}") from e
        if r.hour < 1 or r.hour > 24:
            raise ValueError(f"Invalid hour found in SOLTERM data: {r.hour}. Expected 1..24.")
        key = (r.month, r.day, r.hour)
        if key in seen_keys:
            raise ValueError(f"Duplicate month/day/hour entry found in SOLTERM data: {key}.")
        seen_keys.add(key)

    if len(seen_keys) != HOURS_IN_YEAR:
        raise ValueError("SOLTERM file does not contain exactly one entry per hour of a non-leap year.")

    return rows


def read_epw(path: Path) -> tuple[list[list[str]], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < EPW_HEADER_LINES + 1:
        raise ValueError(f"EPW file seems too short: {path}")

    header = rows[:EPW_HEADER_LINES]
    data = rows[EPW_HEADER_LINES:]

    if len(data) != HOURS_IN_YEAR:
        raise ValueError(f"Base EPW must have {HOURS_IN_YEAR} data rows, got {len(data)}.")

    for idx, row in enumerate(data, start=1):
        if len(row) < EPW_MIN_DATA_COLUMNS:
            raise ValueError(f"EPW data row {idx} has too few columns: {len(row)}.")

    return header, data


def parse_epw_location(header_rows: list[list[str]]) -> tuple[float, float, float]:
    location = header_rows[0]
    if len(location) < 9 or location[0].strip().upper() != "LOCATION":
        raise ValueError("Invalid EPW LOCATION line in header.")
    lat = float(location[6])
    lon = float(location[7])
    utc = float(location[8])
    return lat, lon, utc


def day_of_year(month: int, day: int) -> int:
    return dt.date(2021, month, day).timetuple().tm_yday


def solar_cos_zenith(month: int, day: int, hour_epw: int, lat_deg: float, lon_deg: float, utc_offset: float) -> float:
    doy = day_of_year(month, day)
    lat = math.radians(lat_deg)

    # Cooper's declination approximation.
    decl = math.radians(23.45 * math.sin(math.radians(360.0 * (284 + doy) / 365.0)))

    # Equation of time (minutes).
    b = math.radians(360.0 * (doy - 81) / 364.0)
    eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)

    # EPW hour is end-of-hour (1..24), use hour center.
    local_clock = hour_epw - 0.5
    lstm = 15.0 * utc_offset
    solar_time = local_clock + (4.0 * (lon_deg - lstm) + eot) / 60.0
    hra = math.radians(15.0 * (solar_time - 12.0))

    cosz = (
        math.sin(lat) * math.sin(decl)
        + math.cos(lat) * math.cos(decl) * math.cos(hra)
    )
    return max(0.0, min(1.0, cosz))


def clip_non_negative(value: float) -> float:
    return value if value > 0.0 else 0.0


def infer_radiation_rows(
    rows: list[SoltermRow],
    lat: float,
    lon: float,
    utc: float,
    column_order: str,
    max_dni: float,
    cosz_min: float,
) -> tuple[dict[tuple[int, int, int], RadiationRow], int]:
    radiation: dict[tuple[int, int, int], RadiationRow] = {}
    clipped_diffuse_count = 0

    for r in rows:
        key = (r.month, r.day, r.hour)
        cosz = solar_cos_zenith(r.month, r.day, r.hour, lat, lon, utc)

        if column_order == "ghi_dhi":
            ghi = clip_non_negative(r.col4)
            dhi = clip_non_negative(r.col5)
            if dhi > ghi:
                dhi = ghi
                clipped_diffuse_count += 1
            beam_horizontal = max(0.0, ghi - dhi)
            dni = beam_horizontal / cosz if (cosz > cosz_min and beam_horizontal > 0.0) else 0.0
        else:  # column_order == "dni_dhi"
            dni = clip_non_negative(r.col4)
            dhi = clip_non_negative(r.col5)
            if cosz <= 0.0:
                dni = 0.0
            ghi = dhi + dni * max(0.0, cosz)

        dni = min(max_dni, max(0.0, dni))
        radiation[key] = RadiationRow(ghi=ghi, dni=dni, dhi=dhi)

    return radiation, clipped_diffuse_count


def replace_epw_solar_columns(
    epw_data_rows: list[list[str]],
    radiation_by_key: dict[tuple[int, int, int], RadiationRow],
    year_override: int | None,
) -> list[list[str]]:
    updated_rows: list[list[str]] = []
    missing_keys: set[tuple[int, int, int]] = set()

    for idx, row in enumerate(epw_data_rows, start=1):
        month = int(row[1])
        day = int(row[2])
        hour = int(row[3])
        key = (month, day, hour)
        rad = radiation_by_key.get(key)
        if rad is None:
            missing_keys.add(key)
            continue

        row_out = list(row)
        if year_override is not None:
            row_out[0] = str(year_override)

        row_out[13] = str(int(round(rad.ghi)))  # glohorrad_Whm2
        row_out[14] = str(int(round(rad.dni)))  # dirnorrad_Whm2
        row_out[15] = str(int(round(rad.dhi)))  # difhorrad_Whm2
        updated_rows.append(row_out)

    if missing_keys:
        sample = sorted(missing_keys)[:10]
        raise ValueError(
            f"Could not find radiation values for {len(missing_keys)} EPW timestamps. Sample: {sample}"
        )
    if len(updated_rows) != HOURS_IN_YEAR:
        raise ValueError(f"Unexpected number of updated EPW rows: {len(updated_rows)}")

    return updated_rows


def write_epw(path: Path, header_rows: list[list[str]], data_rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(header_rows)
        writer.writerows(data_rows)


def main() -> None:
    args = parse_args()
    solterm_path = Path(args.solterm)
    base_epw_path = Path(args.base_epw)
    output_epw_path = Path(args.output_epw)

    if not solterm_path.exists():
        raise FileNotFoundError(f"SOLTERM file not found: {solterm_path}")
    if not base_epw_path.exists():
        raise FileNotFoundError(f"Base EPW file not found: {base_epw_path}")

    solterm_rows = read_solterm(solterm_path)
    epw_header, epw_data = read_epw(base_epw_path)
    base_lat, base_lon, base_utc = parse_epw_location(epw_header)

    lat = base_lat if args.lat is None else args.lat
    lon = base_lon if args.lon is None else args.lon
    utc = base_utc if args.utc is None else args.utc

    radiation_by_key, clipped_diffuse_count = infer_radiation_rows(
        rows=solterm_rows,
        lat=lat,
        lon=lon,
        utc=utc,
        column_order=args.column_order,
        max_dni=args.max_dni,
        cosz_min=args.cosz_min,
    )

    updated_epw_data = replace_epw_solar_columns(
        epw_data_rows=epw_data,
        radiation_by_key=radiation_by_key,
        year_override=args.year,
    )

    # Keep a short trace of provenance in COMMENTS 2.
    stamp = dt.date.today().isoformat()
    provenance = (
        f"Converted from {solterm_path.name} using {base_epw_path.name} on {stamp}; "
        f"column-order={args.column_order}"
    )
    if len(epw_header) > 6:
        epw_header[6] = ["COMMENTS 2", provenance]

    write_epw(output_epw_path, epw_header, updated_epw_data)

    print("Conversion finished.")
    print(f"  Input SOLTERM: {solterm_path}")
    print(f"  Base EPW:      {base_epw_path}")
    print(f"  Output EPW:    {output_epw_path}")
    print(f"  Location used: lat={lat}, lon={lon}, utc={utc}")
    if clipped_diffuse_count:
        print(f"  Note: clipped DHI>GHI in {clipped_diffuse_count} rows.")


if __name__ == "__main__":
    main()
