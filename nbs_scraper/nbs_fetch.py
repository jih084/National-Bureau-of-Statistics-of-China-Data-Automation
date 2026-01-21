#!/usr/bin/env python3
"""
NBS Provincial Power Generation Scraper (Wide Export)
====================================================

What this script does
---------------------
Downloads provincial monthly power generation data from NBS EasyQuery (data.stats.gov.cn)
for selected energy types and exports in an NBS-like *wide* table format.

NEW (current version)
---------------------
- Input regions by *name* (can pass multiple, comma-separated) and resolve codes via --regions_csv
- Multi-region wide output in one file per energy (layout similar to the screenshot)
- Keep ONLY: {energy}发电量累计值(万千瓦时)
  - Drops other rows (当期值 / 同比增长 / 累计同比 etc)
- Unit conversion: 亿千瓦时 -> 万千瓦时 by multiplying values by 10000
- Optional IME adjustment: if 内蒙古自治区 (150000) is included, adds an extra block "内蒙古自治区_IME"
  in the SAME output file (no separate output file).

Typical usage
-------------
python nbs_fetch.py --reg "内蒙古自治区,黑龙江省,吉林省" --energies thermal --time "2011,-2023" --regions_csv nbs_regions.csv
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

BASE = "https://data.stats.gov.cn/easyquery.htm"

INDICATORS = {
    "thermal": {"base": "A03010H", "cn": "火力"},
    "wind":    {"base": "A03010K", "cn": "风力"},
    "solar":   {"base": "A03010L", "cn": "太阳能"},
}


def cumulative_zb_code(energy_type: str) -> str:
    # NBS pattern: base + "02" = 累计值
    return INDICATORS[energy_type]["base"] + "02"


HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://data.stats.gov.cn/easyquery.htm?cn=E0101",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def safe_tag(s: str) -> str:
    """Make a string safe for filenames."""
    return re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")


def parse_time_to_yyyymm_range(time_value: str) -> Tuple[str, str, str]:
    """
    Convert NBS time filter strings to (start_yyyymm, end_yyyymm, label).

    Supports:
      - "2011,-2023" -> ("201101", "202312", "2011-2023")
      - "LAST13" -> returns empty range ("", "", "LAST13") (caller should provide start/end)
    """
    tv = time_value.strip().strip('"').strip("'")
    m = re.fullmatch(r"(\d{4}),-(\d{4})", tv)
    if m:
        y1, y2 = m.group(1), m.group(2)
        return f"{y1}01", f"{y2}12", f"{y1}-{y2}"
    return "", "", tv


def normalize_nbs_response(data):
    """
    NBS sometimes returns JSON where 'returndata' is a JSON string, or returns
    an error payload. This normalizes returndata into a dict or raises a helpful error.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            raise RuntimeError(f"NBS response was a plain string (not JSON object): {data[:200]}")

    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected NBS response type: {type(data)}")

    rd = data.get("returndata", None)

    # returndata sometimes comes back as a JSON-encoded string
    if isinstance(rd, str):
        try:
            rd = json.loads(rd)
        except Exception:
            raise RuntimeError(f"NBS returned error string in 'returndata': {rd}")
        data["returndata"] = rd

    if "returndata" not in data or not isinstance(data.get("returndata"), dict):
        raise RuntimeError(f"Unexpected 'returndata' type: {type(data.get('returndata'))}")

    return data


def query_data(reg_code: str, energy_type: str, time_value: str) -> Dict:
    """
    Query the NBS EasyQuery API for one province and one energy type.

    reg_code: province code like "230000"
    energy_type: one of INDICATORS keys
    time_value: NBS time filter string (e.g. "2011,-2023" or "LAST13")
    """
    if energy_type not in INDICATORS:
        raise ValueError(f"Unknown energy_type={energy_type}. Use one of {list(INDICATORS)}")

    # IMPORTANT: valuecode must be the indicator CODE, not the whole dict
    indicator_code = INDICATORS[energy_type]["base"]

    wds = [{"wdcode": "reg", "valuecode": reg_code}]
    dfwds = [{"wdcode": "zb", "valuecode": indicator_code}]
    if time_value:
        dfwds.append({"wdcode": "sj", "valuecode": time_value})

    params = {
        "m": "QueryData",
        "dbcode": "fsyd",
        "rowcode": "zb",
        "colcode": "sj",
        "wds": json.dumps(wds, ensure_ascii=False),
        "dfwds": json.dumps(dfwds, ensure_ascii=False),
        "h": "1",
    }

    r = SESSION.get(BASE, params=params, timeout=30)
    r.raise_for_status()

    # Sometimes the server returns JSON-as-string; normalize it
    data = r.json()
    data = normalize_nbs_response(data)
    return data


def parse_to_long(resp: Dict, reg_code: str, energy_type: str) -> pd.DataFrame:
    """
    Convert NBS returndata -> long DataFrame.

    Columns:
      reg_code, energy_type, zb_code, zb_name, sj_code, sj_name, value
    """
    rd = resp.get("returndata", {})
    datanodes = rd.get("datanodes", [])
    wdnodes = rd.get("wdnodes", [])

    # (wdcode, code) -> name
    label_map: Dict[Tuple[str, str], str] = {}
    for wd in wdnodes:
        wdcode = wd.get("wdcode")
        for node in wd.get("nodes", []):
            code = node.get("code")
            name = node.get("name")
            if wdcode and code is not None:
                label_map[(wdcode, str(code))] = name

    rows = []
    for dn in datanodes:
        wds = {x["wdcode"]: str(x["valuecode"]) for x in dn.get("wds", [])}
        value = dn.get("data", {}).get("data", None)

        zb_code = wds.get("zb")
        sj_code = wds.get("sj")

        rows.append({
            "reg_code": reg_code,
            "energy_type": energy_type,
            "zb_code": zb_code,
            "zb_name": label_map.get(("zb", zb_code), ""),
            "sj_code": sj_code,
            "sj_name": label_map.get(("sj", sj_code), ""),
            "value": value,
        })

    return pd.DataFrame(rows)


def to_nbs_multi_region_wide_table(
    df_long_all: pd.DataFrame,
    time_label: str,
    sort_newest_left: bool = True,
) -> pd.DataFrame:
    """
    Multi-region wide output (layout similar to screenshot).

    df_long_all must include:
      reg_name, zb_name, sj_code, sj_name, value
    and typically already filtered to ONE indicator row per region (cumulative only).
    """
    df = df_long_all.copy()
    df["sj_code"] = df["sj_code"].astype(str)
    df["month_label"] = df["sj_name"].astype(str)

    # Determine column order by sj_code
    order = (
        df[["sj_code", "month_label"]]
        .drop_duplicates()
        .sort_values("sj_code", ascending=not sort_newest_left)["month_label"]
        .tolist()
    )

    wide = df.pivot_table(
        index=["reg_name", "zb_name"],
        columns="month_label",
        values="value",
        aggfunc="first",
    ).reindex(columns=order)

    wide = wide.reset_index()

    # Build output rows with section headers
    cols = ["指标"] + order
    blank = {c: "" for c in cols}

    out_rows = []
    r1 = blank.copy(); r1["指标"] = "数据库：分省月度数据"
    r2 = blank.copy(); r2["指标"] = f"时间：{time_label}"
    out_rows.extend([r1, r2])

    for reg_name in wide["reg_name"].drop_duplicates().tolist():
        r_header = blank.copy()
        r_header["指标"] = f"地区：{reg_name}"
        out_rows.append(r_header)

        sub = wide[wide["reg_name"] == reg_name]
        for _, row in sub.iterrows():
            r = blank.copy()
            r["指标"] = row["zb_name"]
            for c in order:
                r[c] = row.get(c, "")
            out_rows.append(r)

    return pd.DataFrame(out_rows, columns=cols)


def load_regions_maps(regions_csv: str) -> tuple[dict, dict]:
    """
    Returns (name->code, code->name) from regions_csv with columns reg_code, reg_name.
    """
    df = pd.read_csv(regions_csv, dtype=str)
    df["reg_code"] = df["reg_code"].astype(str).str.strip()
    df["reg_name"] = df["reg_name"].astype(str).str.strip()

    name_to_code = dict(zip(df["reg_name"], df["reg_code"]))
    code_to_name = dict(zip(df["reg_code"], df["reg_name"]))
    return name_to_code, code_to_name


def resolve_regions_from_names(reg_names: list[str], name_to_code: dict) -> list[tuple[str, str]]:
    """
    Input: ["内蒙古自治区", "黑龙江省"]
    Output: [("150000","内蒙古自治区"), ("230000","黑龙江省")]
    """
    out = []
    for n in reg_names:
        n2 = str(n).strip()
        if n2 not in name_to_code:
            raise ValueError(f"Region name not found in regions_csv: {n2}")
        out.append((name_to_code[n2], n2))
    return out


def keep_only_cum_and_convert_unit(df_long: pd.DataFrame, energy_type: str) -> pd.DataFrame:
    """
    Keep ONLY the 累计值 series for the given energy_type and convert
    from 亿千瓦时 -> 万千瓦时 by multiplying by 10000.
    """
    df = df_long.copy()

    want_code = cumulative_zb_code(energy_type)
    df = df[df["zb_code"].astype(str) == want_code].copy()

    v = pd.to_numeric(df["value"], errors="coerce")
    df["value"] = (v * 10000).where(v.notna(), df["value"])

    df["zb_name"] = f'{INDICATORS[energy_type]["cn"]}发电量累计值(万千瓦时)'
    return df


def fill_missing_jan_cum_by_ratio(
    df_long: pd.DataFrame,
    jan_weight: float = 1.0,
    feb_weight: float = 0.85,
    zero_is_missing: bool = True,
) -> pd.DataFrame:
    """
    Fill January cumulative (累计值) using February cumulative and the strict ratio Jan:Feb = 1:0.85.
    Treat 0 as missing if zero_is_missing=True.
    """
    df = df_long.copy()

    df["sj_code"] = df["sj_code"].astype(str)
    year = df["sj_code"].str.slice(0, 4)
    month = df["sj_code"].str.slice(4, 6)

    total = jan_weight + feb_weight
    jan_frac = jan_weight / total

    zb = df["zb_name"].astype(str)
    # Works with labels like "火力发电量_累计值"
    is_cum = zb.str.contains("累计值", na=False) & ~zb.str.contains("同比", na=False)

    val = pd.to_numeric(df["value"], errors="coerce")

    feb_mask = is_cum & month.eq("02") & val.notna()
    feb_map = pd.Series(
        val[feb_mask].values,
        index=pd.MultiIndex.from_frame(pd.DataFrame({
            "reg_code": df.loc[feb_mask, "reg_code"].values,
            "energy_type": df.loc[feb_mask, "energy_type"].values,
            "zb_code": df.loc[feb_mask, "zb_code"].values,
            "year": year[feb_mask].values,
        }))
    )

    eps = 1e-12
    jan_need = val.isna() | (val.abs() <= eps) if zero_is_missing else val.isna()
    jan_mask = is_cum & month.eq("01") & jan_need

    if jan_mask.any():
        jan_keys = pd.MultiIndex.from_frame(pd.DataFrame({
            "reg_code": df.loc[jan_mask, "reg_code"].values,
            "energy_type": df.loc[jan_mask, "energy_type"].values,
            "zb_code": df.loc[jan_mask, "zb_code"].values,
            "year": year[jan_mask].values,
        }))
        filled = feb_map.reindex(jan_keys) * jan_frac
        df.loc[jan_mask, "value"] = filled.values

    return df


def load_ime_ratio_map(csv_path: str) -> dict:
    """
    Load year -> ratio from the IME CSV.
    Supports the format where the first row is: Year | IME ratio (for coal).
    """
    df = pd.read_csv(csv_path, dtype=str)

    if df.shape[1] >= 2 and str(df.iloc[0, 0]).strip().lower() == "year":
        df = df.iloc[1:, :2].copy()
        df.columns = ["year", "ratio"]
    else:
        df = df.iloc[:, :2].copy()
        df.columns = ["year", "ratio"]

    df["year"] = df["year"].astype(str).str.strip()
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    df = df.dropna(subset=["ratio"])

    return dict(zip(df["year"], df["ratio"]))


def apply_year_ratio_ime(df_long: pd.DataFrame, year_ratio: dict) -> pd.DataFrame:
    """
    Multiply ALL numeric 'value' by year_ratio[YYYY] based on sj_code=YYYYMM.
    """
    out = df_long.copy()
    out["sj_code"] = out["sj_code"].astype(str)
    out["_year"] = out["sj_code"].str[:4]

    ratio = out["_year"].map(year_ratio)
    val = pd.to_numeric(out["value"], errors="coerce")

    mask = ratio.notna() & val.notna()
    out.loc[mask, "value"] = (val[mask] * ratio[mask]).values

    return out.drop(columns=["_year"])


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Fetch NBS provincial power generation data and export wide CSV.")
    ap.add_argument(
        "--reg", required=True,
        help='Region name(s), comma-separated, e.g. "内蒙古自治区" or "内蒙古自治区,黑龙江省"'
    )
    ap.add_argument(
        "--energies", default="thermal,wind,solar",
        help="Comma-separated: thermal,wind,solar (default: all three)"
    )
    ap.add_argument(
        "--time", default="2011,-2023",
        help='NBS time filter string, e.g. "2011,-2023" or "LAST13"'
    )
    ap.add_argument("--start", default="", help="Override start YYYYMM for local filtering, e.g. 201101")
    ap.add_argument("--end", default="", help="Override end YYYYMM for local filtering, e.g. 202312")
    ap.add_argument("--outdir", default="outputs", help="Output directory (default: outputs)")
    ap.add_argument(
        "--regions_csv", required=True,
        help="CSV with reg_code,reg_name (required; used to resolve names -> codes)"
    )
    ap.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between requests (default: 0.2)")
    ap.add_argument("--save_long", action="store_true", help="Also save the long/tidy CSV (debug/useful)")

    args = ap.parse_args()

    reg_names = parse_csv_list(args.reg)
    name_to_code, _code_to_name = load_regions_maps(args.regions_csv)
    regions = resolve_regions_from_names(reg_names, name_to_code)  # list[(reg_code, reg_name)]

    energies = parse_csv_list(args.energies)

    start_guess, end_guess, time_label = parse_time_to_yyyymm_range(args.time)
    start_yyyymm = args.start.strip() or start_guess
    end_yyyymm = args.end.strip() or end_guess

    os.makedirs(args.outdir, exist_ok=True)

    ime_map = load_ime_ratio_map("IME_IM - Sheet1.csv")

    for energy in energies:
        all_rows = []

        for reg_code, reg_name in regions:
            print(f"Fetching {energy} for {reg_name} ({reg_code})...")

            resp = query_data(reg_code, energy, args.time)
            df_long = parse_to_long(resp, reg_code, energy)

            # strict local time filter
            if start_yyyymm and end_yyyymm:
                df_long["sj_code"] = df_long["sj_code"].astype(str)
                df_long = df_long[df_long["sj_code"].between(start_yyyymm, end_yyyymm)]

            # Fix January cumulative (treat 0 as missing, strict ratio)
            df_long = fill_missing_jan_cum_by_ratio(df_long, jan_weight=1.0, feb_weight=0.85, zero_is_missing=True)

            # Keep ONLY cumulative + convert unit to 万千瓦时 + relabel
            df_long = keep_only_cum_and_convert_unit(df_long, energy)

            # Attach region name for multi-region output formatting
            df_long["reg_name"] = reg_name
            all_rows.append(df_long)

            # If this is Inner Mongolia, append IME-adjusted rows in SAME output file
            if str(reg_code) == "150000":
                df_long_ime = apply_year_ratio_ime(df_long, ime_map)
                df_long_ime["reg_name"] = f"{reg_name}_IME"
                all_rows.append(df_long_ime)

            time.sleep(args.sleep)

        df_all = pd.concat(all_rows, ignore_index=True)

        wide_multi = to_nbs_multi_region_wide_table(
            df_long_all=df_all,
            time_label=time_label,
            sort_newest_left=False,
        )


        out_wide = os.path.join(args.outdir, f"nbs_MULTI_{energy}_{safe_tag(time_label)}_CUM_WANKWH.csv")
        wide_multi.to_csv(out_wide, index=False, encoding="utf-8-sig")
        print(f"saved wide multi-region: {out_wide}")

        if args.save_long:
            out_long = os.path.join(args.outdir, f"nbs_MULTI_{energy}_{safe_tag(time_label)}_CUM_WANKWH_LONG.csv")
            df_all.to_csv(out_long, index=False, encoding="utf-8-sig")
            print(f"saved long multi-region: {out_long}")


if __name__ == "__main__":
    main()
