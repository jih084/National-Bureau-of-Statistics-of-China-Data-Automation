#!/usr/bin/env python3
"""
NBS Provincial Power Generation Scraper (Wide Export)
====================================================

What this script does
---------------------
Downloads provincial monthly power generation data from NBS EasyQuery (data.stats.gov.cn)
for selected energy types and exports in an NBS-like *wide* table format.

UPDATED (this version)
----------------------
- Output CURRENT monthly value (当期值) in GWh, not cumulative.
- Prefer NBS "当期值" series; if missing, fallback to cumulative series diff.
- If Jan cumulative is missing/0, fill it from Feb cumulative using Jan:Feb = 1:0.85 (your existing rule),
  then diff to get Jan monthly value.
- Keeps all months in requested time range (no longer keeps only Jan/Feb).
- Unit conversion: 亿千瓦时 -> GWh by multiplying values by 100.
- Inner Mongolia (150000): outputs extra blocks:
    - "<reg_name> 蒙东" = IME = IM * ratio(year)
    - "<reg_name> 蒙西" = IMW = IM - IME
  (Removed duplicated _IME block to avoid double output.)

Typical usage
-------------
python nbs_fetch.py --reg "内蒙古自治区,黑龙江省,吉林省" --energies thermal --time "2011,-2023" --regions_csv nbs_regions.csv
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Tuple

import pandas as pd
import requests

BASE = "https://data.stats.gov.cn/easyquery.htm"

INDICATORS = {
    "thermal": {"base": "A03010H", "cn": "火力"},
    "wind":    {"base": "A03010K", "cn": "风力"},
    "solar":   {"base": "A03010L", "cn": "太阳能"},
    "generation": {"base": "A03010G", "cn": "发电量"},
}

def current_zb_code(energy_type: str) -> str:
    # NBS pattern: base + "01" = 当期值
    return INDICATORS[energy_type]["base"] + "01"

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
    return re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")


def parse_time_to_yyyymm_range(time_value: str) -> Tuple[str, str, str]:
    tv = time_value.strip().strip('"').strip("'")
    m = re.fullmatch(r"(\d{4}),-(\d{4})", tv)
    if m:
        y1, y2 = m.group(1), m.group(2)
        return f"{y1}01", f"{y2}12", f"{y1}-{y2}"
    return "", "", tv


def normalize_nbs_response(data):
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            raise RuntimeError(f"NBS response was a plain string (not JSON object): {data[:200]}")

    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected NBS response type: {type(data)}")

    rd = data.get("returndata", None)

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
    if energy_type not in INDICATORS:
        raise ValueError(f"Unknown energy_type={energy_type}. Use one of {list(INDICATORS)}")

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

    data = r.json()
    data = normalize_nbs_response(data)
    return data


def parse_to_long(resp: Dict, reg_code: str, energy_type: str) -> pd.DataFrame:
    rd = resp.get("returndata", {})
    datanodes = rd.get("datanodes", [])
    wdnodes = rd.get("wdnodes", [])

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
    df = df_long_all.copy()
    df["sj_code"] = df["sj_code"].astype(str)
    df["month_label"] = df["sj_name"].astype(str)

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


def to_vertical_region_columns(df_long_all: pd.DataFrame) -> pd.DataFrame:
    df = df_long_all.copy()
    df["sj_code"] = df["sj_code"].astype(str)
    df["Year"] = df["sj_code"].str[:4].astype(int)
    df["Month"] = df["sj_code"].str[4:6].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    piv = df.pivot_table(
        index=["Year", "Month"],
        columns="reg_name",
        values="value",
        aggfunc="first",
    ).sort_index()

    piv = piv.reset_index()
    region_cols = sorted([c for c in piv.columns if c not in ("Year", "Month")])
    return piv[["Year", "Month"] + region_cols]


def load_regions_maps(regions_csv: str) -> tuple[dict, dict]:
    df = pd.read_csv(regions_csv, dtype=str)
    df["reg_code"] = df["reg_code"].astype(str).str.strip()
    df["reg_name"] = df["reg_name"].astype(str).str.strip()

    name_to_code = dict(zip(df["reg_name"], df["reg_code"]))
    code_to_name = dict(zip(df["reg_code"], df["reg_name"]))
    return name_to_code, code_to_name


def resolve_regions_from_names(reg_names: list[str], name_to_code: dict) -> list[tuple[str, str]]:
    out = []
    for n in reg_names:
        n2 = str(n).strip()
        if n2 not in name_to_code:
            raise ValueError(f"Region name not found in regions_csv: {n2}")
        out.append((name_to_code[n2], n2))
    return out


def fill_missing_jan_cum_by_ratio(
    df_long: pd.DataFrame,
    jan_weight: float = 1.0,
    feb_weight: float = 0.85,
    zero_is_missing: bool = True,
) -> pd.DataFrame:
    df = df_long.copy()

    df["sj_code"] = df["sj_code"].astype(str)
    year = df["sj_code"].str.slice(0, 4)
    month = df["sj_code"].str.slice(4, 6)

    total = jan_weight + feb_weight
    jan_frac = jan_weight / total

    val = pd.to_numeric(df["value"], errors="coerce")

    feb_mask = month.eq("02") & val.notna()
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
    jan_mask = month.eq("01") & jan_need

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


def build_current_series_from_cur_and_cum(df_long: pd.DataFrame, energy_type: str) -> pd.DataFrame:
    """
    Build ONE series: 当期值(GWh).
    Prefer NBS 当期值; fallback to 累计值差分.
    """
    df = df_long.copy()
    df["sj_code"] = df["sj_code"].astype(str)

    cur_code = current_zb_code(energy_type)
    cum_code = cumulative_zb_code(energy_type)

    df_cur = df[df["zb_code"].astype(str) == cur_code].copy()
    df_cum = df[df["zb_code"].astype(str) == cum_code].copy()

    # numeric + unit convert: 亿千瓦时 -> GWh
    df_cur["value"] = pd.to_numeric(df_cur["value"], errors="coerce") #* 100
    df_cum["value"] = pd.to_numeric(df_cum["value"], errors="coerce") #* 100

    # fill Jan cumulative if needed (your rule), then diff -> monthly fallback
    df_cum = fill_missing_jan_cum_by_ratio(df_cum, jan_weight=1.0, feb_weight=0.85, zero_is_missing=True)

    df_cum = df_cum.sort_values(["reg_code", "energy_type", "sj_code"])
    df_cum["_cum_prev"] = df_cum.groupby(["reg_code", "energy_type"])["value"].shift(1)
    df_cum["value_from_cumdiff"] = df_cum["value"] - df_cum["_cum_prev"]
    df_cum.loc[df_cum["_cum_prev"].isna(), "value_from_cumdiff"] = df_cum["value"]  # Jan

    key_cols = ["reg_code", "energy_type", "sj_code", "sj_name"]
    merged = pd.merge(
        df_cur[key_cols + ["value"]],
        df_cum[key_cols + ["value_from_cumdiff"]],
        on=key_cols,
        how="outer"
    )

    merged["value_final"] = merged["value"]
    merged.loc[merged["value_final"].isna(), "value_final"] = merged["value_from_cumdiff"]
    # -----------------------------
    # Repair Jan/Feb monthly values using Feb cumulative:
    # Jan = CumFeb/1.85*1, Feb = CumFeb/1.85*0.85
    # Only applies when Jan/Feb monthly are missing/0 and Feb cumulative is available.
    # -----------------------------
    jan_w, feb_w = 1.0, 0.85
    total = jan_w + feb_w  # 1.85

    merged["_year"] = merged["sj_code"].str[:4]
    merged["_mm"] = merged["sj_code"].str[-2:]

    # Build Feb cumulative lookup from df_cum (already numeric + unit-handled above)
    # df_cum: reg_code, energy_type, sj_code, value (cumulative)
    feb_cum = df_cum[df_cum["sj_code"].astype(str).str.endswith("02")].copy()
    feb_cum["year"] = feb_cum["sj_code"].astype(str).str[:4]
    feb_cum["cum_feb"] = pd.to_numeric(feb_cum["value"], errors="coerce")
    feb_cum = feb_cum[["reg_code", "energy_type", "year", "cum_feb"]]

    merged = merged.merge(
        feb_cum,
        left_on=["reg_code", "energy_type", "_year"],
        right_on=["reg_code", "energy_type", "year"],
        how="left"
    ).drop(columns=["year"], errors="ignore")

    v = pd.to_numeric(merged["value_final"], errors="coerce")
    cum_feb = pd.to_numeric(merged["cum_feb"], errors="coerce")

    # treat 0 as missing for monthly values (because NBS sometimes returns 0 as missing)
    eps = 1e-12
    missing_monthly = v.isna() | (v.abs() <= eps)

    # Only compute when Feb cumulative is valid (>0)
    ok_cum = cum_feb.notna() & (cum_feb > eps)

    # Fill Jan
    jan_mask = (merged["_mm"] == "01") & missing_monthly & ok_cum
    merged.loc[jan_mask, "value_final"] = (cum_feb[jan_mask] / total) * jan_w

    # Fill Feb
    feb_mask = (merged["_mm"] == "02") & missing_monthly & ok_cum
    merged.loc[feb_mask, "value_final"] = (cum_feb[feb_mask] / total) * feb_w

    merged = merged.drop(columns=["cum_feb", "_year", "_mm"], errors="ignore")

    out = merged[key_cols].copy()
    out["value"] = merged["value_final"]
    out["zb_code"] = cur_code
    out["zb_name"] = f'{INDICATORS[energy_type]["cn"]}发电量当期值(GWh)'
    return out

from functools import lru_cache

ENERGY_TO_RATIO_COL = {
    "wind": "wind_IME_ratio",
    "solar": "solar_IME_ratio",
    "thermal": "coal_IME_ratio",     # alias: your thermal uses coal ratio
    "coal": "coal_IME_ratio",        # (optional)
    "generation": "generation_IME_ratio",
}

@lru_cache()
def load_ime_ratio_table(csv_path: str = "IME_IM_Ratio.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["year"] = df["year"].astype(int)
    return df

def apply_energy_year_ratio_ime(df_long: pd.DataFrame, ratio_df: pd.DataFrame, energy_type: str) -> pd.DataFrame:
    out = df_long.copy()
    out["sj_code"] = out["sj_code"].astype(str)
    out["_year"] = out["sj_code"].str[:4].astype(int)

    col = ENERGY_TO_RATIO_COL.get(energy_type)
    if col is None:
        raise ValueError(f"No IME ratio column mapping for energy_type='{energy_type}'")

    # build year -> ratio map for this energy
    year_to_ratio = dict(zip(ratio_df["year"].astype(int), pd.to_numeric(ratio_df[col], errors="coerce")))

    ratio = out["_year"].map(year_to_ratio)
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
        "--energies", default="generation,thermal,wind,solar",
        help="Comma-separated: generation,thermal,wind,solar (default: all)"
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
    ap.add_argument("--format", default="wide", choices=["wide", "vertical"],
                    help="Output format: wide (NBS-like blocks) or vertical (rows=year-month, cols=regions)")

    args = ap.parse_args()

    reg_names = parse_csv_list(args.reg)
    name_to_code, _ = load_regions_maps(args.regions_csv)
    regions = resolve_regions_from_names(reg_names, name_to_code)

    energies = parse_csv_list(args.energies)

    start_guess, end_guess, time_label = parse_time_to_yyyymm_range(args.time)
    start_yyyymm = args.start.strip() or start_guess
    end_yyyymm = args.end.strip() or end_guess

    os.makedirs(args.outdir, exist_ok=True)

    ratio_df = load_ime_ratio_table("IME_IM_Ratio.csv")

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

            # Build CURRENT monthly series (当期值优先，累计差分兜底)
            df_cur = build_current_series_from_cur_and_cum(df_long, energy)

            df_cur["reg_name"] = reg_name
            all_rows.append(df_cur)

            # Inner Mongolia: 蒙东/蒙西
            if str(reg_code) == "150000":
                df_ime = apply_energy_year_ratio_ime(df_cur, ratio_df, energy)
                df_ime["reg_name"] = f"{reg_name} 蒙东"
                all_rows.append(df_ime)

                df_imw = df_cur.copy()
                df_imw["value"] = (
                    pd.to_numeric(df_cur["value"], errors="coerce")
                    - pd.to_numeric(df_ime["value"], errors="coerce")
                )
                df_imw["reg_name"] = f"{reg_name} 蒙西"
                all_rows.append(df_imw)


            time.sleep(args.sleep)

        df_all = pd.concat(all_rows, ignore_index=True)

        if args.format == "wide":
            wide_multi = to_nbs_multi_region_wide_table(
                df_long_all=df_all,
                time_label=time_label,
                sort_newest_left=False,
            )
            out_wide = os.path.join(args.outdir, f"nbs_MULTI_{energy}_{safe_tag(time_label)}_WIDE.csv")
            wide_multi.to_csv(out_wide, index=False, encoding="utf-8-sig")
            print(f"saved wide multi-region: {out_wide}")
        else:
            vert = to_vertical_region_columns(df_all)
            out_v = os.path.join(args.outdir, f"nbs_MULTI_{energy}_{safe_tag(time_label)}_VERTICAL.csv")
            vert.to_csv(out_v, index=False, encoding="utf-8-sig")
            print(f"saved vertical (regions as columns): {out_v}")

        if args.save_long:
            out_long = os.path.join(args.outdir, f"nbs_MULTI_{energy}_{safe_tag(time_label)}_CUR_GWh_LONG.csv")
            df_all.to_csv(out_long, index=False, encoding="utf-8-sig")
            print(f"saved long multi-region: {out_long}")


if __name__ == "__main__":
    main()
