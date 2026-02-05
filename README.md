# NBS Provincial Power Generation Scraper (Multi‑Region, Wide Export)

`nbs_fetch.py` downloads **provincial monthly power generation data** from the **National Bureau of Statistics of China (NBS)** EasyQuery API and exports CSVs in either:

- **Wide (NBS‑like)** blocks per region (default), or
- **Vertical** format (rows = year‑month, columns = regions)

It is designed for power/energy analysis workflows where you want **multiple regions in one file** and a consistent, reproducible export.

---

## What it outputs (important)

This script builds **one monthly series per energy type: 当期值 (monthly / “current”)**.

It **prefers NBS 当期值** when available, and **falls back to累计值差分** (cumulative month‑to‑month differences) when 当期值 is missing.

Inner Mongolia (`150000`) is split into:

- `内蒙古自治区 蒙东` (IME) = IM × (year‑specific ratio)
- `内蒙古自治区 蒙西` (IMW) = IM − IME

---

## Key Features

- **Input regions by Chinese name** (comma‑separated), not code  
  Example: `内蒙古自治区,黑龙江省,吉林省`
- **Multiple regions in ONE output file** per energy type
- **Monthly series construction**
  - Prefer 当期值
  - Else use 累计值差分
  - Repairs Jan/Feb when monthly values are missing/0 but Feb cumulative exists (see Methodology)
- **Two output layouts**
  - `--format wide` (default): NBS‑style blocks by region
  - `--format vertical`: one row per month with regions as columns
- **Inner Mongolia IME/IMW split**
  - Uses a **year‑specific ratio table** in `IME_IM_Ratio.csv`

---

## Requirements

- Python 3.9+
- Packages:
  ```bash
  pip install pandas requests
  ```

---

## Required Files

### 1) Region mapping CSV (**required**)

You must provide `--regions_csv` with at least these columns:

```csv
reg_code,reg_name
150000,内蒙古自治区
230000,黑龙江省
220000,吉林省
```

### 2) IME ratio CSV (**required** if you include Inner Mongolia)

The script expects this file name in the working directory:

```
IME_IM_Ratio.csv
```

Required columns (lowercase `year` is required):

```csv
year,coal_IME_ratio,wind_IME_ratio,solar_IME_ratio,generation_IME_ratio
2011,0.82,0.44,0.40,0.80
2012,0.83,0.45,0.41,0.81
...
```

Mapping used by the code:
- `thermal` → `coal_IME_ratio`
- `wind` → `wind_IME_ratio`
- `solar` → `solar_IME_ratio`
- `generation` → `generation_IME_ratio`

---

## Usage

### Basic example (wide output, thermal only)

```bash
python nbs_fetch.py \
  --reg "内蒙古自治区,黑龙江省,吉林省" \
  --energies thermal \
  --time "2011,-2023" \
  --regions_csv nbs_regions.csv
```

### Multiple energy types

```bash
python nbs_fetch.py \
  --reg "内蒙古自治区,黑龙江省" \
  --energies generation,thermal,wind,solar \
  --time "2011,-2023" \
  --regions_csv nbs_regions.csv
```

### Save the long/tidy intermediate output (optional)

```bash
python nbs_fetch.py \
  --reg "内蒙古自治区" \
  --energies wind \
  --time "2011,-2023" \
  --regions_csv nbs_regions.csv \
  --save_long
```

### Output format: vertical (regions as columns)

```bash
python nbs_fetch.py \
  --reg "黑龙江省,吉林省" \
  --energies solar \
  --time "2011,-2023" \
  --regions_csv nbs_regions.csv \
  --format vertical
```

### Time filtering notes

- `--time` is passed to NBS (e.g. `"2011,-2023"` or `"LAST13"`).
- You can also enforce a **local** month filter with:
  - `--start YYYYMM` and `--end YYYYMM` (e.g., `201101` / `202312`)

---

## Output Files

Outputs go to `--outdir` (default: `outputs/`).

For each energy type, the script writes either:

- **Wide**
  ```
  outputs/nbs_MULTI_<energy>_<timeLabel>_WIDE.csv
  ```
- **Vertical**
  ```
  outputs/nbs_MULTI_<energy>_<timeLabel>_VERTICAL.csv
  ```

If `--save_long` is enabled:
```
outputs/nbs_MULTI_<energy>_<timeLabel>_CUR_GWh_LONG.csv
```

Notes:
- For `--time "2011,-2023"`, `timeLabel` becomes `2011_2023`.

---

## Output Format Details

### Wide format (NBS‑style blocks)

The wide file contains repeated blocks like:

```
地区：内蒙古自治区
火力发电量当期值(GWh) | 2011年1月 | 2011年2月 | ...

地区：内蒙古自治区 蒙东
火力发电量当期值(GWh) | 2011年1月 | 2011年2月 | ...
```

### Vertical format

A tidy “matrix” table:

- One row per `YYYYMM`
- One column per region name (including `内蒙古自治区 蒙东/蒙西` when applicable)

---

## Methodology & Assumptions

### 1) 当期值优先，累计差分兜底
For each energy and region, the script attempts:

1. Use NBS **当期值** series (`base + "01"`)
2. If missing, compute monthly values from **累计值差分** (`base + "02"`)

### 2) Jan cumulative repair (when NBS gives Jan cumulative as 0/missing)
If January cumulative is missing/0 but February cumulative exists, it reconstructs:

- `JanCum = FebCum / (1 + 0.85) * 1`
- `FebMonthly = FebCum / (1 + 0.85) * 0.85`

### 3) Jan/Feb monthly repair (when monthly values are missing/0)
If Jan/Feb monthly are missing/0 but Feb cumulative exists, it fills:

- `Jan = CumFeb / 1.85 * 1`
- `Feb = CumFeb / 1.85 * 0.85`

### 4) Unit note
The code labels outputs as **GWh**. In `build_current_series_from_cur_and_cum`, the unit conversion from `亿千瓦时 → GWh (×100)` is currently **commented out** in the script.

If your NBS values are in `亿千瓦时`, you should uncomment the `* 100` conversion lines in:
- `df_cur["value"] = ... # * 100`
- `df_cum["value"] = ... # * 100`

---

## CLI Arguments (quick reference)

- `--reg` (required): comma‑separated Chinese region names
- `--regions_csv` (required): CSV mapping names → codes
- `--energies`: `generation,thermal,wind,solar` (default: all)
- `--time`: NBS time filter (default: `2011,-2023`)
- `--start`, `--end`: local YYYYMM filter overrides
- `--outdir`: output directory (default: `outputs`)
- `--format`: `wide` or `vertical` (default: `wide`)
- `--save_long`: also write long/tidy CSV
- `--sleep`: seconds between API requests (default: `0.2`)

---

## Notes / Caveats

- NBS EasyQuery is undocumented and can change.
- The IME split only triggers for `reg_code == 150000`.
- If you run the script from a different working directory, ensure `IME_IM_Ratio.csv` is discoverable (or update the path in code).
