# NBS Provincial Power Generation Scraper (Multi‑Region, Cumulative Only)

This script downloads **provincial monthly power generation data** from the **National Bureau of Statistics of China (NBS)** EasyQuery API and exports a **multi‑region wide table** suitable for analysis.

The current version is tailored for **power‑system / energy analysis workflows**, with strict control over indicators, units, and output layout.

---

## Key Features

- **Input regions by name** (Chinese), not code  
  - Example: `内蒙古自治区,黑龙江省,吉林省`
  - Region codes are resolved automatically via a CSV lookup
- **Multiple regions in ONE output file** (per energy type)
- **Only keeps cumulative generation**
  - `{energy}发电量累计值`
- **Unit conversion**
  - From `亿千瓦时` → `万千瓦时` (× 10,000)
- **Chronological column order**
  - `2011年1月 → 2011年12月 → … → 2023年12月`
- **IME adjustment (Inner Mongolia only)**
  - If `内蒙古自治区` is included, an additional block  
    `内蒙古自治区_IME` is added **in the same file**
  - Values are scaled by year‑specific ratios from an external CSV
- Robust handling of NBS quirks:
  - Fixes missing January cumulative values using a strict Feb‑based ratio
  - Handles malformed NBS responses (`returndata` returned as string)

---

## Output Format (Wide)

Each output file contains:

1. Metadata rows:
   - `数据库：分省月度数据`
   - `时间：2011–2023`
2. Region blocks:
   ```
   地区：内蒙古自治区
   火力发电量累计值(万千瓦时) | 2011年1月 | 2011年2月 | ...

   地区：内蒙古自治区_IME
   火力发电量累计值(万千瓦时) | 2011年1月 | 2011年2月 | ...
   ```

---

## Requirements

- Python 3.9+
- Packages:
  ```bash
  pip install pandas requests
  ```

---

## Required Files

### 1. Region mapping CSV (required)
Example: `nbs_regions.csv`

```csv
reg_code,reg_name
150000,内蒙古自治区
230000,黑龙江省
220000,吉林省
```

### 2. IME ratio CSV (required if using Inner Mongolia)
File name (expected by default):
```
IME_IM - Sheet1.csv
```

Format:
```csv
Year,IME ratio (for coal)
2011,0.82
2012,0.83
...
```

---

## Usage

### Basic example (multi‑region, thermal only)

```bash
python nbs_fetch_modified.py   --reg "内蒙古自治区,黑龙江省,吉林省"   --energies thermal   --time "2011,-2023"   --regions_csv nbs_regions.csv
```

### Multiple energy types

```bash
python nbs_fetch_modified.py   --reg "内蒙古自治区,黑龙江省"   --energies thermal,wind,solar   --time "2011,-2023"   --regions_csv nbs_regions.csv
```

### Save long (tidy) intermediate output (optional)

```bash
python nbs_fetch_modified.py   --reg "内蒙古自治区"   --energies thermal   --time "2011,-2023"   --regions_csv nbs_regions.csv   --save_long
```

---

## Output Files

For each energy type, one file is produced:

```
outputs/
└── nbs_MULTI_thermal_2011_2023_CUM_WANKWH.csv
└── nbs_MULTI_wind_2011_2023_CUM_WANKWH.csv
└── nbs_MULTI_solar_2011_2023_CUM_WANKWH.csv
```

---

## Notes on Methodology

- **January reconstruction**
  - NBS often reports January cumulative as `0`
  - We reconstruct January as:
    ```
    Jan = Feb / (1 + 0.85)
    ```
- **IME adjustment**
  - Applied to **all numeric values**
  - Implemented as a proportional allocation by year
- **No monthly values or growth rates**
  - This script intentionally drops:
    - 当期值
    - 同比增长
    - 累计同比

---

## Intended Use

This pipeline is designed for:
- Power system modeling
- Regional energy accounting
- Decarbonization and transition analysis
- Research workflows requiring **clean, reproducible, long‑range provincial data**

---

## Author / Maintainer Notes

- API is undocumented and unstable; expect occasional NBS changes
- Script is modular and easy to extend (e.g., single‑month views, province aggregation, new indicators)

If you want:
- Per‑year tables
- Single‑month cross‑year slices (e.g., February only)
- Aggregation across regions

…the current structure already supports it cleanly.
