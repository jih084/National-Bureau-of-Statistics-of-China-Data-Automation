import json
import time
import requests
import pandas as pd

BASE = "https://data.stats.gov.cn/easyquery.htm"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://data.stats.gov.cn/easyquery.htm?cn=E0101",
}

# Use ONE indicator as context so the API knows which dataset slice you mean
ZB_FOR_CONTEXT = "A03010H"  # 火力发电量

def get_regions(dbcode: str = "fsyd", zb_code: str = ZB_FOR_CONTEXT) -> pd.DataFrame:
    """
    Returns region list for dimension reg (地区) as codes like 510000 (四川省).
    """
    params = {
        "m": "getOtherWds",
        "dbcode": dbcode,

        # IMPORTANT: keep the same row/col layout used by QueryData
        "rowcode": "zb",
        "colcode": "sj",

        # IMPORTANT: do NOT set reg here; we want the server to return reg options
        "wds": "[]",

        # Provide indicator context; otherwise it may return unrelated dimension values
        "dfwds": json.dumps([{"wdcode": "zb", "valuecode": zb_code}], ensure_ascii=False),

        # cache buster
        "k1": str(int(time.time() * 1000)),
    }

    r = requests.get(BASE, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    # returndata is usually a list of dimension blocks
    # We want the one whose wdcode == "reg"
    blocks = data.get("returndata", [])
    reg_block = None
    for b in blocks:
        if b.get("wdcode") == "reg":
            reg_block = b
            break

    if reg_block is None:
        # Debug helper: show what dimensions came back
        dims = [b.get("wdcode") for b in blocks]
        raise RuntimeError(f"Did not find reg in returndata. Got dimensions: {dims}")

    nodes = reg_block.get("nodes", [])
    df = pd.DataFrame(nodes).rename(columns={"code": "reg_code", "name": "reg_name"})
    df = df[["reg_code", "reg_name"]].drop_duplicates()

    # Optional: keep only 6-digit numeric codes (filters out many aggregates)
    df = df[df["reg_code"].astype(str).str.fullmatch(r"\d{6}")]

    return df.sort_values("reg_code").reset_index(drop=True)

if __name__ == "__main__":
    df = get_regions()
    df.to_csv("nbs_regions.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows -> nbs_regions.csv")
    print(df.head(10))
