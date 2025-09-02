import argparse
from dataclasses import dataclass
from datetime import datetime, date, timedelta, time
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd

TENORS = ["ON", "1W", "1M", "3M", "6M", "12M"]
WEEKEND = {4, 5}  # Fri, Sat (KSA)

# ---------- Markup (min(9% * SAIBID, 20 bps)) ----------
def compute_markup_delta_auto(saibid: float) -> float:
    """
    Returns the markup increment in the SAME UNIT as 'saibid'.
      - If SAIBID looks like 3.98 (whole %), delta is 0.20 for 20 bps.
      - If SAIBID looks like 0.0398 (decimal), delta is 0.0020 for 20 bps.
    Always uses: min( 9% of SAIBID, 20 bps ).
    """
    if pd.isna(saibid):
        return 0.0

    # Whole-percent vs decimal detection
    whole_percent_mode = saibid > 1.0

    if whole_percent_mode:
        # SAIBID like 3.98 => 9% of SAIBID is 0.358; 20bps is 0.20
        nine_pct = saibid * 0.09
        twenty_bps = 0.20
        return min(nine_pct, twenty_bps)
    else:
        # SAIBID like 0.0398 => 9% of SAIBID is 0.003582; 20bps is 0.0020
        nine_pct = saibid * 0.09
        twenty_bps = 0.0020
        return min(nine_pct, twenty_bps)

# ---------- Calendar ----------
def load_calendar(cal_df: pd.DataFrame) -> pd.DataFrame:
    cal_df = cal_df.copy()
    cal_df["Date"] = pd.to_datetime(cal_df["Date"], errors="coerce").dt.date
    if "IsWorkingDay" not in cal_df.columns:
        cal_df["IsWorkingDay"] = (~cal_df["Date"].map(lambda d: d.weekday() in WEEKEND))
    return cal_df[["Date", "IsWorkingDay"]].dropna()

def is_working_day(d: date, cal: pd.DataFrame) -> bool:
    row = cal.loc[cal["Date"] == d]
    if row.empty:
        return d.weekday() not in WEEKEND
    return bool(row["IsWorkingDay"].iloc[0])

def prev_working_day(d: date, cal: pd.DataFrame) -> date:
    cur = d - timedelta(days=1)
    while not is_working_day(cur, cal):
        cur -= timedelta(days=1)
    return cur

def biz_days_diff(start: date, end: date, cal: pd.DataFrame) -> int:
    if pd.isna(start) or pd.isna(end):
        return 0
    s = start.date() if hasattr(start, "date") else start
    e = end.date() if hasattr(end, "date") else end
    if e <= s:
        return 0
    cur = s + timedelta(days=1)
    days = 0
    while cur <= e:
        if is_working_day(cur, cal):
            days += 1
        cur += timedelta(days=1)
    return days

@dataclass
class TenorRule:
    tenor: str
    basis: str      # 'business' or 'calendar'
    min_days: int
    max_days: int

def load_tenor_rules(df: pd.DataFrame) -> Dict[str, TenorRule]:
    rules = {}
    for _, r in df.iterrows():
        rules[str(r["Tenor"]).upper()] = TenorRule(
            tenor=str(r["Tenor"]).upper(),
            basis=str(r["Basis"]).lower(),
            min_days=int(r["MinDays"]),
            max_days=int(r["MaxDays"])
        )
    return rules

def classify_tenor(start, end, rules: Dict[str, TenorRule], cal: pd.DataFrame) -> Optional[str]:
    if pd.isna(start) or pd.isna(end):
        return None
    caldays = (end - start).days
    bizdays = biz_days_diff(start, end, cal)
    for t, rule in rules.items():
        n = bizdays if rule.basis == "business" else caldays
        if rule.min_days <= n <= rule.max_days:
            return t
    return None

# ---------- Selection (lookback & thresholds) ----------
def select_deals_for_tenor(deals_t: pd.DataFrame, reporting_date: date,
                           lookback_days: int, ten: str,
                           min_ticket_10m: bool, aggregate_50m: bool) -> Tuple[pd.DataFrame, str]:
    df = deals_t.copy()
    if min_ticket_10m:
        df = df.loc[pd.to_numeric(df["InvestmentAmount_SAR"], errors="coerce") >= 10_000_000]

    window_dates = [reporting_date - timedelta(days=i) for i in range(0, lookback_days)]
    selected = []
    counterparties = set()
    cutoff = None
    for d in window_dates:
        day_df = df.loc[pd.to_datetime(df["TradeDate"], errors="coerce").dt.date == d]
        if not day_df.empty:
            selected.append(day_df)
            counterparties |= set(day_df["Counterparty"].astype(str))
            if len(counterparties) >= 2:
                cutoff = d
                break

    if cutoff is None:
        return pd.DataFrame(columns=df.columns), "Use L2 or L3"

    out = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=df.columns)

    if aggregate_50m:
        if pd.to_numeric(out["InvestmentAmount_SAR"], errors="coerce").sum() < 50_000_000:
            return pd.DataFrame(columns=df.columns), "Use L2 or L3"

    return out, "L1"

# ---------- L3 calculation from uploaded spreads ----------
def l3_from_spreads(l3_input: pd.DataFrame, tenor: str) -> Optional[Dict[str, float]]:
    """ Expects L3_Input with columns:
        Tenor, Day1_SAIBOR, Day1_TBILL, ..., Day5_SAIBOR, Day5_TBILL
        Units should match your SAIBOR/SAIBID unit (whole % like 4.18 or decimals like 0.0418).
    """
    row = l3_input.loc[l3_input["Tenor"].astype(str).str.upper() == tenor]
    if row.empty:
        return None

    # Assume whole % if numbers are > 1, else decimals
    def _val(x):
        return float(x) if pd.notna(x) else np.nan

    spreads = []
    for i in range(1, 6):
        sb = _val(row.get(f"Day{i}_SAIBOR", pd.NA).iloc[0])
        tb = _val(row.get(f"Day{i}_TBILL", pd.NA).iloc[0])
        if pd.isna(sb) or pd.isna(tb):
            return None
        spreads.append(sb - tb)

    avg_spread = float(np.mean(spreads))
    latest_tbill = _val(row["Day5_TBILL"].iloc[0])
    saibor = latest_tbill + avg_spread
    # SAIBID = SAIBOR - min(avg_spread, 20 bps)  (units respected automatically)
    # If whole % mode (values > 1), 20bps = 0.20; else 0.0020
    twenty_bps = 0.20 if saibor > 1.0 else 0.0020
    saibid = saibor - min(avg_spread, twenty_bps)
    return {"avg_spread": avg_spread, "latest_tbill": latest_tbill, "saibor": saibor, "saibid": saibid}

# ---------- Core engine ----------
def run_engine(xlsx_path: str, reporting_date: Optional[str], out_path: Optional[str] = None):
    print("Loading workbook:", xlsx_path)
    xl = pd.ExcelFile(xlsx_path)

    # Required sheets
    deals = xl.parse("Deals")
    cal = load_calendar(xl.parse("BusinessCalendar"))
    rules = load_tenor_rules(xl.parse("TenorRules"))
    cfg_raw = xl.parse("Config")
    cfg = {str(k): str(v) for k, v in cfg_raw.values}

    # Optional sheet (for L3). If missing, we’ll still write an empty L3_Results.
    try:
        l3_input = xl.parse("L3_Input")
    except Exception:
        l3_input = pd.DataFrame()

    # Config
    rep = pd.to_datetime(reporting_date or cfg.get("reporting_date", "")).date()
    lookback = int(cfg.get("lookback_days", "5"))
    tenors_10m = set([t.strip().upper() for t in cfg.get("min_ticket_10m_tenors", "ON,1W,1M,3M").split(",")])
    tenors_50m = set([t.strip().upper() for t in cfg.get("aggregate_50m_tenors", "6M,12M").split(",")])

    # Clean & types
    deals.columns = deals.columns.map(lambda c: str(c).strip())
    for c in ["TradeDate", "StartDate", "MaturityDate"]:
        deals[c] = pd.to_datetime(deals[c], errors="coerce")

    # Bucket by tenor (per business/calendar rules)
    deals["TenorBucket"] = deals.apply(
        lambda r: classify_tenor(r["StartDate"], r["MaturityDate"], rules, cal), axis=1
    )

    comp_rows: List[Dict] = []
    agg_rows: List[Dict] = []

    # Main loop per tenor
    for ten in TENORS:
        t_df = deals.loc[deals["TenorBucket"] == ten].copy()
        if t_df.empty:
            # Try L3 fallback (we still show Level text in Summary)
            res = l3_from_spreads(l3_input, ten) if not l3_input.empty else None
            if res:
                agg_rows.append({
                    "Tenor": ten, "Level": "L3",
                    "SAIBID": res["saibid"], "SAIBOR": res["saibor"],
                    "TotalNotional": 0.0, "SelectedDeals": 0
                })
            else:
                agg_rows.append({
                    "Tenor": ten, "Level": "Use L2 or L3",
                    "SAIBID": np.nan, "SAIBOR": np.nan,
                    "TotalNotional": 0.0, "SelectedDeals": 0
                })
            continue

        sel, level = select_deals_for_tenor(
            t_df, reporting_date=rep, lookback_days=lookback, ten=ten,
            min_ticket_10m=(ten in tenors_10m), aggregate_50m=(ten in tenors_50m)
        )

        if sel.empty and level != "L1":
            # Try L3 when L1 didn’t materialize
            res = l3_from_spreads(l3_input, ten) if not l3_input.empty else None
            if res:
                agg_rows.append({
                    "Tenor": ten, "Level": "L3",
                    "SAIBID": res["saibid"], "SAIBOR": res["saibor"],
                    "TotalNotional": 0.0, "SelectedDeals": 0
                })
            else:
                agg_rows.append({
                    "Tenor": ten, "Level": "Use L2 or L3",
                    "SAIBID": np.nan, "SAIBOR": np.nan,
                    "TotalNotional": 0.0, "SelectedDeals": 0
                })
            continue

        # Weighted average SAIBID from selected deals
        sel["SumProduct"] = pd.to_numeric(sel["InvestmentAmount_SAR"], errors="coerce") * pd.to_numeric(sel["BankRate"], errors="coerce")
        total_notional = pd.to_numeric(sel["InvestmentAmount_SAR"], errors="coerce").sum()
        saibid = sel["SumProduct"].sum() / total_notional if total_notional > 0 else np.nan

        # Apply markup (min(9% * SAIBID, 20bps)) in the same unit as SAIBID
        delta = compute_markup_delta_auto(saibid)
        saibor = saibid + delta

        # Keep deal-level details
        for _, r in sel.iterrows():
            comp_rows.append({
                "TenorBucket": ten,
                "TradeDate": r["TradeDate"],
                "StartDate": r["StartDate"],
                "MaturityDate": r["MaturityDate"],
                "Counterparty": r.get("Counterparty", ""),
                "InvestmentAmount_SAR": r["InvestmentAmount_SAR"],
                "BankRate": r["BankRate"],
                "SumProduct": r["SumProduct"],
                "SelectedForL1": True
            })

        agg_rows.append({
            "Tenor": ten, "Level": level,
            "SAIBID": saibid, "SAIBOR": saibor,
            "TotalNotional": total_notional, "SelectedDeals": len(sel)
        })

    comp_df = pd.DataFrame(comp_rows)
    summary = pd.DataFrame(agg_rows).sort_values("Tenor")

    # L3_Results (populate if L3_Input present)
    l3_rows = []
    if not l3_input.empty:
        for ten in TENORS:
            res = l3_from_spreads(l3_input, ten)
            if res:
                l3_rows.append({"Tenor": ten, **res})
    l3_df = pd.DataFrame(l3_rows)

    # RunMeta
    runmeta_df = pd.DataFrame([{
        "reporting_date": rep.isoformat(),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z"
    }])

    # Write all five sheets
    out_xlsx = out_path or (xlsx_path.replace(".xlsx", f"_OUT_{rep.isoformat()}.xlsx"))
#   Write all five sheets
    out_xlsx = out_path or (xlsx_path.replace(".xlsx", f"_OUT_{rep.isoformat()}.xlsx"))
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        deals.to_excel(xw, sheet_name="DealsProcessed", index=False)
        comp_df.to_excel(xw, sheet_name="ComputationResults", index=False)
        summary.to_excel(xw, sheet_name="Summary", index=False)
        l3_df.to_excel(xw, sheet_name="L3_Results", index=False)
        runmeta_df.to_excel(xw, sheet_name="RunMeta", index=False)

    print("Done ->", out_xlsx)

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="SAIBOR engine")
    p.add_argument("--xlsx", required=True)
    p.add_argument("--reporting-date", required=True)
    p.add_argument("--out", required=False)
    args = p.parse_args()
    run_engine(args.xlsx, args.reporting_date, args.out)

if __name__ == "__main__":
    main()
