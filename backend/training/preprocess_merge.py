#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
from pathlib import Path

DATA_DIR = Path("data")
OUT = DATA_DIR / "credit_risk_merged.csv"

# ---------- Taiwan ----------
def load_taiwan():
    p = DATA_DIR / "taiwan_credit_default.csv"
    df = pd.read_csv(p)
    # nhận diện tên nhãn linh hoạt
    label_col = None
    for cand in ["default.payment.next.month", "default payment next month", "default", "DEFAULT"]:
        if cand in df.columns:
            label_col = cand; break
    if label_col is None:
        raise ValueError("Taiwan: không tìm thấy cột nhãn (default*).")

    bill_cols = [c for c in df.columns if str(c).startswith("BILL_AMT")]

    out = pd.DataFrame()
    out["age"] = df["AGE"].clip(18,100)
    out["income"] = df["LIMIT_BAL"].clip(lower=0)
    out["liabilities"] = (df[bill_cols].mean(axis=1) if bill_cols else 0).clip(lower=0)
    out["credit_history_months"] = 6 if bill_cols else 0
    out["target_default_12m"] = df[label_col].astype(int)
    return out

# ---------- LendingClub ----------
def emp_to_years(v):
    v = str(v).strip().lower()
    if v in ("nan","n/a"): return np.nan
    if "10+" in v: return 10.0
    if "<" in v: return 0.5
    try: return float(v.split()[0])
    except: return np.nan

def load_lendingclub():
    p = DATA_DIR / "lendingclub" / "accepted_2007_to_2018Q4.csv"
    df = pd.read_csv(p, low_memory=False)

    need = ["loan_status","annual_inc","loan_amnt","emp_length","earliest_cr_line","issue_d"]
    miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"LendingClub: thiếu cột {miss}")

    bad = {"Charged Off","Default","Does not meet the credit policy. Status:Charged Off","Late (31-120 days)","In Grace Period"}
    y = df["loan_status"].astype(str).apply(lambda s: 1 if s in bad else 0)

    emp_years = df["emp_length"].apply(emp_to_years)
    ecl = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
    iss = pd.to_datetime(df["issue_d"], errors="coerce")
    months_hist = ((iss - ecl).dt.days / 30.44).clip(lower=0)

    out = pd.DataFrame()
    out["age"] = (22 + emp_years).clip(18,80).fillna(30)
    out["income"] = (df["annual_inc"]/12.0).clip(lower=0).fillna(0)
    out["liabilities"] = df["loan_amnt"].clip(lower=0).fillna(0)
    out["credit_history_months"] = months_hist.fillna(0).round()
    out["target_default_12m"] = y.astype(int)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

def main():
    parts = []
    # có file nào thì load file đó
    if (DATA_DIR / "taiwan_credit_default.csv").exists():
        print("[INFO] Load Taiwan…"); parts.append(load_taiwan())
    if (DATA_DIR / "lendingclub" / "accepted_2007_to_2018Q4.csv").exists():
        print("[INFO] Load LendingClub…"); parts.append(load_lendingclub())
    if not parts:
        raise SystemExit("Không thấy file nguồn nào trong data/. Dừng.")

    df_all = pd.concat(parts, ignore_index=True)
    # đảm bảo đúng cột & kiểu
    cols = ["age","income","liabilities","credit_history_months","target_default_12m"]
    df_all = df_all[cols]
    df_all["age"] = df_all["age"].astype(float)
    df_all["income"] = df_all["income"].astype(float)
    df_all["liabilities"] = df_all["liabilities"].astype(float)
    df_all["credit_history_months"] = df_all["credit_history_months"].astype(float)
    df_all["target_default_12m"] = df_all["target_default_12m"].astype(int)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUT, index=False)
    print(f"[DONE] Ghi {OUT} | shape={df_all.shape}")

if __name__ == "__main__":
    main()
