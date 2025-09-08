import argparse, json, math
from pathlib import Path 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve
import joblib

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--flavor", choices=["taiwan","lendingclub","custom"], required="true")
    p.add_argument("--data", required=True, help="CSV or XLSX file")
    p.add_argument("--artifacts", default="./artifacts", help="Thư mục ghi model")
    p.add_argument("--test-size", type=float, default=0.2, help="Tỷ lệ dữ liệu kiểm tra")
    p.add_argument("--random-state", type=int, default=42, help="Giá trị random_state")
    p.add_argument("--calibration", choices=["isotonic","sigmoid"], default="isotonic")
    p.add_argument("--max_rows", type=int, default=250_000, help="Giới hạn số dòng đọc (hữu ích cho LendingClub)")
    return p.parse_args()

def ks_stat(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))

def parse_issue(line):
    # ví dụ: "Jan-2015" hoặc "2015-01-01"
    try:
        return pd.to_datetime(line, errors="coerce")
    except Exception:
        return pd.NaT
    
def map_taiwan(df: pd.DataFrame) -> pd.DataFrame:
    required = ["AGE", "LIMIT_BAL"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Thiếu cột {c} trong dữ liệu Taiwan.")

    # Chuẩn hoá tên cột: hạ thấp, bỏ khoảng trắng thừa, thay khoảng trắng -> dấu chấm
    def norm(s):
        return str(s).strip().lower().replace("  ", " ").replace(" ", ".")
    norm_map = {norm(c): c for c in df.columns}

    # Các ứng viên tên nhãn phổ biến
    candidates = [
        "default.payment.next.month",    # bản UCI gốc
        "default.payment.next.month.",   # đôi khi có dấu chấm thừa
        "default.payment.next.month ",   # có khoảng trắng cuối
        "default.payment.next.months",
        "default.payment.nextmonth",
        "default.payment.next",          # an toàn
        "default.payment",               # an toàn
        "default",                       # Kaggle CSV đôi khi chỉ 'default'
        "default.payment.next.month".replace(".", " "),  # "default payment next month"
    ]

    label_key = None
    # 1) thử khớp trực tiếp các candidates đã chuẩn hoá
    for cand in candidates:
        if norm(cand) in norm_map:
            label_key = norm_map[norm(cand)]
            break
    # 2) fallback: tìm bất kỳ cột nào chứa 'default' trong tên chuẩn hoá
    if label_key is None:
        for k, orig in norm_map.items():
            if "default" in k:
                label_key = orig
                break
    if label_key is None:
        # In trợ giúp để bạn thấy toàn bộ cột
        cols_preview = ", ".join([repr(c) for c in df.columns.tolist()])
        raise ValueError(f"Không tìm thấy cột nhãn (chứa 'default'). Các cột đang có: {cols_preview}")

    bill_cols = [c for c in df.columns if str(c).startswith("BILL_AMT")]

    out = pd.DataFrame()
    out["age"] = df["AGE"].clip(18, 100)
    out["income"] = df["LIMIT_BAL"].clip(lower=0)
    out["liabilities"] = (df[bill_cols].fillna(0).mean(axis=1) if bill_cols else 0).clip(lower=0)
    out["credit_history_months"] = 6 if bill_cols else 0
    out["target_default_12m"] = df[label_key].astype(int)
    return out

def map_custom(df: pd.DataFrame) -> pd.DataFrame:
    need = ["age","income","liabilities","credit_history_months","target_default_12m"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CUSTOM: thiếu cột {missing}. File merged phải có đúng 5 cột {need}.")
    out = df[need].copy()
    # ép kiểu an toàn
    out["age"] = out["age"].astype(float).clip(18, 120)
    out["income"] = out["income"].astype(float).clip(lower=0)
    out["liabilities"] = out["liabilities"].astype(float).clip(lower=0)
    out["credit_history_months"] = out["credit_history_months"].astype(float).clip(lower=0)
    out["target_default_12m"] = out["target_default_12m"].astype(int).clip(0,1)
    return out

def map_lendingclub(df: pd.DataFrame, max_rows: int = 250_000) -> pd.DataFrame:
    """
    Chọn/biến đổi tối thiểu để đưa về schema:
      - age: xấp xỉ từ emp_length (năm đi làm) + 22, clip [18..80]
      - income: annual_inc / 12
      - liabilities: loan_amnt
      - credit_history_months: months between earliest_cr_line ↔ issue_d
      - target_default_12m: map từ loan_status
    """
    need_cols = [
        "loan_status","annual_inc","loan_amnt","emp_length",
        "earliest_cr_line","issue_d"
    ]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột trong LendingClub: {missing}")

    # Map nhãn
    bad = {
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
        "Late (31-120 days)",
        "In Grace Period"
    }
    y = df["loan_status"].astype(str).apply(lambda s: 1 if s in bad else 0)

    # emp_length: "10+ years", "3 years", "< 1 year", "n/a"
    def emp_to_years(v):
        v = str(v).strip().lower()
        if v == "n/a" or v == "nan": return np.nan
        if "10+" in v: return 10.0
        if "<" in v: return 0.5
        try:
            return float(v.split()[0])
        except Exception:
            return np.nan

    emp_years = df["emp_length"].apply(emp_to_years)

    # thời gian tín dụng
    ecl = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
    iss = pd.to_datetime(df["issue_d"], errors="coerce")
    months_hist = ((iss - ecl).dt.days / 30.44).clip(lower=0).fillna(0).round()

    out = pd.DataFrame()
    # age proxy = 22 + emp_years (clip)
    out["age"] = (22 + emp_years).clip(lower=18, upper=80).fillna(30)
    out["income"] = (df["annual_inc"] / 12.0).clip(lower=0).fillna(0)
    out["liabilities"] = df["loan_amnt"].clip(lower=0).fillna(0)
    out["credit_history_months"] = months_hist.fillna(0)
    out["target_default_12m"] = y.astype(int)

    # loại NA & giới hạn số dòng để dễ train
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    if max_rows and len(out) > max_rows:
        out = out.sample(max_rows, random_state=42)
    return out

def map_to_online_schema(df: pd.DataFrame, flavor: str, max_rows: int):
    if flavor == "taiwan":
        return map_taiwan(df)
    elif flavor == "lendingclub":
        return map_lendingclub(df, max_rows=max_rows)
    elif flavor == "custom":
        return map_custom(df)
    else:
        raise ValueError("Flavor không hỗ trợ.")
    
def main():
    args = parse_args()
    data_path = Path(args.data)
    art_dir = Path(args.artifacts); art_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Đọc dữ liệu: {data_path.resolve()}")
    if data_path.suffix in [".xls",".xlsx"]:
        df_raw = pd.read_excel(data_path, header=1)
    else:
        # Với file to của LendingClub, bật low_memory để giảm RAM
        df_raw = pd.read_csv(data_path, low_memory=False)

    df = map_to_online_schema(df_raw, args.flavor, args.max_rows)
    features = ["age","income","liabilities","credit_history_months"]
    target = "target_default_12m"
    X, y = df[features], df[target]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.random_state))
    ])
    pipe.fit(Xtr, ytr)
    p_raw = pipe.predict_proba(Xte)[:,1]
    auc_raw = roc_auc_score(yte, p_raw)

    calib = CalibratedClassifierCV(estimator=pipe, method=args.calibration, cv=5)
    calib.fit(Xtr, ytr)
    p_cal = calib.predict_proba(Xte)[:,1]
    auc_cal = roc_auc_score(yte, p_cal)
    ks = ks_stat(yte, p_cal)
    gini = 2*auc_cal - 1

    print(f"[KQ] AUC raw={auc_raw:.4f} | AUC calib={auc_cal:.4f} | KS={ks:.4f} | Gini={gini:.4f}")

    # save artifacts
    (art_dir/"feature_order.json").write_text(json.dumps(features))
    joblib.dump(pipe, art_dir/"pd_model.pkl")
    joblib.dump(calib, art_dir/"calibrator.pkl")
    joblib.dump(None, art_dir/"scaler.pkl")
    joblib.dump(None, art_dir/"encoder.pkl")
    (art_dir/"metrics.json").write_text(json.dumps({
        "auc_raw": float(auc_raw),
        "auc_calibrated": float(auc_cal),
        "ks": float(ks),
        "gini": float(gini),
        "flavor": args.flavor
    }, indent=2, ensure_ascii=False))

    print(f"[DONE] Artifacts lưu ở: {art_dir.resolve()}")

if __name__ == "__main__":
    main()