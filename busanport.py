# busanport.py
import re
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, Tuple, List, Optional

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ===== 0) 설정 =====
DATA_DIR = Path("./data")
YEARS = range(2020, 2025)  # 2020~2024

# 파일명 패턴
PATTERN_MONTHLY_CONTAINER = re.compile(r"\((\d{4})년\).*월별.*컨테이너.*\.xlsx", re.I)
PATTERN_CONTAINER_EXTRA   = re.compile(r"\((\d{4})년\).*선사별부두별.*컨테이너.*\.xlsx", re.I)

PATTERN_MONTHLY_VESSEL    = re.compile(r"\((\d{4})년\).*월별.*입출항.*\.xlsx", re.I)
PATTERN_SHIPTYPE_VESSEL   = re.compile(r"\((\d{4})년\).*선종별.*입출항.*\.xlsx", re.I)
PATTERN_TONNAGE_VESSEL    = re.compile(r"\((\d{4})년\).*톤급별.*입출항.*\.xlsx", re.I)

# ===== 유틸 =====
def clean_numeric(s):
    if pd.isna(s): return np.nan
    if isinstance(s, (int, float)): return float(s)
    s = str(s).strip().replace(",", "")
    s = re.sub(r"[^\d\.-]", "", s)
    try: return float(s)
    except: return np.nan

def find_files(patterns: Iterable[re.Pattern]) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for p in DATA_DIR.glob("*.xlsx"):
        for pat in patterns:
            m = pat.search(p.name)
            if m:
                y = int(m.group(1))
                if y in YEARS:
                    out.append((y, p))
                break
    return sorted(out)

def read_excel_best_sheet(path: Path) -> Optional[pd.DataFrame]:
    try:
        x = pd.ExcelFile(path)
    except Exception as e:
        print(f"[open error] {path.name}: {e}")
        return None

    best_df, best_score = None, -1
    month_regex = re.compile(r"^\s*(0?[1-9]|1[0-2])\s*월\s*$")
    for sheet in x.sheet_names:
        try:
            raw = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")
        except Exception as e:
            print(f"[read error] {path.name}:{sheet}: {e}")
            continue

        header_idx, score = None, -1
        for i in range(min(15, len(raw))):
            row = raw.iloc[i].astype(str).str.replace("\n", " ").str.strip()
            joined = " ".join(row)
            kscore = sum(k in joined for k in ["연월","년월","월","Month","month"])
            mcols  = sum(bool(month_regex.fullmatch(c)) for c in row)
            sc = kscore*3 + mcols*2
            if sc > score:
                score = sc; header_idx = i

        if header_idx is None: continue
        df = raw.copy()
        cols = df.iloc[header_idx].astype(str).str.replace("\n"," ").str.strip().tolist()
        df = df.iloc[header_idx+1:].reset_index(drop=True)
        df.columns = cols
        if score > best_score: best_df, best_score = df, score
    return best_df

def _safe_parse_ym(text: str, year_hint: Optional[int]):
    s = str(text)
    m = re.search(r"(20\d{2}).{0,5}?(1[0-2]|0?[1-9])\s*월", s) \
        or re.search(r"(20\d{2})[^\d]{0,3}(1[0-2]|0?[1-9])(\D|$)", s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return pd.Timestamp(f"{y}-{mo:02d}-01")
    if year_hint:
        m2 = re.search(r"(1[0-2]|0?[1-9])\s*월", s) or re.search(r"\b(1[0-2]|0?[1-9])\b", s)
        if m2:
            mo = int(m2.group(1))
            return pd.Timestamp(f"{year_hint}-{mo:02d}-01")
    return pd.NaT

def normalize_monthly(df: pd.DataFrame, year_hint: Optional[int] = None) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if any(k in c for k in ["연월","년월","월","Month","month","date"])), None)

    if date_col:
        df = df[~df[date_col].astype(str).str.contains(r"합계|총계|계\b")]
        df["date"] = df[date_col].apply(lambda v: _safe_parse_ym(v, year_hint))
    else:
        mcols = [c for c in df.columns if re.fullmatch(r"\s*(0?[1-9]|1[0-2])\s*월\s*", c)]
        if mcols:
            idv = [c for c in df.columns if c not in mcols]
            df = df.melt(id_vars=idv, value_vars=mcols, var_name="월", value_name="value")
            df["월"] = df["월"].str.extract(r"(\d{1,2})").astype(int)
            df["date"] = pd.to_datetime(f"{year_hint}-" + df["월"].astype(str) + "-01", errors="coerce")

    if "date" not in df.columns: return pd.DataFrame()
    for i, c in enumerate(df.columns):
        if c == "date": continue
        col = df.iloc[:, i]
        if col.dtype == object:
            sample = col.head(20).dropna().astype(str)
            if len(sample) and (sample.str.contains(r"\d").mean() > 0.6):
                df.iloc[:, i] = col.apply(clean_numeric)
    return df.dropna(subset=["date"]).drop_duplicates().sort_values("date")

# ===== 폭형(01월 척수/GT …) 파서 =====
def _detect_month_pairs(cols: List[str]) -> List[Tuple[int,int,Optional[int]]]:
    """컬럼명에서 월 열과 GT 짝 위치를 찾음. (중복 월명, 또는 오른쪽 3칸 GT 키워드)"""
    cols = [str(c).strip() for c in cols]
    month_pat = re.compile(r"^\s*(0?[1-9]|1[0-2])\s*월\s*$")
    idx = [(i, int(month_pat.fullmatch(c).group(1))) for i,c in enumerate(cols) if month_pat.fullmatch(c)]

    pos = []
    k = 0
    while k < len(idx):
        i, m1 = idx[k]
        if k+1 < len(idx) and idx[k+1][1]==m1 and idx[k+1][0]==i+1:
            pos.append((m1, i, i+1)); k += 2; continue
        gt_idx = None
        for j in range(i+1, min(i+4, len(cols))):
            if re.search(r"GT|총톤|총톤수|톤", cols[j], re.I):
                gt_idx = j; break
        pos.append((m1, i, gt_idx)); k += 1
    return pos

def parse_monthly_total(df: pd.DataFrame, year_hint: int) -> pd.DataFrame:
    """표 전체 합계를 월별(척수/GT)로."""
    if df is None or df.empty: return pd.DataFrame()
    cols = [str(c).strip() for c in df.columns]
    df = df.copy(); df.columns = cols

    total = df[df.astype(str).apply(lambda s: s.str.contains(r"총계|합계", na=False)).any(axis=1)]
    if total.empty: total = df

    month_pos = _detect_month_pairs(cols)
    if not month_pos: return pd.DataFrame()

    out = []
    for month, idx_cnt, idx_gt in month_pos:
        cnt = pd.to_numeric(total.iloc[:, idx_cnt].astype(str).str.replace(",", ""), errors="coerce").sum()
        rec = {"date": pd.Timestamp(f"{year_hint}-{month:02d}-01"), "척수": float(cnt)}
        if idx_gt is not None:
            gt = pd.to_numeric(total.iloc[:, idx_gt].astype(str).str.replace(",", ""), errors="coerce").sum()
            rec["GT"] = float(gt)
        out.append(rec)
    return pd.DataFrame(out).sort_values("date")

def parse_monthly_by_class(
    df: pd.DataFrame,
    year_hint: int,
    label_candidates=("톤","톤급","선박","선종","종류","구분"),
    label_name: str = "톤급",
) -> pd.DataFrame:
    """분류(톤급/선종 등)별 월별 시계열 → long 형태."""
    if df is None or df.empty: return pd.DataFrame()
    cols = [str(c).strip() for c in df.columns]
    df = df.copy(); df.columns = cols

    label_col = None
    for c in cols:
        if any(k in c for k in label_candidates):
            label_col = c; break
    if label_col is None: label_col = cols[0]

    body = df[~df.astype(str).apply(lambda s: s.str.contains(r"총계|합계", na=False)).any(axis=1)].copy()
    if body.empty: return pd.DataFrame()

    month_pos = _detect_month_pairs(cols)
    if not month_pos: return pd.DataFrame()

    rows = []
    for _, r in body.iterrows():
        name = str(r[label_col]).strip()
        if not name or name == 'nan': continue
        for month, idx_cnt, idx_gt in month_pos:
            cnt = clean_numeric(r.iloc[idx_cnt])
            gt  = clean_numeric(r.iloc[idx_gt]) if idx_gt is not None else np.nan
            rec = {
                "date": pd.Timestamp(f"{year_hint}-{month:02d}-01"),
                label_name: name,
                "척수": cnt if pd.notna(cnt) else np.nan
            }
            if pd.notna(gt): rec["GT"] = gt
            rows.append(rec)

    if not rows: return pd.DataFrame()
    df_out = pd.DataFrame(rows).sort_values([label_name,"date"])
    num_cols = [c for c in ["척수","GT"] if c in df_out.columns]
    return (df_out.groupby([label_name,"date"], as_index=False)[num_cols]
                 .sum(min_count=1).sort_values([label_name,"date"]))

# ===== 컨테이너/선박 빌더(기존) =====
def build_container(files: List[Tuple[int, Path]]) -> pd.DataFrame:
    rows = []
    for year, p in files:
        best = read_excel_best_sheet(p)
        if best is None:
            print(f"[skip container] {p.name}"); continue
        ts = normalize_monthly(best, year_hint=year)
        if ts.empty:
            print(f"[empty container] {p.name}"); continue
        keep = ["date"] + [c for c in ts.columns if any(k in c for k in
                ["합계","총계","수입","수출","환적","TEU","컨테이너","Full","Empty"])]
        ts = ts.loc[:, [c for c in keep if c in ts.columns]]
        ts["year"] = year; ts["source"] = p.name
        rows.append(ts)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def build_vessel(files: List[Tuple[int, Path]]) -> pd.DataFrame:
    rows = []
    for year, p in files:
        best = read_excel_best_sheet(p)
        if best is None:
            print(f"[skip vessel] {p.name}"); continue
        ts = normalize_monthly(best, year_hint=year)
        if ts.empty:
            ts = parse_monthly_total(best, year_hint=year)
            if ts.empty:
                print(f"[empty vessel] {p.name}"); continue
        keep = ["date"] + [c for c in ts.columns if c in ["척수","GT","입항","출항","내항","외항","합계","총계"]]
        ts = ts.loc[:, [c for c in keep if c in ts.columns]]
        ts["year"] = year; ts["source"] = p.name
        rows.append(ts)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# ===== 신규: shiptype 총합/선종별, tonnage 톤급별 =====
def build_shiptype_monthly_total(files: List[Tuple[int, Path]]) -> pd.DataFrame:
    rows = []
    for year, p in files:
        best = read_excel_best_sheet(p)
        if best is None:
            print(f"[skip shiptype] {p.name}"); continue
        ts = parse_monthly_total(best, year_hint=year)
        if ts.empty:
            ts = normalize_monthly(best, year_hint=year)
            if ts.empty:
                print(f"[empty shiptype] {p.name}"); continue
            num_cols = [c for c in ts.columns if c not in ["date"]]
            ts = ts.groupby("date", as_index=False)[num_cols].sum(min_count=1)
        rows.append(ts)
    if not rows: return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    num_cols = [c for c in ["척수","GT"] if c in df.columns]
    return (df.groupby("date", as_index=False)[num_cols].sum(min_count=1)
              .sort_values("date"))

def build_shiptype_byclass(files: List[Tuple[int, Path]]) -> pd.DataFrame:
    rows = []
    for year, p in files:
        best = read_excel_best_sheet(p)
        if best is None:
            print(f"[skip shiptype-byclass] {p.name}"); continue
        ts = parse_monthly_by_class(
            best, year_hint=year,
            label_candidates=("선종","선박종류","종류","구분","선박"),
            label_name="선"
        )
        if ts.empty:
            print(f"[empty shiptype-byclass] {p.name}"); continue
        rows.append(ts)
    if not rows: return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    num_cols = [c for c in ["척수","GT"] if c in df.columns]
    return (df.groupby(["선","date"], as_index=False)[num_cols]
              .sum(min_count=1).sort_values(["선","date"]))

def build_tonnage_byclass(files: List[Tuple[int, Path]]) -> pd.DataFrame:
    rows = []
    for year, p in files:
        best = read_excel_best_sheet(p)
        if best is None:
            print(f"[skip tonnage] {p.name}"); continue
        ts = parse_monthly_by_class(
            best, year_hint=year,
            label_candidates=("톤","톤급","선박","종류","구분"),
            label_name="톤급"
        )
        if ts.empty:
            print(f"[empty tonnage] {p.name}"); continue
        rows.append(ts)
    if not rows: return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    num_cols = [c for c in ["척수","GT"] if c in df.columns]
    return (df.groupby(["톤급","date"], as_index=False)[num_cols]
              .sum(min_count=1).sort_values(["톤급","date"]))

# ===== 디버그 =====
def debug_scan():
    print("[scan] in ./data")
    pats = {
        "월별컨테이너": PATTERN_MONTHLY_CONTAINER, "선사별부두별컨테이너": PATTERN_CONTAINER_EXTRA,
        "월별입출항": PATTERN_MONTHLY_VESSEL, "선종별입출항": PATTERN_SHIPTYPE_VESSEL,
        "톤급별입출항": PATTERN_TONNAGE_VESSEL
    }
    files = list(DATA_DIR.glob("*.xlsx"))
    for p in files:
        tags = [k for k,pat in pats.items() if pat.search(p.name)]
        print("-", p.name, "=>", tags or "NO MATCH")
    print(f"[scan] total xlsx = {len(files)}")

# ===== 메인 =====
if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    debug_scan()

    cont_files     = find_files([PATTERN_MONTHLY_CONTAINER, PATTERN_CONTAINER_EXTRA])
    vess_files     = find_files([PATTERN_MONTHLY_VESSEL, PATTERN_SHIPTYPE_VESSEL, PATTERN_TONNAGE_VESSEL])
    shiptype_files = find_files([PATTERN_SHIPTYPE_VESSEL])
    tonnage_files  = find_files([PATTERN_TONNAGE_VESSEL])

    cont = build_container(cont_files)
    vess = build_vessel(vess_files)
    shiptype_total = build_shiptype_monthly_total(shiptype_files)
    shiptype_bycls = build_shiptype_byclass(shiptype_files)
    tonnage_bycls  = build_tonnage_byclass(tonnage_files)

    print(f"\n[container] rows={len(cont)}")
    print(f"[vessel]    rows={len(vess)}")
    print(f"[shiptype-total] rows={len(shiptype_total)}")
    print(f"[shiptype-byclass] rows={len(shiptype_bycls)}")
    print(f"[tonnage-byclass] rows={len(tonnage_bycls)}")

    if not cont.empty:
        cont.to_csv("busan_container_2020_2024.csv", index=False, encoding="utf-8-sig")
        print("[saved] busan_container_2020_2024.csv")
    if not vess.empty:
        vess.to_csv("busan_vessel_2020_2024.csv", index=False, encoding="utf-8-sig")
        print("[saved] busan_vessel_2020_2024.csv")
    if not shiptype_total.empty:
        shiptype_total.to_csv("busan_vessel_shiptype_monthly_total_2020_2024.csv", index=False, encoding="utf-8-sig")
        print("[saved] busan_vessel_shiptype_monthly_total_2020_2024.csv")
    if not shiptype_bycls.empty:
        shiptype_bycls.to_csv("busan_vessel_shiptype_byclass_2020_2024.csv", index=False, encoding="utf-8-sig")
        print("[saved] busan_vessel_shiptype_byclass_2020_2024.csv")
    if not tonnage_bycls.empty:
        tonnage_bycls.to_csv("busan_vessel_tonnage_byclass_2020_2024.csv", index=False, encoding="utf-8-sig")
        print("[saved] busan_vessel_tonnage_byclass_2020_2024.csv")
