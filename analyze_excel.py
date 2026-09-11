import pandas as pd
import warnings, sys, io
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

FILE = r"D:\TOLAND FOOTBALL FEDERATION\PLOFA-2025-2026.COM\PLOFA 2025-2026 II.xlsx"

xls = pd.ExcelFile(FILE, engine='openpyxl')
print("=" * 100)
print(f"FILE: PLOFA 2025-2026 II.xlsx")
print(f"SHEET NAMES ({len(xls.sheet_names)}):")
for i, name in enumerate(xls.sheet_names, 1):
    print(f"  {i:2d}. {name}")
print("=" * 100)

METRIC_KEYWORDS = [
    'goal', 'assist', 'rating', 'score', 'match', 'appear',
    'minute', 'yellow', 'red', 'clean sheet', 'save', 'tackle',
    'interception', 'pass', 'dribble', 'cross', 'shot', 'foul',
    'offside', 'sub', 'start', 'win', 'draw', 'loss', 'point',
    'fifa', 'ovr', 'pac', 'sho', 'pas', 'dri', 'def', 'phy',
    'position', 'club', 'team', 'player', 'name', 'overall',
    'pace', 'shooting', 'passing', 'dribbling', 'defending',
    'physical', 'gk', 'momentum', 'clutch', 'impact'
]

for sheet_name in xls.sheet_names:
    print(f"\n{'=' * 100}")
    print(f"SHEET: {sheet_name}")
    print(f"{'=' * 100}")

    df = pd.read_excel(xls, sheet_name=sheet_name)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nCOLUMN NAMES ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:3d}. {col}")

    print(f"\nFIRST 5 ROWS (key columns only):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 300)
    pd.set_option('display.max_colwidth', 25)

    # For wide sheets, just show first 8 columns + any metric columns
    metric_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        for kw in METRIC_KEYWORDS:
            if kw in col_lower:
                metric_cols.append(col)
                break

    if df.shape[1] <= 15:
        try:
            print(df.head(5).to_string(index=True))
        except:
            print(df.head(5).to_string(index=True, max_colwidth=20))
    else:
        # Wide sheet: show first 6 cols + metric cols
        show_cols = list(df.columns[:6])
        for mc in metric_cols:
            if mc not in show_cols:
                show_cols.append(mc)
        try:
            print(df[show_cols].head(5).to_string(index=True))
        except:
            print(df[show_cols].head(5).to_string(index=True, max_colwidth=20))
        if metric_cols:
            print(f"\n  (+ {len(metric_cols)} metric-related columns identified)")

    # Identify performance metrics
    if metric_cols:
        print(f"\nPERFORMANCE METRIC COLUMNS:")
        for col in metric_cols:
            nunique = df[col].nunique()
            nonnull = df[col].notna().sum()
            if df[col].dtype in ['float64', 'int64']:
                print(f"  >> {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}, non-null={nonnull}/{len(df)}")
            else:
                print(f"  >> {col}: {nunique} unique values, non-null={nonnull}/{len(df)}")

    # Numeric summary
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols and len(numeric_cols) <= 30:
        print(f"\nNUMERIC SUMMARY:")
        print(df[numeric_cols].describe().round(2).to_string())

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
