import pandas as pd
import os

SMALLCAP_CSV = 'data/nifty_smallcap_250.csv'
SME_CSV = 'data/sme_watchlist.csv'

def check_smallcap_250_csv():
    os.makedirs('data', exist_ok=True)
    if not os.path.exists(SMALLCAP_CSV):
        print("📥 Smallcap CSV not found. Please download it from:")
        print("https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv")
        print(f"and save it as: {SMALLCAP_CSV}")
    else:
        print(f"✅ Smallcap CSV found: {SMALLCAP_CSV}")

def update_sme_watchlist_csv():
    os.makedirs('data', exist_ok=True)
    if not os.path.exists(SME_CSV):
        with open(SME_CSV, 'w') as f:
            f.write("Symbol\n")
        print(f"📄 Created new SME watchlist at: {SME_CSV}")
    else:
        print(f"✅ SME watchlist found: {SME_CSV}")

def get_smallcap_symbols(path=SMALLCAP_CSV):
    try:
        df = pd.read_csv(path)
        symbol_col = next((col for col in df.columns if 'symbol' in col.lower()), None)
        if not symbol_col:
            raise ValueError("❌ Could not find 'Symbol' column in Smallcap CSV.")
        symbols = df[symbol_col].dropna().astype(str).str.strip().tolist()
        symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in symbols]
        return symbols
    except Exception as e:
        print(f"❌ Error reading smallcap symbols: {e}")
        return []

def get_sme_symbols(path=SME_CSV):
    try:
        df = pd.read_csv(path)
        if 'Symbol' not in df.columns:
            raise ValueError("❌ SME CSV must have a 'Symbol' column.")
        symbols = df['Symbol'].dropna().astype(str).str.strip().tolist()
        symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in symbols]
        return symbols
    except Exception as e:
        print(f"❌ Error reading SME symbols: {e}")
        return []

def load_stock_lists():
    return get_smallcap_symbols(), get_sme_symbols()

