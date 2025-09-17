# app_streamlit.py
import os
from io import BytesIO
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st

# -------- Optional viz libs --------
SEABORN_OK = False
try:
    import seaborn as sns
    SEABORN_OK = True
    sns.set_theme(
        context="talk",
        style="whitegrid",
        rc={
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "grid.color": "#e5e7eb",
            "axes.edgecolor": "#e5e7eb",
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
        },
    )
except Exception:
    SEABORN_OK = False

PLOTLY_OK = False
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# -------- Optional PDF export --------
REPORTLAB_OK = False
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# -----------------------
# Page setup + styles
# -----------------------
st.set_page_config(page_title="NordicAI — Fashion Starter", layout="wide")

PRIMARY   = "#162159"   # deep royal
PRIMARY_2 = "#0E173F"   # deeper
ACCENT    = "#3558FF"   # accent
TEXT_ON_DARK = "#FFFFFF"
TEXT_DARK    = "#0B1228"
TEXT_MID     = "#334155"

CUSTOM_CSS = f"""
<style>
.stApp {{
  background: radial-gradient(100% 120% at 0% 0%, {PRIMARY_2} 0%, {PRIMARY} 60%, #0A102C 100%);
  color: {TEXT_ON_DARK};
}}
html, body, [class*="css"]  {{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial,
               "Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol", sans-serif;
}}
.hero {{
  border-radius: 20px; padding: 22px 22px 16px 22px;
  background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
  border: 1px solid rgba(255,255,255,0.18);
  box-shadow: 0 10px 30px rgba(0,0,0,0.25), inset 0 0 1px rgba(255,255,255,0.35);
  color: {TEXT_ON_DARK};
}}
.hero h1 {{ margin: 0; font-size: 28px; font-weight: 800; letter-spacing: .2px; color: {TEXT_ON_DARK}; }}
.chip {{ display: inline-block; padding: 6px 12px; border-radius: 999px; background: {ACCENT};
        color: white; font-size: 12px; letter-spacing: .2px; font-weight: 700;
        box-shadow: 0 2px 8px rgba(53,88,255,0.4); }}
.small {{ color: rgba(255,255,255,0.88); font-size: 12px; }}

.block-card, .metric {{
  background: #ffffff; border-radius: 16px; padding: 16px;
  box-shadow: 0 6px 24px rgba(13,25,60,0.12); border: 1px solid rgba(13,25,60,0.08);
  color: {TEXT_DARK};
}}
.metric .label {{ font-size: 12px; color: {TEXT_MID}; font-weight: 600; }}
.metric .value {{ font-size: 24px; font-weight: 800; color: {PRIMARY}; }}
.hr {{ height: 1px; background: rgba(255,255,255,0.25); margin: 14px 0 10px 0; }}

.stTabs [role="tablist"] div[role="tab"] {{ color: {TEXT_ON_DARK}; font-weight: 600; }}
.stTabs [aria-selected="true"] {{ border-bottom: 2px solid {ACCENT} !important; }}

.block-card .stDataFrame, .block-card table, .metric * {{ color: {TEXT_DARK} !important; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------
# Plain-language helpers
# -----------------------
GLOSSARY = {
    "SKU": "The product code for one item (often one size or color).",
    "Revenue": "Money earned from sales (before costs).",
    "Units sold": "How many pieces were sold.",
    "On hand": "How many pieces you have in stock now.",
    "Avg weekly units": "Average pieces sold per week (over the period shown).",
    "Sell-through": "Percent sold = sold / (sold + in stock). Higher = selling better.",
    "Weeks of Cover (WOC)": "How many weeks current stock would last at recent sales speed.",
    "ABC": "A = top sellers (~80% of revenue), B = mid, C = low.",
    "Suggested markdown": "The price reduction we suggest (e.g., 0.10 = 10% off).",
    "Reorder candidate": "Selling fast and may run out soon → order more.",
    "Markdown candidate": "Too much stock for current sales → discount to move.",
    "Phase-out": "Low demand item → consider ending or replacing.",
    "Trend score": "How closely an item matches selected trends.",
    "Lifecycle": "Launch (<8 weeks), Core (8–40), End-of-life (>40) since launch date.",
    "Gross profit": "Revenue minus estimated product cost.",
    "Performance score": "Combined score (revenue + sell-through + low WOC). Higher = better.",
}

# --- simple password gate (set APP_PASSWORD in env/secrets) ---
def _auth_gate():
    pwd = os.getenv("APP_PASSWORD", "")
    if not pwd:
        return  # no password configured
    if st.session_state.get("authed"):
        return
    st.title("🔒 Protected")
    with st.form("auth"):
        secret = st.text_input("Enter access password", type="password")
        ok = st.form_submit_button("Enter")
    if ok:
        if secret == pwd:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password")
            st.stop()
    else:
        st.stop()

_auth_gate()

# --- secrets helper: works locally (env var) and on Streamlit Cloud (st.secrets)
def get_secret(name: str, default: str = "") -> str:
    try:
        val = os.getenv(name, "")
        if not val:
            # st.secrets exists only on Streamlit Cloud (and locally if secrets.toml is present)
            val = str(st.secrets.get(name, default))
        return val.strip()
    except Exception:
        return default




def note(text: str):
    """Show a gentle caption only if plain-language mode is on."""
    try:
        if lang_simple:
            st.caption(text)
    except NameError:
        st.caption(text)

# -----------------------
# Data helpers
# -----------------------
def load_default():
    """Load NordicAI sample data if present."""
    here = Path(__file__).resolve().parent
    for base in [here / "data", Path.cwd() / "data"]:
        if (base / "sample_orders.csv").exists():
            orders  = pd.read_csv(base / "sample_orders.csv", parse_dates=["order_date"])
            inv     = pd.read_csv(base / "sample_inventory.csv")
            prod    = pd.read_csv(base / "product_master.csv", parse_dates=["launch_date"])
            trends  = pd.read_csv(base / "trend_weights.csv")
            return orders, inv, prod, trends
    return None, None, None, None

def _col(df, *cands):
    for c in cands:
        if c in df.columns: return c
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None

def normalize_shopify_orders(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return None
    c_date = _col(df, "Created at", "Created At", "created_at")
    c_sku  = _col(df, "Lineitem sku", "Lineitem SKU", "SKU", "Variant SKU", "Sku")
    c_qty  = _col(df, "Lineitem quantity", "Quantity")
    c_price= _col(df, "Lineitem price", "Price")
    c_disc = _col(df, "Lineitem discount", "Discount Amount", "Discounts")
    if not all([c_date, c_sku, c_qty, c_price]):
        if all(c in df.columns for c in ["order_date","sku","units","unit_price"]):
            out = df.copy()
            out["discount_rate"] = out.get("discount_rate", 0).fillna(0).astype(float)
            out["order_date"] = pd.to_datetime(out["order_date"])
            return out[["order_date","sku","units","unit_price","discount_rate"]]
        return None
    out = pd.DataFrame({
        "order_date": pd.to_datetime(df[c_date], errors="coerce"),
        "sku": df[c_sku].astype(str),
        "units": pd.to_numeric(df[c_qty], errors="coerce").fillna(0).astype(int),
        "unit_price": pd.to_numeric(df[c_price], errors="coerce").fillna(0.0).astype(float),
    })
    if c_disc and c_disc in df.columns:
        disc_abs = pd.to_numeric(df[c_disc], errors="coerce").fillna(0.0)
        denom = out["unit_price"] * out["units"]
        out["discount_rate"] = np.where(denom>0, (disc_abs/denom).clip(0,1), 0.0)
    else:
        out["discount_rate"] = 0.0
    out = out[out["sku"].str.strip()!=""]
    return out.dropna(subset=["order_date"])

def normalize_shopify_products(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return None
    c_sku  = _col(df, "Variant SKU", "SKU", "Sku")
    c_title= _col(df, "Title")
    c_type = _col(df, "Type", "Product Category")
    c_price= _col(df, "Variant Price", "Price")
    c_cost = _col(df, "Variant Cost", "Cost per item", "Cost Per Item", "Cost")
    c_pub  = _col(df, "Published", "Published At", "Published at", "Created at", "Created At")

    c_opt1n= _col(df, "Option1 Name"); c_opt1v=_col(df, "Option1 Value")
    c_opt2n= _col(df, "Option2 Name"); c_opt2v=_col(df, "Option2 Value")
    c_opt3n= _col(df, "Option3 Name"); c_opt3v=_col(df, "Option3 Value")

    if not c_sku:
        if "sku" in df.columns:
            out = df.copy()
            if "launch_date" in out.columns:
                out["launch_date"] = pd.to_datetime(out["launch_date"], errors="coerce")
            return out
        return None

    def pick_opt(name_col, val_col, *labels):
        if (name_col and val_col) and (name_col in df.columns and val_col in df.columns):
            mask = df[name_col].astype(str).str.lower().isin([l.lower() for l in labels])
            vals = df[val_col].where(mask, "")
            return vals.astype(str)
        return pd.Series([""]*len(df))

    size = pick_opt(c_opt1n,c_opt1v,"size")
    if size.eq("").all(): size = pick_opt(c_opt2n,c_opt2v,"size")
    if size.eq("").all(): size = pick_opt(c_opt3n,c_opt3v,"size")

    color = pick_opt(c_opt1n,c_opt1v,"color","colour")
    if color.eq("").all(): color = pick_opt(c_opt2n,c_opt2v,"color","colour")
    if color.eq("").all(): color = pick_opt(c_opt3n,c_opt3v,"color","colour")

    material = pick_opt(c_opt1n,c_opt1v,"material","fabric")
    if material.eq("").all(): material = pick_opt(c_opt2n,c_opt2v,"material","fabric")
    if material.eq("").all(): material = pick_opt(c_opt3n,c_opt3v,"material","fabric")

    silhouette = pick_opt(c_opt1n,c_opt1v,"style","fit","shape","silhouette")
    if silhouette.eq("").all(): silhouette = pick_opt(c_opt2n,c_opt2v,"style","fit","shape","silhouette")
    if silhouette.eq("").all(): silhouette = pick_opt(c_opt3n,c_opt3v,"style","fit","shape","silhouette")

    out = pd.DataFrame({
        "sku": df[c_sku].astype(str),
        "style_name": df[c_title].astype(str) if c_title else "",
        "category": df[c_type].astype(str) if c_type else "",
        "price": pd.to_numeric(df[c_price], errors="coerce").fillna(0.0),
        "cost": pd.to_numeric(df[c_cost], errors="coerce").fillna(0.0) if c_cost else 0.0,
        "launch_date": pd.to_datetime(df[c_pub], errors="coerce") if c_pub else pd.NaT,
        "size": size,
        "color": color,
        "material": material,
        "silhouette": silhouette,
    })
    out = out[out["sku"].str.strip()!=""]
    return out

def normalize_shopify_inventory(inv_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    if inv_df is not None and not inv_df.empty:
        c_sku = _col(inv_df, "SKU", "Variant SKU", "Sku")
        c_av  = _col(inv_df, "Available", "Quantity", "On Hand", "On-hand")
        c_size = _col(inv_df, "Size")
        if c_sku and c_av:
            out = inv_df[[c_sku, c_av] + ([c_size] if c_size else [])].copy()
            out.columns = ["sku","on_hand"] + (["size"] if c_size else [])
            out["on_hand"] = pd.to_numeric(out["on_hand"], errors="coerce").fillna(0).astype(int)
            out["sku"] = out["sku"].astype(str)
            if "size" in out.columns:
                out = out.groupby(["sku","size"], as_index=False)["on_hand"].sum()
            else:
                out = out.groupby("sku", as_index=False)["on_hand"].sum()
            return out[out["sku"].str.strip()!=""]

    if products_df is not None and not products_df.empty:
        c_sku = _col(products_df, "Variant SKU", "SKU", "Sku")
        c_qty = _col(products_df, "Variant Inventory Qty", "Inventory Quantity", "Inventory qty", "Variant Inventory Quantity")
        if c_sku and c_qty:
            out = products_df[[c_sku, c_qty]].copy()
            out.columns = ["sku","on_hand"]
            out["on_hand"] = pd.to_numeric(out["on_hand"], errors="coerce").fillna(0).astype(int)
            out["sku"] = out["sku"].astype(str)
            out = out.groupby("sku", as_index=False)["on_hand"].sum()
            return out[out["sku"].str.strip()!=""]

    if products_df is not None and not products_df.empty:
        c_sku = _col(products_df, "Variant SKU", "SKU", "Sku")
        if c_sku:
            return pd.DataFrame({"sku": products_df[c_sku].astype(str).unique(), "on_hand": 0})
    return None

# -----------------------
# Metrics & trends
# -----------------------
def compute_metrics(orders, inv, prod):
    orders["revenue"] = orders["units"] * orders["unit_price"] * (1 - orders["discount_rate"])
    sku_sales  = orders.groupby("sku").agg(units_sold=("units","sum"), revenue=("revenue","sum")).reset_index()
    sku_onhand = inv.groupby("sku")["on_hand"].sum().reset_index()
    base_cols = [c for c in ["sku","category","style_name","price","cost","launch_date"] if c in prod.columns]
    metrics = sku_sales.merge(sku_onhand, on="sku", how="outer").fillna({"units_sold":0,"revenue":0,"on_hand":0})
    metrics = metrics.merge(prod[base_cols], on="sku", how="left")

    min_date, max_date = orders["order_date"].min(), orders["order_date"].max()
    weeks_covered = max(1.0, (max_date - min_date).days / 7.0)
    metrics["sell_through_pct"] = metrics["units_sold"] / (metrics["units_sold"] + metrics["on_hand"]).replace(0, np.nan)
    metrics["sell_through_pct"] = metrics["sell_through_pct"].fillna(0.0)
    metrics["avg_weekly_units"] = metrics["units_sold"] / weeks_covered if weeks_covered>0 else 0.0
    metrics["weeks_of_cover"] = np.where(metrics["avg_weekly_units"]>0, metrics["on_hand"] / metrics["avg_weekly_units"], 999)

    metrics = metrics.sort_values("revenue", ascending=False).reset_index(drop=True)
    metrics["cum_revenue"] = metrics["revenue"].cumsum()
    total_rev = metrics["revenue"].sum()
    metrics["cum_pct"] = metrics["cum_revenue"] / total_rev if total_rev>0 else 0.0
    metrics["ABC"] = metrics["cum_pct"].apply(lambda p: "A" if p<=0.80 else ("B" if p<=0.95 else "C"))

    def markdown_from_woc(woc):
        if woc >= 12: return 0.25
        if woc >= 8:  return 0.15
        if woc >= 6:  return 0.10
        return 0.00
    metrics["suggested_markdown"] = metrics["weeks_of_cover"].apply(markdown_from_woc)

    def action_row(row):
        if (row["sell_through_pct"] > 0.70) and (row["weeks_of_cover"] < 4): return "Reorder"
        if row["weeks_of_cover"] >= 8: return "Mark down"
        if row["sell_through_pct"] < 0.40: return "Phase out"
        return "Monitor"
    metrics["action"] = metrics.apply(action_row, axis=1)

    metrics["gross_profit"] = metrics["revenue"] - (metrics.get("cost", 0).fillna(0) * metrics["units_sold"])
    def _ranknorm(s): return s.rank(pct=True, method="average").fillna(0)
    woc_clip = metrics["weeks_of_cover"].replace([np.inf, np.nan], 999).clip(0, 30)
    metrics["performance_score"] = 0.50*_ranknorm(metrics["revenue"]) + 0.30*_ranknorm(metrics["sell_through_pct"]) + 0.20*(1 - _ranknorm(woc_clip))

    if "launch_date" in metrics.columns:
        metrics["weeks_since_launch"] = ((max_date - pd.to_datetime(metrics["launch_date"])).dt.days / 7).clip(lower=0)
        def life(w): return "Launch" if w<8 else ("Core" if w<40 else "EOL")
        metrics["lifecycle"] = metrics["weeks_since_launch"].apply(lambda w: "Unknown" if pd.isna(w) else life(w))
    else:
        metrics["weeks_since_launch"] = np.nan
        metrics["lifecycle"] = "Unknown"

    return metrics, (min_date, max_date, weeks_covered)

def compute_trends(prod, trends):
    trends_list = trends["trend"].tolist()
    flags = pd.DataFrame({"sku": prod["sku"]})
    for t in trends_list:
        flags[t] = prod.apply(lambda r: 1 if (str(r.get("color","")).lower()==t.lower() or 
                                              str(r.get("material","")).lower()==t.lower() or 
                                              str(r.get("silhouette","")).lower()==t.lower()) else 0, axis=1)
    weight_map = dict(zip(trends["trend"], trends["weight"]))
    score = np.zeros(len(flags))
    for t in trends_list:
        score += flags[t].values * weight_map.get(t, 0.0)
    flags["trend_score"] = score
    return flags.merge(prod[["sku","style_name","category","color","material","silhouette"]], on="sku", how="left")

def compute_size_gaps(orders, inv, prod):
    orders_sz = orders.copy()
    if "size" not in orders_sz.columns and "size" in prod.columns:
        orders_sz = orders_sz.merge(prod[["sku","size"]], on="sku", how="left")

    inv_sz = inv.copy()
    if "size" not in inv_sz.columns and "size" in prod.columns:
        inv_sz = inv_sz.merge(prod[["sku","size"]], on="sku", how="left")

    if "order_date" in orders_sz.columns and pd.api.types.is_datetime64_any_dtype(orders_sz["order_date"]):
        cut = orders_sz["order_date"].max() - pd.Timedelta(weeks=8)
        recent = orders_sz[orders_sz["order_date"] >= cut].copy()
        if recent.empty: recent = orders_sz.copy()
    else:
        recent = orders_sz.copy()

    if "size" not in recent.columns and "size" not in inv_sz.columns:
        return None, None

    demand = recent.groupby("size", dropna=False)["units"].sum().reset_index().rename(columns={"units":"units_sold_8w"})
    if "size" in inv_sz.columns:
        onhand = inv_sz.groupby("size", dropna=False)["on_hand"].sum().reset_index()
    else:
        onhand = pd.DataFrame({"size": demand["size"], "on_hand": 0})

    size_summary = demand.merge(onhand, on="size", how="outer").fillna({"units_sold_8w":0,"on_hand":0})
    total = (size_summary["units_sold_8w"] + size_summary["on_hand"]).replace(0, np.nan)
    size_summary["sell_through_pct"] = (size_summary["units_sold_8w"] / total).fillna(0)
    weekly = size_summary["units_sold_8w"] / 8.0
    size_summary["weeks_of_cover"] = np.where(weekly>0, size_summary["on_hand"]/weekly, 999)

    if "size" in inv_sz.columns:
        last8 = recent.groupby(["sku","size"], dropna=False)["units"].sum().reset_index().rename(columns={"units":"units_sold_8w"})
        oh = inv_sz.groupby(["sku","size"], dropna=False)["on_hand"].sum().reset_index()
        oos = last8.merge(oh, on=["sku","size"], how="left").fillna({"on_hand":0})
        oos = oos[(oos["units_sold_8w"]>0) & (oos["on_hand"]==0)].sort_values("units_sold_8w", ascending=False).head(50)
    else:
        oos = None

    return size_summary.sort_values("units_sold_8w", ascending=False), oos

# ===================== SIDEBAR (FULL) =====================
with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose input format", ["NordicAI template (4 CSVs)", "Shopify exports"], index=0)

    st.header("Uploads")
    if source.startswith("NordicAI"):
        orders_file = st.file_uploader("Orders CSV", type="csv")
        inv_file    = st.file_uploader("Inventory CSV", type="csv")
        prod_file   = st.file_uploader("Products CSV", type="csv")
        trends_file = st.file_uploader("Trends CSV", type="csv", help="Columns: trend, weight (0.0–1.0)")
    else:
        shop_orders    = st.file_uploader("Shopify Orders export CSV", type="csv")
        shop_products  = st.file_uploader("Shopify Products export CSV", type="csv")
        shop_inventory = st.file_uploader("Shopify Inventory by location CSV (optional)", type="csv")
        trends_file    = st.file_uploader("Trends CSV (NordicAI format)", type="csv", help="Columns: trend, weight (0.0–1.0)")

    st.header("Chart style")
    style = st.selectbox(
        "Charts",
        ["Interactive (Plotly)", "Static (Seaborn)"],
        index=0 if PLOTLY_OK else 1,
        help="Interactive = hover/zoom/export. Static = lightweight images."
    )

    st.header("Thresholds")
    reorder_st = st.slider("Reorder if Sell-through >", 0.5, 0.9, 0.70, 0.01,
                           help="If an item’s sell-through is above this, it’s selling well.")
    reorder_woc = st.slider("and WOC <", 1, 8, 4, 1,
                            help="WOC = Weeks of Cover: how many weeks current stock lasts at recent sales speed.")
    md_woc_1 = st.slider("Markdown 10% if WOC >=", 4, 12, 6, 1)
    md_woc_2 = st.slider("Markdown 15% if WOC >=", 6, 16, 8, 1)
    md_woc_3 = st.slider("Markdown 25% if WOC >=", 8, 20, 12, 1)

    lang_simple = st.checkbox(
        "Show simple explanations",
        value=True,
        help="Turn on short, non-technical explanations under each chart/table."
    )

    # --- Copilot status & diagnostics (GPT-5) ---
# --- Copilot status & diagnostics (GPT-5 ready) ---
st.divider()
_api_key = get_secret("OPENAI_API_KEY")

st.markdown("#### Copilot status  " + ("✅ OpenAI key detected" if _api_key else "⚠️ No API key found"))

with st.expander("LLM diagnostics", expanded=False):
    try:
        from openai import __version__ as _openai_ver
        st.caption(f"openai lib: v{_openai_ver}")
    except Exception as e:
        st.caption(f"openai lib: not available ({e})")

    colA, colB = st.columns(2)
    with colA:
        if st.button("Ping GPT-5 (Responses)"):
            if not _api_key:
                st.error("No OPENAI_API_KEY in environment.")
            else:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=_api_key)
                    r = client.responses.create(
                        model="gpt-5",
                        input="Say OK",
                        max_completion_tokens=5,   # <-- Responses param
                        temperature=0
                    )
                    txt = getattr(r, "output_text", None)
                    if not txt and getattr(r, "output", None):
                        txt = r.output[0].content[0].text
                    st.success(f"LLM OK (Responses): {txt or 'OK'}")
                except Exception as e:
                    st.error(f"Responses ping failed: {e}")

    with colB:
        if st.button("Ping gpt-4o-mini (Chat)"):
            if not _api_key:
                st.error("No OPENAI_API_KEY in environment.")
            else:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=_api_key)
                    r = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"user","content":"Say OK"}],
                        max_tokens=5,              # <-- Chat Completions param
                        temperature=0
                    )
                    st.success(f"LLM OK (Chat): {r.choices[0].message.content}")
                except Exception as e:
                    st.error(f"Chat ping failed: {e}")

# ===================== END SIDEBAR =====================

# -----------------------
# Load/normalize data
# -----------------------
if source.startswith("NordicAI"):
    orders, inv, prod, trends = load_default()
    if orders_file: orders = pd.read_csv(orders_file, parse_dates=["order_date"])
    if inv_file:    inv    = pd.read_csv(inv_file)
    if prod_file:   prod   = pd.read_csv(prod_file, parse_dates=["launch_date"])
    if trends_file: trends = pd.read_csv(trends_file)
else:
    orders, inv, prod, trends = None, None, None, None
    products_raw = None
    inventory_raw = None
    if 'shop_orders' in locals() and shop_orders is not None:
        orders_raw = pd.read_csv(shop_orders, low_memory=False)
        orders = normalize_shopify_orders(orders_raw)
    if 'shop_products' in locals() and shop_products is not None:
        products_raw = pd.read_csv(shop_products, low_memory=False)
        prod = normalize_shopify_products(products_raw)
    if 'shop_inventory' in locals() and shop_inventory is not None:
        inventory_raw = pd.read_csv(shop_inventory, low_memory=False)
    if prod is not None:
        inv = normalize_shopify_inventory(inventory_raw, products_raw if products_raw is not None else None)
    if trends_file:
        trends = pd.read_csv(trends_file)

if any(x is None for x in [orders, inv, prod, trends]):
    st.info("Missing data. Upload the required CSVs (or place sample data in ./data).")
    st.stop()

# Apply user thresholds to actions/markdown
def apply_thresholds(metrics):
    def markdown_from_woc_custom(woc):
        if woc >= md_woc_3: return 0.25
        if woc >= md_woc_2: return 0.15
        if woc >= md_woc_1: return 0.10
        return 0.00
    metrics["suggested_markdown"] = metrics["weeks_of_cover"].apply(markdown_from_woc_custom)
    def action_row_custom(row):
        if (row["sell_through_pct"] > reorder_st) and (row["weeks_of_cover"] < reorder_woc): return "Reorder"
        if row["weeks_of_cover"] >= md_woc_2: return "Mark down"
        if row["sell_through_pct"] < 0.40: return "Phase out"
        return "Monitor"
    metrics["action"] = metrics.apply(action_row_custom, axis=1)
    return metrics

metrics, (min_date, max_date, weeks_covered) = compute_metrics(orders, inv, prod)
metrics = apply_thresholds(metrics)
trend_scores = compute_trends(prod, trends)
size_summary, oos_by_sku = compute_size_gaps(orders, inv, prod)

# -----------------------
# Header + KPIs
# -----------------------
st.markdown(f"""
<div class="hero">
  <div class="chip">NordicAI</div>
  <h1>Fashion Starter — Live Insights</h1>
  <div class="small">Date range: {min_date.date()} → {max_date.date()} • Weeks covered: {weeks_covered:.1f}</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="metric"><div class="label">Revenue</div><div class="value">{metrics["revenue"].sum():,.0f}</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric"><div class="label">Units sold</div><div class="value">{metrics["units_sold"].sum():,.0f}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric"><div class="label">On hand</div><div class="value">{metrics["on_hand"].sum():,.0f}</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="metric"><div class="label">Avg weekly units</div><div class="value">{metrics["avg_weekly_units"].mean():.2f}</div></div>', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------
# Tabs
# -----------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Overview", "Details", "Trends", "Sizes", "Lifecycle", "Simulator", "Copilot"]
)

def thousands(x, pos):
    try: return f"{int(x):,}"
    except: return str(x)

# --- Overview ---
with tab1:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("Top 10 SKUs by Revenue")
    note("These are the 10 products that brought in the most money in the selected period.")
    top10 = metrics.sort_values("revenue", ascending=False).head(10)
    if PLOTLY_OK and style.startswith("Interactive"):
        fig = px.bar(top10, x="sku", y="revenue", text="revenue", color_discrete_sequence=[ACCENT])
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
        fig.update_layout(template="simple_white", xaxis_title="", yaxis_title="Revenue",
                          margin=dict(l=20, r=20, t=40, b=20), font=dict(size=14))
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        if SEABORN_OK:
            fig, ax = plt.subplots(figsize=(9, 4.8)); sns.barplot(data=top10, x="sku", y="revenue", ax=ax, color=ACCENT)
        else:
            fig, ax = plt.subplots(figsize=(9, 4.8)); ax.bar(top10["sku"], top10["revenue"], color=ACCENT)
        ax.set_xlabel(""); ax.set_ylabel("Revenue"); ax.set_title("Top 10 SKUs by Revenue")
        ax.xaxis.set_tick_params(rotation=30); ax.yaxis.set_major_formatter(FuncFormatter(thousands))
        for p in ax.patches:
            v = p.get_height()
            ax.annotate(f"{v:,.0f}", (p.get_x()+p.get_width()/2, v), ha="center", va="bottom", fontsize=9, xytext=(0,3), textcoords="offset points")
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("Weeks of Cover (Inventory Risk)")
    note("WOC tells you how many weeks your current stock would last at recent sales speed. Higher WOC = more stock sitting.")
    woc_clip = np.clip(metrics["weeks_of_cover"], None, 30)
    if PLOTLY_OK and style.startswith("Interactive"):
        fig = px.histogram(x=woc_clip, nbins=15, color_discrete_sequence=[ACCENT])
        fig.update_layout(template="simple_white", xaxis_title="WOC", yaxis_title="Count of SKUs",
                          margin=dict(l=20, r=20, t=40, b=20), font=dict(size=14))
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        if SEABORN_OK:
            fig, ax = plt.subplots(figsize=(9, 4.8)); sns.histplot(woc_clip, bins=15, ax=ax, color=ACCENT, edgecolor="#e5e7eb")
        else:
            fig, ax = plt.subplots(figsize=(9, 4.8)); ax.hist(woc_clip, bins=15, color=ACCENT)
        ax.set_title("Weeks of Cover (clipped at 30)"); ax.set_xlabel("WOC"); ax.set_ylabel("Count of SKUs")
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Details ---
with tab2:
    cA, cB = st.columns(2)
    with cA:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Reorder Candidates")
        note("These items are selling fast and could run out soon. Consider ordering more.")
        st.dataframe(metrics[metrics["action"]=="Reorder"][["sku","style_name","category","sell_through_pct","weeks_of_cover","revenue"]].head(30), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with cB:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Markdown Candidates")
        note("These items have more stock than their sales rate. A discount can help free up cash.")
        st.dataframe(metrics[metrics["action"]=="Mark down"][["sku","style_name","category","sell_through_pct","weeks_of_cover","suggested_markdown","revenue"]].head(30), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("Potential Phase-out")
    note("Low-demand items. Consider ending, replacing, or moving to clearance.")
    st.dataframe(metrics[metrics["action"]=="Phase out"][["sku","style_name","category","sell_through_pct","weeks_of_cover","revenue"]].head(30), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.download_button("Download metrics CSV", data=metrics.to_csv(index=False).encode("utf-8"), file_name="metrics.csv")

# --- Trends ---
with tab3:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("Trend Heatmap (Weighted Adoption)")
    note("Shows how well your catalog matches selected trends. Higher score = more items in that trend and/or higher importance.")
    adoption = []
    for t in trends["trend"].tolist():
        cnt = ((prod["color"].str.lower()==t.lower()) | (prod["material"].str.lower()==t.lower()) | (prod["silhouette"].str.lower()==t.lower())).sum()
        w = float(trends.loc[trends["trend"]==t, "weight"].iloc[0])
        adoption.append([t, int(cnt), w, cnt*w])
    ad_df = pd.DataFrame(adoption, columns=["trend","sku_count","weight","weighted"])
    if PLOTLY_OK and style.startswith("Interactive"):
        fig = px.bar(ad_df, x="trend", y="weighted", color_discrete_sequence=[ACCENT])
        fig.update_layout(template="simple_white", xaxis_title="", yaxis_title="Weighted score",
                          margin=dict(l=20, r=20, t=40, b=20), font=dict(size=14))
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        if SEABORN_OK:
            fig, ax = plt.subplots(figsize=(9, 4.8)); sns.barplot(data=ad_df, x="trend", y="weighted", ax=ax, color=ACCENT)
        else:
            fig, ax = plt.subplots(figsize=(9, 4.8)); ax.bar(ad_df["trend"], ad_df["weighted"], color=ACCENT)
        ax.set_title("Trend Heatmap (Weighted Adoption Score)"); ax.set_xlabel(""); ax.set_ylabel("Weighted score")
        ax.xaxis.set_tick_params(rotation=30); st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("Trend Score by SKU")
    note("Items ranked by how closely they match the chosen trends.")
    show = trend_scores.sort_values("trend_score", ascending=False)[["sku","style_name","category","trend_score"]].head(50)
    st.dataframe(show, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Sizes ---
# --- Sizes ---
with tab4:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("Size Demand (last 8 weeks)")
    note("Which sizes sold most in the last 8 weeks and how much stock you have in each size.")

    if size_summary is None:
        st.info("No 'size' data found — upload size-level data (or ensure Products include a Size option).")
    else:
        # Table
        st.dataframe(
            size_summary[["size","units_sold_8w","on_hand","sell_through_pct","weeks_of_cover"]],
            use_container_width=True
        )

        # Chart (Plotly or Seaborn)
        data = size_summary.sort_values("units_sold_8w", ascending=False)  # <-- fixed here

        if PLOTLY_OK and style.startswith("Interactive"):
            fig = px.bar(data, x="size", y="units_sold_8w", color_discrete_sequence=[ACCENT])
            fig.update_layout(
                template="simple_white",
                xaxis_title="Size",
                yaxis_title="Units sold (8w)",
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(size=14)
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)
        else:
            if SEABORN_OK:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                sns.barplot(data=data, x="size", y="units_sold_8w", ax=ax, color=ACCENT)
            else:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.bar(data["size"], data["units_sold_8w"], color=ACCENT)
            ax.set_xlabel("Size")
            ax.set_ylabel("Units sold (8w)")
            st.pyplot(fig)

        # Stockouts by SKU/size
        if oos_by_sku is not None and not oos_by_sku.empty:
            st.subheader("OOS by SKU & Size (had demand in last 8w)")
            note("Styles where a size had sales demand but zero stock — likely missed sales.")
            st.dataframe(oos_by_sku, use_container_width=True)
        else:
            st.caption("No size-level stockouts detected or no size-level inventory uploaded.")

    st.markdown('</div>', unsafe_allow_html=True)


# --- Lifecycle ---
with tab5:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("Newness & Lifecycle")
    note("Launch = new items (<8 weeks). Core = stable items. EOL = older items. Fix slow launches and reduce old stock risk.")
    lc = metrics.groupby("lifecycle")["sku"].count().reset_index().rename(columns={"sku":"count"}).sort_values("count", ascending=False)
    st.dataframe(lc, use_container_width=True)
    st.caption("Rules: Launch <8 weeks · Core 8–40 · EOL >40 (from launch_date)")

    slow_launch = metrics[(metrics["lifecycle"]=="Launch") & (metrics["sell_through_pct"]<0.35)].sort_values("sell_through_pct").head(20)
    aging = metrics[(metrics["lifecycle"]=="EOL") & (metrics["weeks_of_cover"]>=8)].sort_values("weeks_of_cover", ascending=False).head(20)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Slow Launches (need support)**")
        st.dataframe(slow_launch[["sku","style_name","category","sell_through_pct","weeks_since_launch"]], use_container_width=True)
    with c2:
        st.markdown("**Aging Inventory (high WOC)**")
        st.dataframe(aging[["sku","style_name","category","weeks_of_cover","sell_through_pct"]], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Markdown Simulator ---
with tab6:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("Markdown Simulator")
    note("Try a discount and see a quick estimate of units, sales and profit. This is a simple model for fast decisions.")
    default_pool = metrics[metrics["action"]=="Mark down"]["sku"].tolist()
    pool = st.multiselect("Select SKUs to simulate", options=metrics["sku"].tolist(), default=default_pool[:20])
    discount = st.slider("Discount %", 0, 50, 15, 1) / 100.0
    elasticity_per_10 = st.slider("Elasticity (units ↑ per +10% discount)", 0.0, 2.0, 0.8, 0.1)

    if pool:
        sim = metrics[metrics["sku"].isin(pool)].copy()
        sim["asp"] = np.where(sim["units_sold"]>0, sim["revenue"]/sim["units_sold"], sim.get("price", 0))
        uplift_mult = 1 + elasticity_per_10 * (discount / 0.10)
        sim["units_new"] = sim["units_sold"] * uplift_mult
        sim["price_new"] = sim["asp"] * (1 - discount)
        sim["rev_new"]   = sim["units_new"] * sim["price_new"]
        sim["profit_new"]= sim["units_new"] * (sim["price_new"] - sim.get("cost", 0).fillna(0))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected SKUs", f"{len(sim):,}")
        c2.metric("Δ Revenue", f"{sim['rev_new'].sum() - sim['revenue'].sum():,.0f}")
        base_profit = (sim["revenue"] - sim.get("cost", 0).fillna(0)*sim["units_sold"]).sum()
        c3.metric("Δ Profit",  f"{sim['profit_new'].sum() - base_profit:,.0f}")
        c4.metric("Units uplift (x)", f"{uplift_mult:.2f}")

        show = sim[["sku","style_name","category","asp","units_sold","revenue","price_new","units_new","rev_new","profit_new"]].copy()
        st.dataframe(show.sort_values("rev_new", ascending=False), use_container_width=True)
        st.download_button("Download simulation CSV", data=show.to_csv(index=False).encode("utf-8"), file_name="markdown_simulation.csv")
    else:
        st.info("Pick one or more SKUs to simulate.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Copilot (GPT-5) ---
# -------------------- Copilot (GPT-5 Responses + fallback to Chat) --------------------
with tab7:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("Data Copilot (Ask a question)")
    note("Ask in plain words, e.g., ‘best performing SKU’, ‘which sizes are selling?’, or ‘what should we reorder?’")

    # Controls (model fixed to GPT-5 Responses)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.text_input("Model", "gpt-5 (Responses API)", disabled=True, help="Primary path uses the Responses API.")
    with col2:
        max_tokens = st.slider("Max tokens", 100, 800, 300, 50, help="Lower = faster (for both paths).")

    local_only = st.checkbox(
        "Local only (no API)",
        value=not bool(os.getenv("OPENAI_API_KEY", "").strip()),
        help="Use the built-in answers only. Turn off to call OpenAI."
    )

    # --- History ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi! Ask me about your top sellers, inventory risk, or which SKUs to reorder or mark down."}
        ]
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # --- Input ---
    prompt = st.chat_input("e.g., Best performing SKU; size gaps; slow launches; Top 3 by profit in knitwear")

    # --- Local heuristic fallback (fast, no API) ---
    def local_answer(q: str) -> str:
        import re
        ql = q.lower()

        def pct(x):
            try:
                return f"{float(x) * 100:.1f}%"
            except Exception:
                return str(x)

        def num(x):
            try:
                xv = float(x)
                return f"{xv:,.0f}" if abs(xv) >= 1000 else f"{xv:,.2f}"
            except Exception:
                return str(x)

        n = 5
        mN = re.search(r"\btop\s*(\d+)\b", ql) or re.search(r"\b(\d+)\s*(best|top)\b", ql)
        if mN:
            n = int(mN.group(1))
        if any(w in ql for w in ["best", "best-performing", "best performing"]) and not mN:
            n = 1

        m_cat = re.search(r"\b(in|for|within)\s+([A-Za-z \-/&]+)", q, flags=re.I)
        cat = (m_cat.group(2).strip() if m_cat else None)
        df = metrics if not cat else metrics[metrics["category"].str.contains(cat, case=False, na=False)]

        m_sku = re.search(r"(sku[\w-]+)", ql)
        if m_sku:
            sku = m_sku.group(1).upper()
            row = metrics[metrics["sku"] == sku]
            if row.empty:
                return f"I can’t find {sku}."
            r = row.iloc[0]
            return (
                f"{sku} — {r['style_name']} ({r['category']})\n"
                f"- Revenue: {num(r['revenue'])}\n"
                f"- Units sold: {num(r['units_sold'])}\n"
                f"- Sell-through: {pct(r['sell_through_pct'])}\n"
                f"- Weeks of cover: {r['weeks_of_cover']:.1f}\n"
                f"- Gross profit (approx): {num(r['gross_profit'])}\n"
                f"- Performance score: {r['performance_score']:.2f}\n"
                f"- Lifecycle: {r['lifecycle']} ({num(r['weeks_since_launch'])} weeks since launch)"
            )

        if any(k in ql for k in ["reorder", "restock", "buy more"]):
            d = df[df["action"] == "Reorder"].nlargest(n, "revenue")
            if d.empty:
                return "No SKUs meet the current reorder rule."
            return "Reorder candidates:\n" + "\n".join(
                f"- {r.sku} {r.style_name} — ST {pct(r.sell_through_pct)}, WOC {r.weeks_of_cover:.1f}, Rev {num(r.revenue)}"
                for _, r in d.iterrows()
            )

        if ("mark" in ql and "down" in ql) or "markdown" in ql or "clearance" in ql:
            d = df[df["action"] == "Mark down"].nlargest(n, "weeks_of_cover")
            if d.empty:
                return "No markdown candidates under current thresholds."
            return "Markdown candidates:\n" + "\n".join(
                f"- {r.sku} {r.style_name} — WOC {r.weeks_of_cover:.1f}, Suggest {pct(r.suggested_markdown)}, Rev {num(r.revenue)}"
                for _, r in d.iterrows()
            )

        if "phase" in ql or "retire" in ql:
            d = df[df["action"] == "Phase out"].nsmallest(n, "sell_through_pct")
            if d.empty:
                return "No clear phase-out items right now."
            return "Potential phase-out:\n" + "\n".join(
                f"- {r.sku} {r.style_name} — ST {pct(r.sell_through_pct)}, WOC {r.weeks_of_cover:.1f}, Rev {num(r.revenue)}"
                for _, r in d.iterrows()
            )

        if "trend" in ql:
            t = trend_scores.sort_values("trend_score", ascending=False).head(n)[["sku", "style_name", "trend_score"]]
            return "Highest trend-score SKUs:\n" + "\n".join(
                [f"- {r.sku} {r.style_name} — score {r.trend_score:.2f}" for _, r in t.iterrows()]
            )

        if "size" in ql and size_summary is not None:
            top_sizes = size_summary.sort_values("units_sold_8w", ascending=False).head(n)
            return "Top sizes (8w):\n" + "\n".join([f"- {r.size}: {int(r.units_sold_8w)} units" for _, r in top_sizes.iterrows()])

        if "launch" in ql or "new" in ql or "lifecycle" in ql:
            d = metrics.sort_values("weeks_since_launch").head(n)
            return "Newest SKUs:\n" + "\n".join([f"- {r.sku} {r.style_name} — {r.weeks_since_launch:.1f} weeks" for _, r in d.iterrows()])

        by_metric = "revenue"
        if any(k in ql for k in ["sell-through", "sell through"]):
            by_metric = "sell_through_pct"
        elif any(k in ql for k in ["profit", "margin"]):
            by_metric = "gross_profit"
        elif any(k in ql for k in ["units", "volume"]):
            by_metric = "units_sold"
        elif any(k in ql for k in ["best performing", "performance", "top performer"]):
            by_metric = "performance_score"

        d = df.sort_values(by_metric, ascending=False).head(n)
        title = {
            "revenue": "Top by revenue",
            "gross_profit": "Top by gross profit (approx)",
            "units_sold": "Top by units sold",
            "sell_through_pct": "Top by sell-through",
            "performance_score": "Top performers (composite)",
        }[by_metric]
        return f"{title} — top {len(d)}:\n" + "\n".join(
            f"- {r.sku} {r.style_name} ({r.category}) — Rev {num(r.revenue)}, Profit {num(r.gross_profit)}, "
            f"ST {r.sell_through_pct*100:.1f}%, WOC {r.weeks_of_cover:.1f}, Score {r.performance_score:.2f}"
            for _, r in d.iterrows()
        )

    # --- On submit: echo user, then answer ---
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        answer = None
        error_text = None
        api_key_env = get_secret("OPENAI_API_KEY")


        if (not local_only) and api_key_env:
            try:
                from openai import OpenAI  # type: ignore
                client = OpenAI(api_key=api_key_env, timeout=20.0, max_retries=1)

                # Small context for speed
                ctx = metrics.sort_values("revenue", ascending=False).head(10)[
                    ["sku", "style_name", "category", "revenue", "sell_through_pct", "weeks_of_cover",
                     "gross_profit", "performance_score", "lifecycle"]
                ].to_dict(orient="records")

                # 1) Try GPT-5 via Responses API (uses max_completion_tokens)
                try:
                    r = client.responses.create(
                        model="gpt-5",
                        input=[
                            {
                                "role": "system",
                                "content": "You are NordicAI's retail data copilot. Reply with short bullet points and concrete numbers using ONLY the provided context."
                            },
                            {
                                "role": "user",
                                "content": f"Question: {prompt}\n\nContext (top 10 rows): {ctx}"
                            }
                        ],
                        temperature=0.2,
                        max_completion_tokens=int(max_tokens)
                    )
                    # Extract text from Responses result
                    txt = getattr(r, "output_text", None)
                    if not txt and getattr(r, "output", None):
                        parts = []
                        for msg in r.output:
                            for c in getattr(msg, "content", []):
                                if getattr(c, "type", "") in ("output_text", "text", "input_text"):
                                    t = getattr(c, "text", "")
                                    if t:
                                        parts.append(t)
                        txt = "\n".join(parts).strip()
                    if txt:
                        with st.chat_message("assistant"):
                            st.write(txt)
                        answer = txt

                except Exception as e_resp:
                    # 2) Fallback: Chat Completions with gpt-4o-mini (uses max_tokens)
                    error_text = f"{e_resp}"
                    try:
                        r2 = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are NordicAI's retail data copilot. Reply with short bullet points and concrete numbers using ONLY the provided context."},
                                {"role": "user", "content": f"Question: {prompt}\n\nContext (top 10 rows): {ctx}"}
                            ],
                            temperature=0.2,
                            max_tokens=int(max_tokens)
                        )
                        txt2 = r2.choices[0].message.content
                        with st.chat_message("assistant"):
                            st.write(txt2)
                        answer = txt2
                        error_text = f"Responses API failed; used Chat instead: {error_text}"
                    except Exception as e_chat:
                        error_text = f"Responses failed: {error_text} | Chat failed: {e_chat}"

            except Exception as e_init:
                error_text = f"OpenAI client not available: {e_init}"
        elif not api_key_env and not local_only:
            error_text = "OPENAI_API_KEY not set."

        # Local fallback if API not used/failed
        if answer is None:
            if error_text:
                with st.expander("LLM error details", expanded=False):
                    st.error(error_text)
            with st.chat_message("assistant"):
                with st.spinner("Working..."):
                    answer = local_answer(prompt)
                    st.write(answer)

        st.session_state["messages"].append({"role": "assistant", "content": answer})

    st.markdown('</div>', unsafe_allow_html=True)


# -------- One-Pager (Markdown + PDF) ----------
summary_lines = [
    "# NordicAI — Brand Pulse",
    f"**Period:** {min_date.date()} → {max_date.date()}  \n**Weeks covered:** {weeks_covered:.1f}",
    "",
    "## Highlights",
    f"- Revenue: **{metrics['revenue'].sum():,.0f}**, Units: **{metrics['units_sold'].sum():,.0f}**, On hand: **{metrics['on_hand'].sum():,.0f}**",
    f"- Avg sell-through: **{metrics['sell_through_pct'].mean()*100:.1f}%**, Avg WOC: **{metrics['weeks_of_cover'].mean():.1f}**",
    "",
    "## Plain-language summary",
]
top_perf = metrics.sort_values("performance_score", ascending=False).head(3)[["sku","style_name","revenue"]]
summary_lines.append("- **Winners:** " + ", ".join([f"{r.sku} ({r.style_name})" for _, r in top_perf.iterrows()]) + " — strong sales now.")
high_woc = metrics.sort_values("weeks_of_cover", ascending=False).head(3)[["sku","style_name","weeks_of_cover"]]
summary_lines.append("- **High stock risk (WOC):** " + ", ".join([f"{r.sku} ({r.style_name}) ~{r.weeks_of_cover:.1f} weeks" for _, r in high_woc.iterrows()]) + " — consider markdown.")
reorder = metrics[metrics["action"]=="Reorder"].nlargest(3, "revenue")[["sku","style_name","sell_through_pct","weeks_of_cover"]]
if not reorder.empty:
    summary_lines.append("- **Reorder now:** " + ", ".join([f"{r.sku} ({r.style_name})" for _, r in reorder.iterrows()]) + " — selling fast; risk of stock-out.")
if 'size' in metrics.columns or 'size' in prod.columns:
    summary_lines.append("- **Sizes:** Focus on top sizes by demand; fix any sizes with demand but zero stock (see Sizes section).")
summary_lines.append("- **Newness:** Support slow launches; reduce aged stock with high WOC.")
summary_lines += ["", "## Top Winners"]
for _, r in metrics.sort_values("performance_score", ascending=False).head(5).iterrows():
    summary_lines.append(f"- {r['sku']} {r['style_name']} — Rev {r['revenue']:,.0f}, ST {r['sell_through_pct']*100:.1f}%, WOC {r['weeks_of_cover']:.1f}, Score {r['performance_score']:.2f}")
summary_lines += ["", "## Action Shortlists"]

def list_block(df, title, n=5):
    lines = [f"### {title}"]
    if df.empty:
        lines.append("_None under current rules._")
    else:
        for _, r in df.head(n).iterrows():
            lines.append(f"- {r['sku']} {r['style_name']} — ST {r['sell_through_pct']*100:.1f}%, WOC {r['weeks_of_cover']:.1f}")
    return lines

summary_lines += list_block(metrics[metrics["action"]=="Reorder"].sort_values("revenue", ascending=False), "Reorder")
summary_lines += list_block(metrics[metrics["action"]=="Mark down"].sort_values("weeks_of_cover", ascending=False), "Markdown")
summary_lines += list_block(metrics[metrics["action"]=="Phase out"].sort_values("sell_through_pct"), "Phase-out")
summary_lines += [
    "",
    "## Glossary (simple terms)",
    "- **SKU**: The product code for one item (often one size or color).",
    "- **Revenue**: Money earned from sales (before costs).",
    "- **Units sold**: How many pieces were sold.",
    "- **On hand**: How many pieces you have in stock now.",
    "- **Sell-through**: Percent sold = sold / (sold + in stock). Higher = selling better.",
    "- **Weeks of Cover (WOC)**: How many weeks current stock would last at recent sales speed.",
    "- **ABC**: A = top sellers (~80% of revenue), B = mid, C = low.",
    "- **Suggested markdown**: The price reduction we suggest (e.g., 0.10 = 10% off).",
    "- **Reorder candidate**: Selling fast and may run out soon → order more.",
    "- **Markdown candidate**: Too much stock for current sales → discount to move.",
    "- **Phase-out**: Low demand item → consider ending or replacing.",
    "- **Trend score**: How closely an item matches selected trends.",
    "- **Lifecycle**: Launch (<8 weeks), Core (8–40), End-of-life (>40) since launch date.",
    "- **Gross profit**: Revenue minus estimated product cost.",
    "- **Performance score**: Combined score (revenue + sell-through + low WOC). Higher = better.",
]
onepager_md = "\n".join(summary_lines)
st.download_button("Download one-pager (Markdown)", data=onepager_md.encode("utf-8"), file_name="brand_pulse.md")

def build_pdf_bytes():
    if not REPORTLAB_OK:
        return None
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1Blue", parent=styles["Heading1"], textColor=colors.HexColor(PRIMARY)))
    styles.add(ParagraphStyle(name="H2Blue", parent=styles["Heading2"], textColor=colors.HexColor(PRIMARY)))
    body = []
    body.append(Paragraph("NordicAI — Brand Pulse", styles["H1Blue"]))
    body.append(Paragraph(f"Period: {min_date.date()} → {max_date.date()}  |  Weeks covered: {weeks_covered:.1f}", styles["Normal"]))
    body.append(Spacer(1, 10))
    hi = [
        ["Revenue", f"{metrics['revenue'].sum():,.0f}"],
        ["Units", f"{metrics['units_sold'].sum():,.0f}"],
        ["On hand", f"{metrics['on_hand'].sum():,.0f}"],
        ["Avg sell-through", f"{metrics['sell_through_pct'].mean()*100:.1f}%"],
        ["Avg WOC", f"{metrics['weeks_of_cover'].mean():.1f}"],
    ]
    t = Table(hi, colWidths=[140, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.whitesmoke),
        ('TEXTCOLOR',(0,0),(-1,-1),colors.black),
        ('GRID',(0,0),(-1,-1),0.25,colors.lightgrey),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica'),
        ('ALIGN',(1,0),(1,-1),'RIGHT'),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.whitesmoke, colors.Color(1,1,1)])
    ]))
    body.append(t); body.append(Spacer(1, 10))

    body.append(Paragraph("Plain-language summary", styles["H2Blue"]))
    top3 = metrics.sort_values("performance_score", ascending=False).head(3)[["sku","style_name"]]
    body.append(Paragraph("• Winners: " + ", ".join([f"{r.sku} ({r.style_name})" for _, r in top3.iterrows()]) + " — strong sales now.", styles["Normal"]))
    hw = metrics.sort_values("weeks_of_cover", ascending=False).head(3)[["sku","style_name","weeks_of_cover"]]
    body.append(Paragraph("• High stock risk (WOC): " + ", ".join([f"{r.sku} ({r.style_name}) ~{r.weeks_of_cover:.1f} weeks" for _, r in hw.iterrows()]) + " — consider markdown.", styles["Normal"]))
    ro = metrics[metrics["action"]=="Reorder"].nlargest(3, "revenue")[["sku","style_name"]]
    if not ro.empty:
        body.append(Paragraph("• Reorder now: " + ", ".join([f"{r.sku} ({r.style_name})" for _, r in ro.iterrows()]) + " — selling fast; risk of stock-out.", styles["Normal"]))
    body.append(Paragraph("• Sizes: Focus on top sizes by demand; fix sizes with demand but zero stock.", styles["Normal"]))
    body.append(Paragraph("• Newness: Support slow launches; reduce aged stock with high WOC.", styles["Normal"]))
    body.append(Spacer(1, 8))

    body.append(Paragraph("Top Winners", styles["H2Blue"]))
    top5 = metrics.sort_values("performance_score", ascending=False).head(5)[["sku","style_name","revenue","sell_through_pct","weeks_of_cover","performance_score"]]
    rows = [["SKU","Style","Revenue","ST","WOC","Score"]] + [
        [r.sku, r.style_name, f"{r.revenue:,.0f}", f"{r.sell_through_pct*100:.1f}%", f"{r.weeks_of_cover:.1f}", f"{r.performance_score:.2f}"]
        for _, r in top5.iterrows()
    ]
    t2 = Table(rows, colWidths=[70,180,80,60,50,50])
    t2.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.lightgrey),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold')]))
    body.append(t2); body.append(Spacer(1, 10))

    def table_for(df, title, n=5):
        body.append(Paragraph(title, styles["H2Blue"]))
        if df.empty:
            body.append(Paragraph("None under current rules.", styles["Normal"]))
            return
        d = df.head(n)[["sku","style_name","sell_through_pct","weeks_of_cover"]]
        rows = [["SKU","Style","ST","WOC"]] + [
            [r.sku, r.style_name, f"{r.sell_through_pct*100:.1f}%", f"{r.weeks_of_cover:.1f}"] for _, r in d.iterrows()
        ]
        tx = Table(rows, colWidths=[70,220,60,60])
        tx.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.lightgrey),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold')]))
        body.append(tx); body.append(Spacer(1, 8))

    table_for(metrics[metrics["action"]=="Reorder"].sort_values("revenue", ascending=False), "Reorder")
    table_for(metrics[metrics["action"]=="Mark down"].sort_values("weeks_of_cover", ascending=False), "Markdown")
    table_for(metrics[metrics["action"]=="Phase out"].sort_values("sell_through_pct"), "Phase-out")

    body.append(Spacer(1, 10))
    body.append(Paragraph("Glossary (simple terms)", styles["H2Blue"]))
    for k in ["SKU","Revenue","Units sold","On hand","Sell-through","Weeks of Cover (WOC)","ABC",
              "Suggested markdown","Reorder candidate","Markdown candidate","Phase-out",
              "Trend score","Lifecycle","Gross profit","Performance score"]:
        body.append(Paragraph(f"• {k}: {GLOSSARY[k]}", styles["Normal"]))

    doc.build(body)
    buf.seek(0)
    return buf.getvalue()

if REPORTLAB_OK:
    pdf_bytes = build_pdf_bytes()
    st.download_button("Download one-pager (PDF)", data=pdf_bytes, file_name="brand_pulse.pdf", mime="application/pdf")
else:
    st.caption("PDF export available if 'reportlab' is installed.")

st.caption("Built with Streamlit · Shopify import + Plain-language mode + One-click PDF · Seaborn/Plotly charts · Sizes/Lifecycle/Simulator · GPT-5 Copilot with streaming · Set OPENAI_API_KEY to enable.")
