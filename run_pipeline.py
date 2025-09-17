#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime as dt

def run_pipeline(orders_path, inventory_path, products_path, trends_path, outdir, brand="Sample Brand"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    orders = pd.read_csv(orders_path, parse_dates=["order_date"])
    inv = pd.read_csv(inventory_path)
    prod = pd.read_csv(products_path, parse_dates=["launch_date"])
    trends = pd.read_csv(trends_path)
    orders["revenue"] = orders["units"] * orders["unit_price"] * (1 - orders["discount_rate"])
    sku_sales = orders.groupby("sku").agg(units_sold=("units","sum"), revenue=("revenue","sum")).reset_index()
    sku_onhand = inv.groupby("sku")["on_hand"].sum().reset_index()
    metrics = sku_sales.merge(sku_onhand, on="sku", how="outer").fillna({"units_sold":0,"revenue":0,"on_hand":0})
    metrics = metrics.merge(prod[["sku","category","style_name","price","cost"]], on="sku", how="left")
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
    def abc(p):
        if p <= 0.80: return "A"
        if p <= 0.95: return "B"
        return "C"
    metrics["ABC"] = metrics["cum_pct"].apply(abc)
    def markdown_from_woc(woc):
        if woc >= 12: return 0.25
        if woc >= 8:  return 0.15
        if woc >= 6:  return 0.10
        return 0.00
    metrics["suggested_markdown"] = metrics["weeks_of_cover"].apply(markdown_from_woc)
    def action_row(row):
        if (row["sell_through_pct"] > 0.70) and (row["weeks_of_cover"] < 4):
            return "Reorder"
        if row["weeks_of_cover"] >= 8:
            return "Mark down"
        if row["sell_through_pct"] < 0.40:
            return "Phase out"
        return "Monitor"
    metrics["action"] = metrics.apply(action_row, axis=1)
    size_curve = orders.pivot_table(index="sku", columns="size", values="units", aggfunc="sum", fill_value=0).reset_index()
    size_order = ["XS","S","M","L","XL"]
    size_cols = [c for c in size_order if c in size_curve.columns]
    size_curve = size_curve[["sku"] + size_cols]
    trends_list = trends["trend"].tolist()
    flags = pd.DataFrame({"sku": prod["sku"]})
    for t in trends_list:
        flags[t] = prod.apply(lambda r: 1 if (str(r["color"]).lower()==t.lower() or 
                                              str(r["material"]).lower()==t.lower() or 
                                              str(r["silhouette"]).lower()==t.lower()) else 0, axis=1)
    weight_map = dict(zip(trends["trend"], trends["weight"]))
    score = np.zeros(len(flags))
    for t in trends_list:
        score += flags[t].values * weight_map.get(t, 0.0)
    flags["trend_score"] = score
    trend_scores = flags.merge(prod[["sku","style_name","category","color","material","silhouette"]], on="sku", how="left")
    recs = metrics.merge(trend_scores[["sku","trend_score"]], on="sku", how="left").fillna({"trend_score":0.0})
    reorder_list = recs[(recs["action"]=="Reorder")].nlargest(10, "revenue")[["sku","style_name","category","sell_through_pct","weeks_of_cover","trend_score","revenue"]]
    markdown_list = recs[(recs["action"]=="Mark down")].nlargest(10, "weeks_of_cover")[["sku","style_name","category","sell_through_pct","weeks_of_cover","suggested_markdown","revenue"]]
    phaseout_list = recs[(recs["action"]=="Phase out")].nsmallest(10, "sell_through_pct")[["sku","style_name","category","sell_through_pct","weeks_of_cover","revenue"]]
    charts_dir = outdir / "charts"
    charts_dir.mkdir(exist_ok=True)
    top10 = metrics.sort_values("revenue", ascending=False).head(10)
    plt.figure(figsize=(8,4.5)); plt.bar(top10["sku"], top10["revenue"]); plt.xticks(rotation=30, ha="right"); plt.title("Top 10 SKUs by Revenue"); plt.tight_layout(); plt.savefig((charts_dir / "top10_revenue.png").as_posix(), dpi=200); plt.close()
    plt.figure(figsize=(8,4.5)); import numpy as np as _np; plt.hist(_np.clip(metrics["weeks_of_cover"], None, 30), bins=15); plt.title("Weeks of Cover (clipped at 30)"); plt.xlabel("WOC"); plt.ylabel("Count of SKUs"); plt.tight_layout(); plt.savefig((charts_dir / "woc_hist.png").as_posix(), dpi=200); plt.close()
    overall_size = orders.groupby("size")["units"].sum().reindex(size_cols).fillna(0)
    plt.figure(figsize=(8,1.6)); plt.imshow([overall_size.values], aspect="auto"); plt.yticks([]); plt.xticks(range(len(overall_size.index)), list(overall_size.index)); plt.title("Overall Size Curve (Units Sold)"); plt.tight_layout(); plt.savefig((charts_dir / "size_curve_heatmap.png").as_posix(), dpi=200); plt.close()
    adoption = []
    for t in trends_list:
        cnt = ((prod["color"].str.lower()==t.lower()) | (prod["material"].str.lower()==t.lower()) | (prod["silhouette"].str.lower()==t.lower())).sum()
        w = weight_map.get(t, 0.0); adoption.append([t, int(cnt), float(w), cnt*w])
    import pandas as _pd; _ad = _pd.DataFrame(adoption, columns=["trend","sku_count","weight","weighted"])
    plt.figure(figsize=(8,4.5)); plt.bar(_ad["trend"], _ad["weighted"]); plt.xticks(rotation=30, ha="right"); plt.title("Trend Heatmap (Weighted Adoption Score)"); plt.tight_layout(); plt.savefig((charts_dir / "trend_heatmap.png").as_posix(), dpi=200); plt.close()
    metrics.to_csv(outdir / "metrics.csv", index=False)
    size_curve.to_csv(outdir / "size_curve.csv", index=False)
    trend_scores.to_csv(outdir / "trend_scores.csv", index=False)
    reorder_list.to_csv(outdir / "reorder_list.csv", index=False)
    markdown_list.to_csv(outdir / "markdown_list.csv", index=False)
    phaseout_list.to_csv(outdir / "phaseout_list.csv", index=False)
    report = outdir / "report.html"
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"<html><head><meta charset='utf-8'><title>{brand} — NordicAI Report</title></head><body>")
        f.write(f"<h1>{brand} — NordicAI Mini-Report</h1>")
        f.write('<img src="charts/top10_revenue.png" style="max-width:100%"/>')
        f.write('<img src="charts/size_curve_heatmap.png" style="max-width:100%"/>')
        f.write('<img src="charts/woc_hist.png" style="max-width:100%"/>')
        f.write('<img src="charts/trend_heatmap.png" style="max-width:100%"/>')
        f.write("<h2>Reorder Candidates</h2>"); f.write(reorder_list.to_html(index=False))
        f.write("<h2>Markdown Candidates</h2>"); f.write(markdown_list.to_html(index=False))
        f.write("<h2>Potential Phase-out</h2>"); f.write(phaseout_list.to_html(index=False))
        f.write("</body></html>")
    print("Done. Outputs at", outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--products", required=True)
    ap.add_argument("--trends", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--brand", default="Sample Brand")
    args = ap.parse_args()
    run_pipeline(args.orders, args.inventory, args.products, args.trends, args.outdir, args.brand)
