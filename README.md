
# NordicAI Fashion Starter — Python Edition

## Quick start (CLI)
```bash
python run_pipeline.py \
  --orders data/sample_orders.csv \
  --inventory data/sample_inventory.csv \
  --products data/product_master.csv \
  --trends data/trend_weights.csv \
  --outdir out \
  --brand "Sample Brand"
```

Outputs in `out/`:
- `metrics.csv`, `size_curve.csv`, `trend_scores.csv`
- `reorder_list.csv`, `markdown_list.csv`, `phaseout_list.csv`
- Charts in `out/charts/`
- `report.html` (open in a browser)

## Streamlit (interactive)
```bash
streamlit run app_streamlit.py
```
Upload your CSVs or use the bundled sample data by doing nothing.

## Data schema
- Orders: `order_date, order_id, sku, size, units, unit_price, channel, discount_rate`
- Inventory: `sku, size, on_hand`
- Products: `sku, style_name, category, color, material, silhouette, price, cost, launch_date`
- Trends: `trend, weight`

## What it calculates
- Sell-through %, Avg weekly units, Weeks of cover
- ABC by cumulative revenue
- Suggested markdown ladder + action labels
- Size curve per SKU
- Trend score via flags × weights

— NordicAI
