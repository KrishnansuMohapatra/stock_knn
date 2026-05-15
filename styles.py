"""
Custom CSS styles for the Streamlit dashboard.
"""

CUSTOM_CSS = """
<style>
    .main { background-color: #0e1117; }

    .signal-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .buy-card  { background: linear-gradient(135deg,#1a472a,#2d6a4f); border:2px solid #52b788; color:#b7e4c7; }
    .sell-card { background: linear-gradient(135deg,#6b0f1a,#b91c1c); border:2px solid #f87171; color:#fecaca; }
    .hold-card { background: linear-gradient(135deg,#78350f,#b45309); border:2px solid #fbbf24; color:#fde68a; }

    .report-box {
        background: #1e2130;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 24px;
        margin-top: 10px;
        line-height: 1.8;
    }
    .score-bar {
        height: 12px;
        border-radius: 6px;
        margin: 6px 0 12px 0;
    }
    .indicator-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #2d3748;
    }
    .bullish  { color: #52b788; font-weight: 600; }
    .bearish  { color: #f87171; font-weight: 600; }
    .neutral  { color: #fbbf24; font-weight: 600; }
    .metric-label { color: #9ca3af; font-size: 13px; }
    .metric-value { color: #f3f4f6; font-size: 15px; font-weight: 600; }

    div[data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 14px;
    }
</style>
"""
