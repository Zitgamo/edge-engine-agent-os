"""Custom CSS for the dashboard."""

CUSTOM_CSS = """
<style>
    /* === Global === */
    .stApp {
        background: #0E1117;
    }
    .main-header {
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00C853, #00E676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    .main-header .subtitle {
        color: #888;
        font-size: 0.9rem;
        margin-top: -0.3rem;
    }
    
    /* === Signal Cards === */
    .signal-grid {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .signal-card {
        flex: 1;
        background: linear-gradient(135deg, #1A1D29 0%, #222638 100%);
        border: 1px solid #2A2E3E;
        border-radius: 16px;
        padding: 1.5rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s, border-color 0.2s;
    }
    .signal-card:hover {
        transform: translateY(-2px);
        border-color: #00C853;
    }
    .signal-card .rank-badge {
        position: absolute;
        top: 0.75rem;
        right: 0.75rem;
        background: #00C853;
        color: #000;
        font-weight: 700;
        font-size: 0.7rem;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
    }
    .signal-card .ticker {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFF;
        margin-bottom: 0.3rem;
    }
    .signal-card .score {
        font-size: 1rem;
        color: #00C853;
        font-weight: 600;
    }
    .signal-card .meta {
        display: flex;
        gap: 1rem;
        margin-top: 0.8rem;
        font-size: 0.8rem;
        color: #888;
    }
    .signal-card .meta span {
        background: #0E1117;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
    }
    .signal-card .meta .sl { color: #FF5252; }
    .signal-card .meta .tp { color: #00C853; }
    
    /* === KPI Cards === */
    .kpi-row {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .kpi-card {
        flex: 1;
        background: #1A1D29;
        border: 1px solid #2A2E3E;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .kpi-card .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FFF;
    }
    .kpi-card .kpi-label {
        font-size: 0.75rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.2rem;
    }
    .kpi-card .kpi-green { color: #00C853; }
    .kpi-card .kpi-red { color: #FF5252; }
    .kpi-card .kpi-blue { color: #448AFF; }

    /* === Section Headers === */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #CCC;
        margin: 2rem 0 0.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2A2E3E;
    }

    /* === Data Tables === */
    .dataframe {
        font-size: 0.85rem;
    }
    .dataframe thead tr th {
        background: #1A1D29 !important;
        color: #888 !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px;
    }
    .dataframe tbody tr {
        background: #0E1117 !important;
    }
    .dataframe tbody tr:nth-child(even) {
        background: #151824 !important;
    }

    /* === Hide Streamlit branding === */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* === Sidebar === */
    section[data-testid="stSidebar"] {
        background: #0E1117;
        border-right: 1px solid #1A1D29;
    }
    section[data-testid="stSidebar"] .sidebar-content {
        padding-top: 1rem;
    }

    /* === Status Badge === */
    .badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .badge-win { background: #00C85322; color: #00C853; border: 1px solid #00C85344; }
    .badge-loss { background: #FF525222; color: #FF5252; border: 1px solid #FF525244; }
    .badge-pending { background: #FFA72622; color: #FFA726; border: 1px solid #FFA72644; }

    /* === Live dot === */
    .live-dot {
        display: inline-block;
        width: 8px; height: 8px;
        background: #00C853;
        border-radius: 50%;
        animation: pulse 2s infinite;
        margin-right: 6px;
    }
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
        100% { opacity: 1; transform: scale(1); }
    }
</style>
"""
