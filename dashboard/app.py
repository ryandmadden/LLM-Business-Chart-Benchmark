"""
Main Streamlit dashboard application.
"""
import streamlit as st
import os
import sys
import warnings

# Suppress Plotly deprecation warnings
warnings.filterwarnings("ignore", message="The keyword arguments have been deprecated and will be removed in a future release. Use `config` instead to specify Plotly configuration options.")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="plotly")
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure page
st.set_page_config(
    page_title="LLM Business Chart Benchmark",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-highlight {
        background-color: #e8f5e8;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #4caf50;
    }
    /* Hide the automatic page navigation */
    [data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">LLM Business Chart Benchmark</h1>', 
            unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Overview", "Deep Dive", "Insights"]
)

# Add refresh button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data", help="Clear cache and reload latest results from S3"):
    st.cache_data.clear()
    st.rerun()

# Page routing
if page == "Overview":
    from pages.overview import show_overview_page
    show_overview_page()
elif page == "Deep Dive":
    from pages.deep_dive import show_deep_dive_page
    show_deep_dive_page()
elif page == "Insights":
    from pages.insights import show_insights_page
    show_insights_page()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**LLM Business Chart Benchmark**")
st.sidebar.markdown("Evaluating vision models on chart interpretation tasks")
st.sidebar.markdown("Built with Streamlit")
