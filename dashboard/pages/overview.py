"""
Overview page showing model comparison and summary statistics.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import (
    load_evaluation_results, get_model_summary_stats, 
    create_performance_comparison_chart, create_cost_performance_scatter,
    format_model_names_in_df, get_full_model_name
)


def show_overview_page():
    """Display the overview page."""
    
    # Load data
    df = load_evaluation_results()
    
    if df is None:
        st.error("No evaluation results found. Please run the evaluation first.")
        st.info("Run: `python scripts/run_evaluation.py`")
        return

    # Show data source info
    total_evaluations = len(df)
    unique_charts = df['chart_id'].nunique() if 'chart_id' in df.columns else 'N/A'
    st.success(f"✅ Loaded {total_evaluations} evaluation results from {unique_charts} charts")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    model_stats = get_model_summary_stats(df)
    model_stats = format_model_names_in_df(model_stats)
    
    # Display summary table
    st.dataframe(
        model_stats.round(3),
        width='stretch',
        hide_index=True
    )
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_model = model_stats.loc[model_stats['Accuracy'].idxmax()]
        st.metric(
            "Best Model",
            best_model['Model'],
            f"{best_model['Accuracy']:.1%} accuracy"
        )
    
    with col2:
        total_cost = df['cost'].sum()
        st.metric(
            "Total Cost",
            f"${total_cost:.2f}",
            f"${df['cost'].mean():.4f} avg per question"
        )
    
    with col3:
        avg_latency = df['latency_seconds'].mean()
        st.metric(
            "Avg Latency",
            f"{avg_latency:.2f}s",
            f"{df['latency_seconds'].median():.2f}s median"
        )
    
    with col4:
        total_questions = len(df)
        correct_answers = (df['score'] >= 0.5).sum()
        st.metric(
            "Overall Accuracy",
            f"{correct_answers/total_questions:.1%}",
            f"{correct_answers}/{total_questions} correct"
        )
    
    
    fig_cost = create_cost_performance_scatter(df)
    st.plotly_chart(fig_cost, use_container_width=True)
    
    
    
    # Key insights
    st.subheader("💡 Key Insights")
    
    insights = []
    
    # Best performing model
    best_model = model_stats.loc[model_stats['Accuracy'].idxmax()]
    insights.append(f"**{best_model['Model']}** achieves the highest overall accuracy ({best_model['Accuracy']:.1%})")
    
    # Most cost-effective
    cost_effective = model_stats.loc[model_stats['Avg Cost'].idxmin()]
    insights.append(f"**{cost_effective['Model']}** is the most cost-effective (${cost_effective['Avg Cost']:.4f} per question)")
    
    # Fastest model
    fastest = model_stats.loc[model_stats['Avg Latency (s)'].idxmin()]
    insights.append(f"**{fastest['Model']}** is the fastest ({fastest['Avg Latency (s)']:.2f}s average)")
    
    # Tier performance
    tier1_best = model_stats.loc[model_stats['Tier 1 Accuracy'].idxmax()]
    tier2_best = model_stats.loc[model_stats['Tier 2 Accuracy'].idxmax()]
    insights.append(f"**{tier1_best['Model']}** excels at factual questions ({tier1_best['Tier 1 Accuracy']:.1%})")
    insights.append(f"**{tier2_best['Model']}** excels at pattern recognition ({tier2_best['Tier 2 Accuracy']:.1%})")
    
    for insight in insights:
        st.info(insight)
    
    # Data summary
    st.subheader("Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Evaluation Details:**")
        st.write(f"• Total Charts: {df['chart_id'].nunique()}")
        st.write(f"• Total Questions: {len(df)}")
        st.write(f"• Models Evaluated: {df['model_name'].nunique()}")
        st.write(f"• Question Tiers: {df['question_tier'].nunique()}")
    
    with col2:
        st.write("**Performance Range:**")
        st.write(f"• Accuracy: {df['score'].min():.3f} - {df['score'].max():.3f}")
        st.write(f"• Cost: \\${df['cost'].min():.4f} - \\${df['cost'].max():.4f}")
        st.write(f"• Latency: {df['latency_seconds'].min():.2f}s - {df['latency_seconds'].max():.2f}s")
