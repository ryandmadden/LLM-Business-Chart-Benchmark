"""
Insights page showing advanced analytics and performance breakdowns.
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import load_evaluation_results, load_ground_truth, get_full_model_name,format_model_names_in_df, MODEL_COLORS


def show_insights_page():
    """Display the insights page."""
    st.header("Advanced Analytics & Insights")

    

    # Load data
    df = load_evaluation_results()
    ground_truth = load_ground_truth()
    
    if df is None:
        st.error("No evaluation results found. Please run the evaluation first.")
        return
    
    if ground_truth is None:
        st.error("No ground truth data found. Please generate charts first.")
        return

    df = format_model_names_in_df(df)
    
    # Merge with ground truth to get chart types
    chart_types = []
    for chart_id in df['chart_id'].unique():
        if chart_id in ground_truth:
            chart_types.append({
                'chart_id': chart_id,
                'chart_type': ground_truth[chart_id].get('chart_type', 'unknown'),
                'category': ground_truth[chart_id].get('category', 'unknown'),
                'difficulty': ground_truth[chart_id].get('difficulty', 'unknown')
            })
    
    chart_type_df = pd.DataFrame(chart_types)
    
    if not chart_type_df.empty:
        # Merge with evaluation results
        df_with_types = df.merge(chart_type_df, on='chart_id', how='left')
        
        # Chart type performance
        type_performance = df_with_types.groupby(['chart_type', 'model_name'])['score'].mean().reset_index()
        type_performance['model_name'] = type_performance['model_name'].apply(get_full_model_name)
        
        fig_type = px.bar(
            type_performance,
            x='chart_type',
            y='score',
            color='model_name',
            title='Average Score by Chart Type',
            barmode='group',
            color_discrete_map=MODEL_COLORS
        )
        fig_type.update_layout(height=400)
        st.plotly_chart(fig_type, use_container_width=True)
        
        # Category performance
        category_performance = df_with_types.groupby(['category', 'model_name'])['score'].mean().reset_index()
        category_performance['model_name'] = category_performance['model_name'].apply(get_full_model_name)
        
        fig_category = px.bar(
            category_performance,
            x='category',
            y='score',
            color='model_name',
            color_discrete_map=MODEL_COLORS,
            title='Average Score by Category',
            barmode='group'
        )
        fig_category.update_layout(height=400)
        st.plotly_chart(fig_category, use_container_width=True)
    
    
    if not chart_type_df.empty:
        difficulty_performance = df_with_types.groupby(['difficulty', 'model_name'])['score'].mean().reset_index()
        difficulty_performance['model_name'] = difficulty_performance['model_name'].apply(get_full_model_name)
        
        # Create difficulty order
        difficulty_order = ['easy', 'medium', 'hard']
        difficulty_performance['difficulty'] = pd.Categorical(
            difficulty_performance['difficulty'], 
            categories=difficulty_order, 
            ordered=True
        )
        difficulty_performance = difficulty_performance.sort_values('difficulty')
        
        fig_difficulty = px.line(
            difficulty_performance,
            x='difficulty',
            y='score',
            color='model_name',
            color_discrete_map=MODEL_COLORS,
            title='Performance Scaling with Difficulty',
            markers=True
        )
        fig_difficulty.update_layout(height=400)
        st.plotly_chart(fig_difficulty, use_container_width=True)
        
        # Difficulty statistics
        difficulty_stats = df_with_types.groupby('difficulty').agg({
            'score': ['mean', 'std', 'count'],
            'cost': 'mean',
            'latency_seconds': 'mean'
        }).round(3)
        
        st.write("**Difficulty Statistics:**")
        st.dataframe(difficulty_stats, width='stretch')
    
    
    # Create cost-efficiency metric
    model_stats = df.groupby('model_name').agg({
        'score': 'mean',
        'cost': 'mean',
        'latency_seconds': 'mean'
    }).reset_index()
    
    model_stats['cost_efficiency'] = model_stats['score'] / model_stats['cost']
    model_stats['speed_efficiency'] = model_stats['score'] / model_stats['latency_seconds']
    
    # Update model names to full names
    model_stats['model_name'] = model_stats['model_name'].apply(get_full_model_name)
    
    # Cost efficiency comparison
    fig_efficiency = px.scatter(
        model_stats,
        x='cost',
        y='score',
        size='latency_seconds',
        color='model_name',
        color_discrete_map=MODEL_COLORS,
        title='Cost vs Performance Trade-off',
        hover_data=['cost_efficiency', 'speed_efficiency']
    )
    fig_efficiency.update_layout(height=400)
    st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Efficiency rankings
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Cost Efficiency Ranking:**")
        efficiency_ranking = model_stats.sort_values('cost_efficiency', ascending=False)
        for i, (_, row) in enumerate(efficiency_ranking.iterrows(), 1):
            st.write(f"{i}. {get_full_model_name(row['model_name'])}: {row['cost_efficiency']:.2f} score/$")
    
    with col2:
        st.write("**Speed Efficiency Ranking:**")
        speed_ranking = model_stats.sort_values('speed_efficiency', ascending=False)
        for i, (_, row) in enumerate(speed_ranking.iterrows(), 1):
            st.write(f"{i}. {get_full_model_name(row['model_name'])}: {row['speed_efficiency']:.3f} score/s")
    
    
    # Latency distribution
    fig_latency = px.box(
        df,
        x='model_name',
        y='latency_seconds',
        title='Latency Distribution by Model',
        color_discrete_map=MODEL_COLORS,
        color='model_name'
    )
    fig_latency.update_layout(height=400)
    fig_latency.update_yaxes(range=[0, 40])  # Set y-axis range to 0-50 seconds
    st.plotly_chart(fig_latency, use_container_width=True)
    
    # Key insights
    st.subheader("Key Insights")
    
    insights = []
    
    # Best overall performer
    best_model = model_stats.loc[model_stats['score'].idxmax()]
    insights.append(f"**{get_full_model_name(best_model['model_name'])}** achieves the highest average score ({best_model['score']:.3f})")
    
    # Most cost-effective
    most_efficient = model_stats.loc[model_stats['cost_efficiency'].idxmax()]
    insights.append(f"**{get_full_model_name(most_efficient['model_name'])}** provides the best cost efficiency ({most_efficient['cost_efficiency']:.2f} score/$)")
    
    # Fastest
    fastest = model_stats.loc[model_stats['latency_seconds'].idxmin()]
    insights.append(f"**{get_full_model_name(fastest['model_name'])}** is the fastest ({fastest['latency_seconds']:.2f}s average)")
    
    
    # Difficulty scaling
    if not chart_type_df.empty:
        difficulty_scores = df_with_types.groupby(['model_name', 'difficulty'])['score'].mean().unstack()
        if 'hard' in difficulty_scores.columns:
            hard_performers = difficulty_scores['hard'].sort_values(ascending=False)
            best_hard = hard_performers.index[0]
            insights.append(f" **{best_hard}** performs best on hard questions ({hard_performers.iloc[0]:.3f})")
    
    for insight in insights:
        st.info(insight)
    
    # Recommendations
    st.subheader("🎯 Recommendations")
    
    recommendations = [
        "**For High Accuracy:** Use the model with the highest overall score for critical applications",
        "**For Cost Optimization:** Use the most cost-efficient model for large-scale deployments",
        "**For Real-time Applications:** Use the fastest model when latency is critical",
        "**For Complex Charts:** Use the model that performs best on hard difficulty questions",
    ]
    
    for rec in recommendations:
        st.write(f"• {rec}")
