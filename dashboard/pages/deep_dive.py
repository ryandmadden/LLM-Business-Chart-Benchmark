"""
Deep dive page for per-chart analysis and detailed model responses.
"""
import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import (
    load_evaluation_results, load_ground_truth, 
    get_chart_image_path, get_full_model_name
)


def show_deep_dive_page():
    """Display the deep dive page."""
    st.header("Deep Dive Analysis")
    
    
    # Load data
    df = load_evaluation_results()
    ground_truth = load_ground_truth()
    
    if df is None:
        st.error("No evaluation results found. Please run the evaluation first.")
        return
    
    if ground_truth is None:
        st.error("No ground truth data found. Please generate charts first.")
        return
    
    # Chart selection
    st.subheader("📊 Select Chart for Analysis")
    
    available_charts = sorted(df['chart_id'].unique())
    selected_chart = st.selectbox(
        "Choose a chart to analyze:",
        available_charts,
        help="Select a chart to view detailed analysis for each model"
    )
    
    if not selected_chart:
        return
    
    # Display chart image
    st.subheader("📈 Chart Visualization")
    
    image_path = get_chart_image_path(selected_chart)
    if image_path:
        st.image(image_path, caption=f"Chart: {selected_chart}", width='stretch')
    else:
        st.warning(f"Chart image not found: {image_path}")
    
    # Chart metadata
    if selected_chart in ground_truth:
        chart_data = ground_truth[selected_chart]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Chart Type", chart_data.get('chart_type', 'Unknown'))
        
        with col2:
            st.metric("Category", chart_data.get('category', 'Unknown'))
        
        with col3:
            st.metric("Difficulty", chart_data.get('difficulty', 'Unknown'))
    
    # Model responses
    st.subheader("🤖 Model Responses")
    
    chart_df = df[df['chart_id'] == selected_chart]
    
    if chart_df.empty:
        st.warning("No evaluation results found for this chart.")
        return
    
    # Create tabs for each model
    model_names = chart_df['model_name'].unique()
    tabs = st.tabs([f"🤖 {model}" for model in model_names])
    
    for i, model_name in enumerate(model_names):
        with tabs[i]:
            model_chart_df = chart_df[chart_df['model_name'] == model_name]
            
            # Model summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_score = model_chart_df['score'].mean()
                st.metric("Avg Score", f"{avg_score:.3f}")
            
            with col2:
                total_cost = model_chart_df['cost'].sum()
                st.metric("Total Cost", f"${total_cost:.4f}")
            
            with col3:
                avg_latency = model_chart_df['latency_seconds'].mean()
                st.metric("Avg Latency", f"{avg_latency:.2f}s")
            
            
            # Question-by-question analysis
            st.subheader("📝 Question Analysis")
            
            for _, row in model_chart_df.iterrows():
                with st.expander(f"Q: {row['question_text']}", expanded=False):
                    
                    # Question details
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Question Details:**")
                        st.write(f"• Tier: {row['question_tier']}")
                        
                        # Format ground truth (handle lists for multi-answer questions)
                        ground_truth_display = row['ground_truth']
                        
                        # Handle string representations of lists (e.g., "[120, 120000]")
                        if isinstance(ground_truth_display, str) and ground_truth_display.startswith('['):
                            try:
                                import ast
                                parsed = ast.literal_eval(ground_truth_display)
                                if isinstance(parsed, list):
                                    ground_truth_display = f"{parsed[0]} (or {parsed[1]})"
                            except:
                                pass
                        elif isinstance(ground_truth_display, list):
                            ground_truth_display = f"{ground_truth_display[0]} (or {ground_truth_display[1]})"
                        
                        st.write(f"• Ground Truth: {ground_truth_display}")
                        st.write(f"• Score: {row['score']:.3f}")
                    
                    with col2:
                        st.write("**Model Response:**")
                        model_response = row.get('model_response', 'N/A')
                        st.write(f"• Response: {model_response}")
                        
                        # Show cost and latency for this question
                        st.write(f"• Cost: ${row['cost']:.4f}")
                        st.write(f"• Latency: {row['latency_seconds']:.2f}s")
                      
                    # Performance indicators
                    if row['score'] >= 0.8:
                        st.success("✅ Excellent response")
                    elif row['score'] >= 0.5:
                        st.info("✓ Acceptable response")
                    else:
                        st.error("❌ Poor response")
    
    # Comparison across models
    st.subheader("Cross-Model Comparison")
    
    comparison_data = []
    for model_name in model_names:
        model_chart_df = chart_df[chart_df['model_name'] == model_name]
        
        comparison_data.append({
            'Model': get_full_model_name(model_name),
            'Avg Score': model_chart_df['score'].mean(),
            'Total Cost': model_chart_df['cost'].sum(),
            'Avg Latency': model_chart_df['latency_seconds'].mean(),
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.round(3), width='stretch', hide_index=True)
    
    # Ground truth reference
    if selected_chart in ground_truth:
        st.subheader("📋 Ground Truth Reference")
        
        chart_data = ground_truth[selected_chart]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Points:**")
            data_points = chart_data.get('data_points', {})
            for key, value in data_points.items():
                st.write(f"• {key}: {value}")
        
        with col2:
            st.write("**Key Facts:**")
            key_facts = chart_data.get('key_facts', {})
            for key, value in key_facts.items():
                st.write(f"• {key}: {value}")
        
        # Questions reference
        st.write("**Questions:**")
        questions = chart_data.get('questions', {})
        
        for tier, tier_questions in questions.items():
            st.write(f"**{tier.replace('_', ' ').title()}:**")
            for q in tier_questions:
                st.write(f"• {q['text']} → {q['answer']}")
