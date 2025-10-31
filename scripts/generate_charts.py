"""
Chart generation script for creating diverse business charts with ground truth data.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, Any, List
import seaborn as sns

# Set style for professional charts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# ============================================================================
# Helper functions for advanced Q5 questions
# ============================================================================

def compute_growth_rates(series: list) -> list:
    """Compute period-over-period growth rates."""
    rates = []
    for i in range(1, len(series)):
        if series[i-1] != 0:
            rates.append((series[i] - series[i-1]) / series[i-1] * 100)
        else:
            rates.append(0)
    return rates


def argmax_growth_rate(series: list, labels: list) -> str:
    """Return label of period with highest growth rate."""
    rates = compute_growth_rates(series)
    if not rates:
        return labels[0]
    return labels[rates.index(max(rates)) + 1]


def windowed_slope(series: list, start_idx: int, end_idx: int) -> float:
    """Average slope between two indices."""
    if end_idx <= start_idx or end_idx >= len(series):
        return 0.0
    return (series[end_idx] - series[start_idx]) / (end_idx - start_idx)


def argmax_abs_decline(series: list, labels: list) -> str:
    """Return label with steepest decline vs previous period."""
    diffs = [series[i] - series[i-1] for i in range(1, len(series))]
    if not diffs:
        return labels[0]
    min_diff_idx = diffs.index(min(diffs))
    return labels[min_diff_idx + 1]


def find_crossing_index(series1: list, series2: list) -> int:
    """Find index where two series cross (sign change in difference)."""
    for i in range(1, min(len(series1), len(series2))):
        if (series1[i-1] - series2[i-1]) * (series1[i] - series2[i]) < 0:
            return i
    return -1


def share_of_total(part: float, parts: list) -> float:
    """Calculate percentage share of total."""
    total = sum(parts)
    return (part / total * 100) if total != 0 else 0.0


def most_consistent_index(matrix: np.ndarray, axis: int = 1) -> int:
    """Return index of row/column with lowest standard deviation."""
    stds = np.std(matrix, axis=axis)
    return int(np.argmin(stds))


def overlap_duration(start1: int, dur1: int, start2: int, dur2: int) -> int:
    """Calculate overlap between two intervals."""
    end1, end2 = start1 + dur1, start2 + dur2
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0, overlap_end - overlap_start)


def trend_excluding_indices(series: list, exclude_indices: list) -> str:
    """Determine trend after removing specific indices."""
    filtered = [series[i] for i in range(len(series)) if i not in exclude_indices]
    if len(filtered) < 2:
        return "insufficient_data"
    if filtered[-1] > filtered[0]:
        return "increasing"
    elif filtered[-1] < filtered[0]:
        return "decreasing"
    return "stable"


# ============================================================================
# Chart generation functions
# ============================================================================

def create_quarterly_revenue_chart(chart_id: str) -> Dict[str, Any]:
    """Create a quarterly revenue line chart."""
    np.random.seed(42)
    
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    revenue = [125000, 148000, 167000, 192000]
    
    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    
    ax.plot(quarters, revenue, marker='o', linewidth=3, markersize=8)
    ax.set_title('Quarterly Revenue Growth', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Quarter', fontsize=14)
    ax.set_ylabel('Revenue ($)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(100000, 200000)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()
    
    data_points_dict = dict(zip(quarters, revenue))
    
    all_valid_numbers = extract_all_valid_numbers(
        data_points_dict,
        additional_numbers=[1, 2, 3, 4, 100, 1000, 2024]  # Quarter numbers + year
    )
    
    return {
        "chart_type": "line",
        "category": "business",
        "difficulty": "easy",
        "data_points": data_points_dict,
        "all_valid_numbers": all_valid_numbers,  
        "key_facts": {
            "max_value": max(revenue),
            "max_period": quarters[revenue.index(max(revenue))],
            "trend": "increasing",
            "growth_rate": round((revenue[-1] - revenue[0]) / revenue[0] * 100, 1)
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "What is the revenue in Q3? Answer in exact dollars.", "answer": revenue[2], "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "Which quarter had the highest revenue?", "answer": quarters[revenue.index(max(revenue))], "type": "categorical"}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "Describe the overall revenue trend.", "answer": "increasing", "type": "trend", "keywords": ["increase", "growth", "upward"]},
                {"id": "q4", "text": "What is the approximate growth rate from Q1 to Q4? Answer in percent.", "answer": round((revenue[-1] - revenue[0]) / revenue[0] * 100, 1), "type": "numeric", "tolerance": 3},
                {"id": "q5", "text": "Which quarter had the highest growth rate compared to its previous quarter?", "answer": argmax_growth_rate(revenue, quarters), "type": "categorical"}
            ]
        }
    }

def extract_all_valid_numbers(data_points: dict, additional_numbers: list = None) -> list:
    """
    Extract all valid numbers from data that models might reasonably mention.
    
    Args:
        data_points: Dictionary of data values
        additional_numbers: Extra numbers to include (e.g., quarter numbers, years)
    
    Returns:
        List of all valid numbers with various representations
    """
    valid_numbers = set()
    
    # Add all data point values
    for value in data_points.values():
        if isinstance(value, (int, float)):
            valid_numbers.add(float(value))
            
            # Add scaled versions for currency (e.g., 167000 → 167)
            if value >= 1000:
                valid_numbers.add(float(value / 1000))  # Thousands
            if value >= 1_000_000:
                valid_numbers.add(float(value / 1_000_000))  # Millions
    
    # Add additional context numbers
    if additional_numbers:
        valid_numbers.update([float(n) for n in additional_numbers])
    
    return sorted(list(valid_numbers))

def create_monthly_kpi_chart(chart_id: str) -> Dict[str, Any]:
    """Create a monthly KPI bar chart."""
    np.random.seed(43)
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    kpis = [85, 92, 78, 95, 88, 91]
    
    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    
    bars = ax.bar(months, kpis, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax.set_title('Monthly KPI Performance', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('KPI Score', fontsize=14)
    ax.set_ylim(70, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, kpis):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(value), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()
    
    data_points_dict = dict(zip(months, kpis))
    
    all_valid_numbers = extract_all_valid_numbers(
        data_points_dict,
        additional_numbers=[1, 2, 3, 4, 5, 6, 70, 100]  # Month numbers + y-axis limits
    )
    
    return {
        "chart_type": "bar",
        "category": "business",
        "difficulty": "easy",
        "data_points": dict(zip(months, kpis)),
        "key_facts": {
            "max_value": max(kpis),
            "max_period": months[kpis.index(max(kpis))],
            "min_value": min(kpis),
            "min_period": months[kpis.index(min(kpis))],
            "average": round(sum(kpis) / len(kpis), 1)
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "What is the KPI score for March? Answer in exact score.", "answer": kpis[2], "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "Which month had the highest KPI score?", "answer": months[kpis.index(max(kpis))], "type": "categorical"}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "What is the average KPI score across all months? Answer in exact score.", "answer": round(sum(kpis) / len(kpis), 1), "type": "numeric", "tolerance": 3},
                {"id": "q4", "text": "Which month had the lowest performance?", "answer": months[kpis.index(min(kpis))], "type": "categorical"},
                {"id": "q5", "text": "Between April and June, which month shows the steepest decline compared to its previous month?", "answer": argmax_abs_decline(kpis[3:6], months[3:6]) if min([kpis[i] - kpis[i-1] for i in range(4, 6)]) < 0 else "None", "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_conversion_funnel(chart_id: str) -> Dict[str, Any]:
    """Create a conversion funnel chart."""
    np.random.seed(44)
    
    stages = ['Visitors', 'Leads', 'Trials', 'Customers']
    values = [10000, 2500, 800, 320]
    
    # --- START: Added code to find the highest drop-off ---
    drop_offs = [values[i] - values[i+1] for i in range(len(values) - 1)]
    max_drop_off_index = np.argmax(drop_offs)
    highest_drop_off_stage = f"{stages[max_drop_off_index]} to {stages[max_drop_off_index + 1]}"
    # --- END: Added code ---
    
    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    
    # Create funnel bars
    widths = [v / max(values) for v in values]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (stage, value, width, color) in enumerate(zip(stages, values, widths, colors)):
        ax.barh(i, width, color=color, alpha=0.8)
        ax.text(width/2, i, f'{stage}\n{value:,}', ha='center', va='center', 
                fontweight='bold', fontsize=12)
    
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    ax.set_xlim(0, 1)
    ax.set_title('Conversion Funnel', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Relative Conversion Rate', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()
    
    data_points_dict = dict(zip(stages, values))
    
    all_valid_numbers = extract_all_valid_numbers(
        data_points_dict,
        additional_numbers=[0, 1, 4]  # Stage indices + funnel stages count
    )
    
    return {
        "chart_type": "funnel",
        "category": "business",
        "difficulty": "medium",
        "data_points": dict(zip(stages, values)),
        "key_facts": {
            "total_visitors": values[0],
            "conversion_rate": round(values[-1] / values[0] * 100, 2),
            "lead_rate": round(values[1] / values[0] * 100, 2),
            "trial_rate": round(values[2] / values[1] * 100, 2)
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "How many visitors entered the funnel? Answer in exact count.", "answer": values[0], "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "How many customers were converted? Answer in exact count.", "answer": values[-1], "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "What is the overall conversion rate from visitors to customers? Answer in percent.", "answer": round(values[-1] / values[0] * 100, 2), "type": "numeric", "tolerance": 3},
                {"id": "q4", "text": "Which stage has the highest drop-off?", "answer": highest_drop_off_stage, "type": "categorical"},
                {"id": "q5", "text": "What percent of visitors become trials? Answer in percent.", "answer": round(values[2] / values[0] * 100, 2), "type": "numeric", "tolerance": 3}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_regional_comparison(chart_id: str) -> Dict[str, Any]:
    """Create a regional comparison chart."""
    np.random.seed(45)
    
    regions = ['North', 'South', 'East', 'West', 'Central']
    sales = [45000, 38000, 52000, 41000, 35000]
    
    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    
    bars = ax.bar(regions, sales, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax.set_title('Regional Sales Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Region', fontsize=14)
    ax.set_ylabel('Sales ($)', fontsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add value labels
    for bar, value in zip(bars, sales):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                f'${value:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()
    
    data_points_dict = dict(zip(regions, sales))
    
    all_valid_numbers = extract_all_valid_numbers(
        data_points_dict,
        additional_numbers=[1, 2, 3, 4, 5]  # Region indices
    )
    
    return {
        "chart_type": "bar",
        "category": "business",
        "difficulty": "easy",
        "data_points": dict(zip(regions, sales)),
        "key_facts": {
            "max_value": max(sales),
            "max_region": regions[sales.index(max(sales))],
            "min_value": min(sales),
            "min_region": regions[sales.index(min(sales))],
            "total_sales": sum(sales)
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "What is the sales amount for the East region? Answer in exact dollars.", "answer": sales[2], "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "Which region has the highest sales?", "answer": regions[sales.index(max(sales))], "type": "categorical"}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "What is the total sales across all regions? Answer in exact dollars.", "answer": sum(sales), "type": "numeric", "tolerance": 3},
                {"id": "q4", "text": "Which region has the lowest sales?", "answer": regions[sales.index(min(sales))], "type": "categorical"},
                {"id": "q5", "text": "Which region shows the largest drop relative to the best-performing region?", "answer": regions[sales.index(min(sales))], "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_user_growth_chart(chart_id: str) -> Dict[str, Any]:
    """Create a user growth curve chart."""
    np.random.seed(46)
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    users = [1200, 1350, 1520, 1680, 1850, 2100, 2350, 2600, 2850, 3100, 3400, 3750]
    
    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    
    ax.plot(months, users, marker='o', linewidth=3, markersize=6, color='#4ECDC4')
    ax.fill_between(months, users, alpha=0.3, color='#4ECDC4')
    ax.set_title('Monthly User Growth', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Number of Users', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()
    
    data_points_dict = dict(zip(months, users))
    
    all_valid_numbers = extract_all_valid_numbers(
        data_points_dict,
        additional_numbers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Month numbers
    )
    
    return {
        "chart_type": "line",
        "category": "tech",
        "difficulty": "medium",
        "data_points": dict(zip(months, users)),
        "key_facts": {
            "start_users": users[0],
            "end_users": users[-1],
            "growth_rate": round((users[-1] - users[0]) / users[0] * 100, 1),
            "max_monthly_growth": max([users[i] - users[i-1] for i in range(1, len(users))])
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "How many users were there in June? Answer in exact count.", "answer": users[5], "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "What was the user count at the end of the year? Answer in exact count.", "answer": users[-1], "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "What is the overall growth rate from January to December? Answer in percent.", "answer": round((users[-1] - users[0]) / users[0] * 100, 1), "type": "numeric", "tolerance": 3},
                {"id": "q4", "text": "Describe the user growth trend.", "answer": "increasing", "type": "trend", "keywords": ["increase", "growth", "upward", "rising"]},
                {"id": "q5", "text": "Which month had the highest absolute month-over-month growth (most new users)?", "answer": months[int(np.argmax([users[i] - users[i-1] for i in range(1, len(users))]) + 1)], "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_ab_test_results(chart_id: str) -> Dict[str, Any]:
    """Create A/B test results chart."""
    np.random.seed(47)
    
    variants = ['Control', 'Variant A', 'Variant B']
    conversion_rates = [12.5, 15.2, 18.7]
    error_bars = [1.2, 1.4, 1.6]
    
    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    
    bars = ax.bar(variants, conversion_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], 
                  yerr=error_bars, capsize=5, alpha=0.8)
    ax.set_title('A/B Test Conversion Rates', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Test Variant', fontsize=14)
    ax.set_ylabel('Conversion Rate (%)', fontsize=14)
    ax.set_ylim(0, 25)
    
    # Add value labels
    for bar, rate in zip(bars, conversion_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{rate}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()
    
    data_points_dict = dict(zip(variants, conversion_rates))
    
    all_valid_numbers = extract_all_valid_numbers(
        data_points_dict,
        additional_numbers=[0, 25, 5]  # Y-axis limits + capsize value
    )
    
    return {
        "chart_type": "bar",
        "category": "tech",
        "difficulty": "medium",
        "data_points": dict(zip(variants, conversion_rates)),
        "key_facts": {
            "control_rate": conversion_rates[0],
            "best_variant": variants[conversion_rates.index(max(conversion_rates))],
            "best_rate": max(conversion_rates),
            "improvement": round(max(conversion_rates) - conversion_rates[0], 1)
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "What is the conversion rate for the control group? Answer in percent.", "answer": conversion_rates[0], "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "Which variant has the highest conversion rate?", "answer": variants[conversion_rates.index(max(conversion_rates))], "type": "categorical"}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "What is the improvement of the best variant over control? Answer in percentage points.", "answer": round(max(conversion_rates) - conversion_rates[0], 1), "type": "numeric", "tolerance": 3},
                {"id": "q4", "text": "What is the conversion rate for Variant B? Answer in percent.", "answer": conversion_rates[2], "type": "numeric", "tolerance": 3},
                {"id": "q5", "text": "What is Variant B's lift over Control in percent? Answer in percent.", "answer": round((conversion_rates[2] - conversion_rates[0]) / conversion_rates[0] * 100, 1), "type": "numeric", "tolerance": 3}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_latency_chart(chart_id: str) -> Dict[str, Any]:
    """Create latency over time chart."""
    np.random.seed(48)
    
    hours = list(range(24))
    latency = [45, 42, 38, 35, 32, 30, 28, 25, 22, 20, 18, 16, 15, 17, 19, 22, 25, 28, 32, 35, 38, 42, 45, 47]
    
    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    
    ax.plot(hours, latency, marker='o', linewidth=2, markersize=4, color='#FF6B6B')
    ax.fill_between(hours, latency, alpha=0.3, color='#FF6B6B')
    ax.set_title('API Latency Over 24 Hours', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Hour of Day', fontsize=14)
    ax.set_ylabel('Latency (ms)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    
    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()
    
    data_points_dict = dict(zip([f"H{h:02d}" for h in hours], latency))
    
    all_valid_numbers = extract_all_valid_numbers(
        data_points_dict,
        additional_numbers=[0, 23, 24]  # X-axis limits + total hours
    )
    
    return {
        "chart_type": "line",
        "category": "tech",
        "difficulty": "medium",
        "data_points": dict(zip([f"H{h:02d}" for h in hours], latency)),
        "key_facts": {
            "min_latency": min(latency),
            "max_latency": max(latency),
            "avg_latency": round(sum(latency) / len(latency), 1),
            "peak_hour": hours[latency.index(max(latency))]
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "What is the minimum latency recorded? Answer in milliseconds.", "answer": min(latency), "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "What is the latency at hour 12? Answer in milliseconds.", "answer": latency[12], "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "What is the average latency across all hours? Answer in milliseconds.", "answer": round(sum(latency) / len(latency), 1), "type": "numeric", "tolerance": 3},
                {"id": "q4", "text": "At what hour does latency peak?", "answer": f"H{hours[latency.index(max(latency))]:02d}", "type": "categorical"},
                {"id": "q5", "text": "Which hour has the steepest latency increase compared to the previous hour?", "answer": f"H{hours[compute_growth_rates(latency).index(max(compute_growth_rates(latency))) + 1]:02d}" if compute_growth_rates(latency) else "H01", "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_stock_performance(chart_id: str) -> Dict[str, Any]:
    """Create stock performance chart."""
    np.random.seed(49)
    
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    prices = [100 + i * 2 + np.random.normal(0, 3) for i in range(30)]
    
    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    
    ax.plot(dates, prices, linewidth=2, color='#45B7D1')
    ax.fill_between(dates, prices, alpha=0.3, color='#45B7D1')
    ax.set_title('Stock Price Performance (30 Days)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price ($)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()
    
    start_price = prices[0]
    end_price = prices[-1]
    
    data_points_dict = {str(dates[i].date()): round(prices[i], 2) for i in range(len(dates))}
    
    all_valid_numbers = extract_all_valid_numbers(
        data_points_dict,
        additional_numbers=[30, 2024]  # Days period + year
    )
    
    return {
        "chart_type": "line",
        "category": "financial",
        "difficulty": "easy",
        "data_points": {str(dates[i].date()): round(prices[i], 2) for i in range(len(dates))},
        "key_facts": {
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "price_change": round(end_price - start_price, 2),
            "percent_change": round((end_price - start_price) / start_price * 100, 2)
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "What was the starting price? Answer in exact dollars.", "answer": round(start_price, 2), "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "What was the ending price? Answer in exact dollars.", "answer": round(end_price, 2), "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "What was the total price change? Answer in exact dollars.", "answer": round(end_price - start_price, 2), "type": "numeric", "tolerance": 3},
                {"id": "q4", "text": "What was the percentage change? Answer in percent.", "answer": round((end_price - start_price) / start_price * 100, 2), "type": "numeric", "tolerance": 3},
                {"id": "q5", "text": "Which day saw the largest single-day gain?", "answer": str(dates[int(np.argmax([prices[i] - prices[i-1] for i in range(1, len(prices))]))+ 1].date()), "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_budget_allocation(chart_id: str) -> Dict[str, Any]:
    """Create budget allocation pie chart."""
    np.random.seed(50)
    
    categories = ['Marketing', 'Development', 'Operations', 'Sales', 'Support']
    amounts = [35, 25, 20, 15, 5]
    
    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    wedges, texts, autotexts = ax.pie(amounts, labels=categories, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    
    ax.set_title('Budget Allocation by Department', fontsize=16, fontweight='bold', pad=20)
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()
    
    data_points_dict = dict(zip(categories, amounts))
    
    all_valid_numbers = extract_all_valid_numbers(
        data_points_dict,
        additional_numbers=[90, 100]  # Start angle + total percentage
    )
    
    return {
        "chart_type": "pie",
        "category": "financial",
        "difficulty": "easy",
        "data_points": dict(zip(categories, amounts)),
        "key_facts": {
            "largest_category": categories[amounts.index(max(amounts))],
            "largest_percentage": max(amounts),
            "smallest_category": categories[amounts.index(min(amounts))],
            "smallest_percentage": min(amounts)
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "What percentage of budget goes to Marketing? Answer in percent.", "answer": amounts[0], "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "Which department gets the largest budget allocation?", "answer": categories[amounts.index(max(amounts))], "type": "categorical"}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "What percentage goes to Development? Answer in percent.", "answer": amounts[1], "type": "numeric", "tolerance": 3},
                {"id": "q4", "text": "Which department has the smallest budget allocation?", "answer": categories[amounts.index(min(amounts))], "type": "categorical"},
                {"id": "q5", "text": "What percent of total budget is Marketing plus Development? Answer in percent.", "answer": amounts[0] + amounts[1], "type": "numeric", "tolerance": 3}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_multi_axis_chart(chart_id: str) -> Dict[str, Any]:
    """Create a multi-axis combo chart (edge case)."""
    np.random.seed(51)
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    revenue = [50000, 52000, 58000, 65000, 71000, 78000]
    customers = [1200, 1350, 1500, 1650, 1800, 1950]
    
    fig, ax1 = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    
    # First y-axis for revenue
    color1 = '#FF6B6B'
    ax1.set_xlabel('Month', fontsize=14)
    ax1.set_ylabel('Revenue ($)', color=color1, fontsize=14)
    line1 = ax1.plot(months, revenue, color=color1, marker='o', linewidth=3, label='Revenue')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Second y-axis for customers
    ax2 = ax1.twinx()

    # --- THE FIX IS HERE ---
    # 1. Set ax1 to be drawn on a higher layer (on top of ax2)
    ax1.set_zorder(1)
    # 2. Make the background of the now-top layer (ax1) transparent
    ax1.patch.set_visible(False)


    color2 = '#4ECDC4'
    ax2.set_ylabel('Number of Customers', color=color2, fontsize=14)
    line2 = ax2.plot(months, customers, color=color2, marker='s', linewidth=3, label='Customers')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add title and legend
    ax1.set_title('Revenue vs Customer Growth', fontsize=16, fontweight='bold', pad=20)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()
    
    # Combine both datasets for all_valid_numbers extraction
    combined_data = dict(zip(months, revenue))
    combined_data.update(dict(zip(months, customers)))
    
    all_valid_numbers = extract_all_valid_numbers(
        combined_data,
        additional_numbers=[1, 2, 3, 4, 5, 6]  # Month numbers
    )
    
    return {
        "chart_type": "combo",
        "category": "edge_case",
        "difficulty": "hard",
        "data_points": {
            "revenue": dict(zip(months, revenue)),
            "customers": dict(zip(months, customers))
        },
        "key_facts": {
            "revenue_growth": round((revenue[-1] - revenue[0]) / revenue[0] * 100, 1),
            "customer_growth": round((customers[-1] - customers[0]) / customers[0] * 100, 1),
            "revenue_per_customer": round(revenue[-1] / customers[-1], 2)
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "What was the revenue in March? Answer in exact dollars.", "answer": 58000, "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "How many customers were there in June? Answer in exact count.", "answer": customers[5], "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "What is the revenue growth rate from January to June? Answer in percent.", "answer": round((revenue[-1] - revenue[0]) / revenue[0] * 100, 1), "type": "numeric", "tolerance": 3},
                {"id": "q4", "text": "What is the revenue per customer in June? Answer in exact dollars.", "answer": round(revenue[-1] / customers[-1], 2), "type": "numeric", "tolerance": 3},
                {"id": "q5", "text": "When Revenue peaks, approximately how many Customers are there? Answer in exact count.", "answer": customers[revenue.index(max(revenue))], "type": "numeric", "tolerance": 3}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_correlation_insights_chart(chart_id: str) -> Dict[str, Any]:
    """Create a scatter plot highlighting noisy correlation patterns."""
    np.random.seed(71)

    campaigns = [f"Campaign {i+1:02d}" for i in range(40)]
    marketing_spend = np.linspace(18000, 120000, len(campaigns))
    noise = np.random.normal(0, 9500, len(campaigns))
    revenue = marketing_spend * 2.3 + noise

    coeffs = np.polyfit(marketing_spend, revenue, 1)
    slope, intercept = coeffs
    correlation = np.corrcoef(marketing_spend, revenue)[0, 1]
    prediction_110k = slope * 110000 + intercept

    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    ax.scatter(marketing_spend, revenue, color='#4ECDC4', edgecolors='black', alpha=0.75, s=120)
    ax.plot(marketing_spend, slope * marketing_spend + intercept, color='#FF6B6B', linewidth=2.5, linestyle='--')
    ax.set_title('Marketing Spend vs Revenue (Noisy Correlation)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Marketing Spend ($)', fontsize=14)
    ax.set_ylabel('Revenue ($)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    data_points = {
        campaign: {
            "marketing_spend": round(spend, 2),
            "revenue": round(rev, 2)
        }
        for campaign, spend, rev in zip(campaigns, marketing_spend, revenue)
    }

    combined_values = {
        campaign: spend
        for campaign, spend in zip(campaigns, marketing_spend)
    }
    combined_values.update({f"rev_{campaign}": rev for campaign, rev in zip(campaigns, revenue)})

    all_valid_numbers = extract_all_valid_numbers(
        combined_values,
        additional_numbers=[correlation, slope, intercept, prediction_110k, 110000]
    )

    corr_answers = [round(correlation, 2), round(correlation, 3), correlation]

    return {
        "chart_type": "scatter",
        "category": "strategic",
        "difficulty": "hard",
        "data_points": data_points,
        "key_facts": {
            "correlation": round(correlation, 2),
            "trend": "strong_positive",
            "steepest_campaign": campaigns[int(np.argmax(revenue))]
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "What is the approximate correlation coefficient between spend and revenue? Answer in decimal.", "answer": round(correlation, 2), "type": "numeric", "tolerance": 0.05},
                {"id": "q2", "text": "Which campaign generated the highest revenue?", "answer": campaigns[int(np.argmax(revenue))], "type": "categorical"}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "Describe the relationship between marketing spend and revenue.", "answer": "strong positive", "type": "trend", "keywords": ["positive", "increase", "direct", "strong"]},
                {"id": "q4", "text": "If marketing spend reaches $110,000, what revenue should be expected? Answer in exact dollars.", "answer": round(prediction_110k, -2), "type": "numeric", "tolerance": 3},
                {"id": "q5", "text": "Is the relationship effectively positive or negative?", "answer": "positive", "type": "trend", "keywords": ["positive", "direct", "increase"]}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_customer_density_heatmap(chart_id: str) -> Dict[str, Any]:
    """Create a heatmap that requires careful cell comparisons."""
    np.random.seed(72)

    regions = ['North', 'South', 'East', 'West', 'Central']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    heatmap_values = np.random.randint(62, 98, size=(len(regions), len(months)))

    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    im = ax.imshow(heatmap_values, cmap='viridis')

    ax.set_xticks(np.arange(len(months)), labels=months)
    ax.set_yticks(np.arange(len(regions)), labels=regions)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    for i in range(len(regions)):
        for j in range(len(months)):
            ax.text(j, i, heatmap_values[i, j], ha='center', va='center', color='white', fontweight='bold')

    fig.colorbar(im, ax=ax, shrink=0.75, label='Customer Support Quality Index')
    ax.set_title('Customer Support Quality by Region and Month', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    flat_values = heatmap_values.flatten()
    max_idx = np.argmax(flat_values)
    min_idx = np.argmin(flat_values)
    max_region = regions[max_idx // len(months)]
    max_month = months[max_idx % len(months)]
    min_region = regions[min_idx // len(months)]
    min_month = months[min_idx % len(months)]
    average_quality = round(flat_values.mean(), 1)

    data_points = {
        region: {month: int(value) for month, value in zip(months, row)}
        for region, row in zip(regions, heatmap_values)
    }

    all_valid_numbers = extract_all_valid_numbers(
        {f"{region}_{month}": value for region, row in data_points.items() for month, value in row.items()},
        additional_numbers=[average_quality]
    )

    return {
        "chart_type": "heatmap",
        "category": "operational",
        "difficulty": "hard",
        "data_points": data_points,
        "key_facts": {
            "best_cell": f"{max_region}-{max_month}",
            "worst_cell": f"{min_region}-{min_month}",
            "average": average_quality
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "Which region-month combination achieved the highest quality score?", "answer": f"{max_region} {max_month}", "type": "categorical"},
                {"id": "q2", "text": "What was the lowest quality score recorded? Answer in exact score.", "answer": int(flat_values.min()), "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "Which region shows the most consistent quality across months?", "answer": regions[int(np.argmin(heatmap_values.std(axis=1)))], "type": "categorical"},
                {"id": "q4", "text": "What is the overall average quality score across the grid? Answer in exact score.", "answer": average_quality, "type": "numeric", "tolerance": 3},
                {"id": "q5", "text": "Which region has the most consistent intensity across months?", "answer": regions[most_consistent_index(heatmap_values, axis=1)], "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_latency_box_plot_chart(chart_id: str) -> Dict[str, Any]:
    """Create a box plot chart emphasising outliers and spread."""
    np.random.seed(73)

    services = ['Auth', 'Search', 'Checkout', 'Analytics']
    auth = np.random.gamma(shape=9, scale=18, size=120)
    search = np.random.gamma(shape=7, scale=20, size=120) + np.random.choice([0, 80], size=120, p=[0.92, 0.08])
    checkout = np.random.gamma(shape=10, scale=14, size=120)
    analytics = np.random.gamma(shape=6, scale=24, size=120) + np.random.choice([0, 140], size=120, p=[0.9, 0.1])

    data = [auth, search, checkout, analytics]

    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    bp = ax.boxplot(data, labels=services, patch_artist=True, notch=True, showfliers=True)

    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#FFEAA7']
    for patch, color in zip(bp['boxes'], colors):
        patch.set(facecolor=color, alpha=0.75)

    ax.set_ylabel('Latency (ms)', fontsize=14)
    ax.set_title('Service Latency Distribution with Outliers', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    medians = [float(np.median(d)) for d in data]
    iqrs = [float(np.percentile(d, 75) - np.percentile(d, 25)) for d in data]
    outlier_counts = [int(np.sum((d > np.percentile(d, 75) + 1.5 * iqr) | (d < np.percentile(d, 25) - 1.5 * iqr))) for d, iqr in zip(data, iqrs)]

    data_points = {
        service: {
            "median": round(median, 1),
            "iqr": round(iqr, 1),
            "outliers": count
        }
        for service, median, iqr, count in zip(services, medians, iqrs, outlier_counts)
    }

    combined_values = {f"median_{service}": median for service, median in zip(services, medians)}
    combined_values.update({f"iqr_{service}": iqr for service, iqr in zip(services, iqrs)})

    all_valid_numbers = extract_all_valid_numbers(
        combined_values,
        additional_numbers=outlier_counts
    )

    slowest_service = services[int(np.argmax(medians))]
    most_variable = services[int(np.argmax(iqrs))]

    return {
        "chart_type": "box",
        "category": "reliability",
        "difficulty": "hard",
        "data_points": data_points,
        "key_facts": {
            "slowest_service": slowest_service,
            "most_variable_service": most_variable,
            "outlier_counts": dict(zip(services, outlier_counts))
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "Which service has the highest median latency?", "answer": slowest_service, "type": "categorical"},
                {"id": "q2", "text": "What is the median latency for the Checkout service? Answer in milliseconds.", "answer": medians[2], "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "Which service experiences the widest latency variability?", "answer": most_variable, "type": "categorical"},
                {"id": "q4", "text": "Approximately how many outliers does the Analytics service have? Answer in exact count.", "answer": outlier_counts[3], "type": "numeric", "tolerance": 2},
                {"id": "q5", "text": "Which service has the largest IQR?", "answer": services[int(np.argmax(iqrs))], "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_financial_waterfall_chart(chart_id: str) -> Dict[str, Any]:
    """Create a waterfall chart showing intricate financial movements."""
    np.random.seed(74)

    components = ['Starting Cash', 'Revenue', 'COGS', 'Marketing', 'R&D', 'Operations', 'Other', 'Ending Cash']
    values = [420000, 580000, -230000, -120000, -95000, -75000, 45000, 530000]

    cumulative = [values[0]]
    for val in values[1:-1]:
        cumulative.append(cumulative[-1] + val)
    cumulative.append(values[-1])

    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)

    running_total = values[0]
    for idx, val in enumerate(values):
        color = '#4ECDC4' if val >= 0 else '#FF6B6B'
        if idx == 0:
            ax.bar(idx, val, color=color, alpha=0.85)
        elif idx == len(values) - 1:
            ax.bar(idx, val, color='#45B7D1', alpha=0.85)
        else:
            ax.bar(idx, running_total + val, bottom=running_total, color=color, alpha=0.85)
            running_total += val

        ax.text(idx, running_total + (15000 if val >= 0 else -30000), f"${val/1000:.0f}K", ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')

    ax.set_title('Cash Flow Waterfall Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Amount ($)', fontsize=14)
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, rotation=45, ha='right')
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    data_points = dict(zip(components, values))

    all_valid_numbers = extract_all_valid_numbers(
        data_points,
        additional_numbers=[val/1000 for val in values] + cumulative
    )

    largest_negative = components[int(np.argmin(values[2:-1])) + 2]
    net_change = values[-1] - values[0]

    return {
        "chart_type": "waterfall",
        "category": "financial",
        "difficulty": "hard",
        "data_points": data_points,
        "key_facts": {
            "net_change": net_change,
            "largest_negative": largest_negative,
            "ending_cash": values[-1]
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "Which component caused the largest decrease in cash?", "answer": largest_negative, "type": "categorical"},
                {"id": "q2", "text": "What is the ending cash balance? Answer in thousands.", "answer": round(values[-1] / 1000, 0), "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "What is the net change from starting to ending cash? Answer in thousands.", "answer": round(net_change / 1000, 0), "type": "numeric", "tolerance": 3},
                {"id": "q4", "text": "Did positive drivers outweigh the negative impacts?", "answer": "yes", "type": "categorical"},
                {"id": "q5", "text": "What percent of net change is explained by Revenue? Answer in percent.", "answer": round(abs(values[1] / net_change * 100), 1), "type": "numeric", "tolerance": 3}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_log_scale_adoption_chart(chart_id: str) -> Dict[str, Any]:
    """Create a log-scale line chart focusing on adoption acceleration."""
    np.random.seed(75)

    years = list(range(2012, 2026))
    adopters = np.array([450, 730, 1180, 1890, 3020, 4860, 7840, 12640, 20400, 32900, 53100, 85750, 138530, 223950])

    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    ax.plot(years, adopters, marker='o', color='#FF6B6B', linewidth=3)
    ax.set_yscale('log')
    ax.set_title('Product Adoption on Logarithmic Scale', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Active Users (log scale)', fontsize=14)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    growth_factor = round((adopters[-1] / adopters[0]) ** (1 / (len(adopters) - 1)), 2)
    doubling_period = round(np.log(2) / np.log(growth_factor), 1)

    data_points = dict(zip(years, adopters))

    all_valid_numbers = extract_all_valid_numbers(
        data_points,
        additional_numbers=[growth_factor, doubling_period]
    )

    return {
        "chart_type": "log_line",
        "category": "growth",
        "difficulty": "hard",
        "data_points": data_points,
        "key_facts": {
            "growth_factor": growth_factor,
            "doubling_period_years": doubling_period,
            "final_year_users": int(adopters[-1])
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "Approximately how many active users were there in 2020? Answer in exact count.", "answer": int(adopters[years.index(2020)]), "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "What is the user count in 2025? Answer in exact count.", "answer": int(adopters[-1]), "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "What is the average annual growth factor across the observed period? Answer in decimal.", "answer": growth_factor, "type": "numeric", "tolerance": 3},
                {"id": "q4", "text": "Roughly how many years does it take for the user base to double? Answer in years.", "answer": doubling_period, "type": "numeric", "tolerance": 1},
                {"id": "q5", "text": "Which year has the highest multiplicative growth over its previous year?", "answer": str(years[int(np.argmax([adopters[i]/adopters[i-1] for i in range(1, len(adopters))]) + 1)]), "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_program_gantt_chart(chart_id: str) -> Dict[str, Any]:
    """Create an overlaid Gantt chart with overlapping initiatives."""
    np.random.seed(76)

    phases = ['Discovery', 'Design', 'Implementation', 'Testing', 'Launch']
    start_offsets = [0, 12, 35, 58, 80]
    durations = [15, 22, 48, 20, 7]
    owners = ['Research', 'UX', 'Engineering', 'QA', 'Operations']

    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7', '#96CEB4']
    for idx, (phase, start, duration, color) in enumerate(zip(phases, start_offsets, durations, colors)):
        ax.barh(idx, duration, left=start, height=0.6, color=color, alpha=0.85)
        ax.text(start + duration / 2, idx, phase, ha='center', va='center', fontweight='bold')

    ax.set_xlabel('Days Since Kickoff', fontsize=14)
    ax.set_title('Program Timeline with Overlapping Phases', fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels(owners)
    ax.grid(True, axis='x', alpha=0.3)

    total_duration = max(start + duration for start, duration in zip(start_offsets, durations))
    critical_path = sum(durations)

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    data_points = {
        phase: {"start": start, "duration": duration, "owner": owner}
        for phase, start, duration, owner in zip(phases, start_offsets, durations, owners)
    }

    all_valid_numbers = extract_all_valid_numbers(
        {phase: duration for phase, duration in zip(phases, durations)},
        additional_numbers=start_offsets + [total_duration, critical_path]
    )

    return {
        "chart_type": "gantt",
        "category": "program",
        "difficulty": "hard",
        "data_points": data_points,
        "key_facts": {
            "total_duration": total_duration,
            "critical_path": critical_path,
            "latest_start": phases[np.argmax(start_offsets)]
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "Which phase begins last in the schedule?", "answer": phases[np.argmax(start_offsets)], "type": "categorical"},
                {"id": "q2", "text": "What is the total number of days from kickoff to launch? Answer in days.", "answer": total_duration, "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "Which two phases overlap the most in time?", "answer": "Implementation and Testing", "type": "categorical"},
                {"id": "q4", "text": "What is the cumulative critical-path duration? Answer in days.", "answer": critical_path, "type": "numeric", "tolerance": 3},
                {"id": "q5", "text": "Which two phases overlap the longest?", "answer": "Implementation Testing" if overlap_duration(start_offsets[2], durations[2], start_offsets[3], durations[3]) > overlap_duration(start_offsets[1], durations[1], start_offsets[2], durations[2]) else "Design Implementation", "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_market_share_bubble_chart(chart_id: str) -> Dict[str, Any]:
    """Create a bubble chart emphasising multi-dimensional comparisons."""
    np.random.seed(77)

    companies = ['Aurora', 'Beacon', 'Cascade', 'Dynamo', 'Everest']
    market_share = [24, 18, 31, 15, 12]
    revenue = [410, 295, 520, 260, 215]  # Millions
    profit_margin = [18.5, 22.1, 16.3, 24.8, 19.7]
    headcount = [1800, 1200, 2300, 950, 760]

    efficiency = [pm / (hc / 1000) for pm, hc in zip(profit_margin, headcount)]
    top_margin_company = companies[np.argmax(profit_margin)]
    most_efficient = companies[np.argmax(efficiency)]

    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    sizes = [hc / 3 for hc in headcount]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7', '#96CEB4']

    for comp, share, rev, margin, size, color in zip(companies, market_share, revenue, profit_margin, sizes, colors):
        ax.scatter(share, margin, s=size, alpha=0.75, color=color, edgecolors='black', linewidth=2)
        ax.text(share, margin, comp, ha='center', va='center', fontweight='bold')

    ax.set_xlabel('Market Share (%)', fontsize=14)
    ax.set_ylabel('Profit Margin (%)', fontsize=14)
    ax.set_title('Market Share vs Profitability (Bubble Size = Headcount)', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    data_points = {
        comp: {
            "market_share": share,
            "revenue_millions": rev,
            "profit_margin": margin,
            "headcount": hc
        }
        for comp, share, rev, margin, hc in zip(companies, market_share, revenue, profit_margin, headcount)
    }

    all_valid_numbers = extract_all_valid_numbers(
        {f"share_{comp}": share for comp, share in zip(companies, market_share)},
        additional_numbers=revenue + profit_margin + headcount
    )

    # Calculate median thresholds for more intuitive quadrants
    median_margin = np.median(profit_margin)  # ~19.7%
    median_share = np.median(market_share)  # ~18%
    
    quadrants = []
    for m, v in zip(profit_margin, market_share):
        if m >= median_margin and v >= median_share:
            quadrants.append('Strategic')
        elif m >= median_margin:
            quadrants.append('Premium')
        elif v >= median_share:
            quadrants.append('Volume')
        else:
            quadrants.append('Commodity')

    dynamo_quadrant = quadrants[companies.index('Dynamo')]

    return {
        "chart_type": "bubble",
        "category": "market",
        "difficulty": "hard",
        "data_points": data_points,
        "key_facts": {
            "largest_share": companies[np.argmax(market_share)],
            "highest_margin": top_margin_company,
            "most_efficient": most_efficient
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "Which company holds the largest market share?", "answer": companies[np.argmax(market_share)], "type": "categorical"},
                {"id": "q2", "text": "Which company has the highest profit margin?", "answer": top_margin_company, "type": "categorical"}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "Which firm achieves the highest profit margin per thousand employees?", "answer": most_efficient, "type": "categorical"},
                {"id": "q4", "text": "Into which quadrant does Dynamo fall based on margin and volume?", "answer": "high margin low volume", "type": "categorical"},
                {"id": "q5", "text": "Which company has the best margin per thousand employees?", "answer": most_efficient, "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_product_radar_chart(chart_id: str) -> Dict[str, Any]:
    """Create a radar chart comparing two products across criteria."""
    np.random.seed(78)

    categories = ['Reliability', 'Integrations', 'Usability', 'Security', 'Insights', 'Support']
    product_alpha = [8.9, 7.4, 8.1, 9.2, 7.8, 8.5]
    product_beta = [7.2, 8.8, 9.1, 7.5, 8.9, 7.1]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    alpha_scores = product_alpha + product_alpha[:1]
    beta_scores = product_beta + product_beta[:1]

    fig, ax = plt.subplots(figsize=(10.92, 10.92), subplot_kw=dict(polar=True), dpi=110)
    ax.plot(angles, alpha_scores, linewidth=2, label='Product Alpha', color='#FF6B6B')
    ax.fill(angles, alpha_scores, alpha=0.25, color='#FF6B6B')
    ax.plot(angles, beta_scores, linewidth=2, label='Product Beta', color='#4ECDC4')
    ax.fill(angles, beta_scores, alpha=0.25, color='#4ECDC4')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], color='grey')
    ax.set_ylim(0, 10)
    ax.set_rlabel_position(0)
    ax.set_title('Capability Comparison: Product Alpha vs Product Beta', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    differences = [round(b - a, 1) for a, b in zip(product_alpha, product_beta)]
    strongest_beta_advantage = categories[int(np.argmax(differences))]
    alpha_average = round(sum(product_alpha) / len(product_alpha), 2)
    beta_average = round(sum(product_beta) / len(product_beta), 2)

    data_points = {
        "Product Alpha": dict(zip(categories, product_alpha)),
        "Product Beta": dict(zip(categories, product_beta))
    }

    all_valid_numbers = extract_all_valid_numbers(
        {f"alpha_{cat}": score for cat, score in zip(categories, product_alpha)},
        additional_numbers=product_beta + differences + [alpha_average, beta_average]
    )

    return {
        "chart_type": "radar",
        "category": "product",
        "difficulty": "hard",
        "data_points": data_points,
        "key_facts": {
            "alpha_average": alpha_average,
            "beta_average": beta_average,
            "beta_strongest_advantage": strongest_beta_advantage
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "In which category does Product Beta outperform Product Alpha the most?", "answer": strongest_beta_advantage, "type": "categorical"},
                {"id": "q2", "text": "What is the average score of Product Alpha across all categories? Answer in exact score.", "answer": alpha_average, "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "Which product shows a more balanced performance overall?", "answer": "Product Alpha", "type": "categorical"},
                {"id": "q4", "text": "What is the score difference in the Support category (Beta minus Alpha)? Answer in exact score.", "answer": differences[-1], "type": "numeric", "tolerance": 3},
                {"id": "q5", "text": "Which category shows the greatest advantage for Product Beta over Alpha?", "answer": strongest_beta_advantage, "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_multi_year_stream_chart(chart_id: str) -> Dict[str, Any]:
    """Create a stacked area chart with shifting dominance by segment."""
    np.random.seed(79)

    years = ['2020', '2021', '2022', '2023']
    enterprise = [32, 41, 49, 58]
    midmarket = [22, 27, 35, 39]
    smb = [15, 18, 22, 24]

    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    ax.stackplot(years, enterprise, midmarket, smb, colors=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.85, labels=['Enterprise', 'Mid-Market', 'SMB'])
    ax.set_title('Annual Revenue Contribution by Segment', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Revenue ($ millions)', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    totals = [sum(values) for values in zip(enterprise, midmarket, smb)]
    total_growth = totals[-1] - totals[0]

    data_points = {
        'Enterprise': dict(zip(years, enterprise)),
        'Mid-Market': dict(zip(years, midmarket)),
        'SMB': dict(zip(years, smb))
    }

    combined_numbers = {f"enterprise_{year}": value for year, value in zip(years, enterprise)}
    combined_numbers.update({f"midmarket_{year}": value for year, value in zip(years, midmarket)})
    combined_numbers.update({f"smb_{year}": value for year, value in zip(years, smb)})

    all_valid_numbers = extract_all_valid_numbers(
        combined_numbers,
        additional_numbers=totals + [total_growth]
    )

    leading_segment_2023 = ['Enterprise', 'Mid-Market', 'SMB'][int(np.argmax([enterprise[-1], midmarket[-1], smb[-1]]))]

    return {
        "chart_type": "stacked_area",
        "category": "revenue",
        "difficulty": "hard",
        "data_points": data_points,
        "key_facts": {
            "total_2023": totals[-1],
            "leading_segment_2023": leading_segment_2023,
            "total_growth": total_growth
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "What is the total revenue across all segments in 2023? Answer in millions.", "answer": totals[-1], "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "Which segment contributes the most revenue in 2023?", "answer": leading_segment_2023, "type": "categorical"}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "Describe the overall trend in total revenue from 2020 to 2023.", "answer": "increasing", "type": "trend", "keywords": ["increase", "upward", "growth"]},
                {"id": "q4", "text": "By how much did total revenue grow from 2020 to 2023? Answer in millions.", "answer": total_growth, "type": "numeric", "tolerance": 3},
                {"id": "q5", "text": "What percent of total in 2022 came from Enterprise? Answer in percent.", "answer": round(enterprise[2] / totals[2] * 100, 1), "type": "numeric", "tolerance": 3}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_double_y_axis_complex(chart_id: str) -> Dict[str, Any]:
    """Create a complex dual-axis chart highlighting inverse metrics."""
    np.random.seed(80)

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
    churn_rate = [5.2, 4.8, 4.5, 4.1, 3.9, 3.6, 3.4, 3.2]
    satisfaction = [72, 74, 76, 79, 81, 84, 86, 88]

    fig, ax1 = plt.subplots(figsize=(10.92, 10.92), dpi=110)

    color1 = '#FF6B6B'
    ax1.set_xlabel('Month', fontsize=14)
    ax1.set_ylabel('Churn Rate (%)', color=color1, fontsize=14)
    line1 = ax1.plot(months, churn_rate, color=color1, marker='o', linewidth=3, label='Churn Rate', markersize=8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 8)

    ax2 = ax1.twinx()
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    color2 = '#4ECDC4'
    ax2.set_ylabel('Customer Satisfaction Score', color=color2, fontsize=14)
    line2 = ax2.plot(months, satisfaction, color=color2, marker='s', linewidth=3, label='Satisfaction', markersize=8)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(60, 100)

    ax1.set_title('Churn Rate vs Customer Satisfaction (Inverse Relationship)', fontsize=16, fontweight='bold', pad=20)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    correlation = np.corrcoef(churn_rate, satisfaction)[0, 1]
    improvement_churn = ((churn_rate[0] - churn_rate[-1]) / churn_rate[0]) * 100
    absolute_drop = round(churn_rate[0] - churn_rate[-1], 1)

    data_points = {
        'churn': dict(zip(months, churn_rate)),
        'satisfaction': dict(zip(months, satisfaction))
    }

    combined_numbers = {f"churn_{month}": value for month, value in zip(months, churn_rate)}
    combined_numbers.update({f"sat_{month}": value for month, value in zip(months, satisfaction)})

    all_valid_numbers = extract_all_valid_numbers(
        combined_numbers,
        additional_numbers=[correlation, improvement_churn, absolute_drop]
    )

    return {
        "chart_type": "dual_axis",
        "category": "customer",
        "difficulty": "hard",
        "data_points": data_points,
        "key_facts": {
            "correlation": round(correlation, 2),
            "churn_percent_improvement": round(improvement_churn, 1),
            "absolute_churn_drop": absolute_drop
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "What was the churn rate in June? Answer in percent.", "answer": churn_rate[5], "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "What was the customer satisfaction score in August? Answer in exact score.", "answer": satisfaction[-1], "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "What is the approximate correlation between churn and satisfaction? Answer in decimal.", "answer": round(correlation, 2), "type": "numeric", "tolerance": 0.05},
                {"id": "q4", "text": "By what percentage did churn improve from January to August? Answer in percent.", "answer": round(improvement_churn, 0), "type": "numeric", "tolerance": 3},
                {"id": "q5", "text": "At the month of minimum churn, what is satisfaction? Answer in exact score.", "answer": satisfaction[churn_rate.index(min(churn_rate))], "type": "numeric", "tolerance": 3}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_forecast_interval_chart(chart_id: str) -> Dict[str, Any]:
    """Create an actual vs forecast chart with confidence intervals."""
    np.random.seed(81)

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    actual = np.array([82, 94, 88, 91, 105, 112, 120, 124, 118, 126, 132, 140])
    forecast = actual * np.random.uniform(0.96, 1.04, len(actual))
    lower = forecast * 0.95
    upper = forecast * 1.05

    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    ax.plot(months, actual, marker='o', linewidth=3, label='Actual', color='#4ECDC4')
    ax.plot(months, forecast, marker='s', linewidth=2, linestyle='--', label='Forecast', color='#FF6B6B')
    ax.fill_between(months, lower, upper, color='#FF6B6B', alpha=0.2, label='Forecast Range')
    ax.set_title('Monthly Sales vs Forecast with Confidence Band', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Revenue ($K)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    data_points = {
        month: {
            "actual": round(act, 1),
            "forecast": round(fore, 1),
            "lower": round(lo, 1),
            "upper": round(up, 1)
        }
        for month, act, fore, lo, up in zip(months, actual, forecast, lower, upper)
    }

    combined_numbers = {
        f"actual_{month}": value["actual"] for month, value in data_points.items()
    }
    combined_numbers.update({f"forecast_{month}": value["forecast"] for month, value in data_points.items()})

    all_valid_numbers = extract_all_valid_numbers(
        combined_numbers,
        additional_numbers=list(lower) + list(upper)
    )

    return {
        "chart_type": "line_interval",
        "category": "planning",
        "difficulty": "very_hard",
        "data_points": data_points,
        "key_facts": {
            "final_actual": round(actual[-1], 1),
            "final_forecast": round(forecast[-1], 1),
            "largest_gap_month": months[int(np.argmax(np.abs(actual - forecast)))]
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "What were the actual sales in August? Answer in thousands.", "answer": round(actual[7], 1), "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "What is the forecasted sales for December? Answer in thousands.", "answer": round(forecast[-1], 1), "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "Which month shows the largest variance between actual and forecast?", "answer": months[int(np.argmax(np.abs(actual - forecast)))], "type": "categorical"},
                {"id": "q4", "text": "Describe the overall relationship between actuals and forecast.", "answer": "closely aligned", "type": "trend", "keywords": ["aligned", "close", "accurate", "tight"]},
                {"id": "q5", "text": "Which month deviates most from forecast (absolute difference)?", "answer": months[int(np.argmax(np.abs(actual - forecast)))], "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_margin_vs_volume_scatter_chart(chart_id: str) -> Dict[str, Any]:
    """Create a scatter chart across product margin and volume quadrants."""
    np.random.seed(82)

    products = [f"Product {chr(65 + i)}" for i in range(30)]
    margin = np.random.uniform(8, 42, len(products))
    volume = np.random.uniform(1800, 9800, len(products))
    quadrants = []
    for m, v in zip(margin, volume):
        if m >= 25 and v >= 6000:
            quadrants.append('Strategic')
        elif m >= 25:
            quadrants.append('Premium')
        elif v >= 6000:
            quadrants.append('Volume')
        else:
            quadrants.append('Commodity')

    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    colors = {'Strategic': '#4ECDC4', 'Premium': '#FF6B6B', 'Volume': '#45B7D1', 'Commodity': '#FFE66D'}
    for quadrant in set(quadrants):
        idx = [i for i, q in enumerate(quadrants) if q == quadrant]
        ax.scatter(margin[idx], volume[idx], s=120, alpha=0.75, edgecolor='black', linewidth=1.5, color=colors[quadrant], label=quadrant)
    ax.axvline(25, color='grey', linestyle='--', alpha=0.7)
    ax.axhline(6000, color='grey', linestyle='--', alpha=0.7)
    ax.set_title('Margin vs Volume Quadrant Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Gross Margin (%)', fontsize=14)
    ax.set_ylabel('Units Sold', fontsize=14)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    data_points = {
        product: {
            "margin": round(margin_val, 2),
            "volume": int(volume_val),
            "quadrant": quadr
        }
        for product, margin_val, volume_val, quadr in zip(products, margin, volume, quadrants)
    }

    all_valid_numbers = extract_all_valid_numbers(
        {product: value['margin'] for product, value in data_points.items()},
        additional_numbers=list(volume)
    )

    top_margin_product = products[int(np.argmax(margin))]
    top_volume_product = products[int(np.argmax(volume))]

    return {
        "chart_type": "scatter_quadrant",
        "category": "strategy",
        "difficulty": "very_hard",
        "data_points": data_points,
        "key_facts": {
            "top_margin_product": top_margin_product,
            "top_volume_product": top_volume_product,
            "strategic_count": quadrants.count('Strategic')
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "Which product delivers the highest gross margin?", "answer": top_margin_product, "type": "categorical"},
                {"id": "q2", "text": "Approximately how many units did the top volume product sell? Answer in exact units.", "answer": int(max(volume)), "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "How many products fall into the Strategic quadrant? Answer in exact count.", "answer": quadrants.count('Strategic'), "type": "numeric", "tolerance": 1},
                {"id": "q4", "text": "Describe the trade-off trend between margin and volume.", "answer": "inverse", "type": "trend", "keywords": ["inverse", "trade-off", "opposite", "negative"]},
                {"id": "q5", "text": "How many products fall in Strategic quadrant? Answer in exact count.", "answer": quadrants.count('Strategic'), "type": "numeric", "tolerance": 1}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_retention_cohort_heatmap(chart_id: str) -> Dict[str, Any]:
    """Create a heatmap showing cohort retention percentages."""
    np.random.seed(83)

    cohorts = [f"Q{i} {year}" for year in [2022, 2023, 2024] for i in range(1, 5)]
    periods = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6']
    retention = np.clip(np.random.normal(0.78, 0.08, (len(cohorts), len(periods))), 0.35, 0.97)

    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    im = ax.imshow(retention, cmap='coolwarm', vmin=0.3, vmax=0.95)
    ax.set_xticks(np.arange(len(periods)), labels=periods)
    ax.set_yticks(np.arange(len(cohorts)), labels=cohorts)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    for i in range(len(cohorts)):
        for j in range(len(periods)):
            value = retention[i, j]
            ax.text(j, i, f"{value:.2f}", ha='center', va='center', color='white' if value < 0.6 else 'black', fontweight='bold')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Retention Rate')
    ax.set_title('Customer Retention Cohort Heatmap', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    data_points = {
        cohort: {period: round(retention_val, 3) for period, retention_val in zip(periods, row)}
        for cohort, row in zip(cohorts, retention)
    }

    all_valid_numbers = extract_all_valid_numbers(
        {f"{cohort}_{period}": value for cohort, row in data_points.items() for period, value in row.items()}
    )

    best_cohort_index = int(np.argmax(retention[:, -1]))

    return {
        "chart_type": "cohort_heatmap",
        "category": "retention",
        "difficulty": "very_hard",
        "data_points": data_points,
        "key_facts": {
            "best_final_retention": round(retention[best_cohort_index, -1], 2),
            "best_cohort": cohorts[best_cohort_index]
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "Which cohort maintains the highest retention by Month 6?", "answer": cohorts[best_cohort_index], "type": "categorical"},
                {"id": "q2", "text": "What is the retention rate for that cohort in Month 6? Answer in decimal (0-1).", "answer": round(retention[best_cohort_index, -1], 2), "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "Describe the general retention trend across cohorts.", "answer": "declining", "type": "trend", "keywords": ["decline", "drop", "fade", "decrease"]},
                {"id": "q4", "text": "How many cohorts retain above 60% by Month 4? Answer in exact count.", "answer": int((retention[:, 3] > 0.6).sum()), "type": "numeric", "tolerance": 1},
                {"id": "q5", "text": "Excluding Month 1, which cohort retains best by Month 6?", "answer": cohorts[best_cohort_index], "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_channel_variability_box_chart(chart_id: str) -> Dict[str, Any]:
    """Create a box plot for response times by support channel."""
    np.random.seed(84)

    channels = ['Email', 'Chat', 'Phone', 'Social']
    email = np.random.normal(14, 4.5, 160)
    chat = np.random.normal(6, 2.2, 160)
    phone = np.random.normal(4, 1.5, 160)
    social = np.random.normal(11, 3.8, 160)

    data = [email, chat, phone, social]

    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    box = ax.boxplot(data, labels=channels, patch_artist=True, showfliers=True)

    palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7']
    for patch, color in zip(box['boxes'], palette):
        patch.set(facecolor=color, alpha=0.75)

    ax.set_ylabel('Response Time (minutes)', fontsize=14)
    ax.set_title('Support Response Time by Channel', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    medians = [round(float(np.median(d)), 2) for d in data]
    iqrs = [round(float(np.percentile(d, 75) - np.percentile(d, 25)), 2) for d in data]
    outlier_counts = [int(np.sum((d > np.percentile(d, 75) + 1.5 * (np.percentile(d, 75) - np.percentile(d, 25))) |
                                 (d < np.percentile(d, 25) - 1.5 * (np.percentile(d, 75) - np.percentile(d, 25)))))
                     for d in data]

    data_points = {
        channel: {
            "median": med,
            "iqr": iqr,
            "outliers": outliers
        }
        for channel, med, iqr, outliers in zip(channels, medians, iqrs, outlier_counts)
    }

    all_valid_numbers = extract_all_valid_numbers(
        {f"median_{channel}": med for channel, med in zip(channels, medians)},
        additional_numbers=iqrs + outlier_counts
    )

    slowest_channel = channels[int(np.argmax(medians))]

    return {
        "chart_type": "box",
        "category": "support",
        "difficulty": "very_hard",
        "data_points": data_points,
        "key_facts": {
            "slowest_channel": slowest_channel,
            "fastest_channel": channels[int(np.argmin(medians))]
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "Which channel has the highest median response time?", "answer": slowest_channel, "type": "categorical"},
                {"id": "q2", "text": "What is the median response time for Chat? Answer in minutes.", "answer": medians[1], "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "Which channel shows the widest variability (largest IQR)?", "answer": channels[int(np.argmax(iqrs))], "type": "categorical"},
                {"id": "q4", "text": "Approximately how many outliers does the Social channel have? Answer in exact count.", "answer": outlier_counts[-1], "type": "numeric", "tolerance": 2},
                {"id": "q5", "text": "Which channel has the most outliers? Answer channel name.", "answer": channels[int(np.argmax(outlier_counts))], "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_supply_demand_dual_axis_chart(chart_id: str) -> Dict[str, Any]:
    """Create a dual-axis chart comparing supply and demand curves."""
    np.random.seed(85)

    weeks = [f"W{i+1}" for i in range(16)]
    demand = np.linspace(5200, 8000, len(weeks)) + np.random.normal(0, 250, len(weeks))
    supply = np.linspace(7800, 5200, len(weeks)) + np.random.normal(0, 200, len(weeks))
    imbalance = demand - supply

    fig, ax1 = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    ax1.plot(weeks, demand, marker='o', color='#FF6B6B', linewidth=3, label='Demand')
    ax1.set_ylabel('Demand Units', color='#FF6B6B')
    ax1.tick_params(axis='y', labelcolor='#FF6B6B')

    ax2 = ax1.twinx()
    ax2.plot(weeks, supply, marker='s', color='#4ECDC4', linewidth=3, label='Supply')
    ax2.set_ylabel('Supply Units', color='#4ECDC4')
    ax2.tick_params(axis='y', labelcolor='#4ECDC4')

    ax1.set_title('Supply vs Demand Alignment', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    data_points = {
        week: {
            "demand": round(dem, 1),
            "supply": round(sup, 1),
            "gap": round(gap, 1)
        }
        for week, dem, sup, gap in zip(weeks, demand, supply, imbalance)
    }

    all_valid_numbers = extract_all_valid_numbers(
        {week: value['demand'] for week, value in data_points.items()},
        additional_numbers=list(supply) + list(imbalance)
    )

    max_gap_week = weeks[int(np.argmax(np.abs(imbalance)))]

    return {
        "chart_type": "dual_axis",
        "category": "operations",
        "difficulty": "very_hard",
        "data_points": data_points,
        "key_facts": {
            "max_gap_week": max_gap_week,
            "final_gap": round(imbalance[-1], 1)
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "What was the demand in Week 1? Answer in exact units.", "answer": round(demand[0], 1), "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "What was the supply in Week 16? Answer in exact units.", "answer": round(supply[-1], 1), "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "In which week is the absolute gap between supply and demand largest?", "answer": max_gap_week, "type": "categorical"},
                {"id": "q4", "text": "Describe the overall relationship between supply and demand.", "answer": "inverse", "type": "trend", "keywords": ["inverse", "opposite", "negative", "supply down", "demand up", "diverging", "cross"]},
                {"id": "q5", "text": "In which week do Supply and Demand lines cross?", "answer": weeks[find_crossing_index(list(supply), list(demand))] if find_crossing_index(list(supply), list(demand)) >= 0 else "W8", "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_anomaly_detection_line_chart(chart_id: str) -> Dict[str, Any]:
    """Create a line chart with highlighted anomalies."""
    np.random.seed(86)

    days = [f"Day {i+1}" for i in range(30)]
    baseline = np.linspace(210, 260, len(days))
    anomalies = baseline.copy()
    anomaly_indices = [5, 14, 22]
    anomalies[anomaly_indices] += [60, -55, 80]

    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    
    # Plot using numeric indices instead of string labels
    day_nums = list(range(len(days)))
    ax.plot(day_nums, anomalies, color='#4ECDC4', linewidth=2.5, label='Observed')
    ax.plot(day_nums, baseline, color='#FF6B6B', linestyle='--', linewidth=2, label='Expected')
    ax.scatter([anomaly_indices[0], anomaly_indices[1], anomaly_indices[2]], 
               [anomalies[anomaly_indices[0]], anomalies[anomaly_indices[1]], anomalies[anomaly_indices[2]]], 
               color='#FF6B6B', s=120, edgecolors='black', linewidth=1.5, label='Anomalies')
    
    ax.set_title('Daily Throughput with Anomalies Highlighted', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Units Processed', fontsize=14)
    ax.set_xlabel('Day', fontsize=14)
    
    # Force matplotlib to use only our specified ticks
    ax.xaxis.set_major_locator(plt.FixedLocator([0, 4, 9, 14, 19, 24, 29]))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(['1', '5', '10', '15', '20', '25', '30']))
    ax.set_xlim(-1, len(days))
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    data_points = {
        day: {
            "baseline": round(base_val, 1),
            "observed": round(obs_val, 1),
            "delta": round(obs_val - base_val, 1)
        }
        for day, base_val, obs_val in zip(days, baseline, anomalies)
    }

    all_valid_numbers = extract_all_valid_numbers(
        {day: value['observed'] for day, value in data_points.items()},
        additional_numbers=list(baseline) + [value['delta'] for value in data_points.values()]
    )

    return {
        "chart_type": "anomaly_line",
        "category": "quality",
        "difficulty": "very_hard",
        "data_points": data_points,
        "key_facts": {
            "anomaly_days": [days[i] for i in anomaly_indices]
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "What was the observed throughput on Day 15? Answer in exact units.", "answer": round(anomalies[14], 1), "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "Which days are flagged as anomalies?", "answer": [days[i] for i in anomaly_indices], "type": "categorical"}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "How much above baseline was the largest positive anomaly? Answer in exact units.", "answer": round(max(anomalies - baseline), 1), "type": "numeric", "tolerance": 3},
                {"id": "q4", "text": "Describe the general throughput trend excluding anomalies.", "answer": "increasing", "type": "trend", "keywords": ["increase", "upward", "growth"]},
                {"id": "q5", "text": "Excluding anomalies, what is the overall trend?", "answer": trend_excluding_indices(list(anomalies), anomaly_indices), "type": "trend", "keywords": ["increase", "upward", "growth"]}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_error_distribution_histogram(chart_id: str) -> Dict[str, Any]:
    """Create a histogram showing error distribution by severity."""
    np.random.seed(87)

    severities = ['Critical', 'High', 'Medium', 'Low']
    counts = np.array([12, 36, 78, 142])
    duration = np.array([5.5, 3.2, 1.4, 0.6])  # avg hours to resolve

    fig, ax1 = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    bars = ax1.bar(severities, counts, color=['#FF6B6B', '#FFA36C', '#FFE66D', '#4ECDC4'], alpha=0.85)
    ax1.set_ylabel('Incident Count', fontsize=14)
    ax1.set_xlabel('Severity Level', fontsize=14)
    ax1.set_title('Incident Volume by Severity', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, axis='y', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(severities, duration, color='#45B7D1', marker='o', linewidth=2, label='Avg Resolution Time (hrs)')
    ax2.set_ylabel('Avg Resolution Time (hrs)', fontsize=14, color='#45B7D1')
    ax2.tick_params(axis='y', labelcolor='#45B7D1')

    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    data_points = {
        severity: {
            "count": int(count),
            "avg_time": round(time, 2)
        }
        for severity, count, time in zip(severities, counts, duration)
    }

    all_valid_numbers = extract_all_valid_numbers(
        {severity: value['count'] for severity, value in data_points.items()},
        additional_numbers=duration.tolist()
    )

    return {
        "chart_type": "histogram_combo",
        "category": "incident",
        "difficulty": "very_hard",
        "data_points": data_points,
        "key_facts": {
            "most_common_severity": severities[int(np.argmax(counts))],
            "slowest_resolution": severities[int(np.argmax(duration))]
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "How many Medium severity incidents occurred? Answer in exact count.", "answer": int(counts[2]), "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "Which severity has the longest average resolution time?", "answer": severities[int(np.argmax(duration))], "type": "categorical"}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "What percent of total incidents were Low severity? Answer in percent.", "answer": round(counts[-1] / counts.sum() * 100, 1), "type": "numeric", "tolerance": 3},
                {"id": "q4", "text": "Describe the relationship between severity and average resolution time.", "answer": "inverse", "type": "trend", "keywords": ["inverse", "higher severity", "faster", "priority"]},
                {"id": "q5", "text": "What percent of all incidents are High plus Critical? Answer in percent.", "answer": round((counts[0] + counts[1]) / counts.sum() * 100, 1), "type": "numeric", "tolerance": 3}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_portfolio_area_chart(chart_id: str) -> Dict[str, Any]:
    """Create a stacked area chart showing portfolio revenue share."""
    np.random.seed(88)

    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    platform = [52, 58, 63, 71]
    services = [18, 21, 24, 28]
    hardware = [12, 10, 9, 7]
    partner = [8, 11, 13, 14]

    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    ax.stackplot(quarters, platform, services, hardware, partner,
                 labels=['Platform', 'Services', 'Hardware', 'Partner'], colors=['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'], alpha=0.9)
    ax.set_title('Revenue Contribution by Portfolio Category', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Revenue ($M)', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    data_points = {
        'Platform': dict(zip(quarters, platform)),
        'Services': dict(zip(quarters, services)),
        'Hardware': dict(zip(quarters, hardware)),
        'Partner': dict(zip(quarters, partner))
    }

    combined_numbers = {f"platform_{q}": v for q, v in zip(quarters, platform)}
    combined_numbers.update({f"services_{q}": v for q, v in zip(quarters, services)})
    combined_numbers.update({f"hardware_{q}": v for q, v in zip(quarters, hardware)})
    combined_numbers.update({f"partner_{q}": v for q, v in zip(quarters, partner)})

    totals = [sum(values) for values in zip(platform, services, hardware, partner)]

    all_valid_numbers = extract_all_valid_numbers(
        combined_numbers,
        additional_numbers=totals
    )

    return {
        "chart_type": "stacked_area",
        "category": "portfolio",
        "difficulty": "very_hard",
        "data_points": data_points,
        "key_facts": {
            "highest_quarter_total": totals[int(np.argmax(totals))],
            "declining_segment": 'Hardware'
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "What is the Platform revenue in Q4? Answer in millions.", "answer": platform[-1], "type": "numeric", "tolerance": 3},
                {"id": "q2", "text": "Which segment shows a downward trend across the year?", "answer": 'Hardware', "type": "categorical"}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "What is the total revenue in Q4? Answer in millions.", "answer": totals[-1], "type": "numeric", "tolerance": 3},
                {"id": "q4", "text": "Describe the overall portfolio mix shift during the year.", "answer": "shifting to platform and services", "type": "trend", "keywords": ["platform", "services", "increasing", "mix"]},
                {"id": "q5", "text": "What percent of total in Q4 is Services? Answer in percent.", "answer": round(services[-1] / totals[-1] * 100, 1), "type": "numeric", "tolerance": 3}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_goal_progress_gantt_chart(chart_id: str) -> Dict[str, Any]:
    """Create a Gantt-style chart showing OKR progress."""
    np.random.seed(89)

    objectives = ['Acquire Users', 'Improve NPS', 'Automate Ops', 'Launch Feature X']
    start_days = [0, 15, 30, 45]
    durations = [40, 35, 50, 28]
    progress = [0.8, 0.55, 0.4, 0.65]

    fig, ax = plt.subplots(figsize=(10.92, 10.92), dpi=110)
    colors = ['#4ECDC4', '#45B7D1', '#FF6B6B', '#FFE66D']
    for idx, (obj, start, length, prog, color) in enumerate(zip(objectives, start_days, durations, progress, colors)):
        ax.barh(idx, length, left=start, color=color, alpha=0.8, height=0.6)
        ax.barh(idx, length * prog, left=start, color='black', alpha=0.15, height=0.6)
        ax.text(start + length / 2, idx, f"{obj}\n{int(prog*100)}%", ha='center', va='center', color='black', fontweight='bold')

    ax.set_xlabel('Days Elapsed', fontsize=14)
    ax.set_title('OKR Progress Timeline', fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(len(objectives)))
    ax.set_yticklabels(objectives)
    ax.grid(True, axis='x', alpha=0.3)

    total_horizon = max(start + length for start, length in zip(start_days, durations))

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    data_points = {
        obj: {
            "start": start,
            "duration": length,
            "progress": prog
        }
        for obj, start, length, prog in zip(objectives, start_days, durations, progress)
    }

    all_valid_numbers = extract_all_valid_numbers(
        {objective: info['duration'] for objective, info in data_points.items()},
        additional_numbers=start_days + [total_horizon] + [p * 100 for p in progress]
    )

    return {
        "chart_type": "gantt",
        "category": "okr",
        "difficulty": "very_hard",
        "data_points": data_points,
        "key_facts": {
            "total_horizon": total_horizon,
            "most_advanced_objective": objectives[int(np.argmax(progress))]
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "Which objective has progressed the furthest?", "answer": objectives[int(np.argmax(progress))], "type": "categorical"},
                {"id": "q2", "text": "What is the total timeline horizon in days? Answer in days.", "answer": total_horizon, "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "Which objective is the least complete?", "answer": objectives[int(np.argmin(progress))], "type": "categorical"},
                {"id": "q4", "text": "Describe the overall progress distribution across objectives.", "answer": "uneven", "type": "trend", "keywords": ["uneven", "varied", "imbalanced", "mixed", "different", "varies", "ranging", "spread"]},
                {"id": "q5", "text": "Which objective has the highest velocity (progress per day)?", "answer": objectives[int(np.argmax([progress[i]/durations[i] for i in range(len(progress))]))], "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def create_competitive_radar_plus_chart(chart_id: str) -> Dict[str, Any]:
    """Create a radar chart comparing three competitors."""
    np.random.seed(90)

    attributes = ['Brand', 'Coverage', 'Innovation', 'Reliability', 'Support', 'Pricing']
    comp_a = [8.2, 7.5, 6.8, 8.9, 7.1, 6.4]
    comp_b = [7.4, 8.7, 7.3, 8.1, 8.4, 7.2]
    comp_c = [6.5, 6.8, 8.9, 7.6, 6.9, 8.5]

    N = len(attributes)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles.append(angles[0])

    comp_a_scores = comp_a + comp_a[:1]
    comp_b_scores = comp_b + comp_b[:1]
    comp_c_scores = comp_c + comp_c[:1]

    fig, ax = plt.subplots(figsize=(10.92, 10.92), subplot_kw=dict(polar=True), dpi=110)
    ax.plot(angles, comp_a_scores, linewidth=2, label='Comp A', color='#FF6B6B')
    ax.fill(angles, comp_a_scores, alpha=0.15, color='#FF6B6B')
    ax.plot(angles, comp_b_scores, linewidth=2, label='Comp B', color='#4ECDC4')
    ax.fill(angles, comp_b_scores, alpha=0.15, color='#4ECDC4')
    ax.plot(angles, comp_c_scores, linewidth=2, label='Comp C', color='#45B7D1')
    ax.fill(angles, comp_c_scores, alpha=0.15, color='#45B7D1')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attributes)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], color='grey')
    ax.set_ylim(0, 10)
    ax.set_rlabel_position(0)
    ax.set_title('Competitive Attributes Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    plt.savefig(f'data/charts/{chart_id}.png', dpi=110, bbox_inches='tight')
    plt.close()

    averages = {
        "Comp A": round(sum(comp_a) / len(comp_a), 2),
        "Comp B": round(sum(comp_b) / len(comp_b), 2),
        "Comp C": round(sum(comp_c) / len(comp_c), 2)
    }

    data_points = {
        "Comp A": dict(zip(attributes, comp_a)),
        "Comp B": dict(zip(attributes, comp_b)),
        "Comp C": dict(zip(attributes, comp_c))
    }

    all_valid_numbers = extract_all_valid_numbers(
        {f"compA_{attr}": score for attr, score in zip(attributes, comp_a)},
        additional_numbers=comp_b + comp_c + list(averages.values())
    )

    best_overall = max(averages, key=averages.get)

    return {
        "chart_type": "radar",
        "category": "competitive",
        "difficulty": "very_hard",
        "data_points": data_points,
        "key_facts": {
            "best_overall": best_overall,
            "averages": averages
        },
        "questions": {
            "tier1_factual": [
                {"id": "q1", "text": "Which competitor has the highest overall average score?", "answer": best_overall, "type": "categorical"},
                {"id": "q2", "text": "What is Comp B's score for Innovation? Answer in exact score.", "answer": comp_b[attributes.index('Innovation')], "type": "numeric", "tolerance": 3}
            ],
            "tier2_pattern": [
                {"id": "q3", "text": "Which attribute shows the greatest spread across competitors?", "answer": attributes[int(np.argmax(np.ptp([comp_a, comp_b, comp_c], axis=0)))], "type": "categorical"},
                {"id": "q4", "text": "Describe the overall positioning of Comp C.", "answer": "innovation leader", "type": "trend", "keywords": ["innovation", "leader", "strong", "high"]},
                {"id": "q5", "text": "Which attribute has the widest spread across competitors?", "answer": attributes[int(np.argmax(np.ptp([comp_a, comp_b, comp_c], axis=0)))], "type": "categorical"}
            ]
        },
        "all_valid_numbers": all_valid_numbers
    }


def generate_all_charts():
    """Generate all charts and create ground truth JSON."""
    os.makedirs('data/charts', exist_ok=True)
    
    chart_generators = [
        ("chart_001", create_quarterly_revenue_chart),
        ("chart_002", create_monthly_kpi_chart),
        ("chart_003", create_conversion_funnel),
        ("chart_004", create_regional_comparison),
        ("chart_005", create_user_growth_chart),
        ("chart_006", create_ab_test_results),
        ("chart_007", create_latency_chart),
        ("chart_008", create_stock_performance),
        ("chart_009", create_budget_allocation),
        ("chart_010", create_multi_axis_chart),
        ("chart_011", create_correlation_insights_chart),
        ("chart_012", create_customer_density_heatmap),
        ("chart_013", create_latency_box_plot_chart),
        ("chart_014", create_financial_waterfall_chart),
        ("chart_015", create_log_scale_adoption_chart),
        ("chart_016", create_program_gantt_chart),
        ("chart_017", create_market_share_bubble_chart),
        ("chart_018", create_product_radar_chart),
        ("chart_019", create_multi_year_stream_chart),
        ("chart_020", create_double_y_axis_complex),
        ("chart_021", create_forecast_interval_chart),
        ("chart_022", create_margin_vs_volume_scatter_chart),
        ("chart_023", create_retention_cohort_heatmap),
        ("chart_024", create_channel_variability_box_chart),
        ("chart_025", create_supply_demand_dual_axis_chart),
        ("chart_026", create_anomaly_detection_line_chart),
        ("chart_027", create_error_distribution_histogram),
        ("chart_028", create_portfolio_area_chart),
        ("chart_029", create_goal_progress_gantt_chart),
        ("chart_030", create_competitive_radar_plus_chart)
    ]
    
    ground_truth = {}
    
    print("Generating charts...")
    for chart_id, generator_func in chart_generators:
        print(f"Creating {chart_id}...")
        ground_truth[chart_id] = generator_func(chart_id)
    
    with open('data/ground_truth.json', 'w') as f:
        json.dump(ground_truth, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else o)
    
    print(f"\nGenerated {len(chart_generators)} charts in data/charts/")
    print("Ground truth saved to data/ground_truth.json")
    
    return ground_truth


if __name__ == "__main__":
    import pandas as pd
    ground_truth = generate_all_charts()
