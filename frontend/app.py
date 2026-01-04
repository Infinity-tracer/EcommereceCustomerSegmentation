"""
Customer Segmentation Frontend
Streamlit application for predicting customer categories
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.customer_predictor import (
    predict_segment,
    get_sample_customers,
    SEGMENT_LABELS
)

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation - E-commerce MLOps",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .segment-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .segment-name {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .recommendation-item {
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        background-color: #1a1a2e;
        border-radius: 8px;
        border-left: 4px solid #00d26a;
        color: #ffffff;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">🛒 Customer Segmentation System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.1rem; color: #666;">
        Enter customer data to predict their segment and get actionable recommendations
    </p>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Sidebar for segment info
    with st.sidebar:
        st.header("📊 Customer Segments")
        st.markdown("---")
        
        for cluster_id, info in SEGMENT_LABELS.items():
            with st.expander(f"{info['icon']} {info['name']}", expanded=False):
                st.markdown(f"<div style='color: {info['color']}; font-weight: bold;'>Cluster {cluster_id}</div>", 
                          unsafe_allow_html=True)
                st.write(info['description'])
        
        st.markdown("---")
        st.info("💡 This system uses KMeans clustering on RFM (Recency, Frequency, Monetary) features to segment customers.")
    
    # Main content - tabs
    tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📁 Upload CSV", "🎯 Sample Predictions"])
    
    with tab1:
        render_manual_input()
    
    with tab2:
        render_csv_upload()
    
    with tab3:
        render_sample_predictions()


def render_manual_input():
    """Render manual input form"""
    st.subheader("Enter Customer Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📅 Recency Metrics**")
        days_since = st.number_input(
            "Days Since Last Purchase",
            min_value=0,
            max_value=365,
            value=30,
            help="Number of days since the customer's last purchase"
        )
        avg_days_between = st.number_input(
            "Avg Days Between Purchases",
            min_value=0.0,
            max_value=365.0,
            value=15.0,
            help="Average number of days between purchases"
        )
        day_of_week = st.selectbox(
            "Preferred Shopping Day",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
            index=2
        )
        hour = st.slider("Preferred Shopping Hour", 0, 23, 12)
    
    with col2:
        st.markdown("**📈 Frequency Metrics**")
        total_transactions = st.number_input(
            "Total Transactions",
            min_value=1,
            max_value=1000,
            value=10,
            help="Total number of transactions"
        )
        total_products = st.number_input(
            "Total Products Purchased",
            min_value=1,
            max_value=10000,
            value=50,
            help="Total number of products purchased"
        )
        unique_products = st.number_input(
            "Unique Products Purchased",
            min_value=1,
            max_value=5000,
            value=20,
            help="Number of unique products purchased"
        )
        cancel_freq = st.number_input(
            "Cancellation Frequency",
            min_value=0,
            max_value=100,
            value=1,
            help="Number of cancelled orders"
        )
        cancel_rate = st.slider(
            "Cancellation Rate",
            0.0, 1.0, 0.05,
            help="Ratio of cancelled orders to total orders"
        )
    
    with col3:
        st.markdown("**💰 Monetary Metrics**")
        total_spend = st.number_input(
            "Total Spend ($)",
            min_value=0.0,
            max_value=100000.0,
            value=1000.0,
            help="Total amount spent by customer"
        )
        avg_transaction = st.number_input(
            "Avg Transaction Value ($)",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            help="Average value per transaction"
        )
        monthly_mean = st.number_input(
            "Monthly Spending Mean ($)",
            min_value=0.0,
            max_value=50000.0,
            value=500.0,
            help="Average monthly spending"
        )
        monthly_std = st.number_input(
            "Monthly Spending Std ($)",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            help="Standard deviation of monthly spending"
        )
        spending_trend = st.slider(
            "Spending Trend",
            -1.0, 1.0, 0.0,
            help="Spending trend: positive = increasing, negative = decreasing"
        )
    
    st.divider()
    
    # Predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔮 Predict Customer Segment", type="primary", use_container_width=True)
    
    if predict_button:
        customer_data = {
            "Days_Since_Last_Purchase": days_since,
            "Total_Transactions": total_transactions,
            "Total_Products_Purchased": total_products,
            "Total_Spend": total_spend,
            "Average_Transaction_Value": avg_transaction,
            "Unique_Products_Purchased": unique_products,
            "Average_Days_Between_Purchases": avg_days_between,
            "Day_Of_Week": day_of_week,
            "Hour": hour,
            "Cancellation_Frequency": cancel_freq,
            "Cancellation_Rate": cancel_rate,
            "Monthly_Spending_Mean": monthly_mean,
            "Monthly_Spending_Std": monthly_std,
            "Spending_Trend": spending_trend
        }
        
        with st.spinner("Analyzing customer data..."):
            result = predict_segment(customer_data)
        
        display_prediction_result(result)


def render_csv_upload():
    """Render CSV upload section"""
    st.subheader("Upload Customer Data CSV")
    
    st.info("""
    📋 **Expected CSV Columns:**
    - Days_Since_Last_Purchase, Total_Transactions, Total_Products_Purchased
    - Total_Spend, Average_Transaction_Value, Unique_Products_Purchased
    - Average_Days_Between_Purchases, Day_Of_Week, Hour
    - Cancellation_Frequency, Cancellation_Rate
    - Monthly_Spending_Mean, Monthly_Spending_Std, Spending_Trend
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} customers")
            
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🔮 Predict All Segments", type="primary"):
                results = []
                progress_bar = st.progress(0)
                
                for idx, row in df.iterrows():
                    customer_data = row.to_dict()
                    result = predict_segment(customer_data)
                    results.append({
                        "Customer": idx + 1,
                        "Segment": result.get("segment_name", "Unknown"),
                        "Cluster ID": result.get("cluster_id", -1)
                    })
                    progress_bar.progress((idx + 1) / len(df))
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Show distribution
                fig = px.pie(
                    results_df, 
                    names='Segment', 
                    title='Customer Segment Distribution',
                    color='Segment',
                    color_discrete_map={
                        "At-Risk Customers": "#e8000b",
                        "Loyal Customers": "#023eff",
                        "Champions": "#1ac938"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")


def render_sample_predictions():
    """Render sample predictions"""
    st.subheader("Sample Customer Predictions")
    st.markdown("Click on a sample customer to see their predicted segment")
    
    samples = get_sample_customers()
    
    cols = st.columns(len(samples))
    
    for idx, (col, sample) in enumerate(zip(cols, samples)):
        with col:
            with st.container():
                st.markdown(f"### 👤 {sample['name']}")
                
                with st.expander("View Data", expanded=False):
                    for key, value in sample['data'].items():
                        st.write(f"**{key}:** {value}")
                
                if st.button(f"Predict Segment", key=f"sample_{idx}", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        result = predict_segment(sample['data'])
                    st.session_state[f'sample_result_{idx}'] = result
                
                if f'sample_result_{idx}' in st.session_state:
                    result = st.session_state[f'sample_result_{idx}']
                    if result['success']:
                        st.markdown(f"""
                        <div style="background-color: {result['color']}20; padding: 1rem; border-radius: 10px; border: 2px solid {result['color']};">
                            <h3 style="color: {result['color']}; margin: 0;">{result['icon']} {result['segment_name']}</h3>
                        </div>
                        """, unsafe_allow_html=True)


def display_prediction_result(result):
    """Display prediction result"""
    st.divider()
    
    if result['success']:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Segment card
            st.markdown(f"""
            <div class="segment-card" style="background-color: {result['color']}20; border: 3px solid {result['color']};">
                <div style="font-size: 4rem;">{result['icon']}</div>
                <div class="segment-name" style="color: {result['color']};">{result['segment_name']}</div>
                <div style="font-size: 1.2rem; color: #666;">Cluster {result['cluster_id']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Radar chart for input features
            features = ['Recency', 'Frequency', 'Monetary', 'Engagement']
            
            # Normalize values for radar chart
            input_data = result['input_data']
            recency_score = max(0, 100 - input_data['Days_Since_Last_Purchase'])
            frequency_score = min(100, input_data['Total_Transactions'] * 2)
            monetary_score = min(100, input_data['Total_Spend'] / 100)
            engagement_score = min(100, (100 - input_data['Cancellation_Rate'] * 100))
            
            values = [recency_score, frequency_score, monetary_score, engagement_score]
            
            # Convert hex color to rgba for fill
            hex_color = result['color'].lstrip('#')
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            fill_rgba = f"rgba({r}, {g}, {b}, 0.25)"
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=features,
                fill='toself',
                fillcolor=fill_rgba,
                line=dict(color=result['color'])
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title="Customer Profile"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📝 Segment Description")
            st.info(result['description'])
            
            st.markdown("### 💡 Recommendations")
            for rec in result['recommendations']:
                st.markdown(f"""
                <div class="recommendation-item">
                    ✓ {rec}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Input Summary")
            input_df = pd.DataFrame([result['input_data']]).T
            input_df.columns = ['Value']
            st.dataframe(input_df, use_container_width=True)
    
    else:
        st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
        st.warning("""
        **Troubleshooting:**
        1. Make sure the MLflow model is available
        2. Check that the processed data files exist in `data/processed/`
        3. Verify the model was trained properly
        """)


if __name__ == "__main__":
    main()
