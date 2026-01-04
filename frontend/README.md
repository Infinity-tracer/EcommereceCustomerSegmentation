# Customer Segmentation Frontend

A Streamlit-based web application for predicting customer segments using the trained KMeans clustering model.

## Features

- **Manual Input**: Enter customer data manually and get instant predictions
- **CSV Upload**: Upload a CSV file with multiple customers for batch predictions
- **Sample Predictions**: Test with pre-defined sample customers
- **Visual Insights**: Radar charts and segment distribution visualizations
- **Actionable Recommendations**: Get tailored recommendations for each segment

## Customer Segments

| Segment | Icon | Description |
|---------|------|-------------|
| At-Risk Customers | ⚠️ | Customers who may be churning - need re-engagement |
| Loyal Customers | 💎 | Engaged customers with good frequency and value |
| Champions | 🏆 | Best customers - high value, frequent buyers |

## Input Features

The model uses the following RFM (Recency, Frequency, Monetary) features:

### Recency Metrics
- `Days_Since_Last_Purchase`: Days since last purchase
- `Average_Days_Between_Purchases`: Average days between purchases

### Frequency Metrics
- `Total_Transactions`: Total number of transactions
- `Total_Products_Purchased`: Total products bought
- `Unique_Products_Purchased`: Unique products bought
- `Cancellation_Frequency`: Number of cancellations
- `Cancellation_Rate`: Ratio of cancellations

### Monetary Metrics
- `Total_Spend`: Total amount spent
- `Average_Transaction_Value`: Average value per transaction
- `Monthly_Spending_Mean`: Average monthly spending
- `Monthly_Spending_Std`: Spending variability
- `Spending_Trend`: Positive/negative spending trend

### Behavioral Metrics
- `Day_Of_Week`: Preferred shopping day (0=Monday, 6=Sunday)
- `Hour`: Preferred shopping hour (0-23)

## Running the Application

### Prerequisites

```bash
pip install streamlit pandas plotly scikit-learn mlflow
```

### Start the App

From the project root directory:

```bash
streamlit run frontend/app.py
```

Or use the run script:

```bash
# Windows
run_frontend.bat

# Linux/Mac
./run_frontend.sh
```

The app will be available at `http://localhost:8501`

## Architecture

```
frontend/
├── __init__.py           # Module initialization
├── app.py                # Main Streamlit application
├── customer_predictor.py # Prediction logic and data processing
└── README.md             # This file
```

## How It Works

1. **Data Input**: Customer data is entered via the UI
2. **Feature Processing**: Data is processed through scaling (StandardScaler + Normalizer)
3. **Dimensionality Reduction**: PCA is applied to reduce features
4. **Prediction**: The trained KMeans model predicts the cluster
5. **Result Display**: Segment information and recommendations are shown
