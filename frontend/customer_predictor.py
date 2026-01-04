"""
Customer Segmentation Predictor Module
Handles data processing and prediction for customer categorization
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
import mlflow

# Project paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_TRACKING_URI = os.path.join(PROJECT_DIR, "mlruns")

# Customer segment labels with descriptions
SEGMENT_LABELS = {
    0: {
        "name": "At-Risk Customers",
        "color": "#e8000b",
        "icon": "⚠️",
        "description": "Customers who haven't purchased recently, have low frequency, or low monetary value. They may be churning and need re-engagement strategies.",
        "recommendations": [
            "Send personalized win-back emails",
            "Offer exclusive discounts",
            "Ask for feedback on their experience"
        ]
    },
    1: {
        "name": "Loyal Customers",
        "color": "#023eff",
        "icon": "💎",
        "description": "Customers who purchase frequently and have good monetary value. They are engaged and valuable to the business.",
        "recommendations": [
            "Offer loyalty rewards program",
            "Provide early access to new products",
            "Send personalized recommendations"
        ]
    },
    2: {
        "name": "Champions",
        "color": "#1ac938",
        "icon": "🏆",
        "description": "Your best customers! They buy often, spend the most, and are recent buyers. They are brand advocates.",
        "recommendations": [
            "Provide VIP treatment",
            "Ask for reviews and referrals",
            "Offer exclusive premium products"
        ]
    }
}


def load_kmeans_model():
    """
    Load the trained KMeans model from MLflow artifacts
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Get the experiment
        experiment = mlflow.get_experiment_by_name("KMeans_Clustering_Ecommerce")
        if experiment is None:
            # Try loading from artifacts directory directly
            model_path = os.path.join(
                PROJECT_DIR, 
                "mlartifacts", 
                "923674514541096298",
                "ef58889a636e40c8bd8103bb7d10db0d",
                "artifacts",
                "model"
            )
            if os.path.exists(model_path):
                return mlflow.sklearn.load_model(model_path)
            raise FileNotFoundError("Model not found in MLflow")
        
        # Get latest run
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if len(runs) == 0:
            raise FileNotFoundError("No runs found in experiment")
        
        latest_run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{latest_run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback: Try to load directly from pickle if available
        fallback_paths = [
            os.path.join(PROJECT_DIR, "mlartifacts", "923674514541096298", 
                        "ef58889a636e40c8bd8103bb7d10db0d", "artifacts", "model", "model.pkl"),
        ]
        for path in fallback_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return pickle.load(f)
        raise FileNotFoundError("Could not load KMeans model")


def load_pca_model():
    """
    Load or create PCA model for feature transformation
    """
    pca_data_path = os.path.join(PROJECT_DIR, "data", "processed", "pca_output.parquet")
    scaler_data_path = os.path.join(PROJECT_DIR, "data", "processed", "scaler_output.parquet")
    
    if os.path.exists(scaler_data_path):
        scaler_data = pd.read_parquet(scaler_data_path)
        # Fit PCA on the scaled data (excluding CustomerID)
        if 'CustomerID' in scaler_data.columns:
            scaler_data = scaler_data.set_index('CustomerID')
        
        # Get the number of components from existing PCA output
        n_components = 3  # Default
        if os.path.exists(pca_data_path):
            pca_data = pd.read_parquet(pca_data_path)
            n_components = len([c for c in pca_data.columns if c.startswith('PC')])
        
        pca = PCA(n_components=n_components)
        pca.fit(scaler_data)
        return pca, scaler_data.columns.tolist()
    
    return None, None


def process_customer_data(customer_data: dict) -> pd.DataFrame:
    """
    Process raw customer data and prepare it for prediction
    
    Args:
        customer_data: Dictionary containing customer metrics
        
    Returns:
        Processed DataFrame ready for prediction
    """
    # Define expected features based on the pipeline
    feature_columns = [
        'Days_Since_Last_Purchase',
        'Total_Transactions',
        'Total_Products_Purchased',
        'Total_Spend',
        'Average_Transaction_Value',
        'Unique_Products_Purchased',
        'Average_Days_Between_Purchases',
        'Day_Of_Week',
        'Hour',
        'Cancellation_Frequency',
        'Cancellation_Rate',
        'Monthly_Spending_Mean',
        'Monthly_Spending_Std',
        'Spending_Trend'
    ]
    
    # Create DataFrame from input
    df = pd.DataFrame([customer_data])
    
    # Ensure all required columns exist (fill with defaults if missing)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the required columns in correct order
    df = df[feature_columns]
    
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standardization and normalization to features
    """
    # Columns to standardize
    standardize_cols = [
        'Days_Since_Last_Purchase', 'Total_Transactions',
        'Total_Spend', 'Average_Transaction_Value',
        'Unique_Products_Purchased', 'Cancellation_Rate',
        'Monthly_Spending_Mean', 'Monthly_Spending_Std', 'Spending_Trend'
    ]
    
    # Columns to normalize
    normalize_cols = [
        'Total_Products_Purchased', 'Average_Days_Between_Purchases',
        'Cancellation_Frequency'
    ]
    
    # Fit scalers on existing data for consistency
    scaler_data_path = os.path.join(PROJECT_DIR, "data", "processed", "scaler_output.parquet")
    
    if os.path.exists(scaler_data_path):
        existing_data = pd.read_parquet(scaler_data_path)
        if 'CustomerID' in existing_data.columns:
            existing_data = existing_data.set_index('CustomerID')
        
        # Standardization
        std_scaler = StandardScaler()
        std_cols_available = [c for c in standardize_cols if c in existing_data.columns and c in df.columns]
        if std_cols_available:
            std_scaler.fit(existing_data[std_cols_available])
            df[std_cols_available] = std_scaler.transform(df[std_cols_available])
        
        # Normalization
        norm_scaler = Normalizer()
        norm_cols_available = [c for c in normalize_cols if c in existing_data.columns and c in df.columns]
        if norm_cols_available:
            norm_scaler.fit(existing_data[norm_cols_available])
            df[norm_cols_available] = norm_scaler.transform(df[norm_cols_available])
    
    return df


def predict_segment(customer_data: dict) -> dict:
    """
    Predict customer segment based on input data
    
    Args:
        customer_data: Dictionary containing customer metrics
        
    Returns:
        Dictionary with segment prediction and details
    """
    try:
        # Process the input data
        df = process_customer_data(customer_data)
        
        # Scale features
        df_scaled = scale_features(df)
        
        # Apply PCA
        pca, pca_columns = load_pca_model()
        if pca is not None:
            # Align columns with PCA training data
            for col in pca_columns:
                if col not in df_scaled.columns:
                    df_scaled[col] = 0
            df_pca = df_scaled[pca_columns]
            features = pca.transform(df_pca)
        else:
            features = df_scaled.values
        
        # Load model and predict
        model = load_kmeans_model()
        cluster = model.predict(features)[0]
        
        # Get segment information
        segment_info = SEGMENT_LABELS.get(cluster, {
            "name": f"Segment {cluster}",
            "color": "#808080",
            "icon": "📊",
            "description": "Customer segment based on RFM analysis",
            "recommendations": ["Analyze customer behavior further"]
        })
        
        return {
            "success": True,
            "cluster_id": int(cluster),
            "segment_name": segment_info["name"],
            "color": segment_info["color"],
            "icon": segment_info["icon"],
            "description": segment_info["description"],
            "recommendations": segment_info["recommendations"],
            "input_data": customer_data
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "input_data": customer_data
        }


def get_sample_customers() -> list:
    """
    Get sample customer data for demo purposes
    """
    return [
        {
            "name": "High Value Customer",
            "data": {
                "Days_Since_Last_Purchase": 5,
                "Total_Transactions": 50,
                "Total_Products_Purchased": 500,
                "Total_Spend": 15000,
                "Average_Transaction_Value": 300,
                "Unique_Products_Purchased": 100,
                "Average_Days_Between_Purchases": 7,
                "Day_Of_Week": 3,
                "Hour": 14,
                "Cancellation_Frequency": 1,
                "Cancellation_Rate": 0.02,
                "Monthly_Spending_Mean": 1250,
                "Monthly_Spending_Std": 200,
                "Spending_Trend": 0.15
            }
        },
        {
            "name": "Regular Customer",
            "data": {
                "Days_Since_Last_Purchase": 30,
                "Total_Transactions": 15,
                "Total_Products_Purchased": 100,
                "Total_Spend": 2500,
                "Average_Transaction_Value": 167,
                "Unique_Products_Purchased": 30,
                "Average_Days_Between_Purchases": 20,
                "Day_Of_Week": 5,
                "Hour": 10,
                "Cancellation_Frequency": 2,
                "Cancellation_Rate": 0.1,
                "Monthly_Spending_Mean": 200,
                "Monthly_Spending_Std": 50,
                "Spending_Trend": 0.02
            }
        },
        {
            "name": "At-Risk Customer",
            "data": {
                "Days_Since_Last_Purchase": 120,
                "Total_Transactions": 3,
                "Total_Products_Purchased": 15,
                "Total_Spend": 250,
                "Average_Transaction_Value": 83,
                "Unique_Products_Purchased": 10,
                "Average_Days_Between_Purchases": 45,
                "Day_Of_Week": 1,
                "Hour": 18,
                "Cancellation_Frequency": 1,
                "Cancellation_Rate": 0.25,
                "Monthly_Spending_Mean": 50,
                "Monthly_Spending_Std": 30,
                "Spending_Trend": -0.1
            }
        }
    ]
