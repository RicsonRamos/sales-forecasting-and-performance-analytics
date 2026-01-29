import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def prepare_modeling_data(sales_raw):
    """
    Prepare data for modeling by converting categorical variables and splitting the dataset.

    Focuses on features identified in the importance study: 
    Unit price, Quantity, Hour, Branch, Customer type, and Product line.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing features and target.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        The four-way split of data for training and testing.
    """
    # Features identified for the predictive engine
    features = ['Unit price', 'Quantity', 'Hour', 'Branch', 'Customer type', 'Product line']
    
    # Converting categorical variables using one-hot encoding
    # drop_first=True is used to avoid the dummy variable trap (multicollinearity)
    X = pd.get_dummies(sales_raw[features], drop_first=True)
    
    # Target variable: Sales
    y = sales_raw['Sales']
    
    # Dividing the data into training and test sets (80/20 split)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train a Random Forest regressor and evaluate its performance.

    Parameters
    ----------
    X_train, X_test : pandas.DataFrame
        Feature sets for training and testing.
    y_train, y_test : pandas.Series
        Target variables for training and testing.

    Returns
    -------
    model : sklearn.ensemble.RandomForestRegressor
        The trained model object.
    predictions : numpy.ndarray
        Model predictions for the test set.
    metrics : dict
        A dictionary containing R² and MAE scores.
    """
    # Initialize and train the regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Generate predictions
    predictions = model.predict(X_test)
    
    # Calculate performance metrics
    metrics = {
        'R2': r2_score(y_test, predictions),
        'MAE': mean_absolute_error(y_test, predictions)
    }
    
    return model, predictions, metrics

def save_model_artifact(model, filename='supermarket_rf_model.pkl'):
    """
    Export the trained model as a binary file for production use.

    Parameters
    ----------
    model : object
        The trained model to be exported.
    filename : str
        The name of the .pkl file.
    """
    output_dir = '../outputs/models'
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    
    joblib.dump(model, path)
    print(f"Model artifact successfully saved at: {path}")

def plot_feature_importance(model, X_test, y_test, output_path='../outputs/figures/feature_importance.png'):
    """
    Generates, displays, and saves a Permutation Importance plot.
    
    Permutation importance is calculated on the test set to show how much 
    the model performance (R²) drops when a specific feature's values are shuffled.

    Parameters
    ----------
    model : object
        The trained model to evaluate.
    X_test : pandas.DataFrame
        Testing features.
    y_test : pandas.Series
        Testing target.
    output_path : str
        Directory path and filename to save the plot.

    Returns
    -------
    result : sklearn.utils.Bunch
        The raw permutation importance results.
    """
    if model is None or X_test is None or y_test is None:
        raise ValueError("Model and test data cannot be null.")

    try:
        # 1. Calculate Permutation Importance
        # n_repeats=10 provides a good balance between stability and speed
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # 2. Sort indices based on mean importance
        sorted_idx = result.importances_mean.argsort()

        # 3. Create the Visualization
        plt.figure(figsize=(12, 8))
        
        # Boxplot shows the distribution of importance across the 10 repeats
        plt.boxplot(
            result.importances[sorted_idx].T, 
            vert=False, 
            labels=X_test.columns[sorted_idx]
        )
        
        # Adding professional chart elements
        plt.title("Feature Importance via Permutation (Test Set)", fontsize=14, pad=20)
        plt.xlabel("Decrease in R² score (when feature is shuffled)", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5) # Reference line at zero
        plt.grid(axis='x', linestyle=':', alpha=0.6)
        
        plt.tight_layout()

        # 4. Save and Export
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300) # Higher DPI for professional reports
        plt.show()
        
        print(f"Importance plot successfully saved at: {output_path}")
        
        return result

    except Exception as e:
        print(f"❌ Error generating importance plot: {e}")
        return None
    

def build_powerbi_dataset(
    base_df,
    cluster_series,
    model,
    features,
    output_path="supermarket_sales_enriched.csv"
):
    """
    Enriches the base dataframe with clustering and sales predictions
    and exports it for Power BI consumption.

    Parameters
    ----------
    base_df : pandas.DataFrame
        The base dataframe containing the original data.
    cluster_series : pandas.Series
        The clustering results for each customer.
    model : sklearn.ensemble.RandomForestRegressor
        The trained model used for sales predictions.
    features : pandas.DataFrame
        The features used for sales predictions.
    output_path : str
        The path and filename to which the dataframe will be exported.

    Returns
    -------
    sales_export : pandas.DataFrame
        The enriched dataframe ready for Power BI consumption.
    """
    # Copying the original dataframe
    sales_export = base_df.copy()

    # Adding clustering results
    sales_export["Cluster_Segment"] = cluster_series

    # Adding sales predictions
    sales_export["Predicted_Sales"] = model.predict(features)

    # Exporting to CSV
    sales_export.to_csv(output_path, index=False)

    # Printing success message
    print(f"Project exported successfully for Power BI → {output_path}")

    return sales_export
