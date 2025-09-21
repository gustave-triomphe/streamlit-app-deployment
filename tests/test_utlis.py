import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import sys
import os

# Add the src directory to the path so we can import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import clean_data, select_data, train_rf, plot_confusion_matrix


def test_clean_data():
    """Test that clean_data removes rows with NaN values"""
    df = pd.DataFrame({
        'bill_length_mm': [39.1, np.nan, 40.3],
        'bill_depth_mm': [18.7, 18.9, 18.0],
        'sex': ['male', 'female', 'male']
    })
    
    result = clean_data(df)
    
    # Should remove the row with NaN
    assert len(result) == 2
    assert result.iloc[0]['bill_length_mm'] == 39.1


def test_select_data():
    """Test that select_data returns correct columns"""
    df = pd.DataFrame({
        'bill_length_mm': [39.1, 39.5],
        'bill_depth_mm': [18.7, 17.4],
        'flipper_length_mm': [181, 186],
        'body_mass_g': [3750, 3800],
        'sex': ['male', 'female']
    })
    
    X, y = select_data(df)
    
    # Check correct columns are selected
    assert list(X.columns) == ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    assert y.name == 'sex'
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_train_rf():
    """Test that train_rf returns correct types"""
    # Create test data using normal distributions for realistic penguin measurements
    np.random.seed(42)  # For reproducible tests
    
    X = pd.DataFrame({
        'bill_length_mm': np.random.normal(50, 5, 100),
        'bill_depth_mm': np.random.normal(20, 2, 100),
        'flipper_length_mm': np.random.normal(200, 20, 100),
        'body_mass_g': np.random.normal(4000, 1000, 100)
    })
    
    y = pd.Series(np.random.choice(['male', 'female'], size=100, p=[0.5, 0.5]))
    
    clf, report, cm, classes = train_rf(X, y)
    
    # Check return types
    assert isinstance(clf, RandomForestClassifier)
    assert isinstance(report, dict)
    assert isinstance(cm, np.ndarray)
    assert isinstance(classes, np.ndarray)


def test_plot_confusion_matrix():
    """Test that plot_confusion_matrix returns a matplotlib figure"""
    cm = np.array([[10, 2], [1, 8]])
    classes = np.array(['male', 'female'])
    
    fig = plot_confusion_matrix(cm, classes)
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)  # Clean up
