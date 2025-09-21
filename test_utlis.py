import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import patch
import sys
import os

# Add the src directory to the path so we can import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import clean_data, select_data, train_rf, plot_confusion_matrix


class TestCleanData:
    """Test suite for the clean_data function"""
    
    def test_clean_data_removes_na_rows(self):
        # Create test data with NaN values
        df = pd.DataFrame({
            'bill_length_mm': [39.1, np.nan, 40.3],
            'bill_depth_mm': [18.7, 18.9, np.nan],
            'flipper_length_mm': [181, 181, 197],
            'body_mass_g': [3750, 3800, 4750],
            'sex': ['male', 'female', 'male']
        })
        
        result = clean_data(df)
        # Should have only 1 row (the first one) since others have NaN
        assert len(result) == 1
    
    def test_clean_data_empty_dataframe(self):
        """Test that clean_data handles empty dataframes"""
        df = pd.DataFrame()
        result = clean_data(df)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
    
    def test_clean_data_all_na_rows(self):
        """Test that clean_data handles dataframes where all rows have NaN"""
        df = pd.DataFrame({
            'col1': [np.nan, np.nan],
            'col2': [np.nan, np.nan]
        })
        
        result = clean_data(df)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)


class TestSelectData:
    """Test suite for the select_data function"""
    
    def test_select_data_returns_correct_columns(self):
        """Test that select_data returns the correct X columns and y column"""
        df = pd.DataFrame({
            'bill_length_mm': [39.1, 39.5],
            'bill_depth_mm': [18.7, 17.4],
            'flipper_length_mm': [181, 186],
            'body_mass_g': [3750, 3800],
            'sex': ['male', 'female'],
            'species': ['Adelie', 'Adelie']  # Extra column that shouldn't be selected
        })
        
        X, y = select_data(df)
        # Check X has correct columns
        expected_columns = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        assert list(X.columns) == expected_columns
        # Check y is the sex column
        assert y.name == 'sex'
    
    def test_select_data_return_types(self):
        """Test that select_data returns pandas DataFrame and Series"""
        df = pd.DataFrame({
            'bill_length_mm': [39.1],
            'bill_depth_mm': [18.7],
            'flipper_length_mm': [181],
            'body_mass_g': [3750],
            'sex': ['male']
        })
        X, y = select_data(df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
    
    def test_select_data_missing_columns(self):
        """Test that select_data raises KeyError when required columns are missing"""
        df = pd.DataFrame({
            'bill_length_mm': [39.1],
            'sex': ['male']
            # Missing other required columns
        })
        
        with pytest.raises(KeyError):
            select_data(df)


class TestTrainRf:
    
    def create_sample_data(self):
        """Helper method to create sample training data"""
        np.random.seed(42)
        X = pd.DataFrame({
            'bill_length_mm': np.random.normal(40, 5, 100),
            'bill_depth_mm': np.random.normal(18, 2, 100),
            'flipper_length_mm': np.random.normal(200, 15, 100),
            'body_mass_g': np.random.normal(4000, 500, 100)
        })
        y = pd.Series(np.random.choice(['male', 'female'], 100))
        return X, y
    
    def test_train_rf_returns_correct_types(self):
        """Test that train_rf returns the correct types"""
        X, y = self.create_sample_data()
        
        clf, report, cm, classes = train_rf(X, y)
        
        assert isinstance(clf, RandomForestClassifier)
        assert isinstance(report, dict)
        assert isinstance(cm, np.ndarray)
        assert isinstance(classes, np.ndarray)
    
    def test_train_rf_model_is_fitted(self):
        X, y = self.create_sample_data()
        
        clf, report, cm, classes = train_rf(X, y)
        
        # Test that model can make predictions (will raise NotFittedError if not fitted)
        predictions = clf.predict(X.iloc[:5])
        assert len(predictions) == 5
        assert all(pred in classes for pred in predictions)
    
    def test_train_rf_report_structure(self):
        """Test that the classification report has expected structure"""
        X, y = self.create_sample_data()
        
        clf, report, cm, classes = train_rf(X, y)
        
        # Check that report contains expected keys
        expected_keys = ['accuracy', 'macro avg', 'weighted avg']
        for key in expected_keys:
            assert key in report
        
        # Check that each class is in the report
        for class_name in classes:
            assert class_name in report
    
    def test_train_rf_confusion_matrix_shape(self):
        """Test that confusion matrix has correct shape"""
        X, y = self.create_sample_data()
        
        clf, report, cm, classes = train_rf(X, y)
        
        # Confusion matrix should be square with dimensions equal to number of classes
        n_classes = len(classes)
        assert cm.shape == (n_classes, n_classes)
    
    def test_train_rf_reproducibility(self):
        """Test that train_rf produces reproducible results with same random_state"""
        X, y = self.create_sample_data()
        
        # Train two models with same parameters
        clf1, report1, cm1, classes1 = train_rf(X, y)
        clf2, report2, cm2, classes2 = train_rf(X, y)
        
        # Results should be identical due to random_state=42
        np.testing.assert_array_equal(cm1, cm2)
        assert report1['accuracy'] == report2['accuracy']


class TestPlotConfusionMatrix:
    """Test suite for the plot_confusion_matrix function"""
    
    def test_plot_confusion_matrix_returns_figure(self):
        """Test that plot_confusion_matrix returns a matplotlib figure"""
        cm = np.array([[10, 2], [1, 8]])
        classes = np.array(['male', 'female'])
        
        fig = plot_confusion_matrix(cm, classes)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)  # Clean up
    
    def test_plot_confusion_matrix_figure_properties(self):
        """Test that the returned figure has correct properties"""
        cm = np.array([[10, 2], [1, 8]])
        classes = np.array(['male', 'female'])
        
        fig = plot_confusion_matrix(cm, classes)
        
        # Check figure size
        assert fig.get_size_inches()[0] == 4
        assert fig.get_size_inches()[1] == 3
        
        # Check that there's one axes
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Predicted"
        assert ax.get_ylabel() == "True"
        assert ax.get_title() == "Confusion Matrix"
        
        plt.close(fig)  # Clean up
    
    def test_plot_confusion_matrix_different_sizes(self):
        """Test that plot_confusion_matrix works with different matrix sizes"""
        # Test 3x3 confusion matrix
        cm = np.array([[5, 1, 0], [0, 4, 1], [0, 0, 3]])
        classes = np.array(['class1', 'class2', 'class3'])
        
        fig = plot_confusion_matrix(cm, classes)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)  # Clean up
    
    def test_plot_confusion_matrix_empty_matrix(self):
        """Test that plot_confusion_matrix handles empty confusion matrix"""
        cm = np.array([]).reshape(0, 0)
        classes = np.array([])
        
        # This should not raise an error
        fig = plot_confusion_matrix(cm, classes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)  # Clean up


class TestIntegration:
    """Integration tests that test the functions working together"""
    
    @patch('utils.load_penguins')
    def test_full_pipeline_integration(self, mock_load_penguins):
        """Test the full pipeline from data loading to model training"""
        # Mock the load_penguins function to return test data
        mock_data = pd.DataFrame({
            'bill_length_mm': [39.1, 39.5, 40.3, np.nan],
            'bill_depth_mm': [18.7, 17.4, 18.0, 18.5],
            'flipper_length_mm': [181, 186, 197, 190],
            'body_mass_g': [3750, 3800, 4750, 3900],
            'sex': ['male', 'female', 'male', 'female'],
            'species': ['Adelie', 'Adelie', 'Adelie', 'Adelie']
        })
        mock_load_penguins.return_value = mock_data
        
        # Run the pipeline
        from utils import load_penguins
        data = load_penguins()
        clean_df = clean_data(data)
        X, y = select_data(clean_df)
        clf, report, cm, classes = train_rf(X, y)
        fig = plot_confusion_matrix(cm, classes)
        
        # Verify the pipeline worked
        assert len(clean_df) == 3  # One row removed due to NaN
        assert X.shape[1] == 4  # Four feature columns
        assert len(y) == 3  # Three samples
        assert isinstance(clf, RandomForestClassifier)
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)  # Clean up


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])
