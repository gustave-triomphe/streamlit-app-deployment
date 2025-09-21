
# Penguin Sex Classifier

A **Streamlit web application** that uses a **Random Forest classifier** to predict penguin sex (male/female) based on body measurements from the Palmer Penguins dataset.

## 🐧 Features

- **Interactive Web App**: User-friendly Streamlit interface
- **Data Processing**: Automatic data cleaning and preprocessing
- **Machine Learning**: Random Forest classifier with stratified sampling
- **Visualization**: Confusion matrix and classification metrics
- **Testing**: Comprehensive unit tests with CI/CD pipeline
- **Containerized**: Docker support for easy deployment

## 📁 Project Structure

```
├── model.py              # Streamlit web application
├── src/
│   └── utils.py          # Core ML functions (data processing, training, plotting)
├── tests/
│   └── test_utlis.py     # Unit tests for all utility functions
├── .github/workflows/
│   └── ci.yml            # GitHub Actions CI pipeline
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
└── README.md           # This file
```

## 🚀 Quick Start

### Option 1: Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the web app
streamlit run model.py
```

### Option 2: Run with Docker
```bash
# Build the container
docker build -t penguin-classifier .

# Run the application
docker run -p 8501:8501 penguin-classifier
```

Then open your browser to `http://localhost:8501`

## 🧪 Testing

Run the unit tests:
```bash
# Install pytest (if not already installed)
pip install pytest

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v
```

## 📊 Model Details

- **Algorithm**: Random Forest Classifier (100 estimators)
- **Features**: 
  - Bill length (mm)
  - Bill depth (mm) 
  - Flipper length (mm)
  - Body mass (g)
- **Target**: Sex (male/female)
- **Data Split**: 80% training, 20% testing (stratified)
- **Evaluation**: Classification report and confusion matrix

## 🔄 CI/CD Pipeline

The project includes automated testing with GitHub Actions:
- Runs on every push and pull request
- Tests all utility functions
- Ensures code quality and reliability

## 📦 Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **palmerpenguins**: Dataset source
- **pytest**: Testing framework

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.
