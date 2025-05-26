# United States Home Price Prediction Model

A time series forecasting system for predicting residential real estate price trends at the ZIP code level using Zillow Home Value Index data and the Chronos-T5-Small transformer model.

## Overview

This research implementation provides ZIP code-level housing market forecasts using historical median home value data from Zillow. The system employs Amazon's Chronos-T5-Small model, a pre-trained time series transformer, to generate predictions for 3, 6, and 12-month forecast horizons.

The approach focuses on market-level trend prediction rather than individual property valuation, which provides more statistically robust results given the aggregated nature of the underlying data source.

## Technical Approach

**Data Source**: Zillow Home Value Index (ZHVI) provides monthly median home values aggregated by ZIP code across the United States.

**Model Architecture**: Chronos-T5-Small is a transformer-based time series forecasting model pre-trained on diverse temporal datasets. The model accepts univariate time series input and generates probabilistic forecasts.

**Forecasting Strategy**: The system treats each ZIP code as an independent time series and generates forecasts using the historical ZHVI sequence for that geographic area.

## Installation

### Setup

1. Clone the repository:
```bash
git clone https://github.com/gil10101/USHPPM.git
cd home-price-predictor
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create environment configuration:
```bash
cp .env.example .env
# Edit .env with your configuration settings
```

5. Download Zillow ZHVI data:
```bash
python scripts/download_data.py
```

## Development

### Project Structure

```
USHPPM/
├── src/                    # Core application code
├── app/                    # FastAPI application
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Unit and integration tests
├── scripts/                # Utility scripts
├── frontend/               # Streamlit web interface
└── data/                   # Data storage
```

### Data Processing Pipeline

The system processes data through several stages:

1. **Raw Data Ingestion**: Download ZHVI data from Zillow Research
2. **Data Validation**: Check for completeness and consistency
3. **Time Series Preparation**: Format data for Chronos model input
4. **Model Inference**: Generate forecasts using pre-trained weights
5. **Post-processing**: Calculate confidence intervals and trend analysis

### Model Performance

The Chronos-T5-Small model has been evaluated on housing market data with the following characteristics:

- **Mean Absolute Percentage Error (MAPE)**: Varies by market volatility and forecast horizon
- **Forecast Horizon Performance**: Generally decreases with longer prediction windows
- **Geographic Coverage**: Performance varies by data availability and market characteristics

## Data Sources

**Zillow Home Value Index (ZHVI)**
- Source: Zillow Research (https://www.zillow.com/research/data/)
- Update Frequency: Monthly
- Geographic Coverage: ZIP codes across the United States
- Methodology: Smoothed, seasonally adjusted median home values
