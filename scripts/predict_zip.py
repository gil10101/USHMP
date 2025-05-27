#!/usr/bin/env python3
"""
ZIP Code Home Price Prediction Script

This script allows users to generate home price forecasts for specific ZIP codes
using the Chronos T5 model. It provides an easy command-line interface for
end users to get predictions without needing to understand the underlying code.

Usage:
    python scripts/predict_zip.py --zip 90210 --months 6
    python scripts/predict_zip.py --zip 77449 --months 3,6,12 --confidence 0.9
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from data_processor import ZillowDataProcessor
from model import ChronosT5Model
from utils import validate_zip_code, format_currency, setup_logging

logger = logging.getLogger(__name__)


class ZipCodePredictor:
    """
    Easy-to-use ZIP code home price predictor.
    
    This class combines data processing and model inference to provide
    simple predictions for specific ZIP codes.
    """
    
    def __init__(self, data_path: Optional[str] = None, model_cache_dir: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            data_path: Path to ZHVI data file
            model_cache_dir: Directory to cache the model
        """
        self.data_path = data_path or "data/raw/zhvi_zip.csv"
        self.model_cache_dir = model_cache_dir or "data/model_cache"
        
        # Initialize components
        self.data_processor = None
        self.model = None
        self.data_loaded = False
        
    def load_data(self) -> bool:
        """Load and process the ZHVI data."""
        try:
            logger.info("Loading ZHVI data...")
            self.data_processor = ZillowDataProcessor(self.data_path)
            self.data_processor.load_data()
            self.data_loaded = True
            logger.info("Data loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load the Chronos T5 model."""
        try:
            logger.info("Loading Chronos T5 model...")
            self.model = ChronosT5Model(cache_dir=self.model_cache_dir)
            self.model.load_model()
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_zip_code(
        self,
        zip_code: str,
        forecast_months: List[int],
        confidence_level: float = 0.8
    ) -> Dict[str, Any]:
        """
        Generate predictions for a specific ZIP code.
        
        Args:
            zip_code: 5-digit ZIP code
            forecast_months: List of forecast horizons in months
            confidence_level: Confidence level for intervals (0-1)
            
        Returns:
            Dictionary containing predictions and metadata
        """
        # Validate inputs
        if not validate_zip_code(zip_code):
            raise ValueError(f"Invalid ZIP code format: {zip_code}")
        
        if not self.data_loaded:
            if not self.load_data():
                raise RuntimeError("Failed to load data")
        
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Failed to load model")
        
        # Get time series data for ZIP code
        try:
            time_series_data = self.data_processor.get_zip_time_series(zip_code)
            if time_series_data is None or len(time_series_data) < 12:
                raise ValueError(f"Insufficient data for ZIP code {zip_code}")
            
            # Get ZIP code metadata
            zip_info = self.data_processor.get_zip_metadata(zip_code)
            
        except Exception as e:
            raise ValueError(f"Error retrieving data for ZIP code {zip_code}: {e}")
        
        # Generate predictions for each forecast horizon
        predictions = []
        
        for months in forecast_months:
            try:
                logger.info(f"Generating {months}-month forecast for ZIP {zip_code}")
                
                # Get prediction
                result = self.model.predict_single_value(
                    time_series=time_series_data,
                    forecast_horizon=months,
                    confidence_level=confidence_level
                )
                
                # Calculate additional statistics
                current_value = float(time_series_data.iloc[-1])
                predicted_value = result['predicted_value']
                change_amount = predicted_value - current_value
                change_percent = (change_amount / current_value) * 100
                
                predictions.append({
                    "forecast_months": months,
                    "current_value": current_value,
                    "predicted_value": predicted_value,
                    "change_amount": change_amount,
                    "change_percent": change_percent,
                    "confidence_interval": result['confidence_interval'],
                    "confidence_level": confidence_level,
                    "standard_deviation": result['std']
                })
                
            except Exception as e:
                logger.error(f"Failed to generate {months}-month forecast: {e}")
                predictions.append({
                    "forecast_months": months,
                    "error": str(e)
                })
        
        # Compile final result
        result = {
            "zip_code": zip_code,
            "zip_info": zip_info,
            "data_points": len(time_series_data),
            "data_start_date": str(time_series_data.index[0]),
            "data_end_date": str(time_series_data.index[-1]),
            "predictions": predictions,
            "model_info": self.model.get_model_info() if self.model else None
        }
        
        return result


def print_prediction_results(results: Dict[str, Any]) -> None:
    """Print prediction results in a user-friendly format."""
    
    print("\n" + "="*80)
    print(f"HOME PRICE FORECAST FOR ZIP CODE {results['zip_code']}")
    print("="*80)
    
    # ZIP code information
    zip_info = results.get('zip_info', {})
    if zip_info:
        print(f"\nLocation: {zip_info.get('City', 'Unknown')}, {zip_info.get('State', 'Unknown')}")
        if 'Metro' in zip_info:
            print(f"Metro Area: {zip_info['Metro']}")
        if 'CountyName' in zip_info:
            print(f"County: {zip_info['CountyName']}")
    
    # Data information
    print(f"\nData Coverage:")
    print(f"  Historical Data Points: {results['data_points']} months")
    print(f"  Data Period: {results['data_start_date']} to {results['data_end_date']}")
    
    # Predictions
    print(f"\nFORECAST RESULTS:")
    print("-" * 80)
    
    for pred in results['predictions']:
        if 'error' in pred:
            print(f"\n{pred['forecast_months']}-Month Forecast: ERROR - {pred['error']}")
            continue
        
        months = pred['forecast_months']
        current = pred['current_value']
        predicted = pred['predicted_value']
        change_amt = pred['change_amount']
        change_pct = pred['change_percent']
        ci_lower, ci_upper = pred['confidence_interval']
        confidence = pred['confidence_level']
        
        print(f"\n{months}-Month Forecast:")
        print(f"  Current Value:     {format_currency(current)}")
        print(f"  Predicted Value:   {format_currency(predicted)}")
        print(f"  Expected Change:   {format_currency(change_amt)} ({change_pct:+.1f}%)")
        print(f"  Confidence Range:  {format_currency(ci_lower)} - {format_currency(ci_upper)}")
        print(f"  Confidence Level:  {confidence*100:.0f}%")
        
        # Risk assessment
        if change_pct > 5:
            trend = "Strong Appreciation Expected"
        elif change_pct > 2:
            trend = "Moderate Appreciation Expected"
        elif change_pct > -2:
            trend = "Stable Market Expected"
        elif change_pct > -5:
            trend = "Moderate Decline Expected"
        else:
            trend = "Significant Decline Expected"
        
        print(f"  Market Outlook:    {trend}")
    
    print("\n" + "="*80)
    print("IMPORTANT DISCLAIMERS:")
    print("• These predictions are based on historical data and statistical models")
    print("• Actual home prices may vary significantly due to local market conditions")
    print("• Economic factors, policy changes, and market events can affect outcomes")
    print("• This forecast should not be used as the sole basis for financial decisions")
    print("="*80)


def save_results(results: Dict[str, Any], output_file: str) -> None:
    """Save results to a JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate home price forecasts for specific ZIP codes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/predict_zip.py --zip 90210 --months 6
  python scripts/predict_zip.py --zip 77449 --months 3,6,12 --confidence 0.9
  python scripts/predict_zip.py --zip 11368 --months 12 --output results.json
        """
    )
    
    parser.add_argument(
        "--zip", 
        required=True,
        help="5-digit ZIP code to predict (e.g., 90210)"
    )
    
    parser.add_argument(
        "--months",
        default="6",
        help="Forecast horizons in months, comma-separated (e.g., 3,6,12). Default: 6"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.8,
        help="Confidence level for prediction intervals (0-1). Default: 0.8"
    )
    
    parser.add_argument(
        "--data-path",
        help="Path to ZHVI data file. Default: data/raw/zhvi_zip.csv"
    )
    
    parser.add_argument(
        "--model-cache",
        help="Directory to cache the model. Default: data/model_cache"
    )
    
    parser.add_argument(
        "--output",
        help="Save results to JSON file (optional)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    try:
        # Parse forecast months
        forecast_months = [int(m.strip()) for m in args.months.split(',')]
        
        # Validate inputs
        if not all(1 <= m <= 60 for m in forecast_months):
            raise ValueError("Forecast months must be between 1 and 60")
        
        if not 0 < args.confidence < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        
        # Initialize predictor
        predictor = ZipCodePredictor(
            data_path=args.data_path,
            model_cache_dir=args.model_cache
        )
        
        # Generate predictions
        logger.info(f"Generating predictions for ZIP code {args.zip}")
        results = predictor.predict_zip_code(
            zip_code=args.zip,
            forecast_months=forecast_months,
            confidence_level=args.confidence
        )
        
        # Display results
        print_prediction_results(results)
        
        # Save results if requested
        if args.output:
            save_results(results, args.output)
        
        logger.info("Prediction completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 