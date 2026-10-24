# Smart Inventory and Demand Forecasting System

## Project Overview

The Smart Inventory and Demand Forecasting System is a Python-based web application developed using Streamlit. It analyzes historical product sales and inventory data to estimate future demand and generate inventory replenishment recommendations.

## Objectives

- Analyze product-wise sales data.
- Forecast future demand using historical sales patterns.
- Monitor inventory levels and estimated inventory value.
- Identify products that may require replenishment.
- Allow users to upload CSV files and download recommendations.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly

## Features

- CSV data upload and validation
- Inventory summary dashboard
- Historical sales visualization
- Demand forecasting
- Inventory status charts
- Reorder recommendations
- Downloadable recommendation reports

## Project Structure

- `app.py` — Main dashboard application
- `requirements.txt` — Python dependencies
- `data/` — Example and prepared test datasets
- `screenshots/` — Dashboard screenshots
- `results/` — Downloaded recommendation results

## How to Run

1. Install Python.
2. Open a terminal in the project folder.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Start the application:

   `python -m streamlit run app.py`

5. Open the local URL displayed in the terminal.

## Dataset

The application supports CSV files containing dates, products, quantities sold, stock levels, unit prices, and supplier lead times.

The demonstration includes synthetic sample data and a prepared historical retail dataset. Historical transaction data does not represent live inventory.

## Forecasting Method

The application uses a Random Forest regression model when sufficient historical data is available and uses a recent-sales average as a fallback in other cases.

Forecasts and reorder recommendations are estimates and should be validated before operational use.

## Limitations

- Forecast accuracy depends on the quality and amount of historical data.
- Supplier lead time may be assumed when it is not supplied.
- Historical inventory does not guarantee current stock availability.
- The project is a demonstration application, not a live enterprise inventory integration.

## Future Enhancements

- Forecast accuracy evaluation using a holdout test set
- Supplier and purchase-order integration
- Automated database storage
- More advanced forecasting methods
- User authentication and role-based access

## Conclusion

The project demonstrates how data analysis, machine learning, and interactive visualizations can support inventory monitoring and demand planning.
