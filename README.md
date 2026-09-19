# Car Price Prediction

A machine learning project that predicts the estimated price of a used car based on its specifications. The project includes data preprocessing, exploratory data analysis, model training, and a FastAPI web interface with a simple HTML frontend.

## Project Overview

This project uses a car sales dataset containing information such as:

- Manufacturer and model
- Production year
- Car category
- Fuel type
- Engine volume
- Mileage
- Number of cylinders
- Gearbox type
- Drive wheels
- Interior type
- Wheel position
- Color
- Number of airbags
- Levy/tax value

The target variable is the car **Price**.

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Remove duplicate records.
3. Convert categorical and numeric values into usable formats.
4. Clean outliers using the IQR method.
5. Create additional features, including car age.
6. Encode categorical features using label encoding and one-hot encoding.
7. Scale numerical features using `StandardScaler`.
8. Train a `RandomForestRegressor` model.
9. Save the trained model and preprocessing objects using `pickle`.
10. Serve predictions through a FastAPI endpoint.

## Model Performance

The trained Random Forest model achieved the following validation results in the training notebook:

- **RMSE:** `5607.76`
- **R² Score:** `0.7611`

These results may vary depending on the dataset, preprocessing changes, and model configuration.

## Repository Structure

```text
Car_Price_Prediction/
├── api/
│   ├── index.html             # Frontend form for entering car information
│   └── main.py                # FastAPI application and prediction endpoint
├── nootbooks/
│   ├── EDA.ipynb              # Exploratory data analysis
│   ├── first_stage.ipynb      # Initial project experiments
│   ├── training.ipynb         # Model training experiments
│   ├── training final.ipynb   # Final training workflow
│   └── training-target-encoding.ipynb
├── scripts/
│   └── preprocessing.py       # Data cleaning and feature engineering functions
├── models/                    # Saved model and preprocessing files
│   ├── model.pkl
│   ├── one_hot_encoder.pkl
│   ├── label_encoders.pkl
│   └── scaler.pkl
├── data/                      # Dataset files
│   └── car_price_prediction.csv
└── README.md
```

> Note: The `models/` and `data/` directories are required by the notebooks and API. Make sure they exist locally before running the project.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Pydantic
- Uvicorn
- HTML
- Bootstrap
- JavaScript

## Installation

Clone the repository:

```bash
git clone https://github.com/Ibraheem902/Car_Price_Prediction.git
cd Car_Price_Prediction
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn fastapi uvicorn pydantic jupyter
```

## Training the Model

1. Place the dataset at:

   ```text
   data/car_price_prediction.csv
   ```

2. Open the final training notebook:

   ```bash
   jupyter notebook "nootbooks/training final.ipynb"
   ```

3. Run the notebook cells in order.

The notebook will generate the preprocessing files and trained model inside the `models/` directory:

- `model.pkl`
- `one_hot_encoder.pkl`
- `label_encoders.pkl`
- `scaler.pkl`

## Running the API

From the project root, start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI automatically provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

## Prediction Endpoint

### Request

Send a `POST` request to:

```text
/predict/
```

Example JSON body:

```json
{
  "Levy": 1399,
  "Manufacturer": "LEXUS",
  "Model": "RX 450",
  "Prode_year": 2020,
  "Category": "Jeep",
  "Leather_interior": "Yes",
  "Fuel_type": "Hybrid",
  "Engine_volume": 3.5,
  "Mileage": 186005,
  "Cylinders": 6,
  "Gear_box_type": "Automatic",
  "Drive_wheels": "4x4",
  "Wheel": "Left wheel",
  "Color": "Silver",
  "Airbags": 12
}
```

### Response

```json
{
  "prediction": 24500.0
}
```

The exact prediction depends on the trained model and the provided vehicle information.

## Running the Frontend

After starting the API server, open the frontend file in a browser:

```text
api/index.html
```

The form sends prediction requests to:

```text
http://127.0.0.1:8000/predict/
```

Make sure the FastAPI server is running before submitting the form.

## Important Notes

- The API loads the model and preprocessing files using relative paths from the `models/` directory.
- The API uses `allow_origins=["*"]` for CORS during development. For production, replace it with the specific frontend domain.
- The model should only be used as an estimate and should not be considered a guaranteed market price.
- Input categories must match values known by the encoders saved during training.
