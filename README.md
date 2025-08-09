# Weather Forecasting using LSTM

This project predicts future weather conditions—specifically temperature—using a **Long Short-Term Memory (LSTM)** neural network.  
It leverages historical weather data, preprocesses it, and applies deep learning to model time-series patterns.


## 🚀 Features
- **Data Preprocessing**  
  - Label encoding for categorical features (e.g., `Summary`)
  - Normalization of features with `MinMaxScaler`
- **Time Series Preparation**  
  - Sliding window approach for supervised learning  
  - Reshaping data for LSTM input `(samples, timesteps, features)`
- **Model Architecture**  
  - Sequential Keras model with:
    - LSTM layer(s)
    - Dropout for regularization
    - Dense output layer
- **Evaluation**  
  - Mean Squared Error (MSE) for model performance  
  - Visual comparison between predicted and actual values

## 📊 Dataset
The project uses the **`weatherHistory.csv`** dataset containing hourly weather data such as:
- Summary
- Temperature
- Humidity
- Pressure
- Wind Speed
- Visibility
- Apparent Temperature

## 🛠️ Requirements
Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn keras tensorflow 
```
## 📈 Results
-Predicts temperature based on past weather patterns

-Generates plots comparing actual vs predicted temperature

-Achieves low error metrics depending on hyperparameters and training data split
