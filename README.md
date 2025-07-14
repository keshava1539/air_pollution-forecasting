## Pollution Level Prediction using Multivariate LSTM
## 📊 Overview
-This project focuses on building a Long Short-Term Memory (LSTM) neural network to forecast future pollution levels based on multivariate time series data. It incorporates comprehensive data preprocessing, advanced feature engineering, and robust model training techniques to effectively capture temporal dependencies and predict environmental conditions.

## 🧠 Project Highlights
-Multivariate Time Series Forecasting: Predicts pollution based on diverse environmental factors.

-Deep Learning with LSTMs: Utilizes a powerful recurrent neural network architecture for sequential data.

-Advanced Data Preprocessing:

-Cyclical Feature Engineering: Transforms time components (hour, month) to capture their periodic nature.

-One-Hot Encoding: Handles categorical wind direction data.

-Min-Max Scaling: Normalizes all input features for optimal model performance.

-Efficient Data Pipelining: Employs tf.keras.preprocessing.sequence.TimeseriesGenerator for memory-efficient batching of sequences.

-Robust Training: Includes EarlyStopping and ReduceLROnPlateau callbacks to enhance model stability and prevent overfitting.

-Comprehensive Evaluation: Assesses model performance using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R 
2
 ) Score.

-Visualization: Provides plots for loss, actual vs. predicted values, and residual analysis.

## 🔍 Techniques Used
-Python: Programming language

-Pandas: Data manipulation and analysis

-NumPy: Numerical computing

-TensorFlow/Keras: Deep Learning framework for LSTM model implementation

-Scikit-learn: Data preprocessing (MinMaxScaler, OneHotEncoder) and metrics

-Matplotlib, Seaborn: Data visualization

-Time Series Analysis: Sequence generation for LSTMs, understanding temporal dependencies.

## 📁 Files
-LSTM-Multivariate_pollution.csv: The primary dataset used for training and evaluation.

-model.ipynb: The main Jupyter Notebook containing all the code for data loading, preprocessing, model building, training, and evaluation.
