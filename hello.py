from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor 
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import StackingRegressor

import numpy as np
import pandas as pd
import joblib


# Đọc dữ liệu từ file csv
file_path = 'Weather Data.csv'
weather_data = pd.read_csv(file_path)
print(weather_data)

# Bỏ cột "Date/Time"
weather_data_cleaned = weather_data.drop(columns=['Date/Time'])

# Mã hóa "Weather" sang dạng số
label_encoder = LabelEncoder()
weather_data_cleaned['Weather'] = label_encoder.fit_transform(weather_data_cleaned['Weather'])
print(weather_data_cleaned)

X = weather_data_cleaned.drop(columns=['Temp_C'])
y = weather_data_cleaned['Temp_C']

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Hiển thị dữ liệu đã được làm sạch và hình dạng của tập huấn luyện/kiểm tra
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Khởi tạo mô hình hồi quy tuyến tính
linear_model = LinearRegression()


# Huấn luyện mô hình
linear_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_linear = linear_model.predict(X_test)

# Đánh giá mô hình
mse_linear = mean_squared_error(y_test, y_pred_linear)
print(f"Linear Regression MSE: {mse_linear}")

# Giả sử mô hình tốt nhất là mô hình hồi quy tuyến tính
joblib.dump(linear_model, 'best_model.pkl')

# Khởi tạo mô hình Lasso
lasso_model = Lasso(alpha=0.1)

# Huấn luyện mô hình
lasso_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_lasso = lasso_model.predict(X_test)

# Đánh giá mô hình
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f"Lasso Regression MSE: {mse_lasso}")

# Khởi tạo MLPRegressor với các tham số phù hợp
mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=50)

# Huấn luyện mô hình
mlp_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_mlp = mlp_model.predict(X_test)

# Đánh giá mô hình
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
print(f"MLPRegressor MSE: {mse_mlp}")

# Khởi tạo Bagging với mô hình hồi quy tuyến tính (hoặc mô hình khác)
bagging_model = BaggingRegressor(base_estimator=LinearRegression(), n_estimators=10, random_state=42)

# Huấn luyện mô hình
bagging_model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred_bagging = bagging_model.predict(X_test)
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
print(f"Bagging MSE: {mse_bagging}")

# Khởi tạo Stacking với các mô hình hồi quy khác nhau
stacking_model = StackingRegressor(
    estimators=[('lr', LinearRegression()), ('lasso', Lasso(alpha=0.1)), ('mlp', MLPRegressor(hidden_layer_sizes=(100,), max_iter=500))],
    final_estimator=LinearRegression()
)

# Huấn luyện mô hình
stacking_model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred_stacking = stacking_model.predict(X_test)
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
print(f"Stacking MSE: {mse_stacking}")