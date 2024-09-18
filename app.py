from flask import Flask, render_template, request
import numpy as np
import joblib  # Thư viện để load mô hình đã lưu

app = Flask(__name__)

# Load mô hình đã huấn luyện
model = joblib.load('best_model.pkl')  
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Lấy dữ liệu từ form nhập vào
        dew_point = float(request.form['dew_point'])
        rel_hum = float(request.form['rel_hum'])
        wind_speed = float(request.form['wind_speed'])
        visibility = float(request.form['visibility'])
        pressure = float(request.form['pressure'])
        weather_condition = int(request.form['weather_condition'])

        # Chuẩn bị dữ liệu đầu vào cho mô hình
        features = np.array([[dew_point, rel_hum, wind_speed, visibility, pressure, weather_condition]])

        # Dự đoán
        prediction = model.predict(features)
        
        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
