import requests

BASE = "http://127.0.0.1:5000/"

body = {
    'Pregnancies':9, 
    'Glucose':185, 
    'BloodPressure':166, 
    'SkinThickness':29, 
    'Insulin':150,
    'BMI':26.6, 
    'DiabetesPedigreeFunction':0.651, 
    'Age':60
}
response = requests.post(BASE ,json=body)
print(response.text)
