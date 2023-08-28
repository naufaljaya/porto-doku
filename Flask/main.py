from helper import make_prediction
from flask import Flask, request, jsonify, make_response

app = Flask(__name__)

# @app.route("/", methods=["GET"])
# def index():
#     dict = {
#         'message': 'API SUCCESS',
#         'status': 'success',
#         'error': False
#     }
#     response = make_response(jsonify(dict))
#     response.headers['Content-Type'] = 'application/json'
#     response.status_code = 200
#     return response

@app.route("/", methods=["POST"])
def index():
    try:
        data = request.get_json() 
        if data:
            Pregnancies = data.get("Pregnancies") 
            Glucose = data.get("Glucose")
            BloodPressure = data.get("BloodPressure")
            SkinThickness = data.get("SkinThickness")
            Insulin = data.get("Insulin")
            BMI = data.get("BMI")
            DiabetesPedigreeFunction = data.get("DiabetesPedigreeFunction")
            Age = data.get("Age")

            prediction, probabilty, advice = make_prediction(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
            
            if prediction == False and not (advice == "Kerja bagus" and prediction == False):
                
                return jsonify({
                    "error": "Anomali data. Apakah data sudah benar?"
                })
            
            return jsonify({
                "prediction":prediction,
                "probabilty":probabilty,
                "advice":advice
                })
        else:
            return "Wrong input"
    except Exception as e:
        return jsonify({"error":str(e)})

if __name__ == "__main__":
    app.run(debug=True)