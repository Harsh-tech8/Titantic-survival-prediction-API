from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

model = joblib.load("Titanic.joblib")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return "Api is running"

    try:
        data = request.get_json()

        print("Received:", data)

        features = [data["features"]]

        prediction = model.predict(features)

        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
