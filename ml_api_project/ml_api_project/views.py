from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target_encoder = joblib.load(os.path.join(BASE_DIR, "..", "ml-classifier-comparison", "target_encoder.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "..", "ml-classifier-comparison", "feature_encoders.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "..", "ml-classifier-comparison", "scaler.pkl"))


# Load all trained models
model_files = {
    "logistic_regression": os.path.join(BASE_DIR, "..", "ml-classifier-comparison", "logistic_regression_model.pkl"),
    "knn": os.path.join(BASE_DIR, "..", "ml-classifier-comparison", "knn_model.pkl"),
    "decision_tree": os.path.join(BASE_DIR, "..", "ml-classifier-comparison", "decision_tree_model.pkl"),
    "random_forest": os.path.join(BASE_DIR, "..", "ml-classifier-comparison", "random_forest_model.pkl")
}

models = {}
for name, file in model_files.items():
    if os.path.exists(file):
        models[name] = joblib.load(file)



@csrf_exempt
def predict(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            features = data.get("features")
            chosen_model = data.get("model")  # optional

            if not features or not isinstance(features, list):
                return JsonResponse({"error": "Invalid input format"}, status=400)

            # Feature names in order (must match training)
            feature_names = [
                "Age", "gender", "R5", "R4", "R3", "R2", "R1",
                "L1", "L2", "L3", "L4", "L5"
            ]

            # Encode categorical features
            processed = []
            for i, val in enumerate(features):
                col = feature_names[i]
                if col in encoders:
                    le = encoders[col]
                    val = le.transform([val])[0]
                processed.append(val)

            # Scale features
            processed = scaler.transform([processed])[0]

            if chosen_model:
                chosen_model = chosen_model.lower().replace(" ", "_")
                if chosen_model not in models:
                    return JsonResponse({"error": f"Model '{chosen_model}' not found"}, status=400)
                numeric_pred = models[chosen_model].predict([processed])[0]
                label = target_encoder.inverse_transform([numeric_pred])[0]
                return JsonResponse({
                    "model": chosen_model,
                    "input": features,
                    "prediction": label
                })

            all_predictions = {}
            for name, model in models.items():
                numeric_pred = model.predict([processed])[0]
                label = target_encoder.inverse_transform([numeric_pred])[0]
                all_predictions[name] = label

            return JsonResponse({
                "input": features,
                "predictions": all_predictions
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"message": "Send a POST request with 'features' list."})
