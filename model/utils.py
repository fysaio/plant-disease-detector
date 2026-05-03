import torch
from torchvision import models

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato_Early_blight",
    "Tomato_healthy"
]

def load_model(model_path):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_disease(model, image_tensor, threshold=0.75):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)
        confidence = top_prob.item()
        class_idx = top_class.item()

        if confidence < threshold:
            return "Unknown or Uncertain", round(confidence * 100, 2)

        return CLASS_NAMES[class_idx], round(confidence * 100, 2)

def batch_predict(model, image_tensors, threshold=0.75):
    results = []
    for tensor in image_tensors:
        label, confidence = predict_disease(model, tensor, threshold)
        results.append({ "label": label, "confidence": confidence })
    return results

def load_and_predict(model_path, image_tensor, threshold=0.75):
    model = load_model(model_path)
    return predict_disease(model, image_tensor, threshold)

def get_top_predictions(model, image_tensor, top_k=3):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)
        predictions = []
        for i in range(top_k):
            idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            predictions.append({
                "label": CLASS_NAMES[idx],
                "confidence": round(prob * 100, 2)
            })
        return predictions
