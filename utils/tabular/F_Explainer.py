import shap

class Explainer:
    def __init__(self, top_n=5):
        self.top_n = top_n

    def explain(self, model, x, y_hat, class_names=None):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(x)

        feature_names = list(x.columns)
        explanations = []

        for i, predicted_class in enumerate(y_hat):
            if shap_values.values.ndim == 3:
                single_shap = shap_values[i, :, int(predicted_class)]
            else:
                single_shap = shap_values[i]

            class_label = (
                class_names[int(predicted_class)]
                if class_names is not None
                else str(int(predicted_class))
            )

            shap_array = single_shap.values
            feature_values = single_shap.data

            ordered_indices = sorted(
                range(len(shap_array)), key=lambda j: abs(shap_array[j]), reverse=True
            )

            top_indices = ordered_indices[: self.top_n]
            top_contributions = [
                {
                    "feature": feature_names[j],
                    "feature_value": float(feature_values[j]),
                    "shap_value": float(shap_array[j]),
                    "direction": "supports prediction" if shap_array[j] > 0 else "against prediction",
                }
                for j in top_indices
            ]

            explanation = {
                "instance_index": i,
                "predicted_class": class_label,
                "base_value": float(single_shap.base_values),
                "final_value": float(single_shap.base_values + shap_array.sum()),
                "top_features": top_contributions,
            }
            explanations.append(explanation)

        return explanations
