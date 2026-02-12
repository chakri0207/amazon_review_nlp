import gradio as gr
import joblib
import os

# ================================
# Load saved TF-IDF and model safely
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf.pkl"))
model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.pkl"))

# ================================
# Prediction function
# ================================
def predict_review(review_text):
    if review_text.strip() == "":
        return "⚠️ Please enter a review text."

    review_vector = tfidf.transform([review_text])
    prediction = model.predict(review_vector)[0]

    if prediction == 1:
        return "✅ Credible Review"
    else:
        return "❌ Fake Review"

# ================================
# Gradio Interface
# ================================
interface = gr.Interface(
    fn=predict_review,
    inputs=gr.Textbox(lines=5, placeholder="Enter an online review here..."),
    outputs="text",
    title="Online Review Credibility Prediction",
    description="Predict whether a review is Credible or Fake using ML"
)

# ================================
# Launch App
# ================================
if __name__ == "__main__":
    interface.launch(share=True)
