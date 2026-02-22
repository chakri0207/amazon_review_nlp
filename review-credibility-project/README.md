# Online Review Credibility Prediction

This project predicts whether an online review is **Credible** or **Fake** using Machine Learning.

## Dataset
- Fake Reviews Dataset (Kaggle)
- Labels:
  - CG → Fake (Computer Generated)
  - OR → Credible (Original Review)

## Models
- TF-IDF (text feature extraction)
- Logistic Regression
- Linear SVM

## Structure
- `data/` → dataset
- `notebooks/`
  - `training.ipynb` → model training & saving
  - `visualization.ipynb` → analysis plots
- `models/` → saved models
- `app.py` → Gradio interface

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train models:
   - Run `notebooks/training.ipynb`
3. Visualize:
   - Run `notebooks/visualization.ipynb`
4. Launch app:
   ```bash
   python app.py
   ```

## Output
- Predicts whether a review is **Credible** or **Fake**
- Accuracy target: 80–90%
