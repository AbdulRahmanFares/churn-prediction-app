# CHURN PREDICTION APP

A Machine Learning web application built with **Streamlit** that predicts whether a bank customer is likely to churn based on customer profile and banking data.

The application uses a trained Neural Network model built with **Keras** and provides a clean two-column UI styled with green (`#86BC24`) and black theme colors.

---

## Live Demo

**Deployed on Streamlit Cloud**

[link]

---

## Features

- Clean two-column user interface
- Styled header and result section (Green `#86BC24` theme)
- Real-time churn probability prediction
- Label Encoding and One-Hot Encoding
- Feature Scaling using StandardScaler
- Neural Network built with Keras
- EarlyStopping and TensorBoard during training
- Managed using `uv`
- Deployed on Streamlit Cloud

---

## Tech Stack

- Python 3.11
- Streamlit
- Pandas
- Scikit-learn
- TensorFlow / Keras
- uv (Python package manager)

---

## Model Architecture

Neural Network Configuration:

- Input Layer
- Hidden Layer 1 → 64 neurons (ReLU)
- Hidden Layer 2 → 32 neurons (ReLU)
- Output Layer → 1 neuron (Sigmoid)
- Optimizer → Adam
- Loss Function → Binary Crossentropy
- Metric → Accuracy
- EarlyStopping enabled

---

## Installation & Setup (Local Development)

### Create Virtual Environment

```bash
uv venv --python 3.11 churn-prediction-app-venv

source churn-prediction-app-venv/bin/activate

uv pip compile requirements.in -o requirements.txt

uv pip install -r requirements.txt

cd churn-prediction-app

streamlit run app/app.py
```

## Training the Model

```bash
python train.py
```

## Issues and Contributions

If you encounter any issues or have suggestions for improvement, feel free to [open an issue](https://github.com/AbdulRahmanFares/YOLOv8-PersonTracker/issues) or submit a pull request.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you find this project helpful or would like to contribute to its continued development, consider supporting me with a coffee! Your support is invaluable.

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/farazzrahman)
