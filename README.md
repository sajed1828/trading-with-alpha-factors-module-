# 🧠 Clever Trade Bot: Alpha Factor-based Bitcoin Prediction

This is a sample trading model that uses **LSTM** and **Transformer** architectures to predict Bitcoin trends and future prices based on **alpha factors** and **technical indicators**.

## 🚀 Project Overview

- Uses historical **Bitcoin data**.
- Incorporates **alpha factors** and **indicators** to generate predictive signals.
- Applies deep learning models like:
  - Long Short-Term Memory (**LSTM**)
  - Transformer encoder model

You can explore how we extract and apply alpha factors to predict both the **trend** and the **next price** of Bitcoin.

## 📄 Learn More

We were inspired by research like:

- **Zero Factors 101**:  
  [`https://arxiv.org/vc/arxiv/papers/1601/1601.00991v1.pdf`](https://arxiv.org/vc/arxiv/papers/1601/1601.00991v1.pdf)  
  _This paper helps build a conceptual roadmap for factor-based trading._

## 📂 Files in the Project

| File                 | Description                                |
|----------------------|--------------------------------------------|
| `main.py`            | Main script to run training or prediction  |
| `alpha_core.py`      | Core logic for factor extraction           |
| `alpha_zero_101.py`  | Experimental logic inspired by the paper   |
| `2.csv`              | Bitcoin historical data                    |
| `model.pth`          | Saved trained model                        |
| `test.ipynb`         | Jupyter notebook for quick testing         |

## 🛠 How to Run

Make sure you have Python installed, then:

```bash
pip install -r requirements.txt
python main.py

 
