# Linear Regression from Scratch

This project is a hands-on implementation of linear regression using only core Python libraries (e.g., NumPy). The goal is to deeply understand the inner workings of one of the most fundamental machine learning algorithms by building it from the ground up—no scikit-learn or machine learning libraries involved.

---

## Why I’m Doing This

As someone passionate about both software engineering and machine learning, I believe that true understanding comes from building things from scratch. This project allows me to:

- **Demystify** the underlying mathematics behind linear regression (e.g., gradient descent, cost functions).
- **Strengthen** my Python and numerical programming skills.
- **Bridge** theoretical machine learning knowledge with practical implementation.
- **Prepare** for more complex models and ML system design by first mastering the basics.

---

## Key Considerations

### Assumptions
- This model assumes a **linear relationship** between features and target variable.
- All features are **numerical and normalized** (or at least scaled reasonably).
- We're using **Mean Squared Error (MSE)** as the cost function.

### Optimization
- Supports **Batch Gradient Descent** (optionally extendable to Stochastic or Mini-batch).
- Customizable **learning rate** and **number of iterations**.

### Evaluation
- Includes functionality to compute and visualize:
  - Cost over iterations (convergence plot)
  - R² score or Mean Absolute Error

---

## Project Structure

```
linear-regression-scratch/
├── data/                 # Sample CSVs or generated data
├── linear_regression.py  # Core model implementation
├── utils.py              # Helpers for plotting, metrics, etc.
├── test_model.py         # Sample usage / evaluation
└── README.md             # This file
```

---

## ▶️ How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/linear-regression-scratch.git
cd linear-regression-scratch

# Run a demo
python test_model.py
```

---

## Sample Output

- Cost function convergence plot  
- Predicted vs actual line on a test set  

_Add visuals here if available._

---

## Future Extensions

- Add support for multivariate linear regression
- Implement L2 regularization (Ridge Regression)
- Compare performance with scikit-learn
- Support live plotting during training

---

## References

- [A Visual Intro to Linear Regression Math](https://bharathikannann.github.io/blogs/a-visual-intro-to-linear-regression-math/)
- [Linear Regression - A Visual Introduction To (Almost) Everything You Should Know](https://mlu-explain.github.io/linear-regression/)
- Personal notes on linear algebra and optimization
