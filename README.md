# **Iris Flower Classification**

This repository contains a project to classify Iris flowers into three species (*Iris-setosa*, *Iris-versicolor*, and *Iris-virginica*) based on their sepal and petal measurements using machine learning.

---

## **Project Overview**

The goal of this project is to build a machine learning model to accurately classify Iris flowers using the Iris dataset. The dataset includes 150 samples, each with the following attributes:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
- Species (Target Variable)

The project demonstrates:
- Exploratory Data Analysis (EDA)
- Model training and evaluation using a Random Forest Classifier
- Performance analysis using metrics and visualizations
- Prediction for new data samples

---

## **Features**
- **Dataset Analysis:** Provides summary statistics and visualizations for better understanding of the data.
- **Data Preprocessing:** Includes target encoding and dataset splitting.
- **Machine Learning Model:** Implements a Random Forest Classifier for classification.
- **Performance Metrics:** Evaluates the model using accuracy, precision, recall, F1-score, and confusion matrix.
- **Prediction:** Supports prediction for new flower measurements.

---

## **Dataset**
The dataset used in this project is the Iris dataset from the [Iris Flower Dataset]([https://www.kaggle.com/code/arshid/support-vector-machine-on-iris-flower-dataset/])). It includes 150 samples with 5 columns:  
1. Sepal Length  
2. Sepal Width  
3. Petal Length  
4. Petal Width  
5. Species (Target)

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/iris-flower-classification.git
   cd iris-flower-classification
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### **Step 1: Run the Script**
Run the main Python script to train the model and perform predictions:
```bash
python main.py
```

### **Step 2: Predict New Samples**
Update the `main.py` file with your own sample data for prediction, e.g.:
```python
new_sample = [[5.0, 3.4, 1.5, 0.2]]
classifier.predict(new_sample)
```

---

## **Project Structure**

```
├── data/
│   └── iris.csv                # Dataset file
├── Notebooks/
│   ├── Iris_classifier.ipynb      # Implementation of IrisClassifier class
│   
├── Scripts/
│   ├── __init__.py
│   └── IrisFlowerClassifier.py                  
├── main.py                     # Main script to execute the project
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
```

---

## **Results**

- **Accuracy:** 100%
- **Performance Metrics:**  
  - Precision, Recall, F1-score: 1.00 for all classes  
- **Confusion Matrix:**  
  Perfect classification with no misclassifications.

### Example Prediction
**Input:**  
```python
[5.0, 3.4, 1.5, 0.2]
```
**Output:**  
```
Predicted Species: Iris-setosa
```

---

## **Technologies Used**

- Python
- Libraries:
  - `pandas` for data manipulation
  - `seaborn` and `matplotlib` for visualization
  - `scikit-learn` for model building and evaluation

---

## **Future Work**
- Deploy the model using a web framework like Flask or Django.
- Test the model on larger and more complex datasets.
- Optimize the model further with hyperparameter tuning.

---

## **Contributing**
Contributions are welcome!  
To contribute:
1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Contact**
For any inquiries or suggestions, please feel free to reach out:
- **Email:** danielhailay72@gmail.com
- **GitHub:** [Daniel Hailay](https://github.com/dani6566)

--- 

Let me know if you'd like adjustments or additional sections!
