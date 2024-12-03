import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class IrisClassifier:
    def __init__(self, file_path):
        """Initialize with dataset file path."""
        self.file_path = file_path
        self.data = None
        self.model = None
        self.label_encoder = LabelEncoder()

    def load_data(self,data):
        """Load the dataset from a CSV file."""
        self.data = data
        print("Data loaded successfully.")
        print(data.head())
    
    def preprocess_data(self):
        """Preprocess the data (split features and encode target)."""
        X = self.data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y = self.data['species']
        # Encode the species column
        y = self.label_encoder.fit_transform(y)
        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Data preprocessing complete.")
        print(f"Classes: {self.label_encoder.classes_}")
    
    def visualize_data(self):
        """Visualize the data."""
        sns.pairplot(self.data, hue="species", markers=["o", "s", "D"])
        plt.show()
    
    def train_model(self):
        """Train a Random Forest classifier."""
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        print("Model training complete.")
    
    def evaluate_model(self):
        """Evaluate the model on the test set."""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:\n", classification_report(self.y_test, y_pred))
        
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def predict(self, new_data):
        """Predict the species for new data."""
        prediction_encoded = self.model.predict(new_data)
        prediction_species = self.label_encoder.inverse_transform(prediction_encoded)
        print(f"Predicted species: {prediction_species[0]}")
        # return prediction_species[0]

# Main Execution
if __name__ == "__main__":
    file_path = "../Data/IRIS.csv"
  
   
