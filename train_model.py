import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def train_career_predictor():
    print("CSV data loading..........")
    # 1. Dataset ko load karo
    df = pd.read_csv("student_careers_dataset.csv")
    
    # 2. X (Inputs) aur y (Output/Label) ko alag karo
    # Inputs: Marks aur Interest scores
    X = df.drop(columns=["target_career"])
    # Output: Jo career hume predict karna hai
    y = df["target_career"]
    
    # 3. Data ko do hisso mein baantein: 80% Training ke liye, 20% Testing ke liye
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Machine Learning Model (Random Forest) is training...")
    # 4. Model ko select aur train (fit) karo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Model ki accuracy check karo test data par
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f" Model Training completed ! Accuracy: {accuracy * 100:.2f}%")
    
    # 6. Is trained model ko ek file mein save (Pickle) kar lo taaki hum ise API mein use kar sakein
    with open("career_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    print(" Model 'career_model.pkl' saved !")

if __name__ == "__main__":
    train_career_predictor()