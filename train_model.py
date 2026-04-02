import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from pywt import wavedec
from skimage.feature import hog
from imblearn.over_sampling import SMOTE

# 1. Data Loading and Preparation
def load_images_and_labels(data_path, image_size=(256, 256)):
    """Load ECG images and labels from directory structure"""
    class_names = sorted([f for f in os.listdir(data_path) 
                          if os.path.isdir(os.path.join(data_path, f)) and not f.startswith('.')])
    
    X = []
    y = []
    
    for label in class_names:
        class_path = os.path.join(data_path, label)
        for file in os.listdir(class_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                    X.append(img)
                    y.append(label)
    return np.array(X), np.array(y), class_names

# 2. Feature Extraction
def extract_features(images):
    """Extract DWT, HOG and statistical features from images"""
    features = []
    for img in images:
        # Wavelet features
        coeffs = wavedec(img, 'db1', level=3)
        dwt_features = np.concatenate([c.flatten() for c in coeffs])[:100]
        
        # HOG features
        hog_features = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
        
        # Statistical features
        stats = [np.mean(img), np.std(img), np.median(img)]
        
        features.append(np.concatenate([dwt_features, hog_features, stats]))
    return np.array(features)

# 3. Model Training
def train_models(X_train, y_train):
    """Initialize and train multiple classifiers"""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10,
                                              min_samples_split=5, random_state=42),
        "SVM": SVC(kernel='rbf', C=5, gamma='auto', probability=True, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.01,
                               max_depth=6, subsample=0.8, random_state=42)
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
    return models

# 4. Evaluation
def evaluate_models(models, X_test, y_test, class_names):
    """Evaluate models and return results"""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print(f"\n{name} Performance:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    return results

# 5. Main Training Pipeline
def main_training_pipeline(data_path):
    """Complete training workflow"""
    # Load and preprocess data
    print("Loading images...")
    X, y, class_names = load_images_and_labels(data_path)
    print(f"Loaded {len(X)} images from {len(class_names)} classes")
    
    # Feature extraction
    print("\nExtracting features...")
    X_features = extract_features(X)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 🚨 CRITICAL FIX: Split data FIRST before applying transformations
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Feature scaling (Fit on TRAIN only, transform Train and Test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA (Fit on TRAIN only, transform Train and Test)
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Handle class imbalance (Apply SMOTE to TRAIN only)
    print("\nApplying SMOTE to handle imbalance...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_pca, y_train)
    
    # Train models
    print("\nTraining models...")
    models = train_models(X_train_res, y_train_res)
    
    # Evaluate models
    print("\nEvaluating models...")
    results = evaluate_models(models, X_test_pca, y_test, class_names)
    
    # Save best model and artifacts
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name} (Accuracy: {results[best_model_name]:.4f})")
    
    artifacts = {
        'model': best_model,
        'encoder': le,
        'scaler': scaler,
        'pca': pca,
        'class_names': class_names
    }
    
    with open('ecg_model_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("Saved model artifacts to 'ecg_model_artifacts.pkl'")

# Example Usage
if __name__ == "__main__":
    # Ensure this path matches the folder containing your ECG classes
    DATA_PATH = r"C:\Users\acer\Downloads\project\ECG_IMAGES_DATASET"
    
    # 1. Run the training pipeline to generate the PKL file
    main_training_pipeline(DATA_PATH)