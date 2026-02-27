import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import streamlit as st


def data_transformation_for_model(df):
    
    raw_df = df.copy()

    # ['DDoS' 'Intrusion' 'Malware']
    smote_malware = raw_df[raw_df['target']==2]
    smote_ddos = raw_df[raw_df['target']==0]
    smote_intrusion = raw_df[raw_df['target']==1]

    return smote_malware, smote_ddos, smote_intrusion

def model_building(malware, ddos, intrusion, preprocessor_path, model_path, pca_path, le_path):
    
    try:
        # Load the trained MLP model
        with open(model_path, 'rb') as f:
            test_mlp = pickle.load(f)
        print("MLP Model loaded")
        
        # Load PCA transformer
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        print("PCA transformer loaded")
        
        # Load preprocessor
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        print("Preprocessor loaded")
        
        # Load label encoder
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
        print("Label encoder loaded")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have saved the model using save_best_mlp_model.py")
        exit(1)


    malware_x = malware.drop('target', axis=1)
    malware_y_encoded = malware['target']

    ddos_x = ddos.drop('target', axis=1)
    ddos_y_encoded = ddos['target']

    intrusion_x = intrusion.drop('target', axis=1)
    intrusion_y_encoded = intrusion['target']

    # # transform each dataset
    # malware_y_encoded = le.transform(malware_y)
    # ddos_y_encoded = le.transform(ddos_y)
    # intrusion_y_encoded = le.transform(intrusion_y)

    # Preprocess each test set
    print("\nPreprocessing Malware test set...")
    X_malware_processed = preprocessor.transform(malware_x)
    print(f"  Shape after preprocessing: {X_malware_processed.shape}")

    print("Preprocessing DDoS test set...")
    X_ddos_processed = preprocessor.transform(ddos_x)
    print(f"  Shape after preprocessing: {X_ddos_processed.shape}")

    print("Preprocessing Intrusion test set...")
    X_intrusion_processed = preprocessor.transform(intrusion_x)
    print(f"  Shape after preprocessing: {X_intrusion_processed.shape}")

    # Apply PCA
    print("\nApplying PCA...")
    X_malware_pca = pca.transform(X_malware_processed)
    print(f"  Malware shape after PCA: {X_malware_pca.shape}")

    X_ddos_pca = pca.transform(X_ddos_processed)
    print(f"  DDoS shape after PCA: { X_ddos_pca.shape}")

    X_intrusion_pca = pca.transform(X_intrusion_processed)
    print(f"  Intrusion shape after PCA: { X_intrusion_pca.shape}")

    # Predictions
    print("\nSTEP 3: MAKE PREDICTIONS")
    print("="*80)

    print("\nPredicting on Malware test set...")
    y_pred_malware = test_mlp.predict(X_malware_pca)
    y_pred_malware_proba = test_mlp.predict_proba(X_malware_pca)
    print(f"  Predictions made: {len(y_pred_malware)} samples")

    print("Predicting on DDoS test set...")
    y_pred_ddos = test_mlp.predict(X_ddos_pca)
    y_pred_ddos_proba = test_mlp.predict_proba(X_ddos_pca)
    print(f"  Predictions made: {len(y_pred_ddos)} samples")

    print("Predicting on Intrusion test set...")
    y_pred_intrusion = test_mlp.predict(X_intrusion_pca)
    y_pred_intrusion_proba = test_mlp.predict_proba(X_intrusion_pca)
    print(f"  Predictions made: {len(y_pred_intrusion)} samples")

    # Evaluate predictions
    print("\nSTEP 4: EVALUATE PREDICTIONS")
    print("="*80)

    # Malware
    malware_acc = accuracy_score(malware_y_encoded, y_pred_malware)
    malware_precision = precision_score(malware_y_encoded, y_pred_malware, average='weighted', zero_division=0)
    malware_recall = recall_score(malware_y_encoded, y_pred_malware, average='weighted', zero_division=0)
    malware_f1 = f1_score(malware_y_encoded, y_pred_malware, average='weighted', zero_division=0)

    print(f"\nMalware Test Set Results:")
    print(f"  Accuracy:  {malware_acc:.4f}")
    print(f"  Precision: {malware_precision:.4f}")
    print(f"  Recall:    {malware_recall:.4f}")
    print(f"  F1-Score:  {malware_f1:.4f}")

    malware = {
    'Accuracy': malware_acc,
    'Precision': malware_precision,
    'Recall': malware_recall,
    'F1-Score': malware_f1
    }

    malware_list = [malware_acc, malware_precision, malware_recall, malware_f1]

    # DDoS
    ddos_acc = accuracy_score(ddos_y_encoded , y_pred_ddos)
    ddos_precision = precision_score(ddos_y_encoded , y_pred_ddos, average='weighted', zero_division=0)
    ddos_recall = recall_score(ddos_y_encoded , y_pred_ddos, average='weighted', zero_division=0)
    ddos_f1 = f1_score(ddos_y_encoded , y_pred_ddos, average='weighted', zero_division=0)

    print(f"\nDDoS Test Set Results:")
    print(f"  Accuracy:  {ddos_acc:.4f}")
    print(f"  Precision: {ddos_precision:.4f}")
    print(f"  Recall:    {ddos_recall:.4f}")
    print(f"  F1-Score:  {ddos_f1:.4f}")

    ddos_list = [ddos_acc, ddos_precision, ddos_recall, ddos_f1]

    ddos = {
    'Accuracy': ddos_acc,
    'Precision': ddos_precision,
    'Recall': ddos_recall,
    'F1-Score': ddos_f1
    }

    # Intrusion
    intrusion_acc = accuracy_score(intrusion_y_encoded , y_pred_intrusion)
    intrusion_precision = precision_score(intrusion_y_encoded, y_pred_intrusion, average='weighted', zero_division=0)
    intrusion_recall = recall_score(intrusion_y_encoded, y_pred_intrusion, average='weighted', zero_division=0)
    intrusion_f1 = f1_score(intrusion_y_encoded, y_pred_intrusion, average='weighted', zero_division=0)

    print(f"\nIntrusion Test Set Results:")
    print(f"  Accuracy:  {intrusion_acc:.4f}")
    print(f"  Precision: {intrusion_precision:.4f}")
    print(f"  Recall:    {intrusion_recall:.4f}")
    print(f"  F1-Score:  {intrusion_f1:.4f}")

    intrusion_list = [intrusion_acc, intrusion_precision, intrusion_recall, intrusion_f1]

    intrusion = {
    'Accuracy': intrusion_acc,
    'Precision': intrusion_precision,
    'Recall': intrusion_recall,
    'F1-Score': intrusion_f1
    }

    # Combine set
    print("\nSTEP 5: COMBINED TEST SET EVALUATION")
    print("="*80)

    # Combine all test data and predictions
    X_test_combined_pca = np.vstack([X_malware_pca, X_ddos_pca, X_intrusion_pca])
    y_test_combined = np.concatenate([malware_y_encoded, ddos_y_encoded, intrusion_y_encoded])
    y_pred_combined = np.concatenate([y_pred_malware, y_pred_ddos, y_pred_intrusion])

    combined_acc = accuracy_score(y_test_combined, y_pred_combined)
    combined_precision = precision_score(y_test_combined, y_pred_combined, average='weighted', zero_division=0)
    combined_recall = recall_score(y_test_combined, y_pred_combined, average='weighted', zero_division=0)
    combined_f1 = f1_score(y_test_combined, y_pred_combined, average='weighted', zero_division=0)
    macro_f1 = f1_score(y_test_combined, y_pred_combined, average='macro')

    combined_list = [combined_acc, combined_precision, combined_recall, combined_f1, macro_f1]

    combined = {
    'Accuracy': combined_acc,
    'Precision': combined_precision,
    'Recall': combined_recall,
    'F1-Score': combined_f1
    }

    print(f"\nCombined Test Set (All 3 Attack Types):")
    print(f"  Accuracy:        {combined_acc:.4f}")
    print(f"  Precision:       {combined_precision:.4f}")
    print(f"  Recall:          {combined_recall:.4f}")
    print(f"  F1-Score:        {combined_f1:.4f}")
    print(f"  Macro Avg F1:    {macro_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test_combined, y_pred_combined)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, 
                yticklabels=le.classes_)
    plt.title('Confusion Matrix - MLP Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    data = {
    'Attack Type': ['Malware', 'DDoS', 'Intrusion', 'Combined'],
    'Accuracy': [malware_acc, ddos_acc, intrusion_acc, combined_acc],
    'Precision': [malware_precision, ddos_precision, intrusion_precision, combined_precision],
    'Recall': [malware_recall, ddos_recall, intrusion_recall, combined_recall],
    'F1-Score': [malware_f1, ddos_f1, intrusion_f1, combined_f1]
    }

    return malware, ddos, intrusion, combined, cm, data


