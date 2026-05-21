import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import json

def main():
    print("Loading data...")
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    
    # 'label' sütununu al
    if 'label' in y_train.columns:
        y = y_train['label']
    else:
        y = y_train.iloc[:, 0]
        
    print(f"X_train shape: {X_train.shape}")
    
    # Eksik veriler için strateji: Medyan kullanımı
    # Test verisinde NaN olma ihtimaline karşı train setinin medyanlarını hesaplayıp ekrana yazdıralım.
    # Bu değerleri preprocess.py içerisinde SABİT (hardcoded) olarak kullanmalıyız.
    medians = X_train.median()
    print("\n--- MISSING VALUE IMPUTATION VALUES FOR preprocess.py ---")
    print("Please hard-code these values into the fillna part in preprocess.py:")
    for col in X_train.columns:
        print(f"'{col}': {medians[col]}")
    print("--------------------------------------------------------------------\n")

    # XGBoost hiperparametreleri
    # Aşırı öğrenmeyi (overfitting) engellemek için max_depth düşük, 
    # subsample ve colsample_bytree 1'den küçük seçilmiştir.
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        n_jobs=-1
    )

    # 5-Katlı Çapraz Doğrulama (Stratified K-Fold)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    acc_scores = []
    f1_scores = []
    
    print("Starting 5-Fold Cross Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y.iloc[val_idx]
        
        # Modeli bu kat için eğit
        model.fit(X_tr, y_tr)
        
        # Tahminler
        preds = model.predict(X_val)
        
        # Metrikleri hesapla
        acc = accuracy_score(y_val, preds)
        mac_f1 = f1_score(y_val, preds, average='macro')
        
        acc_scores.append(acc)
        f1_scores.append(mac_f1)
        
        print(f"Fold {fold+1} - Accuracy: {acc:.4f} | Macro F1: {mac_f1:.4f} | Composite: {(0.5 * acc) + (0.5 * mac_f1):.4f}")

    mean_acc = np.mean(acc_scores)
    mean_f1 = np.mean(f1_scores)
    composite_score = 0.5 * mean_acc + 0.5 * mean_f1
    
    print("\n--- CROSS VALIDATION (CV) RESULTS ---")
    print(f"Mean Accuracy : {mean_acc:.4f}")
    print(f"Mean Macro F1 : {mean_f1:.4f}")
    print(f"Composite Score      : {composite_score:.4f}")
    print("---------------------------------------\n")
    
    # Final modelin tüm veri üzerinde eğitilmesi
    print("Training final model using all training data...")
    final_model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train, y)
    
    # Final modelin kaydedilmesi (ödev formatına uygun olarak Universal Binary JSON .ubj)
    model_filename = 'model.ubj'
    final_model.save_model(model_filename)
    print(f"Model successfully saved as '{model_filename}'.")

if __name__ == "__main__":
    main()
