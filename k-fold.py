import datetime
import shutil
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold

# ================= CONFIGURATION (Updated) =================
BASE_DIR = '/content/drive/MyDrive/PK-YOLO-main'

# Jis dataset par K-Fold chalana hai:
# 'coronal_t1wce_2_class', 'axial_t1wce_2_class', 'sagittal_t1wce_2_class'
DATASET_NAME = 'axial_t1wce_2_class' 

# Detection Model Config
model_cfg = f'{BASE_DIR}/models/detect/pk-yolo.yaml' 

# --- FIXED WEIGHTS PATH ---
# Hum pichli training ka 'best.pt' use karenge taaki transfer learning ho
weights = f'{BASE_DIR}/runs/train/coronal_refined_final7/weights/best.pt'
# Agar ye file na mile to fallback ke liye neechay check lagaya hai
# --------------------------

# Training Parameters
k_folds = 5
img_size = 640
batch_size = 4
epochs = 100 
project_name = f'kfold_{DATASET_NAME}'
# ==========================================================

# --- Safety Check for Weights ---
if not os.path.exists(weights):
    print(f"❌ Error: Weights file nahi mili!\nPath: {weights}")
    print("👉 Please 'weights' variable ko kisi valid .pt file se update karein.")
    exit()
else:
    print(f"✅ Weights Found: {weights}")

# --- Paths Setup ---
dataset_images_path = f'{BASE_DIR}/datasets/brain_tumor/{DATASET_NAME}/images'
original_yaml_path = f'{BASE_DIR}/datasets/brain_tumor/{DATASET_NAME}/{DATASET_NAME}.yaml'

print(f"📂 Dataset: {DATASET_NAME}")
print(f"🖼️ Images: {dataset_images_path}")

# 1. Image List Generate Karna
dataset_path_obj = Path(dataset_images_path)
image_files = sorted(
    list(dataset_path_obj.rglob('*.jpg')) + 
    list(dataset_path_obj.rglob('*.png')) + 
    list(dataset_path_obj.rglob('*.jpeg'))
)

if len(image_files) == 0:
    print(f"❌ Error: Koi images nahi milin! Path check karein.")
    exit()

print(f"✅ Total Images: {len(image_files)}")

# 2. K-Fold Loop
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
    current_fold = fold + 1
    print(f"\n{'='*40}")
    print(f"🚀 Processing Fold {current_fold}/{k_folds}")
    print(f"{'='*40}")
    
    # --- Step A: Train/Val Lists ---
    train_files = [str(image_files[i]) for i in train_idx]
    val_files = [str(image_files[i]) for i in val_idx]
    
    # Fold folder setup
    fold_dir = Path(f'{BASE_DIR}/runs/{project_name}/fold_{current_fold}')
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    train_txt_path = fold_dir / 'train.txt'
    val_txt_path = fold_dir / 'val.txt'
    
    with open(train_txt_path, 'w') as f:
        f.write('\n'.join(train_files))
    with open(val_txt_path, 'w') as f:
        f.write('\n'.join(val_files))
        
    # --- Step B: Data YAML ---
    if os.path.exists(original_yaml_path):
        with open(original_yaml_path, 'r') as f:
            data_dict = yaml.safe_load(f)
            nc = data_dict.get('nc', 2)
            names = data_dict.get('names', ['0', '1'])
    else:
        print(f"⚠️ YAML missing, using defaults.")
        nc = 2
        names = ['Negative', 'Positive']

    fold_yaml_data = {
        'path': str(fold_dir.absolute()), 
        'train': str(train_txt_path.absolute()),
        'val': str(val_txt_path.absolute()),     
        'nc': nc,
        'names': names
    }
    
    fold_data_yaml = fold_dir / 'data_fold.yaml'
    with open(fold_data_yaml, 'w') as f:
        yaml.dump(fold_yaml_data, f)
        
    # --- Step C: Run Training ---
    run_name = f"fold_{current_fold}_run"
    
    cmd = (
        f"python '{BASE_DIR}/train_dual.py' "
        f"--batch {batch_size} "
        f"--epochs {epochs} "
        f"--data '{fold_data_yaml}' "
        f"--weights '{weights}' "
        f"--cfg '{model_cfg}' "
        f"--project '{BASE_DIR}/runs/{project_name}' "
        f"--name '{run_name}' "
        f"--device 0 "
        f"--patience 10 "
        f"--exist-ok" 
    )
    
    print(f"Starting Training for Fold {current_fold}...")
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"❌ Error in Fold {current_fold}. Check logs above.")
        break

    # --- Step D: Read Results ---
    results_csv = Path(f'{BASE_DIR}/runs/{project_name}/{run_name}/results.csv')
    
    if results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            df.columns = [c.strip() for c in df.columns]
            
            # Find best mAP50 epoch
            best_idx = df['metrics/mAP50(B)'].idxmax()
            best_map50 = df.loc[best_idx, 'metrics/mAP50(B)']
            best_map50_95 = df.loc[best_idx, 'metrics/mAP50-95(B)']
            
            print(f"✅ Fold {current_fold} Best mAP50: {best_map50:.4f}")
            fold_metrics.append({
                'Fold': current_fold,
                'mAP50': best_map50,
                'mAP50-95': best_map50_95
            })
        except Exception as e:
             print(f"⚠️ Error reading CSV: {e}")
    else:
        print(f"⚠️ Results CSV not found for Fold {current_fold}")

# ================= FINAL REPORT =================
print("\n" + "="*40)
print(f"📊 K-FOLD FINAL RESULTS: {DATASET_NAME}")
print("="*40)

if len(fold_metrics) > 0:
    results_df = pd.DataFrame(fold_metrics)
    print(results_df)
    print("-" * 30)
    print(f"🏆 Average mAP50:    {results_df['mAP50'].mean():.4f} (+/- {results_df['mAP50'].std():.4f})")
    print(f"🏆 Average mAP50-95: {results_df['mAP50-95'].mean():.4f}")
    
    save_path = f'{BASE_DIR}/runs/{project_name}/kfold_summary.csv'
    results_df.to_csv(save_path, index=False)
    print(f"\nSummary saved to: {save_path}")
else:
    print("No results collected.")