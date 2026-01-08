# FINE TUNE PART 2
import os
import json
import random
import glob

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = "phase3_training"

# 1. CORRECT PATH FOR AZH (Based on your folder check)
AZH_ROOT = os.path.join(BASE_DIR, "datasets/azh_wound/dataset/Train")

# 2. CORRECT PATH FOR DFU (Based on your folder check)
DFU_ROOT = os.path.join(BASE_DIR, "datasets/diabetic-foot-ulcer-dfu/DFU/Patches")

OUTPUT_TRAIN = os.path.join(BASE_DIR, "dataset_specialist_train.json")
OUTPUT_VAL = os.path.join(BASE_DIR, "dataset_specialist_val.json")

# ==========================================
# RICH CLINICAL DEFINITIONS
# ==========================================

AZH_CLASS_MAP = {
    "D": {
        "name": "Diabetic Foot Ulcer",
        "short_desc": "diabetic foot ulcer",
        "full_desc": "This image shows a diabetic foot ulcer, typically located on the foot or heel. Common characteristics include callus formation, slough, and potential signs of neuropathy.",
        "management": "Offloading pressure (specialized footwear/casting), strict glycemic control, debridement of non-viable tissue, and vascular assessment.",
        "risk_factors": "Peripheral neuropathy, peripheral arterial disease, and poor glycemic control."
    },
    "V": {
        "name": "Venous Leg Ulcer",
        "short_desc": "venous leg ulcer",
        "full_desc": "This image shows a venous leg ulcer, typically located on the medial lower leg. Features include irregular borders, hemosiderin staining (brown discoloration), and lipodermatosclerosis.",
        "management": "Compression therapy (multilayer bandaging), leg elevation, and moisture management.",
        "risk_factors": "Chronic venous insufficiency, DVT history, and venous reflux."
    },
    "P": {
        "name": "Pressure Injury",
        "short_desc": "pressure injury",
        "full_desc": "This image shows a pressure injury (bed sore) over a bony prominence. It represents localized damage to skin and underlying tissue from prolonged pressure.",
        "management": "Immediate pressure redistribution (turning schedule), nutritional support, and stage-appropriate wound care.",
        "risk_factors": "Immobility, malnutrition, sensory impairment, and incontinence."
    },
    "S": {
        "name": "Surgical Wound",
        "short_desc": "surgical wound",
        "full_desc": "This image shows a surgical wound site. Assessment focuses on edge approximation, signs of dehiscence, or surgical site infection (SSI).",
        "management": "Maintain clean/dry environment, monitor for infection (erythema, purulent drainage), and ensure adequate nutrition.",
        "risk_factors": "Diabetes, smoking, obesity, and immunosuppression."
    },
    "N": {
        "name": "Normal Skin",
        "short_desc": "healthy intact skin",
        "full_desc": "This image shows healthy, intact skin with no visible wounds, ulcerations, or pathological changes.",
        "management": "Routine skin care: moisturizing, inspection, and pressure redistribution prevention.",
        "risk_factors": "N/A"
    }
}

DFU_CLASS_MAP = {
    "Normal(Healthy skin)": {
        "desc": "This image shows a diabetic patient's foot with no active ulceration. The skin appears intact.",
        "clinical_note": "Preventive care is critical: daily inspection and proper footwear."
    },
    "Abnormal(Ulcer)": {
        "desc": "This image shows an active diabetic foot ulcer. The skin integrity is compromised.",
        "clinical_note": "Immediate clinical assessment required for infection, depth, and perfusion."
    }
}

# ==========================================
# GENERATOR FUNCTIONS
# ==========================================

def generate_azh_qa(folder_code):
    """Generate Q&A for AZH types"""
    qa = []
    info = AZH_CLASS_MAP.get(folder_code)
    if not info: return []

    # 1. Classification
    qa.append({
        "question": "What type of wound is this?",
        "answer": f"This is a {info['name']}. {info['full_desc']}"
    })
    
    # 2. Management
    qa.append({
        "question": "What is the recommended management?",
        "answer": info['management']
    })

    # 3. Detection
    is_wound = "No" if folder_code == "N" else "Yes"
    qa.append({
        "question": "Is there an active wound?",
        "answer": f"{is_wound}. {info['short_desc']}"
    })
    
    return qa

def generate_dfu_qa(category):
    """Generate Q&A for DFU types"""
    qa = []
    info = DFU_CLASS_MAP.get(category)
    if not info: return []

    is_abnormal = "Abnormal" in category
    
    # 1. Detection
    qa.append({
        "question": "Is a diabetic foot ulcer present?",
        "answer": "Yes, an ulcer is visible." if is_abnormal else "No, the skin is intact."
    })

    # 2. Assessment
    qa.append({
        "question": "Provide a clinical assessment.",
        "answer": f"{info['desc']} {info['clinical_note']}"
    })

    return qa

# ==========================================
# MAIN LOGIC
# ==========================================

def main():
    final_data = []
    
    print("🚀 STARTING DATASET GENERATION...")

    # --- 1. PROCESS AZH ---
    print(f"\n📂 Scanning AZH: {AZH_ROOT}")
    if os.path.exists(AZH_ROOT):
        # The folders inside AZH/dataset/Train are D, V, P, S, N
        for code in ["D", "V", "P", "S", "N"]:
            folder = os.path.join(AZH_ROOT, code)
            if not os.path.exists(folder):
                print(f"   ⚠️ Folder {code} not found.")
                continue
            
            images = glob.glob(os.path.join(folder, "*"))
            print(f"   - {code}: Found {len(images)} images")

            for img_path in images:
                if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                
                # Create clean relative path
                rel_path = os.path.relpath(img_path, BASE_DIR)
                qa_list = generate_azh_qa(code)

                # Add each QA pair as a sample
                for qa in qa_list:
                    final_data.append({
                        "id": str(random.getrandbits(32)),
                        "image": rel_path,
                        "conversations": [
                            {"from": "human", "value": qa['question']},
                            {"from": "gpt", "value": qa['answer']}
                        ]
                    })
    else:
        print(f"❌ AZH Root Path not found: {AZH_ROOT}")

    # --- 2. PROCESS DFU (Patches) ---
    print(f"\n📂 Scanning DFU: {DFU_ROOT}")
    if os.path.exists(DFU_ROOT):
        # The specific subfolders inside Patches
        target_folders = ["Abnormal(Ulcer)", "Normal(Healthy skin)"]
        
        for cat in target_folders:
            folder = os.path.join(DFU_ROOT, cat)
            if not os.path.exists(folder):
                print(f"   ⚠️ Folder {cat} not found.")
                continue
            
            images = glob.glob(os.path.join(folder, "*"))
            print(f"   - {cat}: Found {len(images)} images")

            for img_path in images:
                if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                
                rel_path = os.path.relpath(img_path, BASE_DIR)
                qa_list = generate_dfu_qa(cat)

                for qa in qa_list:
                    final_data.append({
                        "id": str(random.getrandbits(32)),
                        "image": rel_path,
                        "conversations": [
                            {"from": "human", "value": qa['question']},
                            {"from": "gpt", "value": qa['answer']}
                        ]
                    })
    else:
        print(f"❌ DFU Root Path not found: {DFU_ROOT}")

    # --- 3. SAVE ---
    if not final_data:
        print("\n❌ CRITICAL: No data generated. Check paths.")
        return

    random.shuffle(final_data)
    
    # Split 90/10
    split = int(len(final_data) * 0.9)
    train_data = final_data[:split]
    val_data = final_data[split:]

    with open(OUTPUT_TRAIN, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(OUTPUT_VAL, "w") as f:
        json.dump(val_data, f, indent=2)

    print("\n✅ SUCCESS!")
    print(f"   Total Training Samples: {len(train_data)}")
    print(f"   Total Validation Samples: {len(val_data)}")
    print(f"   Saved to: {OUTPUT_TRAIN}")

if __name__ == "__main__":
    main()