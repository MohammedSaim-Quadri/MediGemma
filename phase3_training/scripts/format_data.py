# phase3_training/scripts/format_data.py
# FINE TUNE PART 1 
import os
import json
import random
import glob
import cv2
import numpy as np

# --- CONFIGURATION ---
BASE_DIR = "phase3_training"
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset.json")

# 1. HELPER: EXTRACT FEATURES FROM MASKS
def extract_wound_features_from_mask(mask_path):
    """
    Analyzes the black-and-white mask to find wound size and shape.
    """
    try:
        # Load mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: return None
        
        # Calculate Size
        total_pixels = mask.size
        wound_pixels = np.count_nonzero(mask)
        coverage_pct = (wound_pixels / total_pixels) * 100
        
        # Calculate Shape (Irregularity)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        is_irregular = False
        circularity = 0
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            
            # Circularity: 1.0 is a perfect circle. < 0.6 is irregular.
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < 0.6: is_irregular = True
        
        # Categorize Size
        size_cat = "small"
        if coverage_pct > 20: size_cat = "large"
        elif coverage_pct > 5: size_cat = "medium"
        
        return {
            "size": size_cat,
            "irregular": is_irregular,
            "pct": coverage_pct,
            "circularity": circularity
        }
    except Exception as e:
        print(f"⚠️ Mask Error {mask_path}: {e}")
        return None


# 2. HELPER: GENERATE RICH DESCRIPTIONS
def generate_wound_description(dataset_name, features=None, wound_type="unspecified"):
    desc = []
    
    # 1. Identity & Context
    if dataset_name == "medetec":
        desc.append(f"This image shows a {wound_type} wound.")
    elif dataset_name in ["fuseg", "wsnet"]:
        desc.append("This image shows a diabetic foot ulcer.")
    else:
        desc.append("This image shows an open wound.")
        
    # 2. Morphology (If mask exists) - RESTORED DETAIL
    if features:
        if features['size'] == "large":
            desc.append("The wound is extensive, covering a significant portion of the tissue.")
        elif features['size'] == "medium":
            desc.append("The wound is moderate in size.")
        elif features['size'] == "small":
            desc.append("The wound is relatively small in surface area.")
            
        if features['irregular']:
            desc.append("The wound borders are irregular and jagged, which may indicate undermining or complex tissue damage.")
        else:
            desc.append("The wound borders appear well-defined and relatively regular.")
            
    # 3. Clinical Note
    desc.append("Clinical assessment is required to evaluate healing progress and check for signs of infection.")
    
    return " ".join(desc)

def generate_qa_pairs(dataset_name, img_path, mask_path=None, wound_type="unspecified"):
    """
    Creates 4 distinct Q&A pairs for every wound image.
    """
    pairs = []
    features = None
    
    if mask_path and os.path.exists(mask_path):
        features = extract_wound_features_from_mask(mask_path)
        
    main_desc = generate_wound_description(dataset_name, features, wound_type)
    
    # Pair 1: Detection (Yes/No)
    pairs.append({
        "q": random.choice([
            "Is there a wound visible in this image?",
            "Does this image show tissue damage?",
            "What is the clinical finding here?",
            "Is the skin intact?"
        ]),
        "a": f"Yes. {main_desc}"
    })
    
    # Pair 2: Detailed Assessment
    pairs.append({
        "q": random.choice([
            "Perform a clinical wound assessment.",
            "Describe the wound characteristics.",
            "What observations can be made about this tissue?"
        ]),
        "a": main_desc
    })
    
    # Pair 3: Size/Extent (If mask data exists)
    if features:
        size_texts = {
            "small": "The wound appears relatively small in extent.",
            "medium": "The wound is moderate in size.",
            "large": "The wound is extensive, covering a significant tissue area."
        }
        pairs.append({
            "q": "What is the approximate size of the wound?",
            "a": size_texts[features['size']]
        })
        
    # Pair 4: Clinical Action / Triage (RESTORED)
    pairs.append({
        "q": "What is the recommended clinical action?",
        "a": "This wound requires evaluation by a healthcare provider for treatment planning and infection monitoring."
    })
    
    return pairs

def generate_healthy_qa_pairs():
    """
    Teaches the model to say 'NO' in 4 different ways.
    """
    pairs = []
    
    # Variation list to prevent robot-speak
    healthy_descs = [
        "The skin appears intact with no visible wounds, ulcerations, or tissue damage.",
        "This shows healthy skin. No signs of wound formation, infection, or necrosis are present.",
        "The tissue appears normal with no clinical abnormalities detected.",
        "No wound is visible in this image. The skin shows normal texture and coloration."
    ]
    
    # Pair 1: Direct detection question
    pairs.append({
        "q": random.choice([
            "Is there a wound present in this image?",
            "Does this image show any tissue damage?",
            "Is there an ulceration visible?"
        ]),
        "a": random.choice([
            "No, there is no wound present. The skin appears healthy and intact.",
            "No wound is detected. The tissue shows normal appearance.",
            "There is no evidence of wound formation."
        ])
    })
    
    # Pair 2: Assessment question
    pairs.append({
        "q": random.choice([
            "Provide a clinical assessment of this image.",
            "Describe the tissue condition.",
            "Analyze this image for wound presence."
        ]),
        "a": random.choice(healthy_descs)
    })
    
    # Pair 3: Recommendation (RESTORED)
    pairs.append({
        "q": random.choice([
            "What clinical action is recommended?",
            "Does this require medical intervention?"
        ]),
        "a": "No immediate clinical intervention is required. The skin is healthy. Continue routine skin care."
    })
    
    # Pair 4: Negative confirmation (RESTORED - Critical for training)
    pairs.append({
        "q": random.choice([
            "Are there any signs of infection or necrosis?",
            "Is there tissue breakdown visible?",
            "Do you observe any pathological changes?"
        ]),
        "a": "No. There are no signs of infection, necrosis, or tissue breakdown. The skin appears healthy."
    })
    
    return pairs


# 3. MAIN BUILDER
def create_dataset():
    final_data = []
    stats = {'medetec':0, 'fuseg':0, 'wsnet':0, 'healthy':0, 'total_pairs':0}
    
    print("🚀 Starting Advanced Dataset Formatting (Full Detail Mode)...")
    
    # --- PROCESS GENERAL / MEDETEC ---
    med_path = os.path.join(BASE_DIR, "datasets/general")
    if os.path.exists(med_path):
        images = glob.glob(os.path.join(med_path, "**", "*"), recursive=True)
        images = [f for f in images if f.lower().endswith(('.jpg','.png','.jpeg')) and "mask" not in f.lower()]
        
        for img in images:
            w_type = "unspecified"
            fname = img.lower()
            if "pressure" in fname: w_type = "pressure"
            elif "venous" in fname: w_type = "venous"
            elif "diabetic" in fname: w_type = "diabetic"
            elif "surgical" in fname: w_type = "surgical"
            elif "burn" in fname: w_type = "burn"
            
            # Find mask
            mask_path = os.path.splitext(img)[0] + "_mask.png"
            if not os.path.exists(mask_path): 
                 mask_path = img.replace(".jpg", "_mask.png").replace(".jpeg", "_mask.png")
            if not os.path.exists(mask_path): mask_path = None
            
            qa_list = generate_qa_pairs("medetec", img, mask_path, w_type)
            
            for qa in qa_list:
                final_data.append({
                    "id": str(random.getrandbits(32)),
                    "image": os.path.relpath(img, BASE_DIR),
                    "conversations": [
                        {"from": "human", "value": qa['q']},
                        {"from": "gpt", "value": qa['a']}
                    ]
                })
                stats['total_pairs'] += 1
            stats['medetec'] += 1

    # --- PROCESS HEALTHY ---
    healthy_path = os.path.join(BASE_DIR, "datasets/healthy")
    if os.path.exists(healthy_path):
        images = glob.glob(os.path.join(healthy_path, "*"))
        for img in images:
            if not img.lower().endswith(('.jpg', '.png', '.jpeg')): continue
            
            qa_list = generate_healthy_qa_pairs()
            for qa in qa_list:
                final_data.append({
                    "id": str(random.getrandbits(32)),
                    "image": os.path.relpath(img, BASE_DIR),
                    "conversations": [
                        {"from": "human", "value": qa['q']},
                        {"from": "gpt", "value": qa['a']}
                    ]
                })
                stats['total_pairs'] += 1
            stats['healthy'] += 1

    # --- FINALIZE & SAVE ---
    if not final_data:
        print("❌ Error: No images found! Check paths.")
        return

    random.shuffle(final_data)
    
    # 80/20 Split
    split = int(len(final_data) * 0.8)
    train_set = final_data[:split]
    val_set = final_data[split:]
    
    with open(os.path.join(BASE_DIR, "dataset_train.json"), "w") as f:
        json.dump(train_set, f, indent=2)
        
    with open(os.path.join(BASE_DIR, "dataset_val.json"), "w") as f:
        json.dump(val_set, f, indent=2)

    print(f"\n✅ Done! Stats:")
    print(f"   Medetec/General Images: {stats['medetec']}")
    print(f"   Healthy Images: {stats['healthy']}")
    print(f"   Total Conversations Generated: {stats['total_pairs']}")
    print(f"   Training Pairs: {len(train_set)}")
    print(f"   Validation Pairs: {len(val_set)}")
    print(f"   Saved to: {os.path.join(BASE_DIR, 'dataset_train.json')}")

if __name__ == "__main__":
    create_dataset()