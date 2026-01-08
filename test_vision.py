from utils import analyze_image
import os

# Point this to a real image in your dataset
TEST_IMAGE = "phase3_training/datasets/diabetic-foot-ulcer-dfu/DFU/TestSet/1 .jpg" 

if __name__ == "__main__":
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Error: Image not found at {TEST_IMAGE}")
        exit()

    print("🚀 Starting Smoke Test for LLaVA...")
    
    # This runs the function exactly as the App would, 
    # but prints the output directly to console.
    result = analyze_image(TEST_IMAGE)
    
    print("\n" + "="*30)
    print("FINAL RESULT:")
    print(result)
    print("="*30)
    
    if "error" in result:
        print("❌ Test FAILED.")
    elif not result.get("raw_output"):
        print("⚠️ Test UNCERTAIN (Empty Output).")
    else:
        print("✅ Test PASSED. Model is speaking!")