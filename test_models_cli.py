import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
from src.engine.test_models import (
    load_medgemma_4b,
    analyze_with_medgemma_4b,
    load_medgemma_27b,
    analyze_with_medgemma,
    load_hulumed,
    analyze_with_hulumed,
    analyze_with_gemma3,
    unload_model_safely,
    log_vram,
    force_vram_cleanup,
    evict_ollama_with_retry
)

IMAGE = "temp_gemma3_image (7).png"
QUESTION = "Describe the wound."
CYCLES = 2  # Reduced to 2 for testing

for cycle in range(CYCLES):
    print(f"\n================ CYCLE {cycle+1} ================\n")

    # ======================================================
    # 1️⃣ HULU-MED 32B (Largest First!)
    # ======================================================
    log_vram("Before Hulu-Med 32B")
    ph, mh = load_hulumed()
    log_vram("After Hulu-Med Load")
    
    if mh:  # Only if loaded successfully
        print(analyze_with_hulumed(IMAGE, QUESTION, ph, mh)[:200])
        mh, ph = unload_model_safely(mh, ph, "Hulu-Med 32B")
    else:
        print("❌ Hulu-Med failed to load - skipping cycle")
        break

    # ======================================================
    # 2️⃣ MEDGEMMA 27B (Large HF)
    # ======================================================
    log_vram("Before MedGemma 27B")
    p27, m27 = load_medgemma_27b()
    log_vram("After MedGemma 27B Load")

    print(analyze_with_medgemma(IMAGE, QUESTION, p27, m27)[:200])
    m27, p27 = unload_model_safely(m27, p27, "MedGemma 27B")

    # ======================================================
    # 3️⃣ GEMMA 3 via OLLAMA
    # ======================================================
    print("\n🔵 Running Gemma3 via Ollama...")
    print(analyze_with_gemma3(IMAGE, QUESTION)[:200])

    # Evict Ollama BEFORE next cycle
    evict_ollama_with_retry(required_free_gb=18.5)
    force_vram_cleanup()
    log_vram("After Ollama eviction")

    # ======================================================
    # 4️⃣ MEDGEMMA 4B (Smallest)
    # ======================================================
    log_vram("Before MedGemma 4B")
    p4, m4 = load_medgemma_4b()
    log_vram("After MedGemma 4B Load")

    print(analyze_with_medgemma_4b(IMAGE, QUESTION, p4, m4)[:200])
    m4, p4 = unload_model_safely(m4, p4, "MedGemma 4B")

    # ======================================================
    # END OF CYCLE CLEANUP
    # ======================================================
    force_vram_cleanup()
    log_vram("End of cycle")

    time.sleep(5)