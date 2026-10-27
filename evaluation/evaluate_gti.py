import argparse
import pandas as pd

from src.engine.engine_core import (
    DataManager,
    LLMEngine,
    ClinicalRAGEngine,
)


def build_gti_query(user_query, state):
    """
    Replicates the Ground Truth Injection used by ClinicalOrchestrator.
    """
    ground_truth = (
        "\n\n[SYSTEM INJECTED GROUND TRUTH - PRIORITY OVER RETRIEVAL]\n"
        f"Latest Encounter Date: {state['last_visit']}\n"
        f"Wound Records at Latest Encounter: {state['wound_record_count']}\n"
        f"Documented Wound Dimensions at Latest Encounter: {state['wound_dims']}\n"
        f"Active Non-Zero Wound Present: {state['active_wound']}\n"
        f"Latest Encounter Notes: {state['narrative']}\n"
        f"INSTRUCTION: Treat {state['last_visit']} as the current encounter. "
        "Do not substitute measurements from older encounters when describing "
        "the current patient state.\n"
    )

    return user_query + ground_truth

def response_to_text(response):
    if hasattr(response, "response"):
        return str(response.response)
    return str(response)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--encounters",
        required=True,
        help="Path to encounters CSV",
    )

    parser.add_argument(
        "--patients",
        default=None,
        help="Optional path to patient demographics CSV",
    )

    parser.add_argument(
        "--cases",
        default="evaluation/gti_cases.csv",
        help="CSV containing patient_id and question",
    )

    parser.add_argument(
        "--output",
        default="evaluation/gti_results.csv",
    )

    args = parser.parse_args()

    print("Loading data...")

    dm = DataManager()
    dm.load_data(args.encounters, args.patients)

    print(f"Loaded {len(dm.df)} rows")
    print(f"Unique patients: {dm.df['Patient_ID'].nunique()}")

    print("Initializing LLM...")

    llm_engine = LLMEngine()
    llm_engine.initialize()

    print("Initializing RAG engine...")

    # dm.load_data() should have rebuilt dm.index.
    rag = ClinicalRAGEngine(index=dm.index, llm=llm_engine.llm)

    cases = pd.read_csv(args.cases, dtype={"patient_id": str})

    results = []

    for i, row in cases.iterrows():
        patient_id = str(row["patient_id"]).strip()
        question = str(row["question"]).strip()

        # Make sure patient ID is explicitly present so RAG metadata
        # filtering behaves identically in both conditions.
        if patient_id not in question:
            question = f"{question} for patient {patient_id}"

        state = dm.get_patient_current_state(patient_id)

        if state is None:
            print(f"[SKIP] No state found for patient {patient_id}")
            continue

        print()
        print("=" * 80)
        print(f"Case {i + 1}/{len(cases)}")
        print(f"Patient: {patient_id}")
        print(f"Question: {question}")
        print(f"Ground truth latest visit: {state['last_visit']}")
        print(f"Latest encounter: {state['last_visit']}")
        print(f"Latest encounter wound dimensions: {state['wound_dims']}")
        print(f"Active non-zero wound present: {state['active_wound']}")
        print(f"Wound record count: {state['wound_record_count']}")

        # ---------------------------------------------------------
        # CONDITION A: RAG ONLY
        # ---------------------------------------------------------
        baseline_response = rag.chat(
            question,
            history=[],
            visual_context=None,
        )

        baseline_text = response_to_text(baseline_response)

        # ---------------------------------------------------------
        # CONDITION B: SAME RAG + GTI
        # ---------------------------------------------------------
        gti_query = build_gti_query(question, state)

        gti_response = rag.chat(
            gti_query,
            history=[],
            visual_context=None,
        )

        gti_text = response_to_text(gti_response)

        results.append(
            {
                "patient_id": patient_id,
                "question": question,
                "ground_truth_date": state["last_visit"],
                "ground_truth_wound_dims": state["wound_dims"],
                "ground_truth_active_wound": state["active_wound"],
                "ground_truth_wound_record_count": state["wound_record_count"],
                "baseline_response": baseline_text,
                "gti_response": gti_text,
            }
        )

        print("\n--- BASELINE ---")
        print(baseline_text)

        print("\n--- GTI ---")
        print(gti_text)

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)

    print()
    print("=" * 80)
    print(f"Saved {len(results_df)} comparisons to:")
    print(args.output)


if __name__ == "__main__":
    main()