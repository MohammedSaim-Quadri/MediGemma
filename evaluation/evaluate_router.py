import csv
from src.core.router import classify_query

INPUT = "evaluation/router_cases.csv"

correct = 0
total = 0
errors = []

with open(INPUT, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        query = row["query"]
        expected = row["expected_intent"].strip().lower()

        predicted = classify_query(query, [])

        total += 1

        if predicted == expected:
            correct += 1
        else:
            errors.append((query, expected, predicted))

accuracy = correct / total if total else 0

print(f"Correct: {correct}/{total}")
print(f"Accuracy: {accuracy:.1%}")

if errors:
    print("\nMisclassified:")
    for query, expected, predicted in errors:
        print(f"- {query}")
        print(f"  Expected: {expected}")
        print(f"  Predicted: {predicted}")