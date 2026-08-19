import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# Project path
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = BASE_DIR / "evaluation" / "questions.json"

sys.path.insert(0, str(BASE_DIR / "src"))

from rag_chain import generate_answer


# ==========================================
# Load evaluation questions
# ==========================================

with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
    questions = json.load(file)


# ==========================================
# Calculate similarity
# ==========================================

def calculate_similarity(expected, actual):

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [expected, actual]
    )

    score = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return score


# ==========================================
# Run evaluation
# ==========================================

results = []

print("\n====================================")
print("COLLEGE ADMISSION RAG EVALUATION")
print("====================================")


for index, item in enumerate(
    questions,
    start=1
):

    question = item["question"]
    expected = item["expected_answer"]

    print(f"\nQuestion {index}: {question}")

    try:

        result = generate_answer(question)

        actual = result["answer"]

        score = calculate_similarity(
            expected,
            actual
        )

        results.append(score)

        print("Generated Answer:")
        print(actual)

        print(f"\nSimilarity Score: {score:.4f}")

    except Exception as error:

        print("Error:", error)


# ==========================================
# Overall evaluation
# ==========================================

if results:

    average_score = sum(results) / len(results)

    print("\n====================================")
    print("EVALUATION SUMMARY")
    print("====================================")

    print(
        f"Questions evaluated: {len(results)}"
    )

    print(
        f"Average similarity score: "
        f"{average_score:.4f}"
    )

else:

    print("\nNo questions were successfully evaluated.")