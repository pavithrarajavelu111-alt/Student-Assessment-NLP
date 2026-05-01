import csv
import random
import os

# Create folder
os.makedirs("data", exist_ok=True)

file_path = "data/student_answers.csv"

f = open(file_path, "w", newline="", encoding="utf-8")
writer = csv.writer(f)

# Header — added similarity_score and doubt_label for ML training
writer.writerow(["USN", "Name", "Topic", "Question", "Correct Answer", "Student Answer",
                 "Similarity_Score", "Doubt_Label"])

# Students
students = [
    ("1DB23IS001", "Aarav"), ("1DB23IS002", "Diya"), ("1DB23IS003", "Rahul"),
    ("1DB23IS004", "Anita"), ("1DB23IS005", "Kiren"), ("1DB23IS006", "Sneha"),
    ("1DB23IS007", "Rohit"), ("1DB23IS008", "Meena"), ("1DB23IS009", "Arjun"),
    ("1DB23IS010", "Pooja"), ("1DB23IS011", "Vikram"), ("1DB23IS012", "Neha"),
    ("1DB23IS013", "Riya"), ("1DB23IS014", "Karthik"), ("1DB23IS015", "Ishita"),
    ("1DB23IS016", "Varun"), ("1DB23IS017", "Ananya"), ("1DB23IS018", "Siddharth"),
    ("1DB23IS019", "Priya"), ("1DB23IS020", "Aditya")
]

# Force some students to fail
fail_students = ("1DB23IS003", "1DB23IS007", "1DB23IS012", "1DB23IS018")

# Questions
questions = [

    # MCQ
    {"type": "mcq", "topic": "ML Basics", "question": "ML uses?", "answer": "Data",
     "options": ["Data", "Rules", "Random", "None"]},

    {"type": "mcq", "topic": "Regression", "question": "Regression predicts?", "answer": "Continuous",
     "options": ["Continuous", "Category", "Noise", "None"]},

    {"type": "mcq", "topic": "Classification", "question": "Classification predicts?", "answer": "Category",
     "options": ["Category", "Value", "Noise", "Cluster"]},

    {"type": "mcq", "topic": "Clustering", "question": "Clustering means?", "answer": "Grouping",
     "options": ["Grouping", "Sorting", "Deleting", "Training"]},

    {"type": "mcq", "topic": "Neural Networks", "question": "NN inspired by?", "answer": "Brain",
     "options": ["Brain", "CPU", "RAM", "Disk"]},

    # SUBJECTIVE (LONG + REALISTIC)
    {"type": "subjective", "topic": "ML Basics", "question": "Define ML",
     "answer": "Machine learning is a branch of artificial intelligence where systems learn patterns from data and improve performance"},

    {"type": "subjective", "topic": "Regression", "question": "Define regression",
     "answer": "Regression is a supervised learning method used to predict continuous numerical values by analyzing relationships"},

    {"type": "subjective", "topic": "Classification", "question": "Define classification",
     "answer": "Classification is a supervised machine learning technique that assigns input data into predefined categories based"},

    {"type": "subjective", "topic": "Clustering", "question": "Define clustering",
     "answer": "Clustering is an unsupervised learning technique that groups similar data points together based on their features"},

    {"type": "subjective", "topic": "Neural Networks", "question": "Define NN",
     "answer": "Neural networks are models inspired by the human brain consisting of interconnected layers of neurons that can lea"},
]

# Wrong answers (human-like)
wrong_answers = {
    "ML Basics": [
        "Machine learning is just writing programs manually without using any data.",
        "It is about storing data in computer and not really learning anything.",
        "ML means system follows fixed instructions only without improvement."
    ],
    "Regression": [
        "Regression is used to divide data into categories not numbers.",
        "It is mainly used for grouping similar items together.",
        "Regression predicts labels instead of continuous values."
    ],
    "Classification": [
        "Classification is used for predicting numerical values instead of categories.",
        "It randomly assigns data without learning patterns.",
        "Classification is same as clustering without labels."
    ],
    "Clustering": [
        "Clustering requires labeled data to group things.",
        "It is mainly used for prediction not grouping.",
        "Clustering sorts data randomly without logic."
    ],
    "Neural Networks": [
        "Neural networks are simple storage systems without learning.",
        "They are just databases without processing capability.",
        "NN does not involve layers or connections."
    ]
}

# Slight human variations (add noise)
def make_human(text):
    variations = [
        text,
        text.lower(),
        text.replace("is", "is basically"),
        text.replace("that", "which"),
        text + " This is important in real world applications.",
        text + " It is widely used in many real life problems.",
    ]
    return random.choice(variations)


# ─────────────────────────────────────────────────────────────────────
# NLP similarity using TF-IDF (computed at data generation time)
# This produces labeled training data for the ML model in app.py
# ─────────────────────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_tfidf_similarity(student_ans, correct_ans):
    """Real TF-IDF cosine similarity — this is the ground-truth label for training."""
    if not HAS_SKLEARN:
        # Fallback: word-overlap Jaccard similarity
        a_words = set(student_ans.lower().split())
        b_words = set(correct_ans.lower().split())
        if not a_words and not b_words:
            return 0.0
        return round(len(a_words & b_words) / len(a_words | b_words), 4)
    vec = TfidfVectorizer()
    try:
        tfidf = vec.fit_transform([student_ans, correct_ans])
        score = sk_cosine(tfidf[0:1], tfidf[1:2])[0][0]
        return round(float(score), 4)
    except Exception:
        return 0.0


def assign_doubt_label(score, student_ans, correct_ans, topic):
    """
    Rule-based doubt labelling using similarity score + NLP keyword signals.
    These labels become the Y-target for the Random Forest classifier in app.py.
    """
    words = student_ans.lower().split()
    n_words = len(words)

    # Keyword confusion signals
    confusion_map = {
        "Regression":      ["class", "categor"],
        "Classification":  ["continu", "regress"],
        "Clustering":      ["supervis", "label"],
        "Neural Networks": ["storage", "database"],
    }
    confused = any(kw in student_ans.lower()
                   for kw in confusion_map.get(topic, []))

    if n_words <= 3:
        return "Incomplete Answer"
    if confused:
        return f"Confused with {topic}"
    if score >= 0.70:
        return "Good Understanding"
    if score >= 0.40:
        return "Partial Understanding"
    return "Concept Unclear"


# ─────────────────────────────────────────────────────────────────────
# Generate data
# ─────────────────────────────────────────────────────────────────────
for usn, name in students:
    for q in questions:
        correct = q["answer"]

        # Accuracy logic
        if usn in fail_students:
            accuracy = 0.3
        else:
            accuracy = 0.75

        # MCQ
        if q["type"] == "mcq":
            if random.random() < accuracy:
                student = correct
            else:
                wrong_opts = [opt for opt in q["options"] if opt != correct]
                student = random.choice(wrong_opts)

            # MCQ similarity: 1.0 if correct else 0.0
            sim_score = 1.0 if student.strip().lower() == correct.strip().lower() else 0.0
            doubt_label = "Good Understanding" if sim_score == 1.0 else "Wrong Answer"

        # SUBJECTIVE (REALISTIC VARIATION)
        else:
            rand = random.random()

            if usn in fail_students:
                # Weak student
                if rand < 0.6:
                    student = random.choice(wrong_answers[q["topic"]])
                else:
                    student = correct[:50]  # incomplete
            else:
                if rand < 0.4:
                    # Topper (full answer)
                    student = make_human(correct)
                elif rand < 0.75:
                    # Average (partial)
                    student = make_human(correct[:90])
                else:
                    # Slightly wrong
                    student = random.choice(wrong_answers[q["topic"]])

            sim_score = compute_tfidf_similarity(student, correct)
            doubt_label = assign_doubt_label(sim_score, student, correct, q["topic"])

        writer.writerow([usn, name, q["topic"], q["question"],
                         correct, student, sim_score, doubt_label])

f.close()
print("✅ ML-LABELED CSV GENERATED SUCCESSFULLY")
print(f"   → {file_path}")
print("   → Columns: USN, Name, Topic, Question, Correct Answer,")
print("              Student Answer, Similarity_Score, Doubt_Label")
