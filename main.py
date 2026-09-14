# ============================================================
# MULTI-FORMAT RAG SYSTEM (PDF, DOCX, TXT)
# ============================================================

import os
import re
from pathlib import Path
import docx
from pypdf import PdfReader
import spacy

# ------------------------------------------------------------
# CONFIGURATION & MODEL LOADING
# ------------------------------------------------------------

CHUNK_SIZE = 500

print("Loading Greek language model...")
nlp = spacy.load("el_core_news_sm")
print("OK.\n")

# ------------------------------------------------------------
# SYNONYM DICTIONARY & STOP WORDS
# ------------------------------------------------------------

SYNONYMS = {
    "χρόνος": {"χρόνος", "διάρκεια", "χρονικό διάστημα"},
    "ανίχνευση": {"ανίχνευση", "εντοπισμός", "αναγνώριση", "detection"},
    "επίθεση": {"επίθεση", "επιθέσεων", "επιθέσεις", "attack", "attacks"},
    "σύστημα": {"σύστημα", "συστήματα", "system", "systems"},
    "απόδοση": {"απόδοση", "επίδοση", "performance"},
    "ταχύτητα": {"ταχύτητα", "ρυθμός", "speed", "rate"},
    "mttd": {
        "mttd",
        "mean time to detect",
        "χρόνος ανίχνευσης",
        "μέσος χρόνος ανίχνευσης",
    },
    "latency": {"latency", "καθυστέρηση", "χρόνος καθυστέρησης"},
    "agents": {"agents", "agent", "πράκτορες", "πράκτορας"},
    "ευπάθεια": {
        "ευπάθεια",
        "ευπάθειες",
        "vulnerability",
        "vulnerabilities",
    },
    "auc": {"auc", "area under the curve", "εμβαδόν κάτω από την καμπύλη"},
    "yaml": {"yaml", "yaml δεν είναι γλώσσα σήμανσης"},
}

STOP_WORDS = {
    "ποιος",
    "ποια",
    "ποιο",
    "ποιες",
    "ποιας",
    "ποιον",
    "τι",
    "τί",
    "πως",
    "πώς",
    "ποσο",
    "πόσο",
    "πόση",
    "πόσοι",
    "πόσες",
    "είναι",
    "ήταν",
    "έχει",
    "έχουν",
    "έγινε",
    "έγιναν",
    "του",
    "της",
    "των",
    "τον",
    "την",
    "το",
    "τα",
    "τη",
    "τις",
    "ένα",
    "μια",
    "μία",
    "έναν",
    "σε",
    "στο",
    "στη",
    "στην",
    "στα",
    "στις",
    "για",
    "με",
    "από",
    "και",
    "ή",
    "ως",
    "ο",
    "η",
    "οι",
    "μας",
    "μου",
    "σου",
    "τους",
}

DEFINITION_INDICATORS = {"σημαίνει", "ορισμός", "τι είναι", "τί είναι", "έννοια"}

# ------------------------------------------------------------
# MULTI-FORMAT FILE READER (PDF, DOCX, TXT)
# ------------------------------------------------------------


def read_document(file_path):
    path = Path(file_path.strip('"').strip("'"))

    if not path.exists():
        raise FileNotFoundError(f"Το αρχείο δεν βρέθηκε: {path}")

    ext = path.suffix.lower()
    text = ""

    if ext == ".txt":
        text = path.read_text(encoding="utf-8")

    elif ext == ".docx":
        doc = docx.Document(path)
        full_text = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(full_text)

    elif ext == ".pdf":
        reader = PdfReader(path)
        page_texts = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                page_texts.append(extracted)
        text = "\n".join(page_texts)

    else:
        raise ValueError(
            f"Μη υποστηριζόμενος τύπος αρχείου ({ext}). Υποστηρίζονται μόνο .txt, .docx, .pdf"
        )

    return text


# ------------------------------------------------------------
# HELPER FUNCTIONS & CHUNKING
# ------------------------------------------------------------


def normalize_word(word):
    word = word.lower().strip()
    return re.sub(r"^[^\w]+|[^\w]+$", "", word, flags=re.UNICODE)


def create_chunks(text):
    lines = text.splitlines()
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_id = 0

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        if current_length + len(line_str) > CHUNK_SIZE and current_chunk:
            chunks.append({"id": chunk_id, "text": "\n".join(current_chunk)})
            chunk_id += 1
            overlap_lines = (
                current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
            )
            current_chunk = list(overlap_lines)
            current_length = sum(len(l) for l in current_chunk)

        current_chunk.append(line_str)
        current_length += len(line_str)

    if current_chunk:
        chunks.append({"id": chunk_id, "text": "\n".join(current_chunk)})

    return chunks


# ------------------------------------------------------------
# QUESTION ANALYSIS & SEARCH
# ------------------------------------------------------------


def analyze_question(question):
    doc = nlp(question)
    terms = []
    is_definition_query = False

    for token in doc:
        word = normalize_word(token.text)
        if not word:
            continue

        if (
            word in DEFINITION_INDICATORS
            or token.lemma_.lower() in DEFINITION_INDICATORS
        ):
            is_definition_query = True

        if token.is_punct or word in STOP_WORDS:
            continue

        if token.pos_ not in {"NOUN", "VERB", "ADJ", "PROPN", "X"}:
            continue

        lemma = normalize_word(token.lemma_)
        if lemma and lemma not in STOP_WORDS and lemma not in terms:
            terms.append(lemma)

    return terms, is_definition_query


def get_synonyms(term):
    if term in SYNONYMS:
        return SYNONYMS[term]
    for group in SYNONYMS.values():
        if term in group:
            return group
    return {term}


def build_concept_groups(terms):
    return [{"original": term, "variants": get_synonyms(term)} for term in terms]


def match_concept(group, text):
    normalized_text = text.lower()
    matches = []
    for variant in group["variants"]:
        pattern = r"(?<!\w)" + re.escape(variant.lower()) + r"(?!\w)"
        if re.search(pattern, normalized_text, flags=re.UNICODE):
            matches.append(variant)
    return matches


def score_chunk(chunk, concept_groups, is_definition_query):
    matched_groups = 0
    synonym_coverage_bonus = 0.0

    for group in concept_groups:
        matches = match_concept(group, chunk["text"])
        if matches:
            matched_groups += 1
            if is_definition_query and len(matches) > 1:
                synonym_coverage_bonus += 0.50

    total = len(concept_groups)
    base_score = matched_groups / total if total > 0 else 0.0

    glossary_bonus = 0.0
    if is_definition_query and matched_groups > 0:
        lines = [l.strip() for l in chunk["text"].splitlines() if l.strip()]
        if len(lines) > 2 and sum(len(l) for l in lines) / len(lines) < 60:
            glossary_bonus = 0.40

    final_score = base_score + synonym_coverage_bonus + glossary_bonus

    return {
        "chunk_id": chunk["id"],
        "score": final_score,
        "matched": matched_groups,
        "total": total,
        "text": chunk["text"],
    }


# ------------------------------------------------------------
# EXACT ANSWER EXTRACTION
# ------------------------------------------------------------


def extract_exact_answer(chunk_text, concept_groups):
    lines = [l.strip() for l in chunk_text.splitlines() if l.strip()]
    is_list_structure = any(len(l) < 80 for l in lines)

    if is_list_structure:
        relevant_lines = []
        for i, line in enumerate(lines):
            for group in concept_groups:
                if match_concept(group, line):
                    relevant_lines.append(line)
                    if i + 1 < len(lines) and lines[i + 1] not in relevant_lines:
                        relevant_lines.append(lines[i + 1])
                    break
        if relevant_lines:
            return "\n".join(relevant_lines)

    doc = nlp(chunk_text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    best_sentences = []
    max_matches = 0

    for sentence in sentences:
        matches = sum(
            1 for group in concept_groups if match_concept(group, sentence)
        )
        if matches > max_matches:
            max_matches = matches
            best_sentences = [sentence]
        elif matches == max_matches and matches > 0:
            best_sentences.append(sentence)

    return " ".join(best_sentences) if best_sentences else chunk_text


# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------


def main():
    print("=" * 70)
    print("DOCUMENT READER & RAG ENGINE")
    print("=" * 70)

    # Ζητάει το αρχείο από το χρήστη
    while True:
        file_input = input(
            "\nΔώστε τη διαδρομή του αρχείου (TXT, DOCX, PDF): "
        ).strip()
        try:
            text = read_document(file_input)
            print(
                f"-> Επιτυχής ανάγνωση αρχείου! (Σύνολο χαρακτήρων: {len(text)})"
            )
            break
        except Exception as e:
            print(f"Σφάλμα: {e}. Παρακαλώ δοκιμάστε ξανά.")

    chunks = create_chunks(text)
    print(f"-> Δημιουργήθηκαν {len(chunks)} chunks.")

    print("\n" + "=" * 70)
    print("READY FOR QUESTIONS")
    print("=" * 70)

    while True:
        question = input("\nΕρώτηση (ή γράψτε EXIT για έξοδο): ").strip()
        if question.upper() == "EXIT":
            break
        if not question:
            continue

        terms, is_def = analyze_question(question)
        concept_groups = build_concept_groups(terms)

        ranked_results = [
            score_chunk(c, concept_groups, is_def) for c in chunks
        ]
        ranked_results.sort(
            key=lambda x: (x["score"], x["matched"]), reverse=True
        )
        best_match = ranked_results[0]

        exact_answer = extract_exact_answer(best_match["text"], concept_groups)

        print("\n" + "=" * 70)
        print(
            f"MATCH RESULTS (Chunk ID: {best_match['chunk_id']} | Score: {best_match['matched']}/{best_match['total']})"
        )
        print("=" * 70)

        print("\n>>> DIRECT ANSWER / EXACT MATCH:")
        print("-" * 70)
        print(exact_answer)
        print("-" * 70)

        print("\n>>> FULL CHUNK CONTEXT:")
        print("-" * 70)
        print(best_match["text"])
        print("-" * 70)


if __name__ == "__main__":
    main()
