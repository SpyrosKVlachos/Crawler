import json

def create_content_inverted_index(file_path, output_file="inverted_index.json"):
    """
    Δημιουργεί και αποθηκεύει έναν ανεστραμμένο ευρετήριο σε αρχείο JSON.

    Args:
        file_path: Η διαδρομή προς το αρχείο JSON με τα δεδομένα.
        output_file: Το όνομα του αρχείου όπου θα αποθηκευτεί ο ευρετήριος (προεπιλογή: "inverted_index.json").
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Σφάλμα: Το αρχείο {file_path} δεν βρέθηκε.")
        return None
    except json.JSONDecodeError:
        print(f"Σφάλμα: Το αρχείο {file_path} δεν είναι έγκυρο JSON.")
        return None

    content_index = {}

    for document in data:
        doc_id = document['id']
        content_words = document['content']

        for word in content_words:
            if word not in content_index:
                content_index[word] = []
            if doc_id not in content_index[word]:
                content_index[word].append(doc_id)
    
    try:
        with open(output_file, "w", encoding='utf-8') as outfile:
            json.dump(content_index, outfile, ensure_ascii=False, indent=4)
        print(f"Ο ανεστραμμένος ευρετήριος αποθηκεύτηκε στο {output_file}") #εμφανιζει μηνυμα οτι αποθηκευτηκε
    except Exception as e: #πιανει οποιοδηποτε αλλο σφαλμα μπορει να προκυψει
        print(f"Σφάλμα κατά την αποθήκευση του αρχείου: {e}")
        return None
    
    return content_index

# Παράδειγμα χρήσης
file_path = "processed_file.json"
content_index = create_content_inverted_index(file_path)

if content_index:
    print("Ευρετήριο περιεχομένου:")
    for word, doc_ids in content_index.items():
        print(f"{word}: {doc_ids}")