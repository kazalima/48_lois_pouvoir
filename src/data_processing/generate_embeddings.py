import yaml
import json
from sentence_transformers import SentenceTransformer
import os

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def split_into_laws(text):
    laws = []
    # Recherche des indices pour chaque loi (LOI 1 à LOI 48)
    indices = []
    for i in range(1, 49):  # De LOI 1 à LOI 48
        search_term = f"loi {i}"
        index = text.lower().find(search_term)
        if index != -1:
            indices.append((i, index))
            print(f"LOI {i} trouvé à l'index {index}: {text[index:index+100]}")
        else:
            print(f"LOI {i} non trouvé")

    # Trie les indices par position dans le texte (au cas où ils ne seraient pas dans l'ordre)
    indices.sort(key=lambda x: x[1])

    # Divise le texte en lois basées sur les indices
    for i in range(len(indices)):
        law_number, start_index = indices[i]
        end_index = indices[i+1][1] if i+1 < len(indices) else len(text)
        law_text = text[start_index:end_index].strip()
        laws.append(law_text)
        print(f"Loi {law_number} extraite : {law_text[:100]}...")

    print(f"Nombre total de lois détectées : {len(laws)}")
    if len(laws) != 48:
        print(f"Attention : {len(laws)} lois détectées au lieu de 48. Vérifiez les titres manquants.")

    return laws

def generate_embeddings(config):
    model_name = config["model"]["embedding_model"]
    text_path = config["data"]["text"]
    embeddings_path = config["data"]["embeddings"]

    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Texte non trouvé à {text_path}")

    # Charger le modèle
    model = SentenceTransformer(model_name)

    # Lire le texte
    with open(text_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Diviser en lois
    laws = split_into_laws(text)
    if not laws:
        raise ValueError("Aucune loi détectée. Vérifiez le format du texte.")

    # Générer les embeddings
    embeddings = model.encode(laws, show_progress_bar=True)

    # Sauvegarder les embeddings
    embeddings_data = [
        {"law": law, "embedding": embedding.tolist()}
        for law, embedding in zip(laws, embeddings)
    ]
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    with open(embeddings_path, "w", encoding="utf-8") as file:
        json.dump(embeddings_data, file, ensure_ascii=False, indent=2)

    return embeddings_data

if __name__ == "__main__":
    config = load_config()
    embeddings_data = generate_embeddings(config)
    print(f"Embeddings générés et sauvegardés dans {config['data']['embeddings']}")
