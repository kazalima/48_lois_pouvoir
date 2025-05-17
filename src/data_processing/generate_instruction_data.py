import json
import random

# Charger les lois
with open("data/processed/embeddings.json", "r", encoding="utf-8") as f:
    laws_data = json.load(f)

# Questions disponibles
questions = {
    "Explain this law in simple terms": "Expliquez cette loi en termes simples",
    "Give a historical example of this law": "Donnez un exemple historique de cette loi",
    "How to apply this law in business?": "Comment appliquer cette loi en entreprise ?",
    "What is the counter-measure for this law?": "Quelle est la contre-mesure à cette loi ?",
    "Summarize this law in one sentence": "Résumez cette loi en une phrase"
}

# Réponses types
explanations = [
    "elle conseille de ne jamais faire trop confiance aux autres.",
    "elle montre que le pouvoir est souvent basé sur la perception.",
    "elle enseigne à ne jamais éclipser le maître.",
    "elle suggère de toujours dire moins que nécessaire.",
    "elle recommande de maîtriser l’art du timing."
]

historical_figures = [
    "Napoléon Bonaparte", "Louis XIV", "Catherine de Médicis",
    "Jules César", "Machiavel", "Otto von Bismarck", "Talleyrand"
]

business_applications = [
    "en gardant certaines informations confidentielles.",
    "en ne montrant jamais toute sa stratégie à ses concurrents.",
    "en laissant ses supérieurs croire qu’ils ont eu l’idée.",
    "en contrôlant son image publique avec soin.",
    "en construisant des alliances tout en gardant le contrôle."
]

counter_measures = [
    "en cultivant la transparence face aux manipulations.",
    "en posant des limites claires aux abus de pouvoir.",
    "en favorisant la collaboration plutôt que la domination.",
    "en gardant un esprit critique face aux flatteries.",
    "en évitant les jeux d’influence toxiques."
]

summaries = [
    "Ne jamais surpasser son supérieur.",
    "Contrôlez l’apparence et les perceptions.",
    "Maîtrisez le pouvoir à travers les alliances.",
    "Faites-en toujours moins que nécessaire.",
    "Ne révélez jamais vos véritables intentions."
]

output_data = []

# Boucle jusqu’à atteindre 1200 exemples
while len(output_data) < 1200:
    law_entry = random.choice(laws_data)
    law_text = law_entry.get("law", "").strip()
    
    if not law_text:
        continue

    # Numéro de la loi
    try:
        law_number = law_text.split("LOI ")[1].split(" ")[0]
    except IndexError:
        law_number = "?"

    en_question, fr_question = random.choice(list(questions.items()))
    input_text = f"Loi {law_number} : {law_text[:150].strip()}..."
    output = ""

    if "Expliquez" in fr_question:
        output = f"La LOI {law_number} signifie que {random.choice(explanations)}"
    elif "exemple historique" in fr_question:
        output = f"Un exemple historique est {random.choice(historical_figures)}, qui a appliqué cette loi pour renforcer son pouvoir."
    elif "appliquer cette loi en entreprise" in fr_question:
        output = f"On peut appliquer cette loi {random.choice(business_applications)}"
    elif "contre-mesure" in fr_question:
        output = f"La meilleure contre-mesure est {random.choice(counter_measures)}"
    elif "une phrase" in fr_question:
        output = f"{random.choice(summaries)}"

    # Vérification : aucun champ vide
    if input_text and output:
        entry = {
            "instruction": fr_question,
            "input": input_text,
            "output": output
        }
        output_data.append(entry)

# Sauvegarde au format JSONL
with open("data/processed/fine_tuning_data.jsonl", "w", encoding="utf-8") as f:
    for entry in output_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"{len(output_data)} exemples générés.")
