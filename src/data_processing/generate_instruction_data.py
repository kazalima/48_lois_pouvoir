import json
import random

# Charger les lois
with open("data/processed/embeddings.json", "r", encoding="utf-8") as f:
    laws_data = json.load(f)

# Questions en français avec types
question_templates = [
    ("Expliquez cette loi en termes simples", "explanation"),
    ("Donnez un exemple historique de cette loi", "historical"),
    ("Comment appliquer cette loi en entreprise ?", "business"),
    ("Quelle est la contre-mesure à cette loi ?", "counter"),
    ("Résumez cette loi en une phrase", "summary")
]

# Réponses types enrichies
explanations = [
    "elle montre qu'une stratégie subtile est souvent plus efficace qu'une démonstration de force.",
    "elle conseille de toujours préserver les apparences pour mieux manipuler les autres.",
    "elle enseigne que trop de transparence peut nuire à votre influence.",
    "elle souligne l’importance de laisser les autres briller pour gagner leur confiance.",
    "elle démontre que la maîtrise de soi est essentielle pour gouverner efficacement."
]

historical_figures = [
    "Napoléon Bonaparte pendant la campagne d’Italie",
    "Machiavel dans son œuvre *Le Prince*",
    "Catherine de Médicis à la cour de France",
    "Talleyrand sous Napoléon et Louis XVIII",
    "Jules César lors de la guerre des Gaules"
]

business_applications = [
    "en laissant le PDG croire que les idées viennent de lui.",
    "en manipulant l’image publique pour gagner la confiance des clients.",
    "en retardant délibérément certaines annonces stratégiques.",
    "en créant des alliances dans l’entreprise tout en gardant l’avantage.",
    "en évitant de dévoiler ses vraies intentions lors des réunions."
]

counter_measures = [
    "en exposant les tactiques de manipulation utilisées par les puissants.",
    "en créant un climat de transparence contrôlée dans son équipe.",
    "en privilégiant la loyauté authentique aux flatteries intéressées.",
    "en posant des limites claires face à l’abus de pouvoir.",
    "en utilisant l’humour pour désamorcer les rapports de force."
]

summaries = [
    "Faites en sorte que votre supérieur brille plus que vous.",
    "Maîtrisez l'art de dissimuler vos intentions.",
    "Le pouvoir repose sur l’image, pas sur la vérité.",
    "L’influence passe par le contrôle de l’apparence.",
    "Il ne faut jamais dire plus que nécessaire."
]

output_data = []

while len(output_data) < 1200:
    law_entry = random.choice(laws_data)
    law_text = law_entry.get("law", "").strip()

    if not law_text:
        continue

    # Identifier la loi
    try:
        law_number = law_text.split("LOI ")[1].split(" ")[0]
    except IndexError:
        law_number = "?"

    fr_question, question_type = random.choice(question_templates)
    input_text = f"LOI {law_number} : {law_text.strip()[:300]}..."

    # Génération de la réponse
    if question_type == "explanation":
        output = f"Cette loi signifie que {random.choice(explanations)}"
    elif question_type == "historical":
        output = f"{random.choice(historical_figures)} est un exemple parfait d’application de cette loi."
    elif question_type == "business":
        output = f"Dans un contexte professionnel, on peut appliquer cette loi {random.choice(business_applications)}"
    elif question_type == "counter":
        output = f"Une bonne stratégie pour résister à cette loi est {random.choice(counter_measures)}"
    elif question_type == "summary":
        output = f"{random.choice(summaries)}"

    if input_text and output:
        entry = {
            "instruction": fr_question,
            "input": input_text,
            "output": output
        }
        output_data.append(entry)

# Sauvegarde
with open("data/processed/fine_tuning_data.jsonl", "w", encoding="utf-8") as f:
    for entry in output_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"{len(output_data)} exemples générés avec succès.")
