import json
from tqdm import tqdm
from src.llm.dataset import format_input

def generate_responses(model, tokenizer, test_data, config, device):
    model.eval()
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)
        inputs = tokenizer(input_text, return_tensors="pt", max_length=config["model"]["context_length"], truncation=True).to(device)
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(input_text):].strip()
        test_data[i]["model_response"] = response
    
    with open(config["data"]["output"], "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    print(f"Réponses sauvegardées sous {config['data']['output']}")
