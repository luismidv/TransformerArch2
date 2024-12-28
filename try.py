from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

texto = "Hola buenos dias"

tok_text = tokenizer(texto)
print(tok_text)
