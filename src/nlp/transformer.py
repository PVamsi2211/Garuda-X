def load_text_model(model_name):
    from transformers import AutoModel,AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModel.from_pretrained(model_name)
    return tokenizer,model
