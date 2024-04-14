import torch
from transformers import BertModel, BertTokenizerFast


MAX_LEN = 100


if __name__ == '__main__':
    sentence = input("Please input a sentence: ")
    model_name = "hfl/chinese-roberta-wwm-ext"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertModel.from_pretrained(model_name).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    inputs = tokenizer(
        sentence,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    # get the BERT latent representation of chinese sentence
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state[:, 0, :]

    print(outputs)
