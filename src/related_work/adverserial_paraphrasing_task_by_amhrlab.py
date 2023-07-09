from transformers import T5Tokenizer, T5ForConditionalGeneration
paraphrasing_tokenizer = T5Tokenizer.from_pretrained("t5-base")
paraphrasing_model = T5ForConditionalGeneration.from_pretrained("coderpotter/T5-for-Adversarial-Paraphrasing")

# this function will take a sentence, top_k and top_p values based on beam search
def generate_paraphrases(sentence, top_k=120, top_p=0.95):
    text = "paraphrase: " + sentence + " </s>"
    encoding = paraphrasing_tokenizer.encode_plus(text, max_length=256, padding="max_length", return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    beam_outputs = paraphrasing_model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=top_k,
        top_p=top_p,
        early_stopping=True,
        num_return_sequences=10,
    )
    final_outputs = []
    for beam_output in beam_outputs:
        sent = paraphrasing_tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)
    return final_outputs