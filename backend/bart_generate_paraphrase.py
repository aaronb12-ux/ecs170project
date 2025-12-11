# not successful, but included for language generation attempt 

# https://huggingface.co/eugenesiow/bart-paraphrase
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bart_model = BartForConditionalGeneration.from_pretrained("eugenesiow/bart-paraphrase").to(device)
bart_tokenizer = BartTokenizer.from_pretrained("eugenesiow/bart-paraphrase")

def paraphrase_phrase(jd_phrase, resume_phrase):
    # input prompt
    input_text = f"{resume_phrase} => {jd_phrase}"

    # pass prompt to tokenizer
    inputs = bart_tokenizer(input_text, return_tensors="pt").to(device)

    # generate language
    generated_ids = bart_model.generate(
        inputs["input_ids"],
        num_beams=5,
        max_length=30,  
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    # decode back to text
    return bart_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
