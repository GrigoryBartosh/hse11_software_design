from transformers import M2M100ForConditionalGeneration as Model
from transformers import M2M100Tokenizer as Tokenizer


class Ru2EnTranslator:
    def __init__(self):
        self.model = Model.from_pretrained("facebook/m2m100_418M")
        self.tokenizer = Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.tokenizer.src_lang = "ru"

    def translate(self, text):
        encoded_ru = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded_ru, forced_bos_token_id=self.tokenizer.get_lang_id("en"))
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]