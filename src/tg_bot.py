import os
import re

from telegram import InputMediaPhoto
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters
)


class StartHandler:
    def __init__(self, start_text):
        self.start_text = start_text

    def __call__(self, update, context):
        context.bot.send_message(chat_id=update.effective_chat.id, text=self.start_text)


class Searcher():
    def __init__(self, translator, clip, embed_db):
        self.translator = translator
        self.clip = clip
        self.embed_db = embed_db

    def is_russian(self, text):
        text = re.sub('[^а-яА-Я]+', '', text)
        return len(text) > 0

    def search(self, text):
        if self.is_russian(text):
            text = self.translator.translate(text)

        embed = self.clip.encode_texts([text])
        paths = self.embed_db.find(embed)
        return paths[0]


class ImageSender():
    def send_images(self, update, context, paths):
        if len(paths) == 0:
            return
        elif len(paths) == 1:
            with open(paths[0], "rb") as file:
                context.bot.send_photo(chat_id=update.effective_chat.id, photo=file)
        else:
            media = []
            for path in paths:
                with open(path, "rb") as file:
                    media += [InputMediaPhoto(file)]
            context.bot.send_media_group(chat_id=update.effective_chat.id, media=media)


class TextHandler(ImageSender):
    def __init__(self, searcher):
        self.searcher = searcher

    def __call__(self, update, context):
        text = update.message.text
        paths = self.searcher.search(text)
        self.send_images(update, context, paths)


class VoiceHandler(ImageSender):
    def __init__(self, asr, searcher):
        self.asr = asr
        self.searcher = searcher

    def download_voice(self, update, context):
        file = context.bot.get_file(file_id=update.message.voice.file_id)

        file_id = update.message.voice.file_id
        path = f"/tmp/{file_id}.ogg"
        file.download(path)

        return path

    def remove_voice(self, path):
        os.remove(path)

    def __call__(self, update, context):
        voice = self.download_voice(update, context)
        text = self.asr.recognize(voice)
        self.remove_voice(voice)

        paths = self.searcher.search(text)
        self.send_images(update, context, paths)


class TgBot:
    def __init__(self, token, start_text, asr, translator, clip, embed_db):
        self.token = token
        self.start_text = start_text
        self.asr = asr
        self.translator = translator
        self.clip = clip
        self.embed_db = embed_db

    def start(self):
        start_handler = StartHandler(self.start_text)

        searcher = Searcher(self.translator, self.clip, self.embed_db)
        text_handler = TextHandler(searcher)
        voice_handler = VoiceHandler(self.asr, searcher)

        updater = Updater(token=self.token)

        dispatcher = updater.dispatcher
        dispatcher.add_handler(CommandHandler("start", start_handler))
        dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), text_handler))
        dispatcher.add_handler(MessageHandler(Filters.voice, voice_handler))

        updater.start_polling()