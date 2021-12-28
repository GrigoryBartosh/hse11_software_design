from asrecognition import ASREngine


class ASR:
    def __init__(self):
        self.asr = ASREngine("ru", model_path="jonatasgrosman/wav2vec2-large-xlsr-53-russian")

    def recognize(self, path):
        res = self.asr.transcribe([path])
        return res[0]["transcription"]