# -*- coding: utf-8 -*-
import sys
import os
import simplejson as json
import tempfile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
from tqdm import tqdm

def transcribe_audio(path):
    output = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio = AudioSegment.from_wav(path)
        chunks = split_on_silence(
            audio, silence_thresh=audio.dBFS-12, min_silence_len=300, keep_silence=300)
        time_from_0 = 0
        silence_segment = AudioSegment.silent(duration=200)
        for index, chunk in enumerate(tqdm(chunks)):
            chunk = chunk + silence_segment
            chunk_name = os.path.join(
                tmp_dir, 'chunk{index}.wav'.format(index=index))
            #chunk_name = os.path.join(
            #    "./prototyping/tmp/", 'chunk{index}.wav'.format(index=index))
            chunk.export(chunk_name, format='wav', codec='wav')
            recognizer = sr.Recognizer()
            with sr.AudioFile(chunk_name) as source:
                audio_listened = recognizer.listen(source)
                try:
                    transcription = recognizer.recognize_google(audio_listened, language="pl")
                    output.append({'text': transcription, 'time': time_from_0})
                except Exception as e:
                    pass
                time_from_0 += len(chunk)
    return output


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python main.py input_video output_json_file.json')

    result = transcribe_audio(sys.argv[1])
    with open(sys.argv[2], 'w', encoding='utf-8') as f:
        s = json.dumps(result, ensure_ascii=False, indent=4).encode('utf-8')
        f.write(s.decode())

