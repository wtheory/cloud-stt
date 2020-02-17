#!/usr/bin/env python
import asyncio
import base64
import datetime
import io
import json
import pdb
import time

import pydub
import websockets
from pydub import AudioSegment
from scipy.io import wavfile

from match_audio import match, match_target_amplitude

SUBTITLES = {}

for pod in ['pod.1.json', 'pod.2.json', 'pod.3.json']:
    with open(pod, 'r') as f:
        SUBTITLES[pod.rstrip('.json')] = json.loads(f.read())


def foo(wav):

    best_station, best_frag_idx = match(wav)

    if best_station == 0:
        best_station = 'pod.1'
    if best_station == 1:
        best_station = 'pod.2'
    if best_station == 2:
        best_station = 'pod.3'

    best_frag_idx = best_frag_idx * 1000.0

    print ('PODCAST', best_station)
    print ('MS', best_frag_idx)

    return best_station, best_frag_idx


async def serve(websocket, path):

    wav = await websocket.recv()

    before = datetime.datetime.now()

    song = AudioSegment.from_file(io.BytesIO(wav), codec="opus")
    song = song.set_frame_rate(44100)
    song = match_target_amplitude(song, -23)
    song.export('buffer.wav', format="wav")

    rate, wav = wavfile.read('buffer.wav')

    print('rate', rate)

    pod, timestamp = foo(wav)

    delay = datetime.datetime.now() - before

    timestamp = timestamp + delay.seconds * 1000 + delay.microseconds / 1000

    for i, entry in enumerate(SUBTITLES[pod]):
        if entry['time'] > timestamp:
            current_index = i - 1
            break

    current_entry = SUBTITLES[pod][current_index]
    next_entry = SUBTITLES[pod][current_index + 1]

    await websocket.send(json.dumps(current_entry['text']))

    to_wait = next_entry['time'] - timestamp

    no_words = 0

    while True:

        print(f'Waiting {to_wait} milliseconds')

        await asyncio.sleep(to_wait / 1000.0 - no_words * 0.1)

        current_index = current_index + 1

        current_entry = SUBTITLES[pod][current_index]
        next_entry = SUBTITLES[pod][current_index + 1]

        to_wait = next_entry['time'] - current_entry['time']

        words = current_entry['text'].split(' ')

        no_words = len(words)

        for word in words:
            print(word)
            await websocket.send(json.dumps(word))
            await asyncio.sleep(0.1)


start_server = websockets.serve(serve, "10.104.13.52", 8081)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
