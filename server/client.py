#!/usr/bin/env python

# WS client example

import asyncio
import base64
import json

import websockets


async def hello():
    uri = "ws://10.104.13.52:8081"
    async with websockets.connect(uri) as websocket:

        await websocket.send(json.dumps({"audio": base64.b64encode("dsadasdas".encode('utf8')).decode('utf8')}))

        while True:
            greeting = await websocket.recv()

            print(greeting)

asyncio.get_event_loop().run_until_complete(hello())
