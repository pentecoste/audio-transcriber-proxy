import socket
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from collections import deque
import math
import whisper

model = whisper.load_model("medium")
executor = ThreadPoolExecutor(max_workers=40)
mutex = Lock()

def listen_for_requests():
    global executor
    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_socket.bind(('', 6666))
    listen_socket.listen(40)
    while True:
        conn, addr = listen_socket.accept()
        executor.submit(handle_request, conn, addr)

def handle_request(conn, addr):
    global mutex
    global model
    data_buffer = deque()
    expected_size = math.inf
    try:
        while len(data_buffer) < expected_size:
            data = conn.recv(1024)
            if len(data) > 0:
                for data_byte in data:
                    data_buffer.append(data_byte)
                if expected_size == math.inf and len(data_buffer) >= 8:
                    expected_size = 0
                    for i in range(8):
                        expected_size += data_buffer[0] << (8*i)
                        data_buffer.popleft()
            else:
                # socket closed too early by client
                conn.close()
                return
    except Exception as e:
        print(e)
        return
    # process the received audio exclusively and send it
    with mutex:
        with open("to_transcribe.opus", "wb") as f_out:
            f_out.write(bytes(data_buffer))
        result = model.transcribe("to_transcribe.opus", language="it")
        try:
            conn.send(bytes(result["text"].encode('utf-8')))
        except:
            conn.close()
            return
    conn.close()

listen_for_requests()