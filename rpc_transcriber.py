from faster_whisper import WhisperModel
import gc
from threading import Lock
from concurrent import futures
import grpc
import audiotranscription_pb2
import audiotranscription_pb2_grpc
import base64

model_size = "large-v3"

model = WhisperModel(model_size, device="cuda", compute_type="int8")
stop = False
mutex = Lock()

class AudioTranscriptionService(audiotranscription_pb2_grpc.AudioTranscriptionServiceServicer):
    def process_audio(self, request, context):
        response = audiotranscription_pb2.TextResponse()
        result = ""
        with mutex:
            if stop:
                response.value = "Errore nel processing dell'audio"
                return response
            with open("to_transcribe.opus", "wb") as f_out:
                f_out.write(bytes(base64.b64decode(request.base64_data)))
            segments, info = model.transcribe("to_transcribe.opus", beam_size=5, language="it")
            result = ""
            for segment in segments:
                result += "[{tstart:.2f} -> {tend:.2f}] {content}\n".format(tstart=segment.start, tend=segment.end, content=segment.text)
        response.value = result
        return response

    def stop_transcriber(self, request, context):
        with mutex:
            stop = True
            model = None
            gc.collect()

    def start_transcriber(self, request, context):
        with mutex:
            stop = False
            model = WhisperModel(model_size, device="cuda", compute_type="int8")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[('grpc.max_receive_message_length', 500 * 1024 * 1024), ('grpc.max_send_message_length', 10 * 1024 * 1024)])
    audiotranscription_pb2_grpc.add_AudioTranscriptionServiceServicer_to_server(AudioTranscriptionService(), server)
    server.add_insecure_port('[::]:6666')
    server.start()
    server.wait_for_termination()

serve()
