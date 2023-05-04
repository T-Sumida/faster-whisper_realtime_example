from typing import Optional, Union

import numpy as np
from faster_whisper import WhisperModel

from whisper_realtime.audio import Audio

FS = 16000
class App:
    def __init__(self, model_size: str, is_cuda: bool, compute_type: str, mic_id: Optional[int]) -> None:
        if is_cuda:
            self._model = WhisperModel(
                model_size, device="cuda", compute_type=compute_type
            )
        else:
            self._model = WhisperModel(
                model_size, device="auto", compute_type=compute_type
            )
        self._audio = Audio()
        self._mic_id = mic_id

    def run(self) -> None:
        self._audio.start_streaming(FS, self._mic_id)
        try:
            while True:
                data = self._audio.get()
                if data is None: continue
                data = np.frombuffer(data, np.int16)
                data = np.asarray(data, np.float32)/32768.0
                segments, info = self._model.transcribe(data, beam_size=5, vad_filter=True, language="ja")
                for segment in segments:
                    print(segment.text)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(e)
        finally:
            self._audio.stop()
