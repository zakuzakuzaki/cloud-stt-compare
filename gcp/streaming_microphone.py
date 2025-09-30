import os
import queue
import sys
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types
import pyaudio

# プロジェクトID
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

# 音声録音パラメータ
RATE = 16000  # サンプリングレート
CHUNK = int(RATE / 10)  # 100ms


class MicrophoneStream:
    """マイクからの音声ストリームをジェネレーターとして提供するクラス"""
    
    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        """コンテキストマネージャーの開始"""
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,  # モノラル
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        """コンテキストマネージャーの終了"""
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """PyAudioのコールバック関数 - 音声データをバッファに追加"""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """音声データのジェネレーター"""
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            
            # バッファに残っているデータもすべて取得
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            
            yield b"".join(data)


def listen_print_loop(responses):
    """認識結果を表示する"""
    num_chars_printed = 0
    
    for response in responses:
        if not response.results:
            continue
        
        # 最初の結果を取得
        result = response.results[0]
        if not result.alternatives:
            continue
        
        # 最も確率の高い転写結果を取得
        transcript = result.alternatives[0].transcript
        
        # 前の出力を上書きするための空白
        overwrite_chars = " " * (num_chars_printed - len(transcript))
        
        if not result.is_final:
            # 暫定結果 - 同じ行に上書き
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()
            num_chars_printed = len(transcript)
        else:
            # 確定結果 - 改行して表示
            print(transcript + overwrite_chars)
            num_chars_printed = 0


def transcribe_streaming_mic(language_code="ja-JP"):
    """マイクからの音声をストリーミング認識する
    
    Args:
        language_code (str): 言語コード (例: "ja-JP", "en-US")
    """
    client = SpeechClient()
    
    # 認識設定
    recognition_config = cloud_speech_types.RecognitionConfig(
        explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
            encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            audio_channel_count=1,
        ),
        language_codes=[language_code],
        model="long",
    )
    
    # ストリーミング設定
    streaming_config = cloud_speech_types.StreamingRecognitionConfig(
        config=recognition_config,
        streaming_features=cloud_speech_types.StreamingRecognitionFeatures(
            interim_results=True  # 暫定結果を有効化
        ),
    )
    
    # 最初の設定リクエスト
    config_request = cloud_speech_types.StreamingRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/global/recognizers/_",
        streaming_config=streaming_config,
    )
    
    def request_generator(config, audio_generator):
        """リクエストのジェネレーター"""
        yield config
        for content in audio_generator:
            yield cloud_speech_types.StreamingRecognizeRequest(audio=content)
    
    print("マイクに向かって話してください... (Ctrl+Cで終了)")
    print("-" * 50)
    
    try:
        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = request_generator(config_request, audio_generator)
            
            # ストリーミング認識を実行
            responses = client.streaming_recognize(requests=requests)
            
            # 結果を処理
            listen_print_loop(responses)
    
    except KeyboardInterrupt:
        print("\n\n音声認識を終了しました。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 使用例
    # 日本語認識の場合
    transcribe_streaming_mic(language_code="ja-JP")
    
    # 英語認識の場合は以下のように呼び出す
    # transcribe_streaming_mic(language_code="en-US")