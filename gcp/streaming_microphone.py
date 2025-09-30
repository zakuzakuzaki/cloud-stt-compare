"""Streaming microphone input to Google Cloud Speech-to-Text v2.

This script records audio from the system default microphone, streams it to a
Google Cloud Speech-to-Text v2 recognizer, and prints transcription results in
real time. It is inspired by Google's streaming recognition example for v2 but
captures audio from a live microphone instead of reading from a local file.

Prerequisites:
  * Install dependencies: `pip install google-cloud-speech sounddevice`.
  * Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to a
    service account JSON file that has permission to use the Speech-to-Text API.
  * Create a recognizer resource in the target project/location.

Usage:
    python streaming_microphone.py \
        --project-id=my-project \
        --location=us-central1 \
        --recognizer-id=my-recognizer

Press Ctrl+C to stop streaming.
"""
from __future__ import annotations

import argparse
import queue
import sys
from typing import Generator, Iterable

import sounddevice as sd
from google.cloud import speech_v2


DEFAULT_SAMPLE_RATE = 16000
CHANNELS = 1
# Send roughly 100ms of audio per request for low latency.
CHUNK_SIZE = int(DEFAULT_SAMPLE_RATE / 10)


class MicrophoneStream:
    """Open a recording stream as a generator yielding raw audio chunks."""

    def __init__(self, rate: int, chunk_size: int) -> None:
        self.rate = rate
        self.chunk_size = chunk_size
        self._buff: "queue.Queue[bytes]" = queue.Queue()
        self._stream: sd.InputStream | None = None

    def __enter__(self) -> "MicrophoneStream":
        self._stream = sd.InputStream(
            samplerate=self.rate,
            channels=CHANNELS,
            dtype="int16",
            blocksize=self.chunk_size,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Signal the generator to stop.
        self._buff.put_nowait(b"")

    def _callback(self, indata, frames, time_info, status) -> None:  # type: ignore[override]
        if status:
            print(f"Microphone status: {status}", file=sys.stderr)
        self._buff.put(indata.tobytes())

    def generator(self) -> Generator[bytes, None, None]:
        while True:
            chunk = self._buff.get()
            if chunk == b"":
                return
            yield chunk


def request_stream(
    recognizer: str,
    rate: int,
    chunk_size: int,
) -> Iterable[speech_v2.StreamingRecognizeRequest]:
    """Yield requests for the streaming recognizer."""

    config = speech_v2.RecognitionConfig(
        language_codes=["ja-JP"],
        model="latest_long",
        features=speech_v2.RecognitionFeatures(enable_automatic_punctuation=True),
        explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
            encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=rate,
            audio_channel_count=CHANNELS,
        ),
    )

    streaming_config = speech_v2.StreamingRecognitionConfig(
        config=config,
        streaming_features=speech_v2.StreamingRecognitionFeatures(
            interim_results=True,
        ),
    )

    # The first request contains the streaming config.
    yield speech_v2.StreamingRecognizeRequest(
        recognizer=recognizer, streaming_config=streaming_config
    )

    with MicrophoneStream(rate, chunk_size) as stream:
        for chunk in stream.generator():
            yield speech_v2.StreamingRecognizeRequest(audio=chunk)


def print_transcripts(
    responses: Iterable[speech_v2.StreamingRecognizeResponse],
) -> None:
    """Print transcripts from streaming responses."""

    for response in responses:
        for result in response.results:
            if not result.alternatives:
                continue
            alternative = result.alternatives[0]
            transcript = alternative.transcript.strip()
            if not transcript:
                continue

            prefix = "(final)" if result.is_final else "(interim)"
            confidence = f" {alternative.confidence:.0%}" if result.is_final else ""
            print(f"{prefix} {transcript}{confidence}")
            # Flush to ensure timely display when piping output.
            sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream microphone audio to Google Cloud Speech-to-Text v2.",
    )
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument(
        "--location",
        default="global",
        help="Location of the recognizer (default: global)",
    )
    parser.add_argument(
        "--recognizer-id",
        required=True,
        help="Recognizer ID (without the projects/... prefix)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Sampling rate for microphone capture (default: 16000)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Chunk size in frames per streaming request (default: 1600)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recognizer = (
        f"projects/{args.project_id}/locations/{args.location}/recognizers/{args.recognizer_id}"
    )

    client = speech_v2.SpeechClient()

    try:
        responses = client.streaming_recognize(
            requests=request_stream(recognizer, args.sample_rate, args.chunk_size)
        )
        print("Listening... Press Ctrl+C to stop.")
        print_transcripts(responses)
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error during streaming recognition: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
