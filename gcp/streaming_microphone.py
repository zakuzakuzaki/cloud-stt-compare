"""Streaming microphone input to Google Cloud Speech-to-Text v2.

This script records audio from the system default microphone, streams it to a
Google Cloud Speech-to-Text v2 recognizer, and prints transcription results in
real time. It is inspired by Google's streaming recognition example for v2 but
captures audio from a live microphone instead of reading from a local file. The
stream automatically leverages voice activity detection (VAD) events to stop
recording after roughly six seconds of silence while allowing speech to extend
the session up to thirty seconds.

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
import threading
import time
from typing import Generator, Iterable

import sounddevice as sd
from google.cloud import speech_v2


DEFAULT_SAMPLE_RATE = 16000
CHANNELS = 1
# Send roughly 100ms of audio per request for low latency.
CHUNK_SIZE = int(DEFAULT_SAMPLE_RATE / 10)
# Base duration to capture before voice activity driven extensions (seconds).
DEFAULT_INITIAL_DURATION = 6.0
# Maximum duration allowed for a single streaming session (seconds).
MAX_STREAM_DURATION = 30.0


class StreamingController:
    """Coordinate stream lifetime based on voice activity events."""

    def __init__(self, base_duration: float, max_duration: float) -> None:
        self.base_duration = base_duration
        self.max_duration = max_duration
        self.start_time = time.monotonic()
        self.last_activity = self.start_time
        self.voice_active = False
        self._lock = threading.Lock()

    def mark_voice_active(self) -> None:
        """Record that the API detected voice activity."""

        with self._lock:
            self.voice_active = True
            self.last_activity = time.monotonic()

    def mark_voice_inactive(self) -> None:
        """Record that the API detected the end of voice activity."""

        with self._lock:
            self.voice_active = False

    def mark_transcript_activity(self) -> None:
        """Record transcript updates as ongoing activity."""

        with self._lock:
            self.last_activity = time.monotonic()

    def should_stop(self) -> bool:
        """Return True when streaming should stop based on timers/VAD."""

        now = time.monotonic()
        with self._lock:
            elapsed = now - self.start_time
            if elapsed >= self.max_duration:
                return True
            if elapsed <= self.base_duration:
                return False
            if self.voice_active:
                return False
            # Allow a small grace period after the last detected activity.
            if (now - self.last_activity) <= 1.5:
                return False
            return True

    def handle_voice_activity_event(
        self, event: speech_v2.StreamingRecognizeResponse.VoiceActivityEvent
    ) -> None:
        """Interpret VAD events from the API and update state."""

        event_type = getattr(event, "voice_activity_event_type", None)
        if event_type is None:
            return

        # The enum may expose either a string representation or an Enum object.
        if hasattr(event_type, "name"):
            name = event_type.name
        else:
            name = str(event_type)

        normalized = name.upper()
        if "UNSPECIFIED" in normalized:
            return
        if "END" in normalized or "STOP" in normalized:
            self.mark_voice_inactive()
        else:
            self.mark_voice_active()

    def handle_speech_event_type(self, event_type: object) -> None:
        """Fallback handling for speech event types when VAD events are absent."""

        if event_type is None:
            return
        if hasattr(event_type, "name"):
            name = event_type.name
        else:
            name = str(event_type)

        normalized = name.upper()
        if "END" in normalized:
            self.mark_voice_inactive()
        elif "SPEECH" in normalized or "START" in normalized:
            self.mark_voice_active()


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
    controller: StreamingController,
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

    streaming_features = speech_v2.StreamingRecognitionFeatures(interim_results=True)
    # Enable voice activity events when the client library exposes the field.
    voice_events_config = getattr(streaming_features, "voice_activity_events", None)
    if voice_events_config is not None:
        voice_events_config.enable_voice_activity_events = True
    elif hasattr(streaming_features, "enable_voice_activity_events"):
        setattr(streaming_features, "enable_voice_activity_events", True)

    streaming_config = speech_v2.StreamingRecognitionConfig(
        config=config,
        streaming_features=streaming_features,
    )

    # The first request contains the streaming config.
    yield speech_v2.StreamingRecognizeRequest(
        recognizer=recognizer, streaming_config=streaming_config
    )

    with MicrophoneStream(rate, chunk_size) as stream:
        for chunk in stream.generator():
            if controller.should_stop():
                break
            yield speech_v2.StreamingRecognizeRequest(audio=chunk)


def print_transcripts(
    responses: Iterable[speech_v2.StreamingRecognizeResponse],
    controller: StreamingController,
) -> None:
    """Print transcripts from streaming responses."""

    for response in responses:
        controller.handle_speech_event_type(getattr(response, "speech_event_type", None))
        for event in getattr(response, "voice_activity_events", []) or []:
            controller.handle_voice_activity_event(event)

        for result in response.results:
            if not result.alternatives:
                continue
            alternative = result.alternatives[0]
            transcript = alternative.transcript.strip()
            if not transcript:
                continue

            controller.mark_transcript_activity()
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
    parser.add_argument(
        "--initial-duration",
        type=float,
        default=DEFAULT_INITIAL_DURATION,
        help="Base recording duration in seconds before VAD can stop the stream",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_STREAM_DURATION,
        help="Maximum recording duration in seconds when audio continues",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_duration < args.initial_duration:
        print(
            "--max-duration must be greater than or equal to --initial-duration",
            file=sys.stderr,
        )
        sys.exit(2)
    recognizer = (
        f"projects/{args.project_id}/locations/{args.location}/recognizers/{args.recognizer_id}"
    )

    client = speech_v2.SpeechClient()
    controller = StreamingController(
        base_duration=args.initial_duration, max_duration=args.max_duration
    )

    try:
        responses = client.streaming_recognize(
            requests=request_stream(
                recognizer, args.sample_rate, args.chunk_size, controller
            )
        )
        print("Listening... Press Ctrl+C to stop.")
        print_transcripts(responses, controller)
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error during streaming recognition: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
