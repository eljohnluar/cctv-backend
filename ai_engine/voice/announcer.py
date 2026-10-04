import queue
import threading
import tempfile
import os
import time
import base64
import subprocess
import sys
from utils.config import settings
from utils.logger import logger
from utils.uniform_policy import get_runtime_controls, get_voice_settings
from websocket_manager import publish_from_worker


class VoiceAnnouncer:
    """
    Queue-based Text-to-Speech announcement system using gTTS.
    Prevents overlapping audio playback and runs in a separate daemon thread.

    Trespasser alerts are rate-limited so they don't repeat every frame —
    one announcement fires, then a cooldown period is enforced before the
    next trespasser alert can play.
    """

    # Minimum seconds between successive trespasser announcements
    TRESPASSER_COOLDOWN = 30.0

    def __init__(self):
        self.message_queue: queue.Queue = queue.Queue()
        self.is_running = True
        self.enabled = settings.VOICE_ENABLED
        self.lang = settings.VOICE_LANGUAGE

        # Tracks the last time a trespasser announcement was enqueued
        self._last_trespasser_at: float = 0.0
        self._lock = threading.Lock()

        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def announce(self, message: str) -> None:
        """Enqueue an arbitrary speech announcement."""
        if not self.enabled or not get_runtime_controls()["announcer_enabled"] or not message.strip():
            return
        logger.info("[Voice Announcer] Queued: '%s'", message)
        self.message_queue.put(message)

    def announce_attendance_marked(self, student_name: str) -> None:
        """
        Announce that a recognised student's attendance has been recorded.
        Example: "Maria Santos, attendance marked."
        """
        self.announce(f"{student_name}, attendance marked.")

    def announce_trespasser(self) -> None:
        """
        Announce an unrecognised face detection.
        Rate-limited to TRESPASSER_COOLDOWN seconds so it does not fire on
        every consecutive camera frame.
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_trespasser_at < self.TRESPASSER_COOLDOWN:
                return
            self._last_trespasser_at = now
        self.announce("Trespasser.")

    def announce_time_out(self, student_name: str) -> None:
        """
        Announce a departure recorded after the Time out.
        Example: "Maria Santos, time out recorded."
        """
        self.announce(f"{student_name}, time out recorded.")

    def announce_security_alert(self, alert_description: str) -> None:
        """Announce a general security alert."""
        self.announce(f"Security alert! {alert_description}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while self.is_running:
            try:
                message = self.message_queue.get(timeout=1.0)
                voice_gender = get_voice_settings()["voice_gender"]
                volume = get_runtime_controls()["announcer_volume"]
                server_audio = self._play_tts(message, voice_gender, volume)
                # The browser only speaks when the server could not reach an
                # audio device. Publishing after playback prevents double speech.
                publish_from_worker({
                    "type": "voice",
                    "message": message,
                    "server_audio": server_audio,
                    "voice_gender": voice_gender,
                    "volume": volume,
                })
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error("Error in announcement worker: %s", exc)

    def _play_tts(self, message: str, voice_gender: str, volume: int = 100) -> bool:
        # Windows SAPI is already present on supported Windows installations.
        # It does not need network access or optional Python audio packages.
        if sys.platform == "win32" and self._play_windows_sapi(message, voice_gender, volume):
            return True

        tmp_path = None
        played = False
        try:
            from gtts import gTTS
            tts = gTTS(text=message, lang=self.lang, slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            tts.save(tmp_path)

            # Fallback for non-Windows environments where optional packages are
            # installed and an audio device is exposed to the backend process.
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.set_volume(volume / 100)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.music.unload()
                played = True
            except Exception as play_err:
                logger.warning("Audio playback unavailable: %s", play_err)

        except ImportError:
            logger.warning("No server audio fallback is installed for: '%s'", message)
        except Exception as exc:
            logger.error("TTS generation error: %s", exc)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        return played

    @staticmethod
    def _play_windows_sapi(message: str, voice_gender: str = "female", volume: int = 100) -> bool:
        """Speak through the built-in Windows voice without shell escaping risks."""
        safe_message = message.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$voice = $speaker.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Gender.ToString() -eq '{voice_gender.title()}' }} | Select-Object -First 1; "
            "if ($voice) { $speaker.SelectVoice($voice.VoiceInfo.Name) }; "
            "$speaker.Rate = 0; "
            f"$speaker.Volume = {max(0, min(100, int(volume)))}; "
            f"$speaker.Speak('{safe_message}');"
        )
        encoded_script = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        try:
            startup_info = subprocess.STARTUPINFO()
            startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup_info.wShowWindow = 0
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded_script],
                capture_output=True,
                text=True,
                timeout=30,
                startupinfo=startup_info,
                check=False,
            )
            if result.returncode == 0:
                logger.info("[Voice Announcer] Spoken with Windows SAPI: '%s'", message)
                return True
            logger.warning("Windows SAPI failed: %s", result.stderr.strip() or "unknown error")
        except Exception as error:
            logger.warning("Windows SAPI playback failed: %s", error)
        return False


voice_announcer = VoiceAnnouncer()
