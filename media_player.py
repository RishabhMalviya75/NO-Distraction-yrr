import os
import logging
import pygame

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

class MediaPlayer:
    """
    Manages non-blocking audio playback based on user activity state using pygame mixer.
    Prevents repeated clip restarts when the state remains unchanged.
    """
    def __init__(self, activity_media_map: dict[str, str | None]):
        self.activity_media_map = activity_media_map
        self.current_state: str | None = None
        self.is_initialized = False
        self._init_mixer()

    def _init_mixer(self):
        """Initialize pygame mixer safely."""
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.is_initialized = True
            logging.info("Pygame audio mixer initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize audio mixer: {e}")
            self.is_initialized = False

    def play_for_state(self, new_state: str):
        """
        Trigger audio playback when state changes.
        Does nothing if the state has not changed.
        """
        if new_state == self.current_state:
            return

        logging.info(f"State transition: '{self.current_state}' -> '{new_state}'")
        self.current_state = new_state

        if not self.is_initialized:
            logging.warning("Mixer not initialized. Skipping audio play.")
            return

        # Stop any currently playing audio on state change
        try:
            pygame.mixer.music.stop()
        except Exception as e:
            logging.warning(f"Error stopping audio playback: {e}")

        # Get media file path mapped to the new state
        media_path = self.activity_media_map.get(new_state)

        if not media_path:
            logging.info(f"No media configured for state '{new_state}' (Idle or Null). Audio stopped.")
            return

        if not os.path.exists(media_path):
            logging.warning(f"Media file not found for state '{new_state}': '{media_path}'. Please check config.json.")
            return

        # Load and play new media file
        try:
            pygame.mixer.music.load(media_path)
            # Loop playback (-1) for continuous alert until state changes
            pygame.mixer.music.play(loops=-1)
            logging.info(f"Playing audio for state '{new_state}': {media_path}")
        except Exception as e:
            logging.error(f"Failed to play audio file '{media_path}' for state '{new_state}': {e}")

    def stop(self):
        """Stop audio playback cleanly."""
        if self.is_initialized:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass
        self.current_state = None
