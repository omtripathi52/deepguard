"""
DeepGuard v2 - Main Entry Point

Launches the overlay and detection engine.

Usage:
    python main.py
    
Press Ctrl+C in terminal or close the overlay to stop.
"""

import sys
import signal
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from config import config
from core.engine import DeepGuardEngine, EngineState
from overlay.window import OverlayWindow


class DeepGuardApp:
    """
    Main application class for DeepGuard v2.
    
    Coordinates the detection engine and overlay UI.
    """
    
    def __init__(self):
        # Create Qt application
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(True)
        
        # Create overlay window
        self.overlay = OverlayWindow()
        
        # Create engine with callback
        self.engine = DeepGuardEngine(on_state_update=self._on_engine_update)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Timer to process signals (needed for Ctrl+C to work)
        self.signal_timer = QTimer()
        self.signal_timer.timeout.connect(lambda: None)
        self.signal_timer.start(100)
    
    def _on_engine_update(self, state: EngineState):
        """Handle engine state updates"""
        # Emit signal to overlay (thread-safe way to update UI)
        self.overlay.state_updated.emit(state)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C"""
        print("\n[APP] Shutting down...")
        self.stop()
    
    def run(self):
        """Start the application"""
        print("=" * 50)
        print("🛡️  DeepGuard v2 - Real-Time Deepfake Detection")
        print("=" * 50)
        print()
        print("Starting detection engine...")
        
        # Start engine
        self.engine.start()
        
        # Show overlay
        self.overlay.show()
        
        print("Overlay is now active!")
        print()
        print("📌 The overlay will appear in the corner of your screen")
        print("📌 Hover over it to see detailed explanation")
        print("📌 Drag to reposition")
        print("📌 Press Ctrl+C in this terminal to stop")
        print()
        
        # Run Qt event loop
        exit_code = self.app.exec()
        
        # Cleanup
        self.stop()
        return exit_code
    
    def stop(self):
        """Stop the application"""
        self.engine.stop()
        self.app.quit()


def main():
    """Entry point"""
    try:
        app = DeepGuardApp()
        sys.exit(app.run())
    except KeyboardInterrupt:
        print("\n[APP] Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
