"""
DeepGuard v2 - Overlay Window (PyQt6)

Floating, always-on-top overlay that displays detection status.
Features:
- Draggable
- Color-coded status
- Hover to expand with explanation
- Smooth transitions
"""

import sys
from typing import Optional
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, 
    QHBoxLayout, QGraphicsDropShadowEffect, QFrame
)
from PyQt6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve,
    pyqtSignal, QPoint, QSize
)
from PyQt6.QtGui import QFont, QColor, QPainter, QBrush, QPen, QCursor

from config import config
from core.confidence import ConfidenceLevel


class OverlayWindow(QWidget):
    """
    Floating overlay window for DeepGuard v2.
    
    Shows real-time detection status in a minimal, non-intrusive UI.
    Expands on hover to show detailed explanation.
    """
    
    # Signal for state updates from engine
    state_updated = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        self.cfg = config.overlay
        
        # State
        self.is_expanded = False
        self.current_level: Optional[ConfidenceLevel] = None
        self.current_text = "Starting..."
        self.current_explanation = ""
        self.current_color = self.cfg.color_uncertain
        
        # Dragging
        self._drag_position: Optional[QPoint] = None
        
        # Setup UI
        self._setup_window()
        self._setup_widgets()
        self._position_window()
        
        # Animation
        self._setup_animations()
        
        # Connect signal
        self.state_updated.connect(self._on_state_update)
    
    def _setup_window(self):
        """Configure window properties"""
        # Frameless, always on top, transparent background
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool  # Don't show in taskbar
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Size - use setMinimumSize/setMaximumSize instead of setFixedSize
        # to avoid UpdateLayeredWindowIndirect warnings during animations
        self.setMinimumWidth(self.cfg.width)
        self.setMaximumWidth(self.cfg.width + 50)  # Allow some buffer
        self.setMinimumHeight(self.cfg.height_collapsed)
        self.setMaximumHeight(self.cfg.height_expanded + 50)  # Allow animation buffer
        self.resize(self.cfg.width, self.cfg.height_collapsed)
        
        # Opacity
        self.setWindowOpacity(self.cfg.opacity)
    
    def _setup_widgets(self):
        """Create UI widgets"""
        
        # Main container with rounded corners
        self.container = QFrame(self)
        self.container.setObjectName("container")
        self.container.setStyleSheet(f"""
            #container {{
                background-color: {self.cfg.color_background};
                border-radius: {self.cfg.corner_radius}px;
                border: 2px solid {self.cfg.color_uncertain};
            }}
        """)
        
        # Layout
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(8)
        
        # Header row (status + close button)
        header_layout = QHBoxLayout()
        
        # Status indicator (emoji)
        self.emoji_label = QLabel("🟡")
        self.emoji_label.setFont(QFont("Segoe UI Emoji", 16))
        header_layout.addWidget(self.emoji_label)
        
        # Status text
        self.status_label = QLabel("STARTING...")
        self.status_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.status_label.setStyleSheet(f"color: {self.cfg.color_text};")
        header_layout.addWidget(self.status_label)
        
        header_layout.addStretch()
        
        # Confidence percentage
        self.confidence_label = QLabel("")
        self.confidence_label.setFont(QFont("Segoe UI", 12))
        self.confidence_label.setStyleSheet(f"color: {self.cfg.color_text}; opacity: 0.8;")
        header_layout.addWidget(self.confidence_label)
        
        # Close button (X)
        self.close_btn = QLabel("✕")
        self.close_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.close_btn.setStyleSheet(f"""
            color: {self.cfg.color_text}; 
            opacity: 0.6;
            padding: 0 5px;
        """)
        self.close_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.close_btn.mousePressEvent = lambda e: self._on_close_click()
        header_layout.addWidget(self.close_btn)
        
        layout.addLayout(header_layout)
        
        # Explanation (hidden by default)
        self.explanation_label = QLabel("")
        self.explanation_label.setFont(QFont("Segoe UI", 10))
        self.explanation_label.setStyleSheet(f"color: {self.cfg.color_text}; opacity: 0.9;")
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setVisible(False)
        layout.addWidget(self.explanation_label)
        
        # Stability indicator
        self.stability_label = QLabel("")
        self.stability_label.setFont(QFont("Segoe UI", 9))
        self.stability_label.setStyleSheet(f"color: {self.cfg.color_text}; opacity: 0.6;")
        self.stability_label.setVisible(False)
        layout.addWidget(self.stability_label)
        
        # Make container fill window
        self.container.setGeometry(0, 0, self.cfg.width, self.cfg.height_collapsed)
        
        # Drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 5)
        self.container.setGraphicsEffect(shadow)
    
    def _setup_animations(self):
        """Setup expand/collapse animation"""
        self.height_animation = QPropertyAnimation(self, b"size")
        self.height_animation.setDuration(200)
        self.height_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    
    def _position_window(self):
        """Position window on screen"""
        screen = QApplication.primaryScreen().geometry()
        
        if self.cfg.position == "top-right":
            x = screen.width() - self.cfg.width - self.cfg.margin
            y = self.cfg.margin
        elif self.cfg.position == "top-left":
            x = self.cfg.margin
            y = self.cfg.margin
        elif self.cfg.position == "bottom-right":
            x = screen.width() - self.cfg.width - self.cfg.margin
            y = screen.height() - self.cfg.height_collapsed - self.cfg.margin
        else:  # bottom-left
            x = self.cfg.margin
            y = screen.height() - self.cfg.height_collapsed - self.cfg.margin
            
        self.move(x, y)
    
    def update_state(
        self,
        level: Optional[ConfidenceLevel],
        confidence_pct: int,
        explanation: str = "",
        stability: str = "",
        faces_detected: int = 0
    ):
        """
        Update the overlay display.
        
        Called from the engine when state changes.
        """
        
        if level is None:
            # No faces detected
            self._set_no_faces_state()
            return
        
        self.current_level = level
        
        # Update color
        color_map = {
            ConfidenceLevel.REAL: self.cfg.color_real,
            ConfidenceLevel.LIKELY_REAL: self.cfg.color_likely_real,
            ConfidenceLevel.UNCERTAIN: self.cfg.color_uncertain,
            ConfidenceLevel.LIKELY_FAKE: self.cfg.color_likely_fake,
            ConfidenceLevel.DEEPFAKE: self.cfg.color_deepfake,
        }
        self.current_color = color_map.get(level, self.cfg.color_uncertain)
        
        # Update emoji
        emoji_map = {
            ConfidenceLevel.REAL: "🟢",
            ConfidenceLevel.LIKELY_REAL: "🟢",
            ConfidenceLevel.UNCERTAIN: "🟡",
            ConfidenceLevel.LIKELY_FAKE: "🟠",
            ConfidenceLevel.DEEPFAKE: "🔴",
        }
        self.emoji_label.setText(emoji_map.get(level, "⚪"))
        
        # Update text
        self.status_label.setText(level.value)
        self.status_label.setStyleSheet(f"color: {self.current_color};")
        
        self.confidence_label.setText(f"{confidence_pct}%")
        self.confidence_label.setStyleSheet(f"color: {self.current_color};")
        
        # Update border color
        self.container.setStyleSheet(f"""
            #container {{
                background-color: {self.cfg.color_background};
                border-radius: {self.cfg.corner_radius}px;
                border: 2px solid {self.current_color};
            }}
        """)
        
        # Update explanation
        self.current_explanation = explanation
        self.explanation_label.setText(explanation)
        
        # Update stability
        self.stability_label.setText(f"{stability} • {faces_detected} face(s)")
    
    def _set_no_faces_state(self):
        """Set state when no faces are detected - show as actively monitoring"""
        self.emoji_label.setText("🛡️")
        self.status_label.setText("SCANNING")
        self.status_label.setStyleSheet(f"color: {self.cfg.color_text}; opacity: 0.7;")
        self.confidence_label.setText("")
        self.explanation_label.setText("Monitoring screen for video content...")
        self.stability_label.setText("Active protection")
        
        self.container.setStyleSheet(f"""
            #container {{
                background-color: {self.cfg.color_background};
                border-radius: {self.cfg.corner_radius}px;
                border: 2px solid #4B5563;
            }}
        """)
    
    def _expand(self):
        """Expand overlay to show explanation"""
        if self.is_expanded:
            return
            
        self.is_expanded = True
        self.explanation_label.setVisible(True)
        self.stability_label.setVisible(True)
        
        # Animate height
        self.height_animation.setStartValue(QSize(self.cfg.width, self.cfg.height_collapsed))
        self.height_animation.setEndValue(QSize(self.cfg.width, self.cfg.height_expanded))
        self.height_animation.start()
        
        # Resize container
        self.container.setFixedHeight(self.cfg.height_expanded)
    
    def _collapse(self):
        """Collapse overlay to minimal state"""
        if not self.is_expanded:
            return
            
        self.is_expanded = False
        
        # Animate height
        self.height_animation.setStartValue(QSize(self.cfg.width, self.cfg.height_expanded))
        self.height_animation.setEndValue(QSize(self.cfg.width, self.cfg.height_collapsed))
        self.height_animation.start()
        
        # Hide extra widgets after animation
        QTimer.singleShot(200, lambda: self._finish_collapse())
    
    def _finish_collapse(self):
        """Finish collapse after animation"""
        if not self.is_expanded:
            self.explanation_label.setVisible(False)
            self.stability_label.setVisible(False)
            self.container.setFixedHeight(self.cfg.height_collapsed)
    
    def _on_close_click(self):
        """Handle close button click - exit application"""
        QApplication.quit()
    
    # --- Event Handlers ---
    
    def enterEvent(self, event):
        """Mouse entered - expand"""
        self._expand()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Mouse left - collapse"""
        self._collapse()
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        """Start dragging"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Drag window"""
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_position:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """Stop dragging"""
        self._drag_position = None
        event.accept()
    
    def _on_state_update(self, state):
        """Handle state update signal from engine"""
        if state.temporal_state:
            self.update_state(
                level=state.temporal_state.result.level,
                confidence_pct=state.temporal_state.result.confidence_pct,
                explanation=state.explanation.text if state.explanation else "",
                stability=state.temporal_state.trend,
                faces_detected=state.faces_detected
            )
        else:
            self.update_state(
                level=None,
                confidence_pct=0,
                explanation="",
                stability="",
                faces_detected=0
            )
