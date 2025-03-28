# File: main_ui.py
import sys
import logging
from typing import Optional, List, Dict, Any

# --- PySide6 Imports ---
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QSplitter, QFrame,
    QStatusBar, QMenuBar, QToolBar, QComboBox, QPlainTextEdit,
    QCheckBox, QSizePolicy
)
from PySide6.QtCore import Qt, Slot, Signal, QTimer # Added QTimer for potential queue polling
from PySide6.QtGui import QAction, QPixmap, QImage # Added QImage for video frame conversion

# --- Placeholder for Async Integration ---
# You'll need to install and import asyncqt or quamash
# Example: from asyncqt import QEventLoop

# --- Placeholder for Backend Imports ---
# import your classes like ChatManager, FelixTrackingClient, config etc.
# from core.chat import ChatManager
# from IseeYou.IseeYou import FelixTrackingClient
# import config


# --- Basic Logging Setup (redirect later) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] %(message)s')


# --- Dark Theme Style Sheet (Example) ---
DARK_STYLE_SHEET = """
    QWidget {
        background-color: #2b2b2b;
        color: #f0f0f0;
        font-size: 10pt;
    }
    QMainWindow {
        border: 1px solid #444;
    }
    QMenuBar, QToolBar {
        background-color: #3c3f41;
        color: #f0f0f0;
    }
    QMenuBar::item:selected, QToolBar::item:selected {
        background-color: #555;
    }
    QMenu {
        background-color: #3c3f41;
        border: 1px solid #555;
    }
    QMenu::item:selected {
        background-color: #555;
    }
    QPushButton {
        background-color: #555;
        border: 1px solid #666;
        padding: 5px;
        min-width: 60px;
    }
    QPushButton:hover {
        background-color: #666;
    }
    QPushButton:pressed {
        background-color: #444;
    }
    QLineEdit, QTextEdit, QPlainTextEdit, QComboBox {
        background-color: #3c3f41;
        border: 1px solid #555;
        padding: 3px;
    }
    QComboBox::drop-down {
        border: none;
    }
    QComboBox::down-arrow {
        image: url(:/qt-project.org/styles/commonstyle/images/down_arrow.png); /* Example */
    }
    QLabel {
        color: #f0f0f0;
    }
    QStatusBar {
        background-color: #3c3f41;
    }
    QSplitter::handle {
        background-color: #444;
    }
    QSplitter::handle:horizontal {
        width: 2px;
    }
    QSplitter::handle:vertical {
        height: 2px;
    }
    QFrame[frameShape="5"] { /* VLine */
        color: #444;
    }
    QCheckBox::indicator {
        width: 13px;
        height: 13px;
    }
    QCheckBox::indicator:unchecked {
        border: 1px solid #777; background-color: #3c3f41;
    }
    QCheckBox::indicator:checked {
        background-color: #55aaff; border: 1px solid #4488cc;
    }
"""

class AssistantMainWindow(QMainWindow):
    """Main application window for the AI Assistant."""

    # --- Define Signals for UI Updates from Backend ---
    # Example signal to add a message to the chat history
    # Arguments: role (str, e.g., "User", "Assistant"), message (str)
    new_chat_message_signal = Signal(str, str)

    # Example signal to add a log entry
    # Arguments: level (int), message (str)
    new_log_message_signal = Signal(int, str)

    # Example signal to update the video frame
    # Arguments: q_image (QImage or QPixmap)
    new_video_frame_signal = Signal(object) # Use object type for flexibility

    # Example signals for status updates
    tracking_status_signal = Signal(bool) # True if active
    connection_status_signal = Signal(bool) # True if connected
    felix_detection_signal = Signal(bool, float) # is_felix, confidence

    # ---------------------------------------------------

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("AI Assistant Control Panel")
        self.setGeometry(100, 100, 1400, 800) # x, y, width, height

        # --- Backend Integration Placeholder ---
        # self.chat_manager = ChatManager(...)
        # self.felix_client = FelixTrackingClient(...)
        # self.config = config
        # ---------------------------------------

        self._create_widgets()
        self._create_layouts()
        self._create_menu_bar()
        self._create_tool_bar() # Optional
        self._create_status_bar()
        self._connect_signals()

        # --- Placeholder for backend signal connections ---
        # self.new_chat_message_signal.connect(self._display_new_message)
        # self.new_log_message_signal.connect(self._add_log_entry)
        # self.new_video_frame_signal.connect(self._update_video_frame)
        # self.tracking_status_signal.connect(self._update_tracking_status_ui)
        # self.connection_status_signal.connect(self._update_connection_status_ui)
        # self.felix_detection_signal.connect(self._update_felix_status_ui)
        # You'll need to modify your backend classes to *emit* these signals
        # ---------------------------------------------------

        logging.info("Main UI Initialized.")
        self.status_bar.showMessage("Ready", 5000) # Show "Ready" for 5 secs

    def _create_widgets(self):
        """Create all the UI widgets."""
        logging.debug("Creating widgets...")
        # --- Left Pane Widgets ---
        self.chat_history_edit = QTextEdit()
        self.chat_history_edit.setReadOnly(True)
        # self.chat_history_edit.setStyleSheet("font-size: 11pt;") # Example styling

        self.user_input_label = QLabel("You:")
        self.user_input_edit = QLineEdit()
        self.user_input_edit.setPlaceholderText("Enter your message here or toggle mic...")
        self.send_button = QPushButton("Send")
        self.mic_toggle_button = QPushButton("🎤") # Placeholder icon/text
        self.mic_toggle_button.setCheckable(True)
        self.mic_toggle_button.setToolTip("Toggle microphone input (STT)")

        self.llm_status_label = QLabel("LLM: --")
        self.speech_status_label = QLabel("Speech: --")

        # --- Right Pane Widgets ---
        self.video_label = QLabel("Video Feed (Not Connected)")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.video_label.setStyleSheet("background-color: black; color: gray;")
        # Set size policy to allow expansion
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


        self.server_conn_label = QLabel("Server: Disc")
        self.tracking_active_label = QLabel("Tracking: Inactive")
        self.felix_detected_label = QLabel("Felix: No (Conf: -.--)" )
        self.camera_combo = QComboBox()
        # Populate camera_combo later (e.g., in setup or dynamically)
        self.camera_combo.addItems(["Cam 0", "Cam 1", "file.mp4"]) # Example items
        self.start_tracking_button = QPushButton("Start Tracking")
        self.stop_tracking_button = QPushButton("Stop Tracking")
        self.stop_tracking_button.setEnabled(False) # Disabled initially

        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.setCurrentText("INFO")
        self.log_filter_edit = QLineEdit()
        self.log_filter_edit.setPlaceholderText("Filter logs...")
        self.log_viewer_edit = QPlainTextEdit()
        self.log_viewer_edit.setReadOnly(True)
        self.log_viewer_edit.setMaximumBlockCount(2000) # Limit history size
        self.clear_log_button = QPushButton("Clear Log")
        self.log_autoscroll_check = QCheckBox("Auto-scroll")
        self.log_autoscroll_check.setChecked(True)

    def _create_layouts(self):
        """Create layouts and arrange widgets."""
        logging.debug("Creating layouts...")
        # --- Main Splitter ---
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left Pane Layout ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Input Area
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.user_input_label)
        input_layout.addWidget(self.user_input_edit, 1) # Stretch input edit
        input_layout.addWidget(self.send_button)
        input_layout.addWidget(self.mic_toggle_button)

        # Status Area
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.llm_status_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.speech_status_label)

        left_layout.addWidget(QLabel("Conversation:")) # Section Title
        left_layout.addWidget(self.chat_history_edit, 1) # Stretch chat history
        left_layout.addLayout(input_layout)
        left_layout.addLayout(status_layout)

        self.main_splitter.addWidget(left_widget)

        # --- Right Pane Layout ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Vision Panel
        vision_panel = QWidget()
        vision_layout = QVBoxLayout(vision_panel)
        vision_panel.setFrameShape(QFrame.Shape.StyledPanel) # Add border

        vision_status_layout = QHBoxLayout()
        vision_status_layout.addWidget(self.server_conn_label)
        vision_status_layout.addWidget(self.tracking_active_label)
        vision_status_layout.addWidget(self.felix_detected_label)
        vision_status_layout.addStretch(1)

        vision_control_layout = QHBoxLayout()
        vision_control_layout.addWidget(QLabel("Camera:"))
        vision_control_layout.addWidget(self.camera_combo)
        vision_control_layout.addStretch(1)
        vision_control_layout.addWidget(self.start_tracking_button)
        vision_control_layout.addWidget(self.stop_tracking_button)

        vision_layout.addLayout(vision_status_layout)
        vision_layout.addLayout(vision_control_layout)

        # Log Viewer Panel
        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_panel.setFrameShape(QFrame.Shape.StyledPanel) # Add border

        log_filter_layout = QHBoxLayout()
        log_filter_layout.addWidget(QLabel("Level:"))
        log_filter_layout.addWidget(self.log_level_combo)
        log_filter_layout.addWidget(QLabel("Filter:"))
        log_filter_layout.addWidget(self.log_filter_edit, 1)
        log_filter_layout.addWidget(self.log_autoscroll_check)
        log_filter_layout.addWidget(self.clear_log_button)

        log_layout.addLayout(log_filter_layout)
        log_layout.addWidget(self.log_viewer_edit, 1) # Stretch log view

        # Assemble Right Pane
        right_layout.addWidget(QLabel("Video Feed:")) # Section Title
        right_layout.addWidget(self.video_label, 2) # Give video more stretch factor
        right_layout.addWidget(vision_panel)
        right_layout.addWidget(log_panel, 1) # Give logs stretch factor

        self.main_splitter.addWidget(right_widget)

        # Configure Splitter
        self.main_splitter.setStretchFactor(0, 1) # Give left pane factor 1
        self.main_splitter.setStretchFactor(1, 2) # Give right pane factor 2 (wider)
        self.main_splitter.setSizes([400, 1000]) # Initial sizes

        self.setCentralWidget(self.main_splitter)

    def _create_menu_bar(self):
        logging.debug("Creating menu bar...")
        self.menu_bar = self.menuBar()

        # File Menu
        file_menu = self.menu_bar.addMenu("&File")
        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Tracking Menu
        tracking_menu = self.menu_bar.addMenu("&Tracking")
        start_action = QAction("Start Tracking", self)
        start_action.triggered.connect(self._start_tracking) # Connect to slot
        self.start_tracking_action = start_action # Keep reference if needed
        tracking_menu.addAction(start_action)

        stop_action = QAction("Stop Tracking", self)
        stop_action.triggered.connect(self._stop_tracking) # Connect to slot
        self.stop_tracking_action = stop_action # Keep reference
        stop_action.setEnabled(False) # Match button state
        tracking_menu.addAction(stop_action)

        # Settings Menu
        settings_menu = self.menu_bar.addMenu("&Settings")
        # Add action to open settings dialog later

        # Help Menu
        help_menu = self.menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        # about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

    def _create_tool_bar(self):
        """Create an optional toolbar for common actions."""
        logging.debug("Creating tool bar...")
        self.tool_bar = QToolBar("Main Toolbar")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.tool_bar)

        # Add actions from menu (or create new ones with icons)
        connect_action = QAction("Connect", self) # Example
        # connect_action.setIcon(QIcon("path/to/connect_icon.png"))
        connect_action.triggered.connect(self._connect_to_server)
        self.tool_bar.addAction(connect_action)

        self.tool_bar.addAction(self.start_tracking_action)
        self.tool_bar.addAction(self.stop_tracking_action)
        # Add more buttons like clear log etc.

    def _create_status_bar(self):
        logging.debug("Creating status bar...")
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initialized")

    def _connect_signals(self):
        """Connect UI widget signals to internal slots."""
        logging.debug("Connecting UI signals...")
        self.send_button.clicked.connect(self._send_message)
        self.user_input_edit.returnPressed.connect(self._send_message) # Send on Enter key
        self.mic_toggle_button.toggled.connect(self._toggle_microphone)

        self.start_tracking_button.clicked.connect(self._start_tracking)
        self.stop_tracking_button.clicked.connect(self._stop_tracking)

        self.clear_log_button.clicked.connect(self.log_viewer_edit.clear)
        self.log_level_combo.currentTextChanged.connect(self._update_log_filter)
        self.log_filter_edit.textChanged.connect(self._update_log_filter)
        # Auto-scroll connection is handled within the log adding slot

    # --- Action/Slot Methods (Placeholders) ---

    @Slot()
    def _send_message(self):
        """Handles sending text input from the user."""
        user_text = self.user_input_edit.text().strip()
        if user_text:
            logging.info(f"UI: Sending message: {user_text}")
            self.status_bar.showMessage("Sending message...")
            # --- Backend Integration ---
            # 1. Append user message to chat history UI immediately
            self._display_new_message("User", user_text)
            # 2. Clear the input field
            self.user_input_edit.clear()
            # 3. Call your ChatManager's processing function
            #    If it's async, use asyncio.create_task
            # Example: asyncio.create_task(self.chat_manager.process_conversation_turn(user_text))
            #    The ChatManager should then emit a signal with the AI response.
            # -------------------------
            # Placeholder: Simulate AI response after delay
            # QTimer.singleShot(1500, lambda: self._display_new_message("Assistant", f"Simulated reply to '{user_text}'"))
        else:
            logging.debug("UI: Send button clicked with empty input.")

    @Slot(bool)
    def _toggle_microphone(self, checked):
        """Handles toggling the microphone for STT."""
        if checked:
            logging.info("UI: Microphone ON (STT Active - Placeholder)")
            self.status_bar.showMessage("Listening...")
            self.mic_toggle_button.setText("...") # Indicate listening
            # --- Backend Integration ---
            # Start your STT process here.
            # When STT has a result, it should perhaps fill the user_input_edit
            # or directly trigger the send process.
            # -------------------------
        else:
            logging.info("UI: Microphone OFF (STT Inactive - Placeholder)")
            self.status_bar.showMessage("Ready")
            self.mic_toggle_button.setText("🎤")
            # --- Backend Integration ---
            # Stop your STT process if it's continuous.
            # -------------------------

    @Slot()
    def _start_tracking(self):
        """Handles starting the Felix video tracking."""
        logging.info("UI: Start Tracking clicked.")
        self.status_bar.showMessage("Starting tracking...")
        selected_source = self.camera_combo.currentText()
        logging.info(f"UI: Selected source: {selected_source}")

        # --- Backend Integration ---
        # Convert selected_source to int if possible, otherwise use as string
        try:
            source_arg = int(selected_source)
        except ValueError:
            source_arg = selected_source

        # Call your FelixTrackingClient's start method
        # Example: Use asyncio.create_task for the async method
        # task = asyncio.create_task(self.felix_client.start_tracking(video_source=source_arg))
        # You might want to handle the boolean result of start_tracking
        # For now, just update UI assuming it will start
        # -------------------------

        # Update UI immediately (backend should confirm via signals)
        self._update_tracking_status_ui(True) # Assume success for now


    @Slot()
    def _stop_tracking(self):
        """Handles stopping the Felix video tracking."""
        logging.info("UI: Stop Tracking clicked.")
        self.status_bar.showMessage("Stopping tracking...")

        # --- Backend Integration ---
        # Call your FelixTrackingClient's stop method
        # Example: Use asyncio.create_task for the async method
        # asyncio.create_task(self.felix_client.stop_tracking())
        # -------------------------

        # Update UI immediately (backend should confirm via signals)
        self._update_tracking_status_ui(False) # Assume success for now

    @Slot()
    def _connect_to_server(self):
        """Placeholder for connecting to the GPU server if needed separately."""
        logging.info("UI: Connect action triggered (Placeholder).")
        self.status_bar.showMessage("Connecting to server...")
        # --- Backend Integration ---
        # Call connect method if your client has one separated from start_tracking
        # Example: asyncio.create_task(self.felix_client.connect())
        # Update self._update_connection_status_ui based on result/signal
        # -------------------------

    @Slot()
    def _update_log_filter(self):
        """Placeholder slot for when log filter controls change."""
        # This logic would typically be in the _add_log_entry slot
        # to decide *if* a log message should be displayed based on current filters.
        level_text = self.log_level_combo.currentText()
        filter_text = self.log_filter_edit.text()
        logging.debug(f"Log filter updated: Level={level_text}, Filter='{filter_text}'")
        # You might need to re-filter existing logs if implementing dynamic filtering


    # --- UI Update Slots (Connected to Backend Signals) ---

    @Slot(str, str)
    def _display_new_message(self, role, message):
        """Appends a new message to the chat history view."""
        # Simple text append
        # For styling, you might use self.chat_history_edit.insertHtml(...)
        styled_message = f"<b>{role}:</b> {message}<br>" # Basic HTML styling
        self.chat_history_edit.append(styled_message) # Use append for QTextEdit
        self.status_bar.showMessage(f"{role} message received", 3000)

    @Slot(int, str)
    def _add_log_entry(self, level, message):
        """Adds a log message to the log viewer, applying filters."""
        level_name = logging.getLevelName(level)

        # Apply Level Filter
        current_level_filter_str = self.log_level_combo.currentText()
        current_level_filter_int = logging.getLevelName(current_level_filter_str)
        if level < current_level_filter_int:
            return # Skip messages below selected level

        # Apply Text Filter
        filter_text = self.log_filter_edit.text().lower()
        if filter_text and filter_text not in message.lower():
            return # Skip messages not matching text filter

        # Format and append
        log_entry = f"[{level_name}] {message}" # Basic format
        # Could add timestamp here: log_entry = f"{datetime.now():%H:%M:%S} [{level_name}] {message}"

        # Add color based on level (requires using HTML or custom formatting)
        # Example basic coloring:
        color = "white"
        if level == logging.WARNING: color = "yellow"
        elif level >= logging.ERROR: color = "red"
        elif level == logging.DEBUG: color = "gray"

        # Using insertHtml for colors in QPlainTextEdit can be slow.
        # A simpler approach is just appending text, or using a QSyntaxHighlighter.
        # For simplicity now, just append plain text. Richer logging can be added later.
        self.log_viewer_edit.appendPlainText(log_entry)

        if self.log_autoscroll_check.isChecked():
             self.log_viewer_edit.verticalScrollBar().setValue(
                 self.log_viewer_edit.verticalScrollBar().maximum()
             )

    @Slot(object)
    def _update_video_frame(self, q_image):
        """Updates the video feed QLabel with a new QImage or QPixmap."""
        if isinstance(q_image, (QImage, QPixmap)):
            # Scale pixmap to fit the label while keeping aspect ratio
            pixmap = QPixmap.fromImage(q_image) if isinstance(q_image, QImage) else q_image
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
        else:
            logging.warning(f"Received invalid type for video frame update: {type(q_image)}")
            self.video_label.setText("Error displaying frame")
            self.video_label.setStyleSheet("background-color: red; color: white;")

    @Slot(bool)
    def _update_tracking_status_ui(self, is_active):
        """Updates UI elements related to tracking status."""
        logging.debug(f"Updating tracking status UI: {'Active' if is_active else 'Inactive'}")
        self.tracking_active_label.setText(f"Tracking: {'Active' if is_active else 'Inactive'}")
        self.start_tracking_button.setEnabled(not is_active)
        self.start_tracking_action.setEnabled(not is_active)
        self.stop_tracking_button.setEnabled(is_active)
        self.stop_tracking_action.setEnabled(is_active)
        self.camera_combo.setEnabled(not is_active) # Disable camera change while active
        if not is_active:
            # Optionally clear video label when tracking stops
            self.video_label.clear()
            self.video_label.setText("Video Feed (Tracking Inactive)")
            self.video_label.setStyleSheet("background-color: black; color: gray;")
            self.felix_detected_label.setText("Felix: -- (Conf: -.--)")


    @Slot(bool)
    def _update_connection_status_ui(self, is_connected):
        """Updates UI elements related to server connection status."""
        logging.debug(f"Updating connection status UI: {'Connected' if is_connected else 'Disconnected'}")
        self.server_conn_label.setText(f"Server: {'Conn' if is_connected else 'Disc'}")
        # Maybe enable/disable tracking buttons based on connection
        if not is_connected:
             self._update_tracking_status_ui(False) # Ensure tracking UI is off if disconnected


    @Slot(bool, float)
    def _update_felix_status_ui(self, is_felix, confidence):
        """Updates the label showing Felix detection status."""
        # Only update if tracking is active
        if self.stop_tracking_button.isEnabled(): # Check if tracking is supposed to be active
             status_text = f"Felix: {'Yes' if is_felix else 'No'} (Conf: {confidence:.2f})"
             self.felix_detected_label.setText(status_text)


    # --- Overrides ---
    def closeEvent(self, event):
        """Handle the window closing event for graceful shutdown."""
        logging.info("Close event triggered. Initiating shutdown...")
        self.status_bar.showMessage("Shutting down...")
        # --- Backend Integration ---
        # Signal your backend components to shut down gracefully
        # Example:
        # if self.felix_client:
        #     self.felix_client.shutdown() # Assuming a method like this exists
        # Stop any running timers or background threads managed by the UI
        # -------------------------
        logging.info("Exiting application.")
        super().closeEvent(event)


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE_SHEET) # Apply the dark theme

    # --- Async Integration ---
    # If using asyncqt:
    # loop = QEventLoop(app)
    # asyncio.set_event_loop(loop)
    # -----------------------

    window = AssistantMainWindow()
    window.show()

    # --- Async Integration ---
    # If using asyncqt:
    # with loop:
    #     sys.exit(loop.run_forever())
    # Else (standard PySide event loop):
    sys.exit(app.exec())
    # -----------------------