import sys
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QFrame, QSizePolicy, QMainWindow)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QCursor, QIcon

from ui import Ui_AudioForgeryDetection


class HomePage(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio_Q 音频伪造检测系统")
        self.setWindowIcon(QIcon("icon.jpg"))
        self.resize(1280, 800)
        self.init_ui()

    def init_ui(self):
        self.main_window = None
        self.ui = None

        self.main_frame = QFrame(self)
        self.main_frame.setObjectName(u"main_frame")
        self.main_frame.setStyleSheet(
            u"QFrame#main_frame{\n"
            "	background-color: rgb(248, 249, 250);\n"
            "border:0px solid red;\n"
            "border-radius: 0px;\n"
            "}"
        )
        self.setCentralWidget(self.main_frame)

        content_layout = QVBoxLayout(self.main_frame)
        content_layout.setSpacing(0)
        content_layout.setContentsMargins(0, 0, 0, 0)

        content_layout.addStretch(50)

        hero_section = QFrame(self.main_frame)
        hero_section.setStyleSheet("QFrame { background-color: rgb(248, 249, 250); }")
        hero_layout = QVBoxLayout(hero_section)
        hero_layout.setSpacing(20)
        hero_layout.setContentsMargins(0, 30, 0, 30)

        title = QLabel("Audio_Q 音频伪造检测系统")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(32)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: rgb(33, 37, 43);")

        subtitle = QLabel("安全可靠 · 多格式支持 · 智能检测")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle_font = QFont()
        subtitle_font.setPointSize(16)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet("color: rgb(108, 117, 125);")

        start_btn = QPushButton("开始体验")
        start_btn.setMinimumSize(QSize(160, 48))
        start_btn.setCursor(QCursor(Qt.PointingHandCursor))
        start_btn.setStyleSheet(
            "QPushButton {"
            "background-color: rgb(13, 110, 253);"
            "color: white;"
            "border: none;"
            "border-radius: 8px;"
            "font-size: 18px;"
            "font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "background-color: rgb(10, 88, 202);"
            "}"
        )
        start_btn.clicked.connect(self.go_to_main_ui)

        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        hero_layout.addWidget(start_btn, alignment=Qt.AlignCenter)

        content_layout.addWidget(hero_section)

        content_layout.addSpacing(30)

        cards_section = QFrame(self.main_frame)
        cards_section.setStyleSheet("QFrame { background-color: white; }")
        cards_layout = QVBoxLayout(cards_section)
        cards_layout.setSpacing(20)
        cards_layout.setContentsMargins(80, 25, 80, 25)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(15)

        card1 = self.create_feature_card(
            "安全可靠",
            "多维度分析与校验，精准识别伪造痕迹。",
            "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgb(13, 110, 253), stop:1 rgb(66, 133, 244))"
        )

        card2 = self.create_feature_card(
            "多格式支持",
            "WAV / MP3 / FLAC 等主流格式。",
            "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgb(13, 110, 253), stop:1 rgb(66, 133, 244))"
        )

        card3 = self.create_feature_card(
            "智能检测",
            "智能分析音频真实性，生成伪造风险报告。",
            "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgb(13, 110, 253), stop:1 rgb(66, 133, 244))"
        )

        cards_row.addWidget(card1)
        cards_row.addWidget(card2)
        cards_row.addWidget(card3)

        cards_layout.addLayout(cards_row)

        content_layout.addWidget(cards_section)

        footer = QFrame(self.main_frame)
        footer.setMinimumSize(QSize(0, 60))
        footer.setStyleSheet(
            "QFrame {"
            "background-color: rgb(248, 249, 250);"
            "border-top: 1px solid rgb(222, 226, 230);"
            "}"
        )
        footer_layout = QHBoxLayout(footer)

        footer_label = QLabel("© Audio_Q音频伪造检测系统")
        footer_label.setStyleSheet("color: rgb(108, 117, 125); font-size: 14px;")

        footer_layout.addWidget(footer_label, alignment=Qt.AlignCenter)

        content_layout.addWidget(footer)

        content_layout.addStretch(30)

    def go_to_main_ui(self):
        self.main_window = QMainWindow()
        self.ui = Ui_AudioForgeryDetection()
        self.ui.setupUi(self.main_window)
        self.main_window.show()
        self.close()

    def create_feature_card(self, title, description, gradient_color):
        card = QFrame()
        card.setMinimumSize(QSize(180, 90))
        card.setCursor(QCursor(Qt.PointingHandCursor))
        card.setStyleSheet(
            f"QFrame {{"
            f"background-color: {gradient_color};"
            f"border-radius: 8px;"
            f"}}"
        )

        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(8)
        card_layout.setContentsMargins(12, 12, 12, 12)

        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: white;")

        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: rgba(255, 255, 255, 200); font-size: 10px;")

        card_layout.addWidget(title_label)
        card_layout.addWidget(desc_label)

        return card


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = HomePage()
    window.show()
    sys.exit(app.exec())