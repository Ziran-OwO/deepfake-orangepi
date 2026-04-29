# -*- coding: utf-8 -*-

from PySide6.QtCore import (
    QCoreApplication,
    QDate,
    QDateTime,
    QLocale,
    QMetaObject,
    QObject,
    QPoint,
    QRect,
    QRectF,
    QSize,
    QTime,
    QUrl,
    Qt,
    Signal,
    Slot,
    Q_ARG,
    QPointF,
    QTimer,
    QEvent,
    QThread,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QCursor,
    QFont,
    QFontDatabase,
    QGradient,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPalette,
    QPixmap,
    QRadialGradient,
    QTransform,
    QPen,
    QPainterPath,
    QFont,
    QColor,
    QBrush,
    QLinearGradient,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QFileDialog,
    QHeaderView,
    QTableWidgetItem,
    QInputDialog,
    QLineEdit,
    QScrollArea
)
import numpy as np
import wave
import struct
import torch
import librosa
import python_speech_features as ps
import os
import warnings
import threading
import sys
import time
import pandas as pd
import json
import textwrap
import umap

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LEMAS import LEMAS
import psycopg2
from psycopg2 import Error

# 模型相关配置
MAX_LEN = 64600
SR = 16000
MEL_BEGIN = 0
MEL_END = 300
NFILT = 40
THRESHOLD_GLOBAL = 0.87  # 全局伪造判断阈值
THRESHOLD_FRAME_PROB = 0.5  # 帧级伪造概率阈值
MODEL_PATH = "./e20_tloss0.0023_dloss0.0051_deer0.1570.pth"

# 滑动窗口参数
SLIDE_STEP = 1600  # 滑动步长0.1秒 = 1600采样点
SEGMENT_DURATION = MAX_LEN / SR  # 4.0375秒
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
torch.set_default_tensor_type(torch.FloatTensor)
model = None


def init_model():
    """初始化LEMAS模型"""
    global model
    if model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

            model = LEMAS().to(device)
            ckpt = torch.load(MODEL_PATH, map_location=device)
            load_dict = ckpt if not isinstance(ckpt, dict) else ckpt.get("state_dict", ckpt)
            model.load_state_dict(load_dict, strict=False)
            model.eval()
            return True
        except Exception as e:
            QMessageBox.critical(None, "模型加载错误", f"加载模型失败：{str(e)}")
            return False
    return True

def init_database():
    """初始化OpenGauss数据库连接，并创建/更新Audio_Q表"""
    try:
        conn = psycopg2.connect(
            host="127.0.0.1",
            port="7654",
            database="postgres",
            user="opengauss",
            password="123@Abc12"
        )
        cursor = conn.cursor()

        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'audio_q'")
        table_exists = cursor.fetchone()

        if table_exists:
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'audio_q'")
            columns = [row[0] for row in cursor.fetchall()]

            if 'fake_frames' in columns or 'frame_threshold' in columns:
                print("检测到旧表结构，删除旧表并创建新表...")
                cursor.execute("DROP TABLE audio_q")

                create_table_sql = """
                CREATE TABLE audio_q (
                    id SERIAL PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    audio_duration TEXT NOT NULL,
                    fake_prob REAL NOT NULL,
                    conclusion TEXT NOT NULL,
                    fake_segments INTEGER NOT NULL,
                    global_threshold REAL NOT NULL,
                    status TEXT NOT NULL
                );
                """
                cursor.execute(create_table_sql)
                print("已创建新表")
            else:
                print("表结构已是最新版本")
        else:
            create_table_sql = """
            CREATE TABLE audio_q (
                id SERIAL PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                audio_duration TEXT NOT NULL,
                fake_prob REAL NOT NULL,
                conclusion TEXT NOT NULL,
                fake_segments INTEGER NOT NULL,
                global_threshold REAL NOT NULL,
                status TEXT NOT NULL
            );
            """
            cursor.execute(create_table_sql)
            print("创建新表")

        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Error as e:
        QMessageBox.critical(None, "数据库操作错误", f"数据库操作失败：{str(e)}")
        return False


def insert_detection_result_to_db(result_data):
    """
    插入检测结果到OpenGauss数据库
    """
    try:
        conn = psycopg2.connect(
            host="127.0.0.1",
            port="7654",
            database="postgres",
            user="opengauss",
            password="123@Abc12"
        )
        cursor = conn.cursor()

        insert_sql = """
        INSERT INTO audio_q (
            file_name, file_path, audio_duration, fake_prob, conclusion,
            fake_segments, global_threshold,
            status
        ) VALUES (
            %(file_name)s, %(file_path)s, %(audio_duration)s, %(fake_prob)s, %(conclusion)s,
            %(fake_segments)s, %(global_threshold)s,
            %(status)s
        );
        """

        cursor.execute(insert_sql, result_data)
        conn.commit()

        cursor.close()
        conn.close()
        return True
    except Error as e:
        QMessageBox.warning(None, "数据库插入错误", f"保存结果到数据库失败：{str(e)}")
        return False


def pad(x, max_len=64600):
    """音频填充/截断"""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def extract_mel(x):
    """提取mel频谱特征"""
    mel_spec = ps.logfbank(x, SR, nfilt=NFILT)
    delta1 = ps.delta(mel_spec, 2)
    delta2 = ps.delta(delta1, 2)
    fea = np.stack([
        mel_spec[MEL_BEGIN:MEL_END, :],
        delta1[MEL_BEGIN:MEL_END, :],
        delta2[MEL_BEGIN:MEL_END, :]
    ], axis=0)
    return fea


def preprocess_audio(file_path):
    """音频预处理，返回模型输入的张量"""
    wave, _ = librosa.load(file_path, sr=SR)
    padded_wave = pad(wave, MAX_LEN)
    spectrogram = extract_mel(padded_wave)
    wave_tensor = torch.tensor(padded_wave, dtype=torch.float32).unsqueeze(0).to(device)
    spec_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).to(device)
    freq_aug = torch.tensor([False], dtype=torch.bool).to(device)

    return wave_tensor, spec_tensor, freq_aug, wave


def preprocess_audio_segment(wave):
    """音频段预处理函数"""
    padded_wave = pad(wave, MAX_LEN)
    spectrogram = extract_mel(padded_wave)
    wave_tensor = torch.tensor(padded_wave, dtype=torch.float32).unsqueeze(0).to(device)
    spec_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).to(device)
    freq_aug = torch.tensor([False], dtype=torch.bool).to(device)

    return wave_tensor, spec_tensor, freq_aug, padded_wave

class WaveformWidget(QWidget):
    """音频波形"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.waveform_data = None
        self.setMinimumSize(QSize(0, 100))
        self.setStyleSheet(
            u"background-color: white;\n"
            "border-radius:10px;\n"
            "padding: 10px;"
        )
    def set_waveform_data(self, waveform):
        """设置波形数据并触发重绘"""
        if waveform is None or len(waveform) == 0:
            self.waveform_data = None
            self.update()
            return

        target_samples = 1000
        if len(waveform) > target_samples:
            step = len(waveform) // target_samples
            self.waveform_data = np.array([
                np.mean(np.abs(waveform[i:i + step]))
                for i in range(0, len(waveform) - step, step)
            ])
        else:
            self.waveform_data = np.abs(waveform)

        max_val = np.max(self.waveform_data)
        if max_val > 0:
            self.waveform_data = self.waveform_data / max_val

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        painter.setPen(QColor(80, 80, 80))
        painter.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
        painter.drawText(15, 25, "音频波形")
        if self.waveform_data is None or len(self.waveform_data) == 0:
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont("Microsoft YaHei UI", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "音频波形展示区")
            return
        draw_width = self.width() - 20
        draw_height = self.height() - 20
        center_y = self.height() // 2
        pen = QPen(QColor(30, 150, 255), 2, Qt.SolidLine)
        brush = QBrush(QColor(30, 150, 255, 50))
        painter.setPen(pen)
        painter.setBrush(brush)
        x_step = draw_width / len(self.waveform_data)
        path = QPainterPath()
        path.moveTo(10, center_y)

        for i, val in enumerate(self.waveform_data):
            x = 10 + i * x_step
            y = center_y - val * draw_height / 2
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        for i in reversed(range(len(self.waveform_data))):
            x = 10 + i * x_step
            val = self.waveform_data[i]
            y = center_y + val * draw_height / 2
            path.lineTo(x, y)

        path.closeSubpath()
        painter.fillPath(path, brush)
        painter.drawPath(path)
        center_pen = QPen(QColor(200, 200, 200), 1, Qt.DashLine)
        painter.setPen(center_pen)
        painter.drawLine(10, center_y, 10 + draw_width, center_y)

class SegmentBarChartWidget(QWidget):
    """滑动窗口检测的连续伪造概率曲线"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.segment_probs = None  # 每段的伪造概率
        self.segment_timestamps = None  # 每段的起始时间
        self.audio_duration = 0  # 音频总时长
        self.threshold = THRESHOLD_GLOBAL  # 全局阈值
        self.slide_step = 1.0  # 滑动步长
        self.use_sliding_window = True  # 是否使用滑动窗口
        self.setMinimumSize(QSize(0, 120))
        self.setStyleSheet(
            u"background-color: white;\n"
            "border-radius:10px;\n"
            "padding: 10px;"
        )

    def set_segment_data(self, segment_probs, segment_timestamps, audio_duration, slide_step=1.0,
                         use_sliding_window=True):
        if segment_probs is None or len(segment_probs) == 0:
            self.segment_probs = None
            self.segment_timestamps = None
            self.audio_duration = 0
            self.use_sliding_window = True
            self.update()
            return

        self.segment_probs = np.array(segment_probs)
        self.segment_timestamps = np.array(segment_timestamps) if segment_timestamps is not None else None
        self.audio_duration = audio_duration
        self.slide_step = slide_step
        self.use_sliding_window = use_sliding_window
        self.update()

    def set_segment_probs(self, segment_probs, segment_duration=4.0):
        if segment_probs is None or len(segment_probs) == 0:
            self.segment_probs = None
            self.update()
            return
        self.segment_probs = np.array(segment_probs)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        if self.segment_probs is None or len(self.segment_probs) == 0:
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont("Microsoft YaHei UI", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "滑动窗口伪造概率曲线")
            return

        if hasattr(self, 'use_sliding_window') and not self.use_sliding_window:
            self._draw_short_audio_ring(painter)
        else:
            self._draw_probability_curve(painter)

    def _draw_probability_curve(self, painter):
        painter.setPen(QColor(80, 80, 80))
        painter.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
        painter.drawText(15, 25, "滑动窗口伪造概率曲线")

        draw_width = self.width() - 80
        draw_height = self.height() - 80
        x_start = 60
        y_start = 40
        threshold_prob = 50.0

        try:
            from scipy.ndimage import gaussian_filter1d
            smoothed_probs = gaussian_filter1d(self.segment_probs, sigma=1.0)
        except ImportError:
            smoothed_probs = self.segment_probs

        axis_pen = QPen(QColor(200, 200, 200), 1)
        painter.setPen(axis_pen)
        painter.drawLine(x_start, y_start, x_start, y_start + draw_height)

        painter.setFont(QFont("Microsoft YaHei UI", 8))
        painter.setPen(QColor(100, 100, 100))
        for i in range(5):
            y_pos = y_start + draw_height - (i * draw_height / 4)
            prob_value = i * 25
            painter.drawText(QRect(0, y_pos - 10, x_start - 5, 20),
                             Qt.AlignRight | Qt.AlignVCenter, f"{prob_value}%")

        threshold_y = y_start + draw_height - (threshold_prob / 100 * draw_height)
        threshold_pen = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
        painter.setPen(threshold_pen)
        painter.drawLine(x_start, int(threshold_y), x_start + draw_width, int(threshold_y))

        painter.setFont(QFont("Microsoft YaHei UI", 8))
        painter.setPen(QColor(255, 0, 0))
        painter.drawText(QRect(x_start + draw_width + 5, int(threshold_y) - 10, 100, 20),
                         Qt.AlignLeft | Qt.AlignVCenter, f"阈值: {threshold_prob:.1f}%")

        num_points = len(smoothed_probs)
        points = []

        # 构建曲线点
        for i, prob in enumerate(smoothed_probs):
            x = x_start + (i / (num_points - 1)) * draw_width if num_points > 1 else x_start
            y = y_start + draw_height - (prob / 100) * draw_height
            points.append(QPointF(x, y))

        # 绘制曲线下的填充区域
        if len(points) > 1:
            # 创建渐变填充
            gradient = QLinearGradient(0, y_start, 0, y_start + draw_height)
            gradient.setColorAt(0, QColor(255, 100, 100, 150))
            gradient.setColorAt(threshold_prob / 100, QColor(255, 200, 100, 100))
            gradient.setColorAt(1, QColor(100, 200, 100, 50))

            # 绘制填充多边形
            fill_path = QPainterPath()
            fill_path.moveTo(points[0].x(), y_start + draw_height)
            for pt in points:
                fill_path.lineTo(pt.x(), pt.y())
            fill_path.lineTo(points[-1].x(), y_start + draw_height)
            fill_path.closeSubpath()

            painter.fillPath(fill_path, QBrush(gradient))

        # 绘制曲线
        curve_pen = QPen(QColor(66, 133, 244), 2)
        painter.setPen(curve_pen)
        if len(points) > 1:
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])

        # 绘制数据点
        painter.setPen(QPen(QColor(66, 133, 244), 3))
        for pt in points:
            painter.drawEllipse(pt, 2, 2)

        # 绘制X轴（时间轴）
        painter.setPen(axis_pen)
        painter.drawLine(x_start, y_start + draw_height, x_start + draw_width, y_start + draw_height)

        # 绘制X轴标签（时间）
        painter.setFont(QFont("Microsoft YaHei UI", 8))
        painter.setPen(QColor(100, 100, 100))

        # 根据音频时长确定刻度间隔
        if self.audio_duration > 0:
            if self.audio_duration <= 10:
                tick_interval = 2
            elif self.audio_duration <= 30:
                tick_interval = 5
            elif self.audio_duration <= 60:
                tick_interval = 10
            else:
                tick_interval = 30

            for t in range(0, int(self.audio_duration) + 1, tick_interval):
                x = x_start + (t / self.audio_duration) * draw_width
                painter.drawText(QRect(int(x) - 20, y_start + draw_height + 5, 40, 20),
                                 Qt.AlignCenter, f"{t}s")

    def _draw_short_audio_ring(self, painter):
        """绘制短音频的圆环图"""
        # 计算中心位置
        center_x = self.width() // 2
        center_y = self.height() // 2
        radius = min(center_x, center_y) - 30
        pen_width = 20

        # 绘制标题
        painter.setPen(QColor(80, 80, 80))
        painter.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
        painter.drawText(15, 25, "伪造概率")

        # 获取伪造概率
        fake_prob = self.segment_probs[0]
        real_prob = 100.0 - fake_prob

        # 绘制背景圆环
        painter.setPen(QPen(QColor(200, 200, 200), pen_width))
        painter.drawEllipse(center_x - radius, center_y - radius, 2 * radius, 2 * radius)

        # 绘制伪造概率部分
        if fake_prob > 0:
            fake_angle = int(fake_prob * 360 / 100)
            painter.setPen(QPen(QColor(255, 100, 100), pen_width))
            painter.drawArc(center_x - radius, center_y - radius, 2 * radius, 2 * radius, 90 * 16, -fake_angle * 16)

        # 绘制真实概率部分
        if real_prob > 0:
            real_angle = int(real_prob * 360 / 100)
            painter.setPen(QPen(QColor(100, 200, 100), pen_width))
            painter.drawArc(center_x - radius, center_y - radius, 2 * radius, 2 * radius, 90 * 16 - fake_angle * 16,
                            -real_angle * 16)

        # 绘制中心文字
        painter.setPen(QColor(80, 80, 80))
        painter.setFont(QFont("Microsoft YaHei UI", 14, QFont.Bold))
        prob_text = f"{fake_prob:.1f}%"
        painter.drawText(QRect(center_x - 40, center_y - 15, 80, 30), Qt.AlignCenter, prob_text)

        # 绘制下方标签
        painter.setFont(QFont("Microsoft YaHei UI", 10))
        painter.setPen(QColor(100, 100, 100))
        label_text = f"真实: {real_prob:.1f}%"
        painter.drawText(QRect(center_x - 50, center_y + 10, 100, 20), Qt.AlignCenter, label_text)

    def _logit_to_fake_prob(self, threshold):
        """将real logit阈值转换为伪造概率"""
        fake_prob = 100 * (np.exp(-threshold) / (np.exp(-threshold) + np.exp(threshold)))
        return fake_prob


#UMAP聚类可视化
class UMAPPlotWidget(QWidget):
    """UMAP聚类可视化控件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.umap_reducer = None  # UMAP降维器
        self.baseline_2d = None  # 基准向量的2D坐标
        self.baseline_labels = None  # 基准向量的标签
        self.upload_2d = None  # 上传音频的2D坐标
        self.setMinimumSize(QSize(0, 120))
        self.setStyleSheet(
            u"background-color: white;\n"
            "border-radius:10px;\n"
            "padding: 10px;"
        )

    def load_baseline_vectors(self):
        """加载基准音频的投影向量"""
        baseline_vectors = []
        baseline_labels = []

        fake_folder = "fake"
        real_folder = "real"

        if os.path.exists(fake_folder):
            for file_name in os.listdir(fake_folder):
                if file_name.endswith('.flac'):
                    file_path = os.path.join(fake_folder, file_name)
                    npy_path = file_path.replace('.flac', '_projected.npy')

                    if os.path.exists(npy_path):
                        vector = np.load(npy_path)
                    else:
                        vector = self.extract_projection_vector(file_path)
                        if vector is not None:
                            np.save(npy_path, vector)

                    if vector is not None:
                        baseline_vectors.append(vector)
                        baseline_labels.append(1)

        if os.path.exists(real_folder):
            for file_name in os.listdir(real_folder):
                if file_name.endswith('.flac'):
                    file_path = os.path.join(real_folder, file_name)
                    npy_path = file_path.replace('.flac', '_projected.npy')

                    if os.path.exists(npy_path):
                        vector = np.load(npy_path)
                    else:
                        vector = self.extract_projection_vector(file_path)
                        if vector is not None:
                            np.save(npy_path, vector)

                    if vector is not None:
                        baseline_vectors.append(vector)
                        baseline_labels.append(0)

        return np.array(baseline_vectors), np.array(baseline_labels)

    def extract_projection_vector(self, file_path):
        """从音频文件提取投影向量"""
        try:
            wave_tensor, spec_tensor, freq_aug, raw_wave = preprocess_audio(file_path)
            with torch.no_grad():
                global_logits, frame_logits, projected_vector = model(wave_tensor, spec_tensor, freq_aug)
            projected_vector_np = projected_vector.squeeze(0).cpu().numpy()
            return projected_vector_np
        except Exception as e:
            print(f"提取投影向量失败 {file_path}: {str(e)}")
            return None

    def set_upload_vector(self, upload_vector):
        """设置上传音频的投影向量并进行UMAP可视化"""
        if upload_vector is None or len(upload_vector) != 128:
            self.baseline_2d = None
            self.baseline_labels = None
            self.upload_2d = None
            self.update()
            return

        # 加载基准向量
        baseline_vectors, baseline_labels = self.load_baseline_vectors()

        if len(baseline_vectors) == 0:
            print("警告：没有找到基准音频向量")
            self.baseline_2d = None
            self.baseline_labels = None
            self.upload_2d = None
            self.update()
            return

        # 如果还没有训练UMAP，或者基准向量数量变化了，重新训练
        if self.umap_reducer is None or self.baseline_2d is None or len(baseline_vectors) != len(self.baseline_labels):
            try:
                # 训练UMAP降维器
                self.umap_reducer = umap.UMAP(n_components=2, random_state=42,
                                              n_neighbors=min(15, len(baseline_vectors) - 1),
                                              min_dist=0.1)
                self.baseline_2d = self.umap_reducer.fit_transform(baseline_vectors)
                self.baseline_labels = baseline_labels
                print(f"UMAP训练完成，基准向量数量: {len(baseline_vectors)}")
            except Exception as e:
                print(f"UMAP训练失败: {str(e)}")
                self.baseline_2d = None
                self.baseline_labels = None
                self.upload_2d = None
                self.update()
                return

        # 使用已训练的UMAP转换上传向量
        try:
            self.upload_2d = self.umap_reducer.transform(upload_vector.reshape(1, -1))
            self.update()
        except Exception as e:
            print(f"UMAP转换失败: {str(e)}")
            self.upload_2d = None
            self.update()

    def paintEvent(self, event):
        """绘制UMAP散点图"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), QColor(255, 255, 255))
        painter.setPen(QColor(80, 80, 80))
        painter.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
        painter.drawText(15, 25, "UMAP聚类可视化")

        if self.baseline_2d is None or self.upload_2d is None:
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont("Microsoft YaHei UI", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "聚类可视化展示区")
            return

        draw_width = self.width() - 60
        draw_height = self.height() - 60
        x_start = 40
        y_start = 40
        all_2d = np.vstack([self.baseline_2d, self.upload_2d])

        x_min, x_max = np.min(all_2d[:, 0]), np.max(all_2d[:, 0])
        y_min, y_max = np.min(all_2d[:, 1]), np.max(all_2d[:, 1])

        if x_max - x_min == 0:
            x_max = x_min + 1
        if y_max - y_min == 0:
            y_max = y_min + 1

        legend_x = self.width() - 120
        legend_y = 20
        legend_items = [
            ("真实", QColor(50, 200, 50)),
            ("伪造", QColor(200, 50, 50)),
            ("上传", QColor(255, 200, 0))
        ]

        painter.setFont(QFont("Microsoft YaHei UI", 9))
        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + i * 25
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(legend_x, y_pos, 10, 10)
            painter.setPen(QColor(80, 80, 80))
            painter.drawText(legend_x + 15, y_pos + 9, label)

        for i, (x, y) in enumerate(self.baseline_2d):
            norm_x = (x - x_min) / (x_max - x_min)
            norm_y = (y - y_min) / (y_max - y_min)
            draw_x = x_start + norm_x * draw_width
            draw_y = y_start + (1 - norm_y) * draw_height
            if self.baseline_labels[i] == 0:  # 真实
                color = QColor(50, 200, 50)
            else:  # 伪造
                color = QColor(200, 50, 50)
            radius = 6

            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(draw_x, draw_y), radius, radius)

        x, y = self.upload_2d[0]
        norm_x = (x - x_min) / (x_max - x_min)
        norm_y = (y - y_min) / (y_max - y_min)
        draw_x = x_start + norm_x * draw_width
        draw_y = y_start + (1 - norm_y) * draw_height

        color = QColor(255, 200, 0)
        radius = 10
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(draw_x, draw_y), radius, radius)

        painter.setPen(QColor(80, 80, 80))
        painter.setFont(QFont("Microsoft YaHei UI", 8))
        coord_text = f"({x:.2f}, {y:.2f})"
        painter.drawText(QRectF(draw_x + 12, draw_y - 10, 80, 20), Qt.AlignLeft | Qt.AlignVCenter, coord_text)


class Ui_AudioForgeryDetection(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"AudioForgeryDetection")
        MainWindow.resize(1280, 680)
        MainWindow.setWindowIcon(QIcon("icon.jpg"))
        self.Main_QW = QWidget(MainWindow)
        self.Main_QW.setObjectName(u"Main_QW")
        self.Main_QW.setStyleSheet("QWidget#Main_QW { border: none; }")
        self.horizontalLayout_14 = QHBoxLayout(self.Main_QW)
        self.horizontalLayout_14.setSpacing(0)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.horizontalLayout_14.setContentsMargins(0, 0, 0, 0)

        self.Main_QF = QFrame(self.Main_QW)
        self.Main_QF.setObjectName(u"Main_QF")
        self.Main_QF.setStyleSheet(
            u"QFrame#Main_QF{\n"
            "	background-color: qlineargradient(x0:0, y0:1, x1:1, y1:1,stop:0.4  rgb(0, 80, 120), stop:1 rgb(10, 100, 180));\n"
            "border:0px solid red;\n"
            "border-radius: 0px;\n"
            "}"
        )
        self.main_qframe = QHBoxLayout(self.Main_QF)
        self.main_qframe.setSpacing(0)
        self.main_qframe.setObjectName(u"main_qframe")
        self.main_qframe.setContentsMargins(0, 0, 0, 0)

        # 左侧菜单栏
        self.LeftMenuBg = QFrame(self.Main_QF)
        self.LeftMenuBg.setObjectName(u"LeftMenuBg")
        self.LeftMenuBg.setMinimumSize(QSize(68, 0))
        self.LeftMenuBg.setMaximumSize(QSize(68, 16777215))
        self.LeftMenuBg.setStyleSheet(
            u"QFrame#LeftMenuBg{\n"
            "	background-color: rgba(255, 255, 255,0);\n"
            "border:0px solid red;\n"
            "border-radius: 0px;\n"
            "}"
        )
        self.LeftMenuBg.setFrameShape(QFrame.NoFrame)
        self.LeftMenuBg.setFrameShadow(QFrame.Raised)

        # 侧边栏内部水平布局：左侧按钮 + 右侧三角形指示器
        self.sidebar_hlayout = QHBoxLayout()
        self.sidebar_hlayout.setSpacing(0)
        self.sidebar_hlayout.setContentsMargins(0, 0, 0, 0)

        # 左侧按钮区域
        self.sidebar_btn_area = QWidget()
        self.sidebar_btn_area.setStyleSheet("background: transparent;")
        self.verticalLayout_2 = QVBoxLayout(self.sidebar_btn_area)
        self.verticalLayout_2.setSpacing(20)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(10, 40, 0, 40)

        # 右侧三角形指示器区域
        self.triangle_indicator_area = QWidget()
        self.triangle_indicator_area.setMinimumSize(QSize(10, 0))
        self.triangle_indicator_area.setMaximumSize(QSize(10, 16777215))
        self.triangle_indicator_area.setStyleSheet("background: transparent;")

        # 垂直布局来放置4个三角形
        self.triangle_vlayout = QVBoxLayout(self.triangle_indicator_area)
        self.triangle_vlayout.setSpacing(20)
        self.triangle_vlayout.setContentsMargins(0, 40, 10, 40)  # 右边距10

        # 4个三角形指示器
        self.single_triangle = QLabel()
        self.single_triangle.setText("◀")
        self.single_triangle.setAlignment(Qt.AlignCenter)
        self.single_triangle.setStyleSheet("border: none; color: transparent; font-size: 12px;")

        self.batch_triangle = QLabel()
        self.batch_triangle.setText("◀")
        self.batch_triangle.setAlignment(Qt.AlignCenter)
        self.batch_triangle.setStyleSheet("border: none; color: transparent; font-size: 12px;")

        self.db_triangle = QLabel()
        self.db_triangle.setText("◀")
        self.db_triangle.setAlignment(Qt.AlignCenter)
        self.db_triangle.setStyleSheet("border: none; color: transparent; font-size: 12px;")

        self.sys_triangle = QLabel()
        self.sys_triangle.setText("◀")
        self.sys_triangle.setAlignment(Qt.AlignCenter)
        self.sys_triangle.setStyleSheet("border: none; color: transparent; font-size: 12px;")

        self.triangle_vlayout.addWidget(self.single_triangle)
        self.triangle_vlayout.addWidget(self.batch_triangle)
        self.triangle_vlayout.addWidget(self.db_triangle)
        self.triangle_vlayout.addWidget(self.sys_triangle)

        # 将左右两部分添加到侧边栏水平布局
        self.sidebar_hlayout.addWidget(self.sidebar_btn_area)
        self.sidebar_hlayout.addWidget(self.triangle_indicator_area)

        # 将侧边栏水平布局设置为 LeftMenuBg 的布局
        self.LeftMenuBg.setLayout(self.sidebar_hlayout)

        # 侧边栏按钮
        # 1. 单音频检测按钮
        self.single_audio_button = QPushButton(self.LeftMenuBg)
        self.single_audio_button.setObjectName(u"single_audio_button")
        self.single_audio_button.setMinimumSize(QSize(48, 48))
        self.single_audio_button.setMaximumSize(QSize(48, 48))
        self.single_audio_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.single_audio_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: rgba(255, 255, 255, 20);\n"
            "border-radius: 8px;\n"
            "color: white;\n"
            "font: 700 10pt \"Nirmala UI\";\n"
            "text-align: center;\n"
            "}\n"
            "QPushButton:hover{\n"
            "background-color: rgba(114, 129, 214, 59);\n"
            "}\n"
            "QPushButton:pressed{\n"
            "background-color: rgba(255, 255, 255, 40);\n"
            "}"
        )
        self.single_audio_button.setText("单音频")
        self.verticalLayout_2.addWidget(self.single_audio_button, alignment=Qt.AlignCenter)

        # 2. 批量音频检测按钮
        self.batch_audio_button = QPushButton(self.LeftMenuBg)
        self.batch_audio_button.setObjectName(u"batch_audio_button")
        self.batch_audio_button.setMinimumSize(QSize(48, 48))
        self.batch_audio_button.setMaximumSize(QSize(48, 48))
        self.batch_audio_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.batch_audio_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: rgba(255, 255, 255, 20);\n"
            "border-radius: 8px;\n"
            "color: white;\n"
            "font: 700 10pt \"Nirmala UI\";\n"
            "text-align: center;\n"
            "}\n"
            "QPushButton:hover{\n"
            "background-color: rgba(114, 129, 214, 59);\n"
            "}\n"
            "QPushButton:pressed{\n"
            "background-color: rgba(255, 255, 255, 40);\n"
            "}"
        )
        self.batch_audio_button.setText("批量")
        self.verticalLayout_2.addWidget(self.batch_audio_button, alignment=Qt.AlignCenter)

        # 3. 数据库管理按钮
        self.database_button = QPushButton(self.LeftMenuBg)
        self.database_button.setObjectName(u"database_button")
        self.database_button.setMinimumSize(QSize(48, 48))
        self.database_button.setMaximumSize(QSize(48, 48))
        self.database_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.database_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: rgba(255, 255, 255, 20);\n"
            "border-radius: 8px;\n"
            "color: white;\n"
            "font: 700 10pt \"Nirmala UI\";\n"
            "text-align: center;\n"
            "}\n"
            "QPushButton:hover{\n"
            "background-color: rgba(114, 129, 214, 59);\n"
            "}\n"
            "QPushButton:pressed{\n"
            "background-color: rgba(255, 255, 255, 40);\n"
            "}"
        )
        self.database_button.setText("数据库")
        self.verticalLayout_2.addWidget(self.database_button, alignment=Qt.AlignCenter)

        # 4. 系统信息按钮
        self.system_info_button = QPushButton(self.LeftMenuBg)
        self.system_info_button.setObjectName(u"system_info_button")
        self.system_info_button.setMinimumSize(QSize(48, 48))
        self.system_info_button.setMaximumSize(QSize(48, 48))
        self.system_info_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.system_info_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: rgba(255, 255, 255, 20);\n"
            "border-radius: 8px;\n"
            "color: white;\n"
            "font: 700 10pt \"Nirmala UI\";\n"
            "text-align: center;\n"
            "}\n"
            "QPushButton:hover{\n"
            "background-color: rgba(114, 129, 214, 59);\n"
            "}\n"
            "QPushButton:pressed{\n"
            "background-color: rgba(255, 255, 255, 40);\n"
            "}"
        )
        self.system_info_button.setText("系统")
        self.verticalLayout_2.addWidget(self.system_info_button, alignment=Qt.AlignCenter)

        self.update_sidebar_triangle("single")

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(self.verticalSpacer)

        # 版本信息
        self.VersionInfo = QFrame(self.LeftMenuBg)
        self.VersionInfo.setObjectName(u"VersionInfo")
        self.VersionInfo.setMinimumSize(QSize(48, 20))
        self.VersionInfo.setMaximumSize(QSize(48, 20))
        self.VersionInfo.setFrameShape(QFrame.StyledPanel)
        self.VersionInfo.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.VersionInfo)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.VersionLabel = QLabel(self.VersionInfo)
        self.VersionLabel.setObjectName(u"VersionLabel")
        self.VersionLabel.setStyleSheet(
            u'font: 900 italic 8pt "Segoe UI";\n' "color: rgba(255, 255, 255, 199);"
        )
        self.VersionLabel.setAlignment(Qt.AlignCenter)
        self.VersionLabel.setText("v1.0")
        self.verticalLayout_3.addWidget(self.VersionLabel)
        self.verticalLayout_2.addWidget(self.VersionInfo, alignment=Qt.AlignCenter)

        self.main_qframe.addWidget(self.LeftMenuBg)

        # 主内容区域
        self.ContentBox = QFrame(self.Main_QF)
        self.ContentBox.setObjectName(u"ContentBox")
        self.ContentBox.setStyleSheet(
            u"QFrame#ContentBox{\n"
            "	background-color: rgb(245, 249, 254);\n"
            "border:0px solid red;\n"
            "border-radius: 0px;\n"
            "}"
        )
        self.ContentBox.setFrameShape(QFrame.StyledPanel)
        self.ContentBox.setFrameShadow(QFrame.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.ContentBox)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)

        # 顶部标题栏
        self.top = QFrame(self.ContentBox)
        self.top.setObjectName(u"top")
        self.top.setMinimumSize(QSize(0, 25))
        self.top.setMaximumSize(QSize(16777215, 25))
        self.top.setStyleSheet(
            u"QFrame#top{\n" "background-color: rgba(255, 255, 255,0);\n" "}"
        )
        self.top.setFrameShape(QFrame.StyledPanel)
        self.top.setFrameShadow(QFrame.Raised)
        self.horizontalLayout = QHBoxLayout(self.top)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(20, 0, 20, 0)

        # 标题文本
        self.explain_title = QLabel(self.top)
        self.explain_title.setObjectName(u"explain_title")
        self.explain_title.setMinimumSize(QSize(0, 25))
        self.explain_title.setMaximumSize(QSize(16777215, 25))
        self.explain_title.setStyleSheet(u'font: 700 italic 11pt "Segoe UI";')
        self.explain_title.setAlignment(Qt.AlignCenter)
        self.explain_title.setText("音频伪造检测系统 - 单音频模式")
        self.horizontalLayout.addWidget(self.explain_title)

        # 窗口控制按钮组
        self.buttons_sf = QFrame(self.top)
        self.buttons_sf.setObjectName(u"buttons_sf")
        self.buttons_sf.setMinimumSize(QSize(150, 30))
        self.buttons_sf.setMaximumSize(QSize(150, 30))
        self.buttons_sf.setFrameShape(QFrame.StyledPanel)
        self.buttons_sf.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.buttons_sf)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(10, 0, 10, 0)

        # 最小化按钮
        self.min_sf = QPushButton(self.buttons_sf)
        self.min_sf.setObjectName(u"min_sf")
        self.min_sf.setMinimumSize(QSize(14, 14))
        self.min_sf.setMaximumSize(QSize(14, 14))
        self.min_sf.setStyleSheet(
            u"QPushButton{\n"
            "	background-color: rgb(4, 180, 0);\n"
            "border:1px solid rgba(113, 17, 15,50);\n"
            "border-radius:6px;\n"
            "}\n"
            "QPushButton:hover {\n"
            "background-color:rgb(139, 29, 31)\n"
            "}\n"
            "QPushButton:pressed {\n"
            "	background-color: rgb(232, 59, 35);\n"
            "}\n"
            ""
        )
        self.horizontalLayout_2.addWidget(self.min_sf)

        # 最大化按钮
        self.max_sf = QPushButton(self.buttons_sf)
        self.max_sf.setObjectName(u"max_sf")
        self.max_sf.setMinimumSize(QSize(14, 14))
        self.max_sf.setMaximumSize(QSize(14, 14))
        self.max_sf.setStyleSheet(
            u"QPushButton{\n"
            "	background-color: rgb(227, 199, 0);\n"
            "border:1px solid rgba(113, 17, 15,50);\n"
            "border-radius:6px;\n"
            "}\n"
            "QPushButton:hover {\n"
            "background-color:rgb(139, 29, 31)\n"
            "}\n"
            "QPushButton:pressed {\n"
            "	background-color: rgb(232, 59, 35);\n"
            "}\n"
            ""
        )
        self.horizontalLayout_2.addWidget(self.max_sf)

        # 关闭按钮
        self.close_button = QPushButton(self.buttons_sf)
        self.close_button.setObjectName(u"close_button")
        self.close_button.setMinimumSize(QSize(14, 14))
        self.close_button.setMaximumSize(QSize(14, 14))
        self.close_button.setStyleSheet(
            u"QPushButton{\n"
            "	background-color: rgb(240, 108, 96);\n"
            "border:1px solid rgba(113, 17, 15,50);\n"
            "border-radius:6px;\n"
            "}\n"
            "QPushButton:hover {\n"
            "background-color:rgb(139, 29, 31)\n"
            "}\n"
            "QPushButton:pressed {\n"
            "	background-color: rgb(232, 59, 35);\n"
            "}\n"
            ""
        )
        self.horizontalLayout_2.addWidget(self.close_button)

        # 工具按钮
        self.tools_button = QPushButton(self.buttons_sf)
        self.tools_button.setObjectName(u"tools_button")
        self.tools_button.setMinimumSize(QSize(14, 14))
        self.tools_button.setMaximumSize(QSize(14, 14))
        self.tools_button.setStyleSheet(
            u"QPushButton{\n"
            "	background-color: rgb(100, 100, 100);\n"
            "border:1px solid rgba(50, 50, 50, 50);\n"
            "border-radius:6px;\n"
            "}\n"
            "QPushButton:hover {\n"
            "background-color:rgb(80, 80, 80)\n"
            "}\n"
            "QPushButton:pressed {\n"
            "	background-color: rgb(60, 60, 60);\n"
            "}\n"
            ""
        )
        self.horizontalLayout_2.addWidget(self.tools_button)
        self.horizontalLayout.addWidget(self.buttons_sf)
        self.verticalLayout_6.addWidget(self.top)

        self.hlayout_content = QHBoxLayout()
        self.hlayout_content.setSpacing(0)
        self.hlayout_content.setContentsMargins(0, 0, 0, 0)

        # 核心内容堆叠窗口
        self.content = QStackedWidget(self.ContentBox)
        self.content.setObjectName(u"content")
        self.content.setStyleSheet(u"")
        self.content.setFrameShape(QFrame.StyledPanel)
        self.content.setFrameShadow(QFrame.Raised)
        self.hlayout_content.addWidget(self.content)

        # 右侧工具面板
        self.tools_panel = QFrame(self.ContentBox)
        self.tools_panel.setObjectName(u"tools_panel")
        self.tools_panel.setMinimumSize(QSize(280, 0))
        self.tools_panel.setMaximumSize(QSize(280, 16777215))
        self.tools_panel.setStyleSheet(
            u"QFrame#tools_panel {\n"
            "background-color: rgb(245, 247, 250);\n"
            "border-left: 2px solid rgb(200, 200, 200);\n"
            "}"
        )
        self.tools_panel.setFrameShape(QFrame.StyledPanel)
        self.tools_panel.setFrameShadow(QFrame.Raised)
        self.tools_panel.hide()  # 默认隐藏

        self.tools_layout = QVBoxLayout(self.tools_panel)
        self.tools_layout.setSpacing(15)
        self.tools_layout.setContentsMargins(15, 20, 15, 20)

        # 工具面板标题
        self.tools_title = QLabel(self.tools_panel)
        self.tools_title.setText("音频工具")
        self.tools_title.setStyleSheet(
            u"font-size: 16px; font-weight: bold; color: #333; padding-bottom: 10px;"
        )
        self.tools_layout.addWidget(self.tools_title)

        # 滑动窗口步进值
        self.tools_slide_step_label = QLabel(self.tools_panel)
        self.tools_slide_step_label.setText("滑动窗口步进值")
        self.tools_slide_step_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555;")
        self.tools_layout.addWidget(self.tools_slide_step_label)

        self.tools_slide_step_slider = QSlider(Qt.Horizontal, self.tools_panel)
        self.tools_slide_step_slider.setObjectName(u"tools_slide_step_slider")
        self.tools_slide_step_slider.setMinimum(0)
        self.tools_slide_step_slider.setMaximum(40)
        self.tools_slide_step_slider.setValue(1)
        self.tools_slide_step_slider.setCursor(QCursor(Qt.PointingHandCursor))
        self.tools_layout.addWidget(self.tools_slide_step_slider)

        self.tools_slide_step_value_label = QLabel(self.tools_panel)
        self.tools_slide_step_value_label.setText("步进值: 0.1秒")
        self.tools_slide_step_value_label.setStyleSheet("font-size: 11px; color: #666;")
        self.tools_layout.addWidget(self.tools_slide_step_value_label)

        self.tools_sample_label = QLabel(self.tools_panel)
        self.tools_sample_label.setText("采样率转换")
        self.tools_sample_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555; margin-top: 10px;")
        self.tools_layout.addWidget(self.tools_sample_label)

        self.tools_sample_combo = QComboBox(self.tools_panel)
        self.tools_sample_combo.setObjectName(u"tools_sample_combo")
        self.tools_sample_combo.setMinimumSize(QSize(0, 35))
        self.tools_sample_combo.setCursor(QCursor(Qt.PointingHandCursor))
        self.tools_sample_combo.setStyleSheet(
            u"QComboBox{\n"
            "background-color: white;\n"
            "border: 1px solid #ddd;\n"
            "border-radius: 8px;\n"
            "padding: 5px 10px;\n"
            "color: #333;\n"
            "font-size: 12px;\n"
            "}\n"
            "QComboBox::drop-down {\n"
            "border: none;\n"
            "width: 25px;\n"
            "}\n"
            "QComboBox QAbstractItemView {\n"
            "background-color: white;\n"
            "color: #333;\n"
            "selection-background-color: #3498db;\n"
            "border-radius: 5px;\n"
            "}"
        )
        self.tools_sample_combo.addItem("原始采样率", 0)
        self.tools_sample_combo.addItem("8000 Hz", 8000)
        self.tools_sample_combo.addItem("16000 Hz", 16000)
        self.tools_sample_combo.addItem("22050 Hz", 22050)
        self.tools_sample_combo.addItem("44100 Hz", 44100)
        self.tools_sample_combo.addItem("48000 Hz", 48000)
        self.tools_layout.addWidget(self.tools_sample_combo)

        self.tools_change_sample_btn = QPushButton(self.tools_panel)
        self.tools_change_sample_btn.setObjectName(u"tools_change_sample_btn")
        self.tools_change_sample_btn.setMinimumSize(QSize(0, 35))
        self.tools_change_sample_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.tools_change_sample_btn.setStyleSheet(
            u"QPushButton{\n"
            "background-color: #3498db;\n"
            "color: white;\n"
            "border-radius: 17px;\n"
            "font-size: 12px;\n"
            "font-weight: bold;\n"
            "}\n"
            "QPushButton:hover{\n"
            "background-color: #2980b9;\n"
            "}"
        )
        self.tools_change_sample_btn.setText("🔄 转换采样率")
        self.tools_layout.addWidget(self.tools_change_sample_btn)

        # 格式转换
        self.tools_format_label = QLabel(self.tools_panel)
        self.tools_format_label.setText("格式转换")
        self.tools_format_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555; margin-top: 10px;")
        self.tools_layout.addWidget(self.tools_format_label)

        self.tools_format_combo = QComboBox(self.tools_panel)
        self.tools_format_combo.setObjectName(u"tools_format_combo")
        self.tools_format_combo.setMinimumSize(QSize(0, 35))
        self.tools_format_combo.setCursor(QCursor(Qt.PointingHandCursor))
        self.tools_format_combo.setStyleSheet(
            u"QComboBox{\n"
            "background-color: white;\n"
            "border: 1px solid #ddd;\n"
            "border-radius: 8px;\n"
            "padding: 5px 10px;\n"
            "color: #333;\n"
            "font-size: 12px;\n"
            "}\n"
            "QComboBox::drop-down {\n"
            "border: none;\n"
            "width: 25px;\n"
            "}\n"
            "QComboBox QAbstractItemView {\n"
            "background-color: white;\n"
            "color: #333;\n"
            "selection-background-color: #3498db;\n"
            "border-radius: 5px;\n"
            "}"
        )
        self.tools_format_combo.addItem("原始格式", "original")
        self.tools_format_combo.addItem("WAV", "wav")
        self.tools_format_combo.addItem("FLAC", "flac")
        self.tools_format_combo.addItem("MP3", "mp3")
        self.tools_layout.addWidget(self.tools_format_combo)

        self.tools_change_format_btn = QPushButton(self.tools_panel)
        self.tools_change_format_btn.setObjectName(u"tools_change_format_btn")
        self.tools_change_format_btn.setMinimumSize(QSize(0, 35))
        self.tools_change_format_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.tools_change_format_btn.setStyleSheet(
            u"QPushButton{\n"
            "background-color: #27ae60;\n"
            "color: white;\n"
            "border-radius: 17px;\n"
            "font-size: 12px;\n"
            "font-weight: bold;\n"
            "}\n"
            "QPushButton:hover{\n"
            "background-color: #229954;\n"
            "}"
        )
        self.tools_change_format_btn.setText("🔁 转换格式")
        self.tools_layout.addWidget(self.tools_change_format_btn)
        self.tools_layout.addStretch()
        self.hlayout_content.addWidget(self.tools_panel)

        #1. 单音频检测页面
        self.single_audio_page = QWidget()
        self.single_audio_page.setObjectName(u"single_audio_page")
        self.single_audio_page.setStyleSheet(u"background: transparent;")
        self.verticalLayout_single = QVBoxLayout(self.single_audio_page)
        self.verticalLayout_single.setSpacing(15)
        self.verticalLayout_single.setObjectName(u"verticalLayout_single")
        self.verticalLayout_single.setContentsMargins(20, 20, 20, 20)

        # 数据统计卡片组
        self.QF_Group = QFrame(self.single_audio_page)
        self.QF_Group.setObjectName(u"QF_Group")
        self.QF_Group.setMinimumSize(QSize(0, 80))
        self.QF_Group.setMaximumSize(QSize(16777215, 100))
        self.QF_Group.setStyleSheet(
            u"QFrame#QF_Group{\n"
            "background-color: rgb(238, 242, 255);\n"
            "border:2px solid rgb(255, 255, 255);\n"
            "border-radius:15px;\n"
            "}"
        )
        self.QF_Group.setFrameShape(QFrame.StyledPanel)
        self.QF_Group.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.QF_Group)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(20, 10, 20, 10)
        self.horizontalLayout_3.setSpacing(20)

        self.ForgeryProb_QF = QFrame(self.QF_Group)
        self.ForgeryProb_QF.setObjectName(u"ForgeryProb_QF")
        self.ForgeryProb_QF.setMinimumSize(QSize(150, 80))
        self.ForgeryProb_QF.setMaximumSize(QSize(150, 80))
        self.ForgeryProb_QF.setStyleSheet(
            u"QFrame#ForgeryProb_QF{\n"
            "color: rgb(255, 255, 255);\n"
            "border-radius: 15px;\n"
            "background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(30, 150, 255),  stop:1 rgb(50, 100, 255));\n"
            "border: 1px outset rgb(40, 125, 255);\n"
            "}"
        )
        self.ForgeryProb_QF.setFrameShape(QFrame.StyledPanel)
        self.ForgeryProb_QF.setFrameShadow(QFrame.Raised)
        self.verticalLayout_7 = QVBoxLayout(self.ForgeryProb_QF)
        self.verticalLayout_7.setSpacing(5)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(10, 10, 10, 10)

        self.label_forgery_prob = QLabel(self.ForgeryProb_QF)
        self.label_forgery_prob.setObjectName(u"label_forgery_prob")
        self.label_forgery_prob.setStyleSheet(
            u"color: rgba(255, 255, 255,210);\n"
            'font: 700 italic 14pt "Segoe UI";'
        )
        self.label_forgery_prob.setAlignment(Qt.AlignCenter)
        self.label_forgery_prob.setText("伪造概率")
        self.verticalLayout_7.addWidget(self.label_forgery_prob)

        self.forgery_prob_value = QLabel(self.ForgeryProb_QF)
        self.forgery_prob_value.setObjectName(u"forgery_prob_value")
        self.forgery_prob_value.setStyleSheet(
            u"color: rgb(255, 255, 255);\n" 'font: 16pt "Microsoft YaHei UI";'
        )
        self.forgery_prob_value.setAlignment(Qt.AlignCenter)
        self.forgery_prob_value.setText("0.00%")
        self.verticalLayout_7.addWidget(self.forgery_prob_value)
        self.horizontalLayout_3.addWidget(self.ForgeryProb_QF)

        self.AudioDuration_QF = QFrame(self.QF_Group)
        self.AudioDuration_QF.setObjectName(u"AudioDuration_QF")
        self.AudioDuration_QF.setMinimumSize(QSize(150, 80))
        self.AudioDuration_QF.setMaximumSize(QSize(150, 80))
        self.AudioDuration_QF.setStyleSheet(
            u"QFrame#AudioDuration_QF{\n"
            "color: rgb(255, 255, 255);\n"
            "border-radius: 15px;\n"
            "background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(253, 139, 133),  stop:1 rgb(248, 194, 152));\n"
            "border: 1px outset rgb(252, 194, 149)\n"
            "}"
        )
        self.AudioDuration_QF.setFrameShape(QFrame.StyledPanel)
        self.AudioDuration_QF.setFrameShadow(QFrame.Raised)
        self.verticalLayout_9 = QVBoxLayout(self.AudioDuration_QF)
        self.verticalLayout_9.setSpacing(5)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(10, 10, 10, 10)

        self.label_audio_duration = QLabel(self.AudioDuration_QF)
        self.label_audio_duration.setObjectName(u"label_audio_duration")
        self.label_audio_duration.setStyleSheet(
            u"color: rgba(255, 255, 255,210);\n"
            'font: 700 italic 14pt "Segoe UI";'
        )
        self.label_audio_duration.setAlignment(Qt.AlignCenter)
        self.label_audio_duration.setText("音频时长")
        self.verticalLayout_9.addWidget(self.label_audio_duration)

        self.audio_duration_value = QLabel(self.AudioDuration_QF)
        self.audio_duration_value.setObjectName(u"audio_duration_value")
        self.audio_duration_value.setStyleSheet(
            u"color: rgb(255, 255, 255);\n" 'font: 16pt "Microsoft YaHei UI";'
        )
        self.audio_duration_value.setAlignment(Qt.AlignCenter)
        self.audio_duration_value.setText("00:00:00")
        self.verticalLayout_9.addWidget(self.audio_duration_value)
        self.horizontalLayout_3.addWidget(self.AudioDuration_QF)

        self.ForgeryFrames_QF = QFrame(self.QF_Group)
        self.ForgeryFrames_QF.setObjectName(u"ForgeryFrames_QF")
        self.ForgeryFrames_QF.setMinimumSize(QSize(150, 80))
        self.ForgeryFrames_QF.setMaximumSize(QSize(150, 80))
        self.ForgeryFrames_QF.setStyleSheet(
            u"QFrame#ForgeryFrames_QF{\n"
            "color: rgb(255, 255, 255);\n"
            "border-radius: 15px;\n"
            "background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(255, 100, 100),  stop:1 rgb(255, 150, 100));\n"
            "border: 1px outset rgb(255, 125, 100)\n"
            "}"
        )
        self.ForgeryFrames_QF.setFrameShape(QFrame.StyledPanel)
        self.ForgeryFrames_QF.setFrameShadow(QFrame.Raised)
        self.verticalLayout_11 = QVBoxLayout(self.ForgeryFrames_QF)
        self.verticalLayout_11.setSpacing(5)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalLayout_11.setContentsMargins(10, 10, 10, 10)

        self.label_forgery_frames = QLabel(self.ForgeryFrames_QF)
        self.label_forgery_frames.setObjectName(u"label_forgery_frames")
        self.label_forgery_frames.setStyleSheet(
            u"color: rgba(255, 255, 255,210);\n"
            'font: 700 italic 14pt "Segoe UI";'
        )
        self.label_forgery_frames.setAlignment(Qt.AlignCenter)
        self.label_forgery_frames.setText("伪造段数")
        self.verticalLayout_11.addWidget(self.label_forgery_frames)

        self.forgery_frames_value = QLabel(self.ForgeryFrames_QF)
        self.forgery_frames_value.setObjectName(u"forgery_frames_value")
        self.forgery_frames_value.setStyleSheet(
            u"color: rgb(255, 255, 255);\n" 'font: 16pt "Microsoft YaHei UI";'
        )
        self.forgery_frames_value.setAlignment(Qt.AlignCenter)
        self.forgery_frames_value.setText("0")
        self.verticalLayout_11.addWidget(self.forgery_frames_value)
        self.horizontalLayout_3.addWidget(self.ForgeryFrames_QF)

        self.Model_QF = QFrame(self.QF_Group)
        self.Model_QF.setObjectName(u"Model_QF")
        self.Model_QF.setMinimumSize(QSize(150, 80))
        self.Model_QF.setMaximumSize(QSize(150, 80))
        self.Model_QF.setStyleSheet(
            u"QFrame#Model_QF{\n"
            "color: rgb(255, 255, 255);\n"
            "border-radius: 15px;\n"
            "background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(66, 150, 192),  stop:1 rgb(62, 100, 193));\n"
            "border: 1px outset rgb(72, 158, 204)\n"
            "}"
        )
        self.Model_QF.setFrameShape(QFrame.StyledPanel)
        self.Model_QF.setFrameShadow(QFrame.Raised)
        self.verticalLayout_13 = QVBoxLayout(self.Model_QF)
        self.verticalLayout_13.setSpacing(5)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.verticalLayout_13.setContentsMargins(10, 10, 10, 10)

        self.label_model = QLabel(self.Model_QF)
        self.label_model.setObjectName(u"label_model")
        self.label_model.setStyleSheet(
            u"color: rgba(255, 255, 255,210);\n"
            'font: 700 italic 14pt "Segoe UI";'
        )
        self.label_model.setAlignment(Qt.AlignCenter)
        self.label_model.setText("使用模型")
        self.verticalLayout_13.addWidget(self.label_model)

        self.model_name = QLabel(self.Model_QF)
        self.model_name.setObjectName(u"model_name")
        self.model_name.setStyleSheet(
            u"color: rgb(255, 255, 255);\n" 'font: 14pt "Microsoft YaHei UI";'
        )
        self.model_name.setAlignment(Qt.AlignCenter)
        self.model_name.setText("LEMAS")
        self.verticalLayout_13.addWidget(self.model_name)
        self.horizontalLayout_3.addWidget(self.Model_QF)
        self.verticalLayout_single.addWidget(self.QF_Group)

        # 结果展示区域
        self.Result_QF = QFrame(self.single_audio_page)
        self.Result_QF.setObjectName(u"Result_QF")
        self.Result_QF.setMinimumSize(QSize(0, 100))
        self.Result_QF.setStyleSheet(
            u"QFrame#Result_QF{\n"
            "background-color: rgb(238, 242, 255);\n"
            "border:2px solid rgb(255, 255, 255);\n"
            "border-radius:15px\n"
            "}"
        )
        self.Result_QF.setFrameShape(QFrame.StyledPanel)
        self.Result_QF.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_result = QHBoxLayout(self.Result_QF)
        self.horizontalLayout_result.setSpacing(10)
        self.horizontalLayout_result.setObjectName(u"horizontalLayout_result")
        self.horizontalLayout_result.setContentsMargins(10, 10, 10, 10)

        # 左侧：音频波形展示 + 播放按钮
        self.left_result_widget = QWidget(self.Result_QF)
        self.left_result_layout = QVBoxLayout(self.left_result_widget)
        self.left_result_layout.setContentsMargins(0, 0, 0, 0)
        self.left_result_layout.setSpacing(5)

        # 波形图
        self.audio_waveform = WaveformWidget(self.left_result_widget)
        self.audio_waveform.setObjectName(u"audio_waveform")
        self.left_result_layout.addWidget(self.audio_waveform)

        # 播放按钮（放在波形图下方）
        self.play_button = QPushButton(self.left_result_widget)
        self.play_button.setObjectName(u"play_button")
        self.play_button.setMinimumSize(QSize(80, 20))
        self.play_button.setMaximumSize(QSize(80, 20))
        self.play_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.play_button.setStyleSheet(
            u"QPushButton{"
            "background-color: #3498db;"
            "color: white;"
            "border-radius: 14px;"
            "font-size: 11px;"
            "}"
            "QPushButton:hover{"
            "background-color: #2980b9;"
            "}"
        )
        self.play_button.setText("▶ 播放")
        self.left_result_layout.addWidget(self.play_button, alignment=Qt.AlignLeft)

        self.horizontalLayout_result.addWidget(self.left_result_widget)

        # 右侧：段级伪造概率柱状图
        self.forgery_heatmap = SegmentBarChartWidget(self.Result_QF)
        self.forgery_heatmap.setObjectName(u"forgery_heatmap")
        self.horizontalLayout_result.addWidget(self.forgery_heatmap)
        self.verticalLayout_single.addWidget(self.Result_QF)

        # 投影向量可视化区域
        self.vector_visualization = QFrame(self.single_audio_page)
        self.vector_visualization.setObjectName(u"vector_visualization")
        self.vector_visualization.setMinimumSize(QSize(0, 80))
        self.vector_visualization.setStyleSheet(
            u"background-color: rgb(238, 242, 255);\n"
            "border:2px solid rgb(255, 255, 255);\n"
            "border-radius:15px"
        )
        self.vector_visualization.setFrameShape(QFrame.StyledPanel)
        self.vector_visualization.setFrameShadow(QFrame.Raised)
        self.verticalLayout_vector = QVBoxLayout(self.vector_visualization)
        self.verticalLayout_vector.setObjectName(u"verticalLayout_vector")
        self.verticalLayout_vector.setContentsMargins(10, 10, 10, 10)

        self.vector_plot = UMAPPlotWidget(self.vector_visualization)
        self.vector_plot.setObjectName(u"vector_plot")
        self.vector_plot.setMinimumSize(QSize(0, 120))
        self.verticalLayout_vector.addWidget(self.vector_plot)
        self.verticalLayout_single.addWidget(self.vector_visualization)

        # 控制按钮区域
        self.Control_QF = QFrame(self.single_audio_page)
        self.Control_QF.setObjectName(u"Control_QF")
        self.Control_QF.setMinimumSize(QSize(0, 50))
        self.Control_QF.setMaximumSize(QSize(16777215, 50))
        self.Control_QF.setFrameShape(QFrame.StyledPanel)
        self.Control_QF.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.Control_QF)
        self.horizontalLayout_4.setSpacing(10)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(20, 10, 20, 10)

        # 音频源选择下拉框
        self.audio_source_combo = QComboBox(self.Control_QF)
        self.audio_source_combo.setObjectName(u"audio_source_combo")
        self.audio_source_combo.setMinimumSize(QSize(140, 40))
        self.audio_source_combo.setCursor(QCursor(Qt.PointingHandCursor))
        self.audio_source_combo.setStyleSheet(
            u"QComboBox{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #59969b, stop:1 #04e7fa);\n"
            "border: none;\n"
            "border-radius:10px;\n"
            "color: white;\n"
            'font: 700 11pt "Microsoft YaHei UI";\n'
            "padding-left: 15px;\n"
            "}\n"
            "QComboBox::drop-down {\n"
            "border: none;\n"
            "width: 30px;\n"
            "}\n"
            "QComboBox::down-arrow {\n"
            "image: url(no-scroll);\n"
            "border: none;\n"
            "}\n"
            "QComboBox QAbstractItemView {\n"
            "background-color: white;\n"
            "color: #333;\n"
            "selection-background-color: #59969b;\n"
            "border-radius: 5px;\n"
            "}"
        )
        self.audio_source_combo.addItem("上传音频文件", "file")
        self.audio_source_combo.addItem("🎤 实时录制", "record")
        self.horizontalLayout_4.addWidget(self.audio_source_combo)

        # 选择音频文件按钮（仅用于上传文件模式）
        self.select_audio_button = QPushButton(self.Control_QF)
        self.select_audio_button.setObjectName(u"select_audio_button")
        self.select_audio_button.setMinimumSize(QSize(120, 40))
        self.select_audio_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.select_audio_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #59969b, stop:1 #04e7fa);\n"
            "border: none;\n"
            "border-radius:10px;\n"
            "color: white;\n"
            'font: 700 11pt "Microsoft YaHei UI";\n'
            "}\n"
            "QPushButton:hover{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #40787c, stop:1 #03b5cc);\n"
            "}"
        )
        self.select_audio_button.setText("选择音频")
        self.horizontalLayout_4.addWidget(self.select_audio_button)

        # 开始检测按钮
        self.detect_button = QPushButton(self.Control_QF)
        self.detect_button.setObjectName(u"detect_button")
        self.detect_button.setMinimumSize(QSize(120, 40))
        self.detect_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.detect_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4CAF50, stop:1 #45a049);\n"
            "border: none;\n"
            "border-radius:10px;\n"
            "color: white;\n"
            'font: 700 11pt "Microsoft YaHei UI";\n'
            "}\n"
            "QPushButton:hover{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3d8b40, stop:1 #39803c);\n"
            "}"
        )
        self.detect_button.setText("开始检测")
        self.horizontalLayout_4.addWidget(self.detect_button)

        # 进度条
        self.progress_bar = QProgressBar(self.Control_QF)
        self.progress_bar.setObjectName(u"progress_bar")
        self.progress_bar.setMinimumSize(QSize(0, 20))
        self.progress_bar.setStyleSheet(
            u"QProgressBar{ \n"
            'font: 700 10pt "Microsoft YaHei UI";\n'
            "color: rgb(253, 143, 134); \n"
            "text-align:center; \n"
            "border:3px solid rgb(255, 255, 255);\n"
            "border-radius: 10px; \n"
            "background-color: rgba(215, 215, 215,100);\n"
            "} \n"
            "QProgressBar:chunk{ \n"
            "border-radius:7px;\n"
            "background: rgba(119, 111, 252, 200);\n"
            "}"
        )
        self.progress_bar.setMaximum(1000)
        self.progress_bar.setValue(0)
        self.horizontalLayout_4.addWidget(self.progress_bar)

        # 保存结果按钮
        self.save_button = QPushButton(self.Control_QF)
        self.save_button.setObjectName(u"save_button")
        self.save_button.setMinimumSize(QSize(120, 40))
        self.save_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.save_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2196F3, stop:1 #0b7dda);\n"
            "border: none;\n"
            "border-radius:10px;\n"
            "color: white;\n"
            'font: 700 11pt "Microsoft YaHei UI";\n'
            "}\n"
            "QPushButton:hover{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1976D2, stop:1 #0967b7);\n"
            "}"
        )
        self.save_button.setText("保存结果")
        self.horizontalLayout_4.addWidget(self.save_button)

        self.verticalLayout_single.addWidget(self.Control_QF)

        # 2. 批量音频检测页面
        self.batch_audio_page = QWidget()
        self.batch_audio_page.setObjectName(u"batch_audio_page")
        self.batch_audio_page.setStyleSheet(u"background: transparent;")
        self.verticalLayout_batch = QVBoxLayout(self.batch_audio_page)
        self.verticalLayout_batch.setSpacing(15)
        self.verticalLayout_batch.setContentsMargins(20, 20, 20, 20)

        #顶部控制栏
        self.batch_control_bar = QFrame(self.batch_audio_page)
        self.batch_control_bar.setObjectName(u"batch_control_bar")
        self.batch_control_bar.setMinimumSize(QSize(0, 50))
        self.batch_control_bar.setStyleSheet(
            u"QFrame#batch_control_bar{\n"
            "background-color: rgb(238, 242, 255);\n"
            "border:2px solid rgb(255, 255, 255);\n"
            "border-radius:15px\n"
            "}"
        )
        self.horizontalLayout_batch_control = QHBoxLayout(self.batch_control_bar)
        self.horizontalLayout_batch_control.setSpacing(10)
        self.horizontalLayout_batch_control.setContentsMargins(20, 10, 20, 10)

        # 选择多个音频文件按钮
        self.select_batch_button = QPushButton(self.batch_control_bar)
        self.select_batch_button.setObjectName(u"select_batch_button")
        self.select_batch_button.setMinimumSize(QSize(120, 40))
        self.select_batch_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.select_batch_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #59969b, stop:1 #04e7fa);\n"
            "border: none;\n"
            "border-radius:10px;\n"
            "color: white;\n"
            'font: 700 11pt "Microsoft YaHei UI";\n'
            "}\n"
            "QPushButton:hover{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #40787c, stop:1 #03b5cc);\n"
            "}"
        )
        self.select_batch_button.setText("选择多个音频")
        self.horizontalLayout_batch_control.addWidget(self.select_batch_button)

        # 一键开始检测按钮
        self.start_batch_button = QPushButton(self.batch_control_bar)
        self.start_batch_button.setObjectName(u"start_batch_button")
        self.start_batch_button.setMinimumSize(QSize(120, 40))
        self.start_batch_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.start_batch_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4CAF50, stop:1 #45a049);\n"
            "border: none;\n"
            "border-radius:10px;\n"
            "color: white;\n"
            'font: 700 11pt "Microsoft YaHei UI";\n'
            "}\n"
            "QPushButton:hover{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3d8b40, stop:1 #39803c);\n"
            "}"
        )
        self.start_batch_button.setText("一键开始检测")
        self.horizontalLayout_batch_control.addWidget(self.start_batch_button)

        # 暂停/继续按钮
        self.pause_resume_button = QPushButton(self.batch_control_bar)
        self.pause_resume_button.setObjectName(u"pause_resume_button")
        self.pause_resume_button.setMinimumSize(QSize(120, 40))
        self.pause_resume_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.pause_resume_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF9800, stop:1 #F57C00);\n"
            "border: none;\n"
            "border-radius:10px;\n"
            "color: white;\n"
            'font: 700 11pt "Microsoft YaHei UI";\n'
            "}\n"
            "QPushButton:hover{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #F57C00, stop:1 #E65100);\n"
            "}"
        )
        self.pause_resume_button.setText("暂停检测")
        self.pause_resume_button.setEnabled(False)
        self.horizontalLayout_batch_control.addWidget(self.pause_resume_button)

        # 取消检测按钮
        self.cancel_batch_button = QPushButton(self.batch_control_bar)
        self.cancel_batch_button.setObjectName(u"cancel_batch_button")
        self.cancel_batch_button.setMinimumSize(QSize(120, 40))
        self.cancel_batch_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.cancel_batch_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #F44336, stop:1 #D32F2F);\n"
            "border: none;\n"
            "border-radius:10px;\n"
            "color: white;\n"
            'font: 700 11pt "Microsoft YaHei UI";\n'
            "}\n"
            "QPushButton:hover{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #D32F2F, stop:1 #B71C1C);\n"
            "}"
        )
        self.cancel_batch_button.setText("取消检测")
        self.cancel_batch_button.setEnabled(False)
        self.horizontalLayout_batch_control.addWidget(self.cancel_batch_button)

        # 导出结果按钮
        self.export_batch_button = QPushButton(self.batch_control_bar)
        self.export_batch_button.setObjectName(u"export_batch_button")
        self.export_batch_button.setMinimumSize(QSize(120, 40))
        self.export_batch_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.export_batch_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2196F3, stop:1 #0b7dda);\n"
            "border: none;\n"
            "border-radius:10px;\n"
            "color: white;\n"
            'font: 700 11pt "Microsoft YaHei UI";\n'
            "}\n"
            "QPushButton:hover{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1976D2, stop:1 #0967b7);\n"
            "}"
        )
        self.export_batch_button.setText("导出结果")
        self.export_batch_button.setEnabled(False)
        self.horizontalLayout_batch_control.addWidget(self.export_batch_button)
        self.verticalLayout_batch.addWidget(self.batch_control_bar)
        #进度展示区域
        self.batch_progress_bar = QFrame(self.batch_audio_page)
        self.batch_progress_bar.setObjectName(u"batch_progress_bar")
        self.batch_progress_bar.setMinimumSize(QSize(0, 60))
        self.batch_progress_bar.setStyleSheet(
            u"QFrame#batch_progress_bar{\n"
            "background-color: rgb(238, 242, 255);\n"
            "border:2px solid rgb(255, 255, 255);\n"
            "border-radius:15px\n"
            "}"
        )
        self.verticalLayout_progress = QVBoxLayout(self.batch_progress_bar)
        self.verticalLayout_progress.setSpacing(10)
        self.verticalLayout_progress.setContentsMargins(20, 10, 20, 10)

        # 整体进度条
        self.global_progress_bar = QProgressBar(self.batch_progress_bar)
        self.global_progress_bar.setObjectName(u"global_progress_bar")
        self.global_progress_bar.setMinimumSize(QSize(0, 20))
        self.global_progress_bar.setStyleSheet(
            u"QProgressBar{ \n"
            'font: 700 10pt "Microsoft YaHei UI";\n'
            "color: rgb(253, 143, 134); \n"
            "text-align:center; \n"
            "border:3px solid rgb(255, 255, 255);\n"
            "border-radius: 10px; \n"
            "background-color: rgba(215, 215, 215,100);\n"
            "} \n"
            "QProgressBar:chunk{ \n"
            "border-radius:7px;\n"
            "background: rgba(119, 111, 252, 200);\n"
            "}"
        )
        self.global_progress_bar.setMaximum(100)
        self.global_progress_bar.setValue(0)
        self.verticalLayout_progress.addWidget(self.global_progress_bar)

        # 进度信息标签（已用时/预估剩余时间）
        self.progress_info_label = QLabel(self.batch_progress_bar)
        self.progress_info_label.setObjectName(u"progress_info_label")
        self.progress_info_label.setStyleSheet(u'font: 700 10pt "Microsoft YaHei UI";')
        self.progress_info_label.setAlignment(Qt.AlignCenter)
        self.progress_info_label.setText("已用时：00:00:00 | 预估剩余：00:00:00 | 当前进度：0/0")
        self.verticalLayout_progress.addWidget(self.progress_info_label)

        self.verticalLayout_batch.addWidget(self.batch_progress_bar)

        #文件列表和结果列表切换标签
        self.batch_tab_widget = QTabWidget(self.batch_audio_page)
        self.batch_tab_widget.setObjectName(u"batch_tab_widget")
        self.batch_tab_widget.setMinimumSize(QSize(0, 220))
        self.batch_tab_widget.setStyleSheet("""
        QTabWidget {
            background-color: #ffffff;  
            border: none;               
        }
        QTabWidget::pane {
            background-color: #ffffff;  
            border: 1px solid #e0e0e0;  
            border-radius: 6px;         
            padding: 5px;               
        }
        QTabBar::tab {
            background-color: #ffffff;  
            color: #333333;             
            padding: 8px 16px;          
            margin-right: 2px;          
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
        }
        QTabBar::tab:selected {
            background-color: #3096fb;  
            color: white;               
        }
        QTabBar {
            background-color: #ffffff;  
            border-bottom: 1px solid #e0e0e0;  
        }
        """)
        self.file_list_page = QWidget()
        self.file_list_page.setObjectName(u"file_list_page")
        self.file_list_page.setStyleSheet("background-color: #ffffff;")  # 新增：标签页背景白色
        self.verticalLayout_file_list = QVBoxLayout(self.file_list_page)
        self.verticalLayout_file_list.setContentsMargins(10, 10, 10, 10)

        # 文件列表表格
        self.file_list_table = QTableWidget(self.file_list_page)
        self.file_list_table.setObjectName(u"file_list_table")
        # 给文件列表表格添加白色背景样式
        self.file_list_table.setStyleSheet("""
        QTableWidget {
            background-color: #ffffff;       
            gridline-color: #e0e0e0;         
            border: 1px solid #ffffff;       
            border-radius: 6px;              
            padding: 4px;
        }
        QTableWidget::item {
            color: #333333;                  
            padding: 6px 8px;
            background-color: #ffffff;       
        }
        QTableWidget::item:selected {
            background-color: #3096fb;       
            color: white;
        }
        QTableWidget::item:alternate {
            background-color: #f5f7fa;       
        }
        QHeaderView::section {
            background-color: #ffffff;       
            color: #333333;
            border: 1px solid #ffffff;      
            padding: 6px;
            font-weight: bold;
        }
        QTableWidget::corner {
            background-color: #ffffff;       
            border: 1px solid #ffffff;       
        }
        QTableWidget::pane {
            border: 1px solid #ffffff;       
            border-radius: 6px;
            background-color: #ffffff;       
        }
        """)
        # 开启交替行颜色
        self.file_list_table.setAlternatingRowColors(True)
        self.file_list_table.setColumnCount(4)
        self.file_list_table.setHorizontalHeaderLabels(["文件名", "文件路径", "文件大小(MB)", "音频时长"])
        # 设置列宽自适应
        self.file_list_table.horizontalHeader().setStretchLastSection(True)
        self.file_list_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalLayout_file_list.addWidget(self.file_list_table)

        self.batch_tab_widget.addTab(self.file_list_page, "待检测文件列表")

        #检测结果列表标签页
        self.result_list_page = QWidget()
        self.result_list_page.setObjectName(u"result_list_page")
        self.result_list_page.setStyleSheet("background-color: #ffffff;")  # 新增：标签页背景白色
        self.verticalLayout_result_list = QVBoxLayout(self.result_list_page)
        self.verticalLayout_result_list.setContentsMargins(10, 10, 10, 10)

        # 结果列表表格
        self.result_list_table = QTableWidget(self.result_list_page)
        self.result_list_table.setObjectName(u"result_list_table")
        self.result_list_table.setStyleSheet("""
        QTableWidget {
            background-color: #ffffff;       
            gridline-color: #e0e0e0;         
            border: 1px solid #ffffff;       
            border-radius: 6px;              
            padding: 4px;
        }
        QTableWidget::item {
            color: #333333;                  
            padding: 6px 8px;
            background-color: #ffffff;       
        }
        QTableWidget::item:selected {
            background-color: #3096fb;       
            color: white;
        }
        QTableWidget::item:alternate {
            background-color: #f5f7fa;       
        }
        QHeaderView::section {
            background-color: #ffffff;       
            color: #333333;
            border: 1px solid #ffffff;       
            padding: 6px;
            font-weight: bold;
        }
        QTableWidget::corner {
            background-color: #ffffff;       
            border: 1px solid #ffffff;       
        }
        QTableWidget::pane {
            border: 1px solid #ffffff;       
            border-radius: 6px;
            background-color: #ffffff;       
        }
        """)
        # 开启交替行颜色
        self.result_list_table.setAlternatingRowColors(True)
        self.result_list_table.setColumnCount(7)
        self.result_list_table.setHorizontalHeaderLabels([
            "文件名", "文件路径", "音频时长", "伪造概率", "判断结论",
            "伪造段数", "检测状态"
        ])
        self.result_list_table.horizontalHeader().setStretchLastSection(True)
        self.result_list_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # 点击行预览功能
        self.result_list_table.clicked.connect(self.preview_result_file)
        self.verticalLayout_result_list.addWidget(self.result_list_table)

        self.batch_tab_widget.addTab(self.result_list_page, "检测结果列表")

        self.verticalLayout_batch.addWidget(self.batch_tab_widget)

        #快速预览区域
        self.quick_preview_area = QFrame(self.batch_audio_page)
        self.quick_preview_area.setObjectName(u"quick_preview_area")
        self.quick_preview_area.setMinimumSize(QSize(0, 120))
        self.quick_preview_area.setStyleSheet(
            u"QFrame#quick_preview_area{\n"
            "background-color: rgb(238, 242, 255);\n"
            "border:2px solid rgb(255, 255, 255);\n"
            "border-radius:15px\n"
            "}"
        )
        self.horizontalLayout_preview = QHBoxLayout(self.quick_preview_area)
        self.horizontalLayout_preview.setSpacing(10)
        self.horizontalLayout_preview.setContentsMargins(10, 10, 10, 10)

        # 预览波形图
        self.preview_waveform = WaveformWidget(self.quick_preview_area)
        self.preview_waveform.setObjectName(u"preview_waveform")
        self.horizontalLayout_preview.addWidget(self.preview_waveform)

        # 预览段级概率显示
        self.preview_heatmap = SegmentBarChartWidget(self.quick_preview_area)
        self.preview_heatmap.setObjectName(u"preview_heatmap")
        self.horizontalLayout_preview.addWidget(self.preview_heatmap)

        # 预览向量图
        self.preview_vector = UMAPPlotWidget(self.quick_preview_area)
        self.preview_vector.setObjectName(u"preview_vector")
        self.horizontalLayout_preview.addWidget(self.preview_vector)

        self.verticalLayout_batch.addWidget(self.quick_preview_area)

        #统计面板
        self.stats_panel = QFrame(self.batch_audio_page)
        self.stats_panel.setObjectName(u"stats_panel")
        self.stats_panel.setMinimumSize(QSize(0, 80))
        self.stats_panel.setStyleSheet(
            u"QFrame#stats_panel{\n"
            "background-color: rgb(238, 242, 255);\n"
            "border:2px solid rgb(255, 255, 255);\n"
            "border-radius:15px\n"
            "}"
        )
        self.horizontalLayout_stats = QHBoxLayout(self.stats_panel)
        self.horizontalLayout_stats.setSpacing(20)
        self.horizontalLayout_stats.setContentsMargins(20, 10, 20, 10)

        # 总数/成功数/等待数
        self.count_stats = QFrame(self.stats_panel)
        self.count_stats.setObjectName(u"count_stats")
        self.count_stats.setMinimumSize(QSize(150, 80))
        self.count_stats.setMaximumSize(QSize(150, 80))
        self.count_stats.setStyleSheet(
            u"QFrame#count_stats{\n"
            "color: rgb(255, 255, 255);\n"
            "border-radius: 15px;\n"
            "background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(30, 150, 255),  stop:1 rgb(50, 100, 255));\n"
            "border: 1px outset rgb(40, 125, 255);\n"
            "}"
        )
        self.verticalLayout_count = QVBoxLayout(self.count_stats)
        self.verticalLayout_count.setSpacing(5)
        self.verticalLayout_count.setContentsMargins(10, 10, 10, 10)
        self.label_count = QLabel(self.count_stats)
        self.label_count.setStyleSheet(u"color: rgba(255, 255, 255,210);\n" 'font: 700 italic 12pt "Segoe UI";')
        self.label_count.setAlignment(Qt.AlignCenter)
        self.label_count.setText("总数/成功/等待")
        self.count_value = QLabel(self.count_stats)
        self.count_value.setStyleSheet(u"color: rgb(255, 255, 255);\n" 'font: 14pt "Microsoft YaHei UI";')
        self.count_value.setAlignment(Qt.AlignCenter)
        self.count_value.setText("0/0/0")
        self.verticalLayout_count.addWidget(self.label_count)
        self.verticalLayout_count.addWidget(self.count_value)
        self.horizontalLayout_stats.addWidget(self.count_stats)

        # 伪造音频占比
        self.fake_ratio_stats = QFrame(self.stats_panel)
        self.fake_ratio_stats.setObjectName(u"fake_ratio_stats")
        self.fake_ratio_stats.setMinimumSize(QSize(150, 80))
        self.fake_ratio_stats.setMaximumSize(QSize(150, 80))
        self.fake_ratio_stats.setStyleSheet(
            u"QFrame#fake_ratio_stats{\n"
            "color: rgb(255, 255, 255);\n"
            "border-radius: 15px;\n"
            "background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(253, 139, 133),  stop:1 rgb(248, 194, 152));\n"
            "border: 1px outset rgb(252, 194, 149)\n"
            "}"
        )
        self.verticalLayout_fake = QVBoxLayout(self.fake_ratio_stats)
        self.verticalLayout_fake.setSpacing(5)
        self.verticalLayout_fake.setContentsMargins(10, 10, 10, 10)
        self.label_fake = QLabel(self.fake_ratio_stats)
        self.label_fake.setStyleSheet(u"color: rgba(255, 255, 255,210);\n" 'font: 700 italic 12pt "Segoe UI";')
        self.label_fake.setAlignment(Qt.AlignCenter)
        self.label_fake.setText("伪造音频占比")
        self.fake_ratio_value = QLabel(self.fake_ratio_stats)
        self.fake_ratio_value.setStyleSheet(u"color: rgb(255, 255, 255);\n" 'font: 14pt "Microsoft YaHei UI";')
        self.fake_ratio_value.setAlignment(Qt.AlignCenter)
        self.fake_ratio_value.setText("0.00%")
        self.verticalLayout_fake.addWidget(self.label_fake)
        self.verticalLayout_fake.addWidget(self.fake_ratio_value)
        self.horizontalLayout_stats.addWidget(self.fake_ratio_stats)

        # 真实音频占比
        self.real_ratio_stats = QFrame(self.stats_panel)
        self.real_ratio_stats.setObjectName(u"real_ratio_stats")
        self.real_ratio_stats.setMinimumSize(QSize(150, 80))
        self.real_ratio_stats.setMaximumSize(QSize(150, 80))
        self.real_ratio_stats.setStyleSheet(
            u"QFrame#real_ratio_stats{\n"
            "color: rgb(255, 255, 255);\n"
            "border-radius: 15px;\n"
            "background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(255, 100, 100),  stop:1 rgb(255, 150, 100));\n"
            "border: 1px outset rgb(255, 125, 100)\n"
            "}"
        )
        self.verticalLayout_real = QVBoxLayout(self.real_ratio_stats)
        self.verticalLayout_real.setSpacing(5)
        self.verticalLayout_real.setContentsMargins(10, 10, 10, 10)
        self.label_real = QLabel(self.real_ratio_stats)
        self.label_real.setStyleSheet(u"color: rgba(255, 255, 255,210);\n" 'font: 700 italic 12pt "Segoe UI";')
        self.label_real.setAlignment(Qt.AlignCenter)
        self.label_real.setText("真实音频占比")
        self.real_ratio_value = QLabel(self.real_ratio_stats)
        self.real_ratio_value.setStyleSheet(u"color: rgb(255, 255, 255);\n" 'font: 14pt "Microsoft YaHei UI";')
        self.real_ratio_value.setAlignment(Qt.AlignCenter)
        self.real_ratio_value.setText("0.00%")
        self.verticalLayout_real.addWidget(self.label_real)
        self.verticalLayout_real.addWidget(self.real_ratio_value)
        self.horizontalLayout_stats.addWidget(self.real_ratio_stats)

        # 伪造概率均值
        self.prob_mean_stats = QFrame(self.stats_panel)
        self.prob_mean_stats.setObjectName(u"prob_mean_stats")
        self.prob_mean_stats.setMinimumSize(QSize(150, 80))
        self.prob_mean_stats.setMaximumSize(QSize(150, 80))
        self.prob_mean_stats.setStyleSheet(
            u"QFrame#prob_mean_stats{\n"
            "color: rgb(255, 255, 255);\n"
            "border-radius: 15px;\n"
            "background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(66, 150, 192),  stop:1 rgb(62, 100, 193));\n"
            "border: 1px outset rgb(72, 158, 204)\n"
            "}"
        )
        self.verticalLayout_prob = QVBoxLayout(self.prob_mean_stats)
        self.verticalLayout_prob.setSpacing(5)
        self.verticalLayout_prob.setContentsMargins(10, 10, 10, 10)
        self.label_prob = QLabel(self.prob_mean_stats)
        self.label_prob.setStyleSheet(u"color: rgba(255, 255, 255,210);\n" 'font: 700 italic 12pt "Segoe UI";')
        self.label_prob.setAlignment(Qt.AlignCenter)
        self.label_prob.setText("伪造概率均值")
        self.prob_mean_value = QLabel(self.prob_mean_stats)
        self.prob_mean_value.setStyleSheet(u"color: rgb(255, 255, 255);\n" 'font: 14pt "Microsoft YaHei UI";')
        self.prob_mean_value.setAlignment(Qt.AlignCenter)
        self.prob_mean_value.setText("0.00%")
        self.verticalLayout_prob.addWidget(self.label_prob)
        self.verticalLayout_prob.addWidget(self.prob_mean_value)
        self.horizontalLayout_stats.addWidget(self.prob_mean_stats)

        self.verticalLayout_batch.addWidget(self.stats_panel)

        #数据库管理页面
        self.database_page = QWidget()
        self.database_page.setObjectName(u"database_page")
        self.database_page.setStyleSheet(u"background: transparent;")
        self.verticalLayout_db = QVBoxLayout(self.database_page)
        self.verticalLayout_db.setSpacing(15)
        self.verticalLayout_db.setContentsMargins(20, 20, 20, 20)

        #数据库控制栏（查询/刷新/删除）
        self.db_control_bar = QFrame(self.database_page)
        self.db_control_bar.setObjectName(u"db_control_bar")
        self.db_control_bar.setMinimumSize(QSize(0, 50))
        self.db_control_bar.setStyleSheet(
            u"QFrame#db_control_bar{\n"
            "background-color: rgb(238, 242, 255);\n"
            "border:2px solid rgb(255, 255, 255);\n"
            "border-radius:15px\n"
            "}"
        )
        self.horizontalLayout_db_control = QHBoxLayout(self.db_control_bar)
        self.horizontalLayout_db_control.setSpacing(10)
        self.horizontalLayout_db_control.setContentsMargins(20, 10, 20, 10)

        # 查询按钮
        self.query_db_button = QPushButton(self.db_control_bar)
        self.query_db_button.setObjectName(u"query_db_button")
        self.query_db_button.setMinimumSize(QSize(120, 40))
        self.query_db_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.query_db_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #59969b, stop:1 #04e7fa);\n"
            "border: none;\n"
            "border-radius:10px;\n"
            "color: white;\n"
            'font: 700 11pt "Microsoft YaHei UI";\n'
            "}\n"
            "QPushButton:hover{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #40787c, stop:1 #03b5cc);\n"
            "}"
        )
        self.query_db_button.setText("查询所有记录")
        self.horizontalLayout_db_control.addWidget(self.query_db_button)

        # 刷新按钮
        self.refresh_db_button = QPushButton(self.db_control_bar)
        self.refresh_db_button.setObjectName(u"refresh_db_button")
        self.refresh_db_button.setMinimumSize(QSize(120, 40))
        self.refresh_db_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.refresh_db_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4CAF50, stop:1 #45a049);\n"
            "border: none;\n"
            "border-radius:10px;\n"
            "color: white;\n"
            'font: 700 11pt "Microsoft YaHei UI";\n'
            "}\n"
            "QPushButton:hover{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3d8b40, stop:1 #39803c);\n"
            "}"
        )
        self.refresh_db_button.setText("刷新列表")
        self.horizontalLayout_db_control.addWidget(self.refresh_db_button)

        # 删除选中记录按钮
        self.delete_db_button = QPushButton(self.db_control_bar)
        self.delete_db_button.setObjectName(u"delete_db_button")
        self.delete_db_button.setMinimumSize(QSize(120, 40))
        self.delete_db_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.delete_db_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #F44336, stop:1 #D32F2F);\n"
            "border: none;\n"
            "border-radius:10px;\n"
            "color: white;\n"
            'font: 700 11pt "Microsoft YaHei UI";\n'
            "}\n"
            "QPushButton:hover{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #D32F2F, stop:1 #B71C1C);\n"
            "}"
        )
        self.delete_db_button.setText("删除选中记录")
        self.horizontalLayout_db_control.addWidget(self.delete_db_button)

        # 导出数据库记录按钮
        self.export_db_button = QPushButton(self.db_control_bar)
        self.export_db_button.setObjectName(u"export_db_button")
        self.export_db_button.setMinimumSize(QSize(120, 40))
        self.export_db_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.export_db_button.setStyleSheet(
            u"QPushButton{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2196F3, stop:1 #0b7dda);\n"
            "border: none;\n"
            "border-radius:10px;\n"
            "color: white;\n"
            'font: 700 11pt "Microsoft YaHei UI";\n'
            "}\n"
            "QPushButton:hover{\n"
            "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1976D2, stop:1 #0967b7);\n"
            "}"
        )
        self.export_db_button.setText("导出所有记录")
        self.horizontalLayout_db_control.addWidget(self.export_db_button)

        self.verticalLayout_db.addWidget(self.db_control_bar)

        #数据库记录列表
        self.db_table_frame = QFrame(self.database_page)
        self.db_table_frame.setObjectName(u"db_table_frame")
        self.db_table_frame.setMinimumSize(QSize(0, 300))
        self.db_table_frame.setStyleSheet(
            u"QFrame#db_table_frame{\n"
            "background-color: rgb(238, 242, 255);\n"
            "border:2px solid rgb(255, 255, 255);\n"
            "border-radius:15px\n"
            "}"
        )
        self.verticalLayout_db_table = QVBoxLayout(self.db_table_frame)
        self.verticalLayout_db_table.setContentsMargins(10, 10, 10, 10)

        # 数据库表格
        self.db_table = QTableWidget(self.db_table_frame)
        self.db_table.setObjectName(u"db_table")
        self.db_table.setStyleSheet("""
        QTableWidget {
            background-color: #ffffff;       
            gridline-color: #e0e0e0;         
            border: 1px solid #ffffff;       
            border-radius: 6px;              
            padding: 4px;
        }
        QTableWidget::item {
            color: #333333;                  
            padding: 6px 8px;
            background-color: #ffffff;       
        }
        QTableWidget::item:selected {
            background-color: #3096fb;       
            color: white;
        }
        QTableWidget::item:alternate {
            background-color: #f5f7fa;       
        }
        QHeaderView::section {
            background-color: #ffffff;       
            color: #333333;
            border: 1px solid #ffffff;       
            padding: 6px;
            font-weight: bold;
        }
        /* 新增这一段修复左上角黑色 */
        QTableWidget::corner {
            background-color: #ffffff;
            border: 1px solid #ffffff;
        }
        QTableWidget::pane {
            border: 1px solid #ffffff;       
            border-radius: 6px;
            background-color: #ffffff;       
        }
        """)
        self.db_table.setAlternatingRowColors(True)
        # 设置列名
        self.db_table.setColumnCount(9)
        self.db_table.setHorizontalHeaderLabels([
            "ID", "文件名", "文件路径", "音频时长", "伪造概率",
            "判断结论", "伪造段数", "全局阈值",
            "状态"
        ])
        self.db_table.horizontalHeader().setStretchLastSection(True)
        self.db_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.verticalLayout_db_table.addWidget(self.db_table)
        self.verticalLayout_db.addWidget(self.db_table_frame)

        #数据库统计面板
        self.db_stats_panel = QFrame(self.database_page)
        self.db_stats_panel.setObjectName(u"db_stats_panel")
        self.db_stats_panel.setMinimumSize(QSize(0, 80))
        self.db_stats_panel.setStyleSheet(
            u"QFrame#db_stats_panel{\n"
            "background-color: rgb(238, 242, 255);\n"
            "border:2px solid rgb(255, 255, 255);\n"
            "border-radius:15px\n"
            "}"
        )
        self.horizontalLayout_db_stats = QHBoxLayout(self.db_stats_panel)
        self.horizontalLayout_db_stats.setSpacing(20)
        self.horizontalLayout_db_stats.setContentsMargins(20, 10, 20, 10)

        # 总记录数
        self.db_total_count = QFrame(self.db_stats_panel)
        self.db_total_count.setObjectName(u"db_total_count")
        self.db_total_count.setMinimumSize(QSize(150, 80))
        self.db_total_count.setMaximumSize(QSize(150, 80))
        self.db_total_count.setStyleSheet(
            u"QFrame#db_total_count{\n"
            "color: rgb(255, 255, 255);\n"
            "border-radius: 15px;\n"
            "background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(30, 150, 255),  stop:1 rgb(50, 100, 255));\n"
            "border: 1px outset rgb(40, 125, 255);\n"
            "}"
        )
        self.verticalLayout_db_total = QVBoxLayout(self.db_total_count)
        self.verticalLayout_db_total.setSpacing(5)
        self.verticalLayout_db_total.setContentsMargins(10, 10, 10, 10)
        self.label_db_total = QLabel(self.db_total_count)
        self.label_db_total.setStyleSheet(u"color: rgba(255, 255, 255,210);\n" 'font: 700 italic 12pt "Segoe UI";')
        self.label_db_total.setAlignment(Qt.AlignCenter)
        self.label_db_total.setText("总记录数")
        self.db_total_value = QLabel(self.db_total_count)
        self.db_total_value.setStyleSheet(u"color: rgb(255, 255, 255);\n" 'font: 14pt "Microsoft YaHei UI";')
        self.db_total_value.setAlignment(Qt.AlignCenter)
        self.db_total_value.setText("0")
        self.verticalLayout_db_total.addWidget(self.label_db_total)
        self.verticalLayout_db_total.addWidget(self.db_total_value)
        self.horizontalLayout_db_stats.addWidget(self.db_total_count)

        # 伪造记录数
        self.db_fake_count = QFrame(self.db_stats_panel)
        self.db_fake_count.setObjectName(u"db_fake_count")
        self.db_fake_count.setMinimumSize(QSize(150, 80))
        self.db_fake_count.setMaximumSize(QSize(150, 80))
        self.db_fake_count.setStyleSheet(
            u"QFrame#db_fake_count{\n"
            "color: rgb(255, 255, 255);\n"
            "border-radius: 15px;\n"
            "background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(253, 139, 133),  stop:1 rgb(248, 194, 152));\n"
            "border: 1px outset rgb(252, 194, 149)\n"
            "}"
        )
        self.verticalLayout_db_fake = QVBoxLayout(self.db_fake_count)
        self.verticalLayout_db_fake.setSpacing(5)
        self.verticalLayout_db_fake.setContentsMargins(10, 10, 10, 10)
        self.label_db_fake = QLabel(self.db_fake_count)
        self.label_db_fake.setStyleSheet(u"color: rgba(255, 255, 255,210);\n" 'font: 700 italic 12pt "Segoe UI";')
        self.label_db_fake.setAlignment(Qt.AlignCenter)
        self.label_db_fake.setText("伪造记录数")
        self.db_fake_value = QLabel(self.db_fake_count)
        self.db_fake_value.setStyleSheet(u"color: rgb(255, 255, 255);\n" 'font: 14pt "Microsoft YaHei UI";')
        self.db_fake_value.setAlignment(Qt.AlignCenter)
        self.db_fake_value.setText("0")
        self.verticalLayout_db_fake.addWidget(self.label_db_fake)
        self.verticalLayout_db_fake.addWidget(self.db_fake_value)
        self.horizontalLayout_db_stats.addWidget(self.db_fake_count)

        # 真实记录数
        self.db_real_count = QFrame(self.db_stats_panel)
        self.db_real_count.setObjectName(u"db_real_count")
        self.db_real_count.setMinimumSize(QSize(150, 80))
        self.db_real_count.setMaximumSize(QSize(150, 80))
        self.db_real_count.setStyleSheet(
            u"QFrame#db_real_count{\n"
            "color: rgb(255, 255, 255);\n"
            "border-radius: 15px;\n"
            "background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(255, 100, 100),  stop:1 rgb(255, 150, 100));\n"
            "border: 1px outset rgb(255, 125, 100)\n"
            "}"
        )
        self.verticalLayout_db_real = QVBoxLayout(self.db_real_count)
        self.verticalLayout_db_real.setSpacing(5)
        self.verticalLayout_db_real.setContentsMargins(10, 10, 10, 10)
        self.label_db_real = QLabel(self.db_real_count)
        self.label_db_real.setStyleSheet(u"color: rgba(255, 255, 255,210);\n" 'font: 700 italic 12pt "Segoe UI";')
        self.label_db_real.setAlignment(Qt.AlignCenter)
        self.label_db_real.setText("真实记录数")
        self.db_real_value = QLabel(self.db_real_count)
        self.db_real_value.setStyleSheet(u"color: rgb(255, 255, 255);\n" 'font: 14pt "Microsoft YaHei UI";')
        self.db_real_value.setAlignment(Qt.AlignCenter)
        self.db_real_value.setText("0")
        self.verticalLayout_db_real.addWidget(self.label_db_real)
        self.verticalLayout_db_real.addWidget(self.db_real_value)
        self.horizontalLayout_db_stats.addWidget(self.db_real_count)

        # 平均伪造概率
        self.db_avg_prob = QFrame(self.db_stats_panel)
        self.db_avg_prob.setObjectName(u"db_avg_prob")
        self.db_avg_prob.setMinimumSize(QSize(150, 80))
        self.db_avg_prob.setMaximumSize(QSize(150, 80))
        self.db_avg_prob.setStyleSheet(
            u"QFrame#db_avg_prob{\n"
            "color: rgb(255, 255, 255);\n"
            "border-radius: 15px;\n"
            "background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(66, 150, 192),  stop:1 rgb(62, 100, 193));\n"
            "border: 1px outset rgb(72, 158, 204)\n"
            "}"
        )
        self.verticalLayout_db_avg = QVBoxLayout(self.db_avg_prob)
        self.verticalLayout_db_avg.setSpacing(5)
        self.verticalLayout_db_avg.setContentsMargins(10, 10, 10, 10)
        self.label_db_avg = QLabel(self.db_avg_prob)
        self.label_db_avg.setStyleSheet(u"color: rgba(255, 255, 255,210);\n" 'font: 700 italic 12pt "Segoe UI";')
        self.label_db_avg.setAlignment(Qt.AlignCenter)
        self.label_db_avg.setText("平均伪造概率")
        self.db_avg_value = QLabel(self.db_avg_prob)
        self.db_avg_value.setStyleSheet(u"color: rgb(255, 255, 255);\n" 'font: 14pt "Microsoft YaHei UI";')
        self.db_avg_value.setAlignment(Qt.AlignCenter)
        self.db_avg_value.setText("0.00%")
        self.verticalLayout_db_avg.addWidget(self.label_db_avg)
        self.verticalLayout_db_avg.addWidget(self.db_avg_value)
        self.horizontalLayout_db_stats.addWidget(self.db_avg_prob)

        self.verticalLayout_db.addWidget(self.db_stats_panel)

        #系统信息页面
        self.system_info_page = QWidget()
        self.system_info_page.setObjectName(u"system_info_page")
        self.system_info_page.setStyleSheet(u"background: transparent;")
        self.scroll_area = QScrollArea(self.system_info_page)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
        QScrollArea {
            border: none;
            background-color: transparent;
        }
        QScrollBar:vertical {
            width: 8px;
            background-color: #f0f0f0;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background-color: #3096fb;
            border-radius: 4px;
        }
        """)

        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet(u"background-color: #ffffff;")
        self.verticalLayout_sys = QVBoxLayout(self.scroll_content)
        self.verticalLayout_sys.setSpacing(15)
        self.verticalLayout_sys.setContentsMargins(20, 20, 20, 20)
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout_sys = QVBoxLayout(self.system_info_page)
        self.main_layout_sys.setContentsMargins(0, 0, 0, 0)
        self.main_layout_sys.addWidget(self.scroll_area)
        self.label_sys_title = QLabel(self.system_info_page)
        self.label_sys_title.setStyleSheet(u'font: 700 16pt "Segoe UI"; color: #333;')
        self.label_sys_title.setAlignment(Qt.AlignCenter)
        self.label_sys_title.setText("系统信息")
        self.verticalLayout_sys.addWidget(self.label_sys_title)

        #模型效果
        self.label_model_effect_title = QLabel(self.system_info_page)
        self.label_model_effect_title.setStyleSheet(u'font: 700 12pt "Segoe UI"; color: #333;')
        self.label_model_effect_title.setAlignment(Qt.AlignLeft)
        self.label_model_effect_title.setText("模型效果")
        self.verticalLayout_sys.addWidget(self.label_model_effect_title)

        #检测精度
        self.accuracy_table = QTableWidget(self.system_info_page)
        self.accuracy_table.setObjectName(u"accuracy_table")
        self.accuracy_table.setStyleSheet("""
        QTableWidget {
            background-color: #ffffff;
            gridline-color: #e0e0e0;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 4px;
        }
        QTableWidget::item {
            color: #333333;
            padding: 6px 8px;
            background-color: #ffffff;
        }
        QTableWidget::item:selected {
            background-color: #3096fb;
            color: white;
        }
        QHeaderView::section {
            background-color: #f5f7fa;
            color: #333333;
            border: 1px solid #e0e0e0;
            padding: 6px;
            font-weight: bold;
        }
        QTableWidget::corner {
            background-color: #ffffff;
            border: 1px solid #ffffff;
        }
        QTableWidget::pane {
            border: 1px solid #ffffff;       
            border-radius: 6px;
            background-color: #ffffff;      
        }
        """)
        self.accuracy_table.setColumnCount(2)
        self.accuracy_table.setHorizontalHeaderLabels(["数据集", "EER值"])

        # 填充数据
        accuracy_data = [
            ["ASVspoof2019 LA", "1.6836"],
            ["ASVspoof2021 LA", "10.9740"],
            ["FAC", "9.8951"],
            ["FOR", "0.7984"],
            ["ITW", "22.3924"]
        ]
        self.accuracy_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.accuracy_table.setMinimumHeight(200)  # 给表格设置最小高度
        self.accuracy_table.setRowCount(len(accuracy_data))
        for row_idx, (dataset, eer) in enumerate(accuracy_data):
            self.accuracy_table.setItem(row_idx, 0, QTableWidgetItem(dataset))
            self.accuracy_table.setItem(row_idx, 1, QTableWidgetItem(eer))
        self.accuracy_table.horizontalHeader().setStretchLastSection(True)
        self.accuracy_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalLayout_sys.addWidget(self.accuracy_table)

        #推理效率
        self.label_inference_title = QLabel(self.system_info_page)
        self.label_inference_title.setStyleSheet(u'font: 700 12pt "Segoe UI"; color: #333;')
        self.label_inference_title.setAlignment(Qt.AlignLeft)
        self.label_inference_title.setText("推理效率")
        self.verticalLayout_sys.addWidget(self.label_inference_title)

        #推理效率
        self.inference_table = QTableWidget(self.system_info_page)
        self.inference_table.setObjectName(u"inference_table")
        self.inference_table.setStyleSheet(self.accuracy_table.styleSheet())
        self.inference_table.setColumnCount(3)
        self.inference_table.setHorizontalHeaderLabels(["模型", "RTF (Orange Pi 5Plus)", "参数量 (M)"])
        # 填充数据
        inference_data = [
            ["AASIST", "0.0955", "0.297"],
            ["RawGAT-ST", "0.1162", "0.437"],
            ["PSDL", "0.1273", "317.7"],
            ["Nes2Net", "0.1580", "317.9"],
            ["AMSDF", "0.2642", "325.4"],
            ["SLS", "0.2732", "340.8"],
            ["Audio_Q", "0.0486", "27.84"]
        ]
        self.inference_table.setRowCount(len(inference_data))
        self.inference_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.inference_table.setMinimumHeight(300)
        for row_idx, (model_name, rtf, params) in enumerate(inference_data):
            self.inference_table.setItem(row_idx, 0, QTableWidgetItem(model_name))
            self.inference_table.setItem(row_idx, 1, QTableWidgetItem(rtf))
            self.inference_table.setItem(row_idx, 2, QTableWidgetItem(params))
        self.inference_table.horizontalHeader().setStretchLastSection(True)
        self.inference_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalLayout_sys.addWidget(self.inference_table)

        #模型优点
        self.label_advantages_title = QLabel(self.system_info_page)
        self.label_advantages_title.setStyleSheet(u'font: 700 12pt "Segoe UI"; color: #333;')
        self.label_advantages_title.setAlignment(Qt.AlignLeft)
        self.label_advantages_title.setText("模型优点")
        self.verticalLayout_sys.addWidget(self.label_advantages_title)

        import textwrap

        self.advantages_text = QTextEdit(self.system_info_page)
        self.advantages_text.setReadOnly(True)
        self.advantages_text.setStyleSheet("""
        QTextEdit {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 10px;
            font: 12pt "Microsoft YaHei UI";
            color: #333;
        }
        """)
        advantages_content = textwrap.dedent("""
            1. 高精度鲁棒性：在ASVspoof2019 LA数据集上EER低至1.6836，对常见伪造攻击（TTS、语音转换）识别准确率高。
            2. 泛化能力强：通过多数据集（ASVspoof系列、FAC、FOR等）验证，在跨数据集场景下仍保持较好性能（FOR数据集EER仅0.7984）。
            3. 高效轻量化：参数量约27.8M，在RK3588上可实现实时推理，兼顾精度与部署效率。
            4. 特征表达能力强：结合卷积与注意力机制，能有效捕捉音频的时序与频谱伪造特征，区分细微的声学差异。
        """).strip()

        self.advantages_text.setPlainText(advantages_content)
        self.verticalLayout_sys.addWidget(self.advantages_text)

        # 将4个页面添加到堆叠窗口
        self.content.addWidget(self.single_audio_page)
        self.content.addWidget(self.batch_audio_page)
        self.content.addWidget(self.database_page)
        self.content.addWidget(self.system_info_page)

        # 将水平布局添加到垂直布局
        self.verticalLayout_6.addLayout(self.hlayout_content)
        self.main_qframe.addWidget(self.ContentBox)
        self.horizontalLayout_14.addWidget(self.Main_QF)
        MainWindow.setCentralWidget(self.Main_QW)

        self.retranslateUi(MainWindow)
        self.content.setCurrentWidget(self.single_audio_page)
        QMetaObject.connectSlotsByName(MainWindow)

        # 绑定侧边栏按钮切换页面
        self.single_audio_button.clicked.connect(
            lambda: [self.content.setCurrentWidget(self.single_audio_page), self.update_sidebar_triangle("single")])
        self.batch_audio_button.clicked.connect(
            lambda: [self.content.setCurrentWidget(self.batch_audio_page), self.update_sidebar_triangle("batch")])
        self.database_button.clicked.connect(
            lambda: [self.content.setCurrentWidget(self.database_page), self.update_sidebar_triangle("db")])
        self.system_info_button.clicked.connect(
            lambda: [self.content.setCurrentWidget(self.system_info_page), self.update_sidebar_triangle("sys")])
        # 批量检测按钮事件绑定
        self.select_batch_button.clicked.connect(self.select_batch_files)
        self.start_batch_button.clicked.connect(self.start_batch_detection)
        self.pause_resume_button.clicked.connect(self.pause_resume_batch)
        self.cancel_batch_button.clicked.connect(self.cancel_batch_detection)
        self.export_batch_button.clicked.connect(self.export_batch_results)

        self.stored_waveform_data = None  # 持久化存储波形数据
        self.audio_file_path = None  # 存储选中的音频文件路径
        self.original_file_path = None  # 存储原始文件路径
        self.original_audio_format = None  # 存储原始音频格式
        self.detection_results = None  # 存储检测结果
        self.batch_file_list = []  # 存储选中的文件列表
        self.batch_results = {}  # 存储批量检测结果 {文件路径: 检测结果}
        self.batch_current_index = 0  # 当前检测到的文件索引
        self.batch_is_paused = False  # 是否暂停
        self.batch_is_cancelled = False  # 是否取消
        self.batch_start_time = None  # 批量检测开始时间
        self.batch_thread = None  # 批量检测线程

        self.audio_playing = False  # 是否正在播放
        self.audio_current_pos = 0  # 当前播放位置
        self.stored_waveform_data = None  # 存储的波形数据
        self.audio_process = None  # 音频播放进程
        self.processed_audio_path = None  # 处理后的音频路径
        self.processed_waveform_data = None  # 处理后的波形数据

        # 绑定按钮点击事件
        self.audio_source_combo.currentIndexChanged.connect(self.on_audio_source_changed)
        self.select_audio_button.clicked.connect(self.select_audio_file)
        self.detect_button.clicked.connect(self.run_real_detection)
        self.play_button.clicked.connect(self.toggle_audio_playback)
        self.save_button.clicked.connect(self.save_results)
        self.min_sf.clicked.connect(MainWindow.showMinimized)
        self.max_sf.clicked.connect(MainWindow.showMaximized)
        self.close_button.clicked.connect(MainWindow.close)
        self.tools_button.clicked.connect(self.toggle_tools_panel)

        # 工具面板功能绑定
        self.tools_slide_step_slider.valueChanged.connect(self.on_slide_step_changed)
        self.tools_change_sample_btn.clicked.connect(self.tools_change_sample_rate)
        self.tools_change_format_btn.clicked.connect(self.tools_change_format)

        # 数据库页面按钮绑定
        self.query_db_button.clicked.connect(self.load_db_records)
        self.refresh_db_button.clicked.connect(self.load_db_records)
        self.delete_db_button.clicked.connect(self.delete_selected_db_record)
        self.export_db_button.clicked.connect(self.export_db_records)

        # 初始化时加载数据库记录
        self.load_db_records()
        # 初始化模型
        init_model()
        init_database()

    def select_audio_file(self):
        """选择音频文件或实时录制"""
        audio_source = self.audio_source_combo.currentData()

        if audio_source == "file":
            # 上传文件模式
            self.select_audio_from_file()
        elif audio_source == "record":
            # 实时录制模式
            self.start_real_time_recording()

    def select_audio_from_file(self):
        """从文件选择音频"""
        # 清空之前的结果和处理数据
        self.clear_results()

        # 清除之前处理的数据
        if hasattr(self, 'processed_audio_path'):
            self.processed_audio_path = None
        if hasattr(self, 'processed_waveform_data'):
            self.processed_waveform_data = None
        if hasattr(self, 'audio_processed'):
            self.audio_processed = False

        # 打开文件选择对话框 - 明确指定WAV和FLAC格式
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "选择音频文件",
            "",
            "Audio Files (*.wav *.flac);;WAV Files (*.wav);;FLAC Files (*.flac)"
        )

        if not file_path:
            return

        self.audio_file_path = file_path
        self.original_file_path = file_path

        # 保存原始音频格式
        import os
        self.original_audio_format = os.path.splitext(file_path)[1].lower().replace('.', '')

        try:
            wave, sr = librosa.load(file_path, sr=SR)
            import soundfile as sf
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, wave, SR)
            self.audio_file_path = temp_file.name
            temp_file.close()
            self.update_waveform_display(wave)
            duration = len(wave) / sr
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            self.audio_duration_value.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

            QMessageBox.information(None, "成功", "音频文件加载成功！")

        except Exception as e:
            QMessageBox.critical(None, "错误", f"加载音频失败：{str(e)}")
            self.audio_file_path = None

    def run_real_detection(self):
        """模型检测"""
        if not self.audio_file_path:
            QMessageBox.warning(None, "警告", "请先选择音频文件！")
            return

        if model is None:
            if not init_model():
                return

        # 启动进度条更新线程
        def update_progress_bar():
            for i in range(0, 1001, 50):
                QMetaObject.invokeMethod(self.progress_bar, "setValue", Qt.QueuedConnection,
                                         Q_ARG(int, i))
                time.sleep(0.1)

        progress_thread = threading.Thread(target=update_progress_bar)
        progress_thread.daemon = True
        progress_thread.start()

        try:
            if hasattr(self, 'processed_waveform_data') and self.processed_waveform_data is not None:
                wave = self.processed_waveform_data.copy()
                audio_file_used = getattr(self, 'processed_audio_path', None) or self.audio_file_path
            else:
                wave, _ = librosa.load(self.audio_file_path, sr=SR)
                audio_file_used = self.audio_file_path

            audio_duration = len(wave) / SR
            num_samples = len(wave)
            use_sliding_window = num_samples >= MAX_LEN

            if use_sliding_window:
                num_segments = max(1, int(np.ceil((num_samples - MAX_LEN) / SLIDE_STEP)) + 1) if SLIDE_STEP > 0 else 1
                segment_probs = []
                fake_segments = 0
                global_logits_list = []
                segment_vectors = []
                segment_timestamps = []  # 记录每段的起始时间（秒）

                for seg_idx in range(num_segments):
                    # 使用滑动步进值
                    start_idx = seg_idx * SLIDE_STEP
                    if start_idx >= num_samples:
                        break
                    end_idx = min(start_idx + MAX_LEN, num_samples)
                    segment_wave = wave[start_idx:end_idx]

                    # 如果最后一段不足MAX_LEN，重复填充
                    if len(segment_wave) < MAX_LEN:
                        num_repeats = int(MAX_LEN / len(segment_wave)) + 1
                        segment_wave = np.tile(segment_wave, (1, num_repeats))[:, :MAX_LEN][0]

                    # 记录段起始时间
                    segment_timestamps.append(start_idx / SR)

                    # 预处理当前段
                    wave_tensor, spec_tensor, freq_aug, _ = preprocess_audio_segment(segment_wave)

                    # 模型推理
                    with torch.no_grad():
                        global_logits, _, projected_vector = model(wave_tensor, spec_tensor, freq_aug)

                    # 转换为numpy
                    global_logits_np = global_logits.squeeze(0).cpu().numpy()
                    print(f">>> 段{seg_idx + 1} logits: fake={global_logits_np[0]:.4f}, real={global_logits_np[1]:.4f}")
                    projected_vector_np = projected_vector.squeeze(0).cpu().numpy()
                    global_logits_list.append(global_logits_np)
                    segment_vectors.append(projected_vector_np)

                    # 计算伪造概率
                    real_logit = global_logits_np[1]
                    fake_prob = 100 / (1 + np.exp(1.5 * (real_logit - THRESHOLD_GLOBAL)))
                    fake_prob = np.clip(fake_prob, 0, 100)
                    segment_probs.append(fake_prob)

                    # 判断是否为伪造段（使用50%阈值）
                    is_fake = fake_prob > 50.0
                    if is_fake:
                        fake_segments += 1

                # 计算整体结果
                global_logits_avg = np.mean(global_logits_list, axis=0)
                projected_vector_avg = np.mean(segment_vectors, axis=0) if len(segment_vectors) > 0 else None
            else:
                if num_samples < MAX_LEN:
                    num_repeats = int(MAX_LEN / num_samples) + 1
                    padded_wave = np.tile(wave, (1, num_repeats))[:, :MAX_LEN][0]
                else:
                    padded_wave = wave[:MAX_LEN]

                # 预处理
                wave_tensor, spec_tensor, freq_aug, _ = preprocess_audio_segment(padded_wave)

                # 模型推理
                with torch.no_grad():
                    global_logits, _, projected_vector = model(wave_tensor, spec_tensor, freq_aug)
                global_logits_avg = global_logits.squeeze(0).cpu().numpy()
                projected_vector_avg = projected_vector.squeeze(0).cpu().numpy()

                # 计算伪造概率
                real_logit = global_logits_avg[1]
                fake_prob = 100 / (1 + np.exp(1.5 * (real_logit - THRESHOLD_GLOBAL)))
                fake_prob = np.clip(fake_prob, 0, 100)

                # 使用50%阈值判断
                fake_segments = 1 if fake_prob > 50.0 else 0

                segment_probs = [fake_prob]
                segment_timestamps = [0.0]
                num_segments = 1

            # 存储结果
            self.detection_results = {
                'global_logits': global_logits_avg,
                'frame_logits': None,
                'projected_vector': projected_vector_avg,
                'audio_duration': audio_duration,
                'waveform_data': wave,
                'segment_probs': segment_probs,
                'fake_segments': fake_segments,
                'num_segments': num_segments,
                'segment_vectors': segment_vectors if use_sliding_window else [],
                'segment_timestamps': segment_timestamps,
                'slide_step': SLIDE_STEP / SR,
                'use_sliding_window': use_sliding_window
            }

            # 更新UI
            self.update_detection_results(self.detection_results)

            # 更新UMAP可视化
            if projected_vector_avg is not None:
                self.vector_plot.set_upload_vector(projected_vector_avg)

            QMessageBox.information(None, "完成", f"检测完成！共 {num_segments} 段，伪造 {fake_segments} 段")

            db_result_data = {
                "file_name": os.path.basename(self.original_file_path),
                "file_path": self.original_file_path,
                "audio_duration": self.audio_duration_value.text(),
                "fake_prob": float(self.forgery_prob_value.text().replace('%', '')),
                "conclusion": "伪造" if float(self.forgery_prob_value.text().replace('%', '')) > 50 else "真实",
                "fake_segments": fake_segments,
                "global_threshold": THRESHOLD_GLOBAL,
                "status": "成功"
            }

            insert_detection_result_to_db(db_result_data)

        except Exception as e:
            QMessageBox.critical(None, "检测错误", f"模型推理失败：{str(e)}")
            self.progress_bar.setValue(0)
    def save_results(self):
        """保存单音频检测结果"""
        if not self.detection_results or not self.audio_file_path:
            QMessageBox.warning(None, "警告", "暂无检测结果可保存！")
            return

        #弹出格式选择对话框
        format_options = ["Excel (.xlsx)", "CSV (.csv)", "TXT (.txt)", "JSON (.json)"]
        choice, ok = QInputDialog.getItem(
            None,
            "选择保存格式",
            "请选择要保存的文件格式：",
            format_options,
            0,
            False
        )
        if not ok:
            return

        #匹配格式后缀和过滤器
        format_map = {
            "Excel (.xlsx)": (".xlsx", "Excel Files (*.xlsx)"),
            "CSV (.csv)": (".csv", "CSV Files (*.csv)"),
            "TXT (.txt)": (".txt", "Text Files (*.txt)"),
            "JSON (.json)": (".json", "JSON Files (*.json)")
        }
        suffix, filter_text = format_map[choice]

        #打开保存对话框
        default_filename = os.path.splitext(os.path.basename(self.audio_file_path))[0] + "_检测结果"
        save_path, _ = QFileDialog.getSaveFileName(
            None,
            "保存检测结果",
            default_filename + suffix,
            f"{filter_text};All Files (*.*)"
        )
        if not save_path:
            return
        if not save_path.endswith(suffix):
            save_path += suffix

        #按格式保存结果
        try:
            segment_probs = self.detection_results.get('segment_probs', [])
            if len(segment_probs) > 0:
                fake_prob = np.mean(segment_probs)
            else:
                fake_prob = 0.0
            is_fake = fake_prob > 50.0
            global_logits = self.detection_results.get('global_logits', [0, 0])

            single_result = {
                'name': os.path.basename(self.audio_file_path),
                'path': self.audio_file_path,
                'duration': self.audio_duration_value.text(),
                'fake_prob': round(fake_prob, 2),
                'conclusion': "伪造" if is_fake else "真实",
                'fake_segments': self.detection_results.get('fake_segments', 0),
                'status': "成功"
            }

            # 按选择的格式保存
            if choice == "Excel (.xlsx)":
                import pandas as pd
                df = pd.DataFrame([single_result])
                df = df[['name', 'path', 'duration', 'fake_prob', 'conclusion', 'fake_segments', 'status']]
                df.to_excel(save_path, index=False)
            elif choice == "CSV (.csv)":
                import pandas as pd
                df = pd.DataFrame([single_result])
                df = df[['name', 'path', 'duration', 'fake_prob', 'conclusion', 'fake_segments', 'status']]
                df.to_csv(save_path, index=False, encoding='utf-8-sig')
            elif choice == "TXT (.txt)":
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write("音频伪造检测结果\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"检测时间：{QDateTime.currentDateTime().toString('yyyy-MM-dd HH:mm:ss')}\n")
                    f.write(f"文件名：{single_result['name']}\n")
                    f.write(f"文件路径：{single_result['path']}\n")
                    f.write(f"音频时长：{single_result['duration']}\n")
                    f.write(f"伪造概率：{single_result['fake_prob']}%\n")
                    f.write(f"判断结论：{single_result['conclusion']}\n")
                    f.write(f"伪造段数：{single_result['fake_segments']}\n")
                    f.write(f"检测状态：{single_result['status']}\n")
            elif choice == "JSON (.json)":
                import json
                export_item = {
                    '文件名': single_result['name'],
                    '文件路径': single_result['path'],
                    '音频时长': single_result['duration'],
                    '伪造概率(%)': float(single_result['fake_prob']),
                    '判断结论': single_result['conclusion'],
                    '伪造段数': int(single_result['fake_segments']),
                    '检测状态': single_result['status']
                }
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(export_item, f, ensure_ascii=False, indent=4)

            QMessageBox.information(None, "成功", f"检测结果已成功保存为{choice}格式！")
        except Exception as e:
            QMessageBox.critical(None, "错误", f"保存结果失败：{str(e)}")

    def update_detection_results(self, results):
        """更新检测结果到UI界面"""
        waveform_data = results.get('waveform_data', None)
        if waveform_data is None:
            QMessageBox.warning(None, "警告", "未找到音频数据")
            return

        #获取段级检测结果
        segment_probs = results.get('segment_probs', [])
        fake_segments = results.get('fake_segments', 0)
        num_segments = results.get('num_segments', 1)
        global_logits = results.get('global_logits', None)
        segment_timestamps = results.get('segment_timestamps', None)
        audio_duration = results.get('audio_duration', 0)
        use_sliding_window = results.get('use_sliding_window', True)

        #计算音频时长
        hours = int(audio_duration // 3600)
        minutes = int((audio_duration % 3600) // 60)
        seconds = int(audio_duration % 60)
        self.audio_duration_value.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

        #计算整体伪造概率
        if len(segment_probs) > 0:
            fake_prob = np.mean(segment_probs)
            self.forgery_prob_value.setText(f"{fake_prob:.2f}%")
        else:
            self.forgery_prob_value.setText("0.00%")

        #显示伪造段数
        self.forgery_frames_value.setText(str(fake_segments))

        #更新概率曲线图（滑动窗口）
        if len(segment_probs) > 0:
            slide_step = results.get('slide_step', 1.0)
            self.forgery_heatmap.set_segment_data(
                segment_probs,
                segment_timestamps,
                audio_duration,
                slide_step,
                use_sliding_window
            )
        else:
            self.forgery_heatmap.set_segment_probs(None)

        #更新进度条为100%
        self.progress_bar.setValue(1000)

        #更新波形图
        self.update_waveform_display(waveform_data)

        #保存检测结果
        self.detection_results = {
            'logits': [round(float(global_logits[0]), 4),
                       round(float(global_logits[1]), 4)] if global_logits is not None else [0, 0],
            'fake_segments': fake_segments,
            'segment_probs': segment_probs,
            'num_segments': num_segments,
            'waveform_data': waveform_data,
            'use_sliding_window': use_sliding_window
        }

    def _logit_to_fake_prob(self, threshold):
        """将real logit阈值转换为伪造概率"""
        fake_prob = 100 * (np.exp(-threshold) / (np.exp(-threshold) + np.exp(threshold)))
        return fake_prob

    def update_progress(self, value):
        """更新进度条"""
        QMetaObject.invokeMethod(self.progress_bar, "setValue", Qt.QueuedConnection,
                                 Q_ARG(int, int(value)))

    def clear_results(self):
        """清空结果"""
        self.forgery_prob_value.setText("0.00%")
        self.audio_duration_value.setText("00:00:00")
        self.forgery_frames_value.setText("0")
        self.progress_bar.setValue(0)
        self.audio_waveform.set_waveform_data(None)
        self.forgery_heatmap.set_segment_probs(None)
        # 重置播放状态
        self.audio_playing = False
        if hasattr(self, 'play_button'):
            self.play_button.setText("▶ 播放")

    def on_audio_source_changed(self, index):
        """音频源切换处理"""
        audio_source = self.audio_source_combo.currentData()

        if audio_source == "file":
            # 上传文件模式，显示选择文件按钮
            self.select_audio_button.setVisible(True)
            self.select_audio_button.setText("选择音频")
        elif audio_source == "record":
            # 实时录制模式
            self.select_audio_button.setText("开始录制")
            # 重置录音状态
            self.recording_active = False

        # 清空当前结果和处理数据
        self.clear_results()
        self.audio_file_path = None
        # 清除处理后的音频数据
        if hasattr(self, 'processed_audio_path'):
            self.processed_audio_path = None
        if hasattr(self, 'processed_waveform_data'):
            self.processed_waveform_data = None

    def start_real_time_recording(self):
        """开始/停止实时录制音频"""
        import platform
        import subprocess
        import tempfile
        import os

        system = platform.system()

        # 检查模型是否加载
        global model
        if model is None and not (hasattr(self, 'recording_active') and self.recording_active):
            QMessageBox.warning(None, "警告", "请先加载模型！")
            return

        # 如果正在录音，点击则停止
        if hasattr(self, 'recording_active') and self.recording_active:
            self.recording_active = False
            self.select_audio_button.setText("开始录制")
            self.select_audio_button.setEnabled(True)

            if system == "Linux" and hasattr(self, 'recording_process'):
                try:
                    import time
                    time.sleep(0.3)
                    self.recording_process.terminate()
                    self.recording_process.wait()
                except:
                    pass

            if hasattr(self, 'recorded_audio_path'):
                self.load_recorded_audio()
            return

        # 开始新的录制
        self.recording_active = True

        # 创建临时文件保存录音
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.recorded_audio_path = temp_file.name
        temp_file.close()

        if system == "Linux":
            try:
                process = subprocess.Popen(
                    ['arecord', '-D', 'plughw:4', '-r', '16000', '-f', 'S16_LE', '-c', '1', '-t', 'wav',
                     self.recorded_audio_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self.recording_process = process

                # 更新UI
                self.select_audio_button.setText("⏹ 停止录音")

                # 启动定时器检查录制状态
                QTimer.singleShot(500, self.check_recording_status_linux)

            except Exception as e:
                QMessageBox.warning(None, "录制错误", f"无法启动录音: {str(e)}")
                self.recording_active = False
        else:
            try:
                import pyaudio
                import wave
                import threading

                CHUNK = 1024
                FORMAT = pyaudio.paInt16
                CHANNELS = 1
                RATE = SR

                p = pyaudio.PyAudio()

                stream = p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                )

                frames = []
                self.pyaudio_stream = stream
                self.pyaudio_frames = frames
                self.pyaudio_pyaudio = p

                # 更新UI
                self.select_audio_button.setText("⏹ 停止录音")

                # 在后台录制
                def record_thread():
                    try:
                        while self.recording_active:
                            data = stream.read(CHUNK, exception_on_overflow=False)
                            frames.append(data)
                    except:
                        pass

                self.record_thread = threading.Thread(target=record_thread, daemon=True)
                self.record_thread.start()

            except ImportError:
                QMessageBox.warning(None, "错误", "请安装 pyaudio: pip install pyaudio")
                self.recording_active = False

    def check_recording_status_linux(self):
        """检查Linux录音状态"""
        if not self.recording_active:
            return

        if hasattr(self, 'recording_process') and self.recording_process.poll() is None:
            QTimer.singleShot(500, self.check_recording_status_linux)
        else:
            self.recording_active = False
            self.select_audio_button.setText("开始录制")
            self.select_audio_button.setEnabled(True)

            # 加载录音文件
            if hasattr(self, 'recorded_audio_path'):
                self.load_recorded_audio()

    def load_recorded_audio(self):
        """加载录制的音频文件"""
        import os
        import librosa

        if not hasattr(self, 'recorded_audio_path') or not os.path.exists(self.recorded_audio_path):
            QMessageBox.warning(None, "错误", "录音文件不存在！")
            return

        try:
            self.audio_file_path = self.recorded_audio_path

            # 读取音频
            wave, sr = librosa.load(self.recorded_audio_path, sr=SR)

            # 更新波形显示
            self.update_waveform_display(wave)

            # 更新音频时长显示
            duration = len(wave) / SR
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            self.audio_duration_value.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

        except Exception as e:
            QMessageBox.warning(None, "错误", f"无法读取录音文件: {str(e)}")

    def check_recording_status_linux(self):
        """检查Linux录音状态"""
        if not self.recording_active:
            return

        # 检查进程是否还在运行
        if hasattr(self, 'recording_process') and self.recording_process.poll() is None:
            # 还在录音，每秒检查一次
            QTimer.singleShot(1000, lambda: self.check_recording_status_linux())
        else:
            # 录音完成
            self.on_recording_finished()

    def on_recording_finished(self):
        """录音完成后的处理"""
        self.recording_active = False
        self.select_audio_button.setText("开始录制")
        self.select_audio_button.setEnabled(True)

        if hasattr(self, 'recorded_audio_path') and os.path.exists(self.recorded_audio_path):
            self.audio_file_path = self.recorded_audio_path

            try:
                # 读取录制的音频
                wave, sr = librosa.load(self.recorded_audio_path, sr=SR)

                # 更新波形显示
                self.update_waveform_display(wave)

                # 更新音频时长显示
                duration = len(wave) / SR
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                self.audio_duration_value.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

            except Exception as e:
                QMessageBox.warning(None, "错误", f"无法读取录音文件: {str(e)}")

        # 清除之前的处理数据
        self.stored_waveform_data = None
        if hasattr(self, 'processed_audio_path'):
            self.processed_audio_path = None
        if hasattr(self, 'processed_waveform_data'):
            self.processed_waveform_data = None
        self.detection_results = None
        self.vector_plot.set_upload_vector(None)

    def toggle_audio_playback(self):
        """切换音频播放/暂停"""
        import subprocess
        import threading
        import time

        if not hasattr(self, 'audio_file_path') or not self.audio_file_path:
            QMessageBox.warning(None, "警告", "请先选择音频文件！")
            return

        if self.audio_playing:
            # 停止播放
            self.audio_playing = False
            self.play_button.setText("▶ 播放")
            if hasattr(self, 'pygame_initialized') and self.pygame_initialized:
                try:
                    import pygame
                    pygame.mixer.music.stop()
                except:
                    pass
        else:
            # 开始播放
            self.audio_playing = True
            self.play_button.setText("⏸ 暂停")

            def play_audio():
                try:
                    try:
                        import pygame

                        # 只在明确处理过音频后才使用处理后的路径
                        if hasattr(self, 'processed_audio_path') and self.processed_audio_path and hasattr(self,
                                                                                                           'audio_processed') and self.audio_processed:
                            play_file = self.processed_audio_path
                        else:
                            play_file = self.audio_file_path

                        if not hasattr(self, 'pygame_initialized') or not self.pygame_initialized:
                            pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=512)
                            self.pygame_initialized = True

                        pygame.mixer.music.load(play_file)
                        pygame.mixer.music.play()

                        while pygame.mixer.music.get_busy() and self.audio_playing:
                            time.sleep(0.1)

                        self.audio_playing = False
                        # 播放完成后在主线程中更新UI
                        self.play_button.setText("▶ 播放")
                        return
                    except Exception as e:
                        print(f"pygame播放失败: {e}")
                    if hasattr(self, 'processed_audio_path') and self.processed_audio_path and hasattr(self,
                                                                                                       'audio_processed') and self.audio_processed:
                        play_file = self.processed_audio_path
                    else:
                        play_file = self.audio_file_path
                    self.audio_process = subprocess.Popen(
                        ['aplay', '-q', play_file],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )

                    self.audio_playing = False
                    self.play_button.setText("▶ 播放")

                except Exception as e:
                    print(f"播放错误: {e}")
                    self.audio_playing = False
                    self.play_button.setText("▶ 播放")

            play_thread = threading.Thread(target=play_audio, daemon=True)
            play_thread.start()

    def toggle_tools_panel(self):
        """切换工具面板显示"""
        if self.tools_panel.isVisible():
            self.tools_panel.hide()
            self.hlayout_content.setStretchFactor(self.content, 1)
            self.hlayout_content.setStretchFactor(self.tools_panel, 0)
        else:
            self.tools_panel.show()
            self.hlayout_content.setStretchFactor(self.content, 1)
            self.hlayout_content.setStretchFactor(self.tools_panel, 0)

    def on_slide_step_changed(self, value):
        """工具面板滑动窗口步进值滑块变化"""
        global SLIDE_STEP
        step_seconds = value / 10.0  # 0-40对应0-4秒
        SLIDE_STEP = int(step_seconds * SR)  # 转换为采样点数
        self.tools_slide_step_value_label.setText(f"步进值: {step_seconds:.1f}秒")

    def tools_change_sample_rate(self):
        """工具面板 - 转换采样率"""
        import tempfile
        import soundfile as sf
        import os

        if not hasattr(self, 'original_file_path') or not self.original_file_path:
            QMessageBox.warning(None, "警告", "请先选择音频文件！")
            return

        try:
            target_sr = self.tools_sample_combo.currentData()
            if target_sr == 0:
                target_sr = SR
            audio_data, _ = librosa.load(self.original_file_path, sr=SR)
            resampled_audio = librosa.resample(audio_data, orig_sr=SR, target_sr=target_sr)
            save_format = getattr(self, 'original_audio_format', None) or 'wav'
            temp_file = tempfile.NamedTemporaryFile(suffix=f'.{save_format}', delete=False)
            self.processed_audio_path = temp_file.name
            temp_file.close()

            sf.write(self.processed_audio_path, resampled_audio, target_sr)

            # 加载处理后的音频数据
            processed_wave, _ = librosa.load(self.processed_audio_path, sr=SR)
            self.processed_waveform_data = processed_wave.copy()
            self.processed_sample_rate = target_sr

            # 更新波形显示
            self.update_waveform_display(processed_wave)

            # 记录已处理
            self.audio_processed = True
            print(f">>> 音频已处理，采样率转换: {target_sr} Hz, 保存路径: {self.processed_audio_path}")

            QMessageBox.information(None, "成功", f"已转换采样率至 {target_sr} Hz！")

        except Exception as e:
            QMessageBox.warning(None, "错误", f"转换采样率失败: {str(e)}")

    def tools_change_format(self):
        """工具面板 - 转换格式"""
        import tempfile
        import soundfile as sf
        import subprocess

        if not hasattr(self, 'original_file_path') or not self.original_file_path:
            QMessageBox.warning(None, "警告", "请先选择音频文件！")
            return

        try:
            # 获取目标格式
            target_format = self.tools_format_combo.currentData()
            if target_format == "original":
                QMessageBox.warning(None, "警告", "请选择目标格式！")
                return

            # 从原始文件读取音频数据，而不是使用已处理的数据
            audio_data, _ = librosa.load(self.original_file_path, sr=SR)
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav_path = temp_wav.name
            temp_wav.close()
            sf.write(temp_wav_path, audio_data, SR)

            # 转换为目标格式
            temp_output = tempfile.NamedTemporaryFile(suffix=f'.{target_format}', delete=False)
            output_path = temp_output.name
            temp_output.close()

            # 使用ffmpeg转换格式
            if target_format == "mp3":
                cmd = ['ffmpeg', '-y', '-i', temp_wav_path, '-codec:a', 'libmp3lame', '-b:a', '192k', output_path]
            elif target_format == "flac":
                cmd = ['ffmpeg', '-y', '-i', temp_wav_path, '-codec:a', 'flac', output_path]
            else:  # wav
                cmd = ['ffmpeg', '-y', '-i', temp_wav_path, '-codec:a', 'pcm_s16le', output_path]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"ffmpeg转换失败: {result.stderr}")

            self.processed_audio_path = output_path

            # 加载处理后的音频数据
            processed_wave, _ = librosa.load(output_path, sr=SR)
            self.processed_waveform_data = processed_wave.copy()
            self.processed_format = target_format

            # 更新波形显示
            self.update_waveform_display(processed_wave)

            # 记录已处理
            self.audio_processed = True
            print(f">>> 音频已处理，格式转换: {target_format.upper()}, 保存路径: {self.processed_audio_path}")

            QMessageBox.information(None, "成功", f"已转换为 {target_format.upper()} 格式！")

        except Exception as e:
            QMessageBox.warning(None, "错误", f"转换格式失败: {str(e)}")

    def update_waveform_display(self, waveform_data):
        """更新波形图显示"""
        self.stored_waveform_data = waveform_data.copy()
        self.audio_waveform.set_waveform_data(waveform_data)
        self.audio_waveform.update()

    def update_forgery_segment_display(self, frame_logits):
        """更新伪造片段热力图显示"""
        pass

    def update_vector_visualization(self, vector_data):
        """更新UMAP聚类可视化"""
        if vector_data is not None and len(vector_data) == 128:
            self.vector_plot.set_upload_vector(vector_data)
        else:
            self.vector_plot.set_upload_vector(None)

    def select_batch_files(self):
        """选择多个音频文件并展示文件列表"""
        # 清空之前的列表
        self.batch_file_list.clear()
        self.file_list_table.setRowCount(0)

        # 打开多文件选择对话框
        file_paths, _ = QFileDialog.getOpenFileNames(
            None,
            "选择多个音频文件",
            "",
            "Audio Files (*.wav *.flac);;WAV Files (*.wav);;FLAC Files (*.flac)"
        )

        if not file_paths:
            return

        # 处理每个文件，获取详细信息
        for file_path in file_paths:
            try:
                # 获取文件名
                file_name = os.path.basename(file_path)
                # 获取文件大小(MB)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                # 获取音频时长
                wave, sr = librosa.load(file_path, sr=SR)
                duration = len(wave) / sr
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                # 添加到文件列表
                self.batch_file_list.append({
                    'path': file_path,
                    'name': file_name,
                    'size': round(file_size, 2),
                    'duration': duration_str,
                    'raw_duration': duration
                })

                # 添加到表格
                row = self.file_list_table.rowCount()
                self.file_list_table.insertRow(row)
                self.file_list_table.setItem(row, 0, QTableWidgetItem(file_name))
                self.file_list_table.setItem(row, 1, QTableWidgetItem(file_path))
                self.file_list_table.setItem(row, 2, QTableWidgetItem(f"{round(file_size, 2)}"))
                self.file_list_table.setItem(row, 3, QTableWidgetItem(duration_str))

            except Exception as e:
                QMessageBox.warning(None, "警告", f"文件{file_path}加载失败：{str(e)}")

        # 更新进度信息
        self.progress_info_label.setText(
            f"已用时：00:00:00 | 预估剩余：00:00:00 | 当前进度：0/{len(self.batch_file_list)}"
        )

        # 启用开始检测按钮
        self.start_batch_button.setEnabled(True)

    def start_batch_detection(self):
        """开始批量检测"""
        if not self.batch_file_list:
            QMessageBox.warning(None, "警告", "请先选择音频文件！")
            return

        if model is None:
            if not init_model():
                return

        # 初始化状态
        self.batch_is_cancelled = False
        self.batch_start_time = time.time()
        self.batch_results.clear()
        self.result_list_table.setRowCount(0)

        # 更新按钮状态
        self.start_batch_button.setEnabled(False)
        self.pause_resume_button.setEnabled(True)
        self.cancel_batch_button.setEnabled(True)
        self.select_batch_button.setEnabled(False)
        self.export_batch_button.setEnabled(False)
        self.pause_resume_button.setText("暂停检测")

        # 串行执行批量检测
        total_files = len(self.batch_file_list)
        self.batch_current_index = 0

        while self.batch_current_index < total_files and not self.batch_is_cancelled:
            # 检查是否暂停
            while self.batch_is_paused and not self.batch_is_cancelled:
                QApplication.processEvents()
                time.sleep(0.1)

            if self.batch_is_cancelled:
                break

            # 获取当前文件信息
            file_info = self.batch_file_list[self.batch_current_index]
            file_path = file_info['path']
            current_idx = self.batch_current_index

            try:
                # 更新进度信息
                self.update_batch_progress(current_idx, total_files, "检测中...")
                QApplication.processEvents()  # 刷新UI

                # 加载原始音频数据（支持滑动窗口和短音频）
                wave, _ = librosa.load(file_path, sr=SR)
                audio_duration = len(wave) / SR

                # 批量检测也支持短音频处理
                num_samples = len(wave)
                use_sliding_window = num_samples >= MAX_LEN

                if use_sliding_window:
                    # 滑动窗口检测：每段4.035秒，使用滑动步进值
                    num_segments = max(1,
                                       int(np.ceil((num_samples - MAX_LEN) / SLIDE_STEP)) + 1) if SLIDE_STEP > 0 else 1
                    segment_probs = []
                    fake_segments = 0
                    global_logits_list = []
                    segment_vectors = []
                    segment_timestamps = []

                    for seg_idx in range(num_segments):
                        # 使用滑动步进值
                        start_idx = seg_idx * SLIDE_STEP
                        if start_idx >= num_samples:
                            break
                        end_idx = min(start_idx + MAX_LEN, num_samples)
                        segment_wave = wave[start_idx:end_idx]

                        # 如果最后一段不足MAX_LEN，重复填充
                        if len(segment_wave) < MAX_LEN:
                            num_repeats = int(MAX_LEN / len(segment_wave)) + 1
                            segment_wave = np.tile(segment_wave, (1, num_repeats))[:, :MAX_LEN][0]

                        # 记录段起始时间
                        segment_timestamps.append(start_idx / SR)

                        # 预处理当前段
                        wave_tensor, spec_tensor, freq_aug, _ = preprocess_audio_segment(segment_wave)

                        # 模型推理
                        with torch.no_grad():
                            global_logits, _, projected_vector = model(wave_tensor, spec_tensor, freq_aug)
                        global_logits_np = global_logits.squeeze(0).cpu().numpy()
                        projected_vector_np = projected_vector.squeeze(0).cpu().numpy()
                        global_logits_list.append(global_logits_np)
                        segment_vectors.append(projected_vector_np)

                        # 计算伪造概率
                        real_logit = global_logits_np[1]
                        fake_prob = 100 / (1 + np.exp(1.5 * (real_logit - THRESHOLD_GLOBAL)))
                        fake_prob = np.clip(fake_prob, 0, 100)
                        segment_probs.append(fake_prob)

                        # 判断是否为伪造段（使用50%阈值）
                        is_fake = fake_prob > 50.0
                        if is_fake:
                            fake_segments += 1

                    # 计算整体结果
                    global_logits_avg = np.mean(global_logits_list, axis=0)
                    projected_vector_avg = np.mean(segment_vectors, axis=0) if len(segment_vectors) > 0 else None
                else:
                    # 短音频：直接整个送入模型
                    if num_samples < MAX_LEN:
                        num_repeats = int(MAX_LEN / num_samples) + 1
                        padded_wave = np.tile(wave, (1, num_repeats))[:, :MAX_LEN][0]
                    else:
                        padded_wave = wave[:MAX_LEN]

                    # 预处理
                    wave_tensor, spec_tensor, freq_aug, _ = preprocess_audio_segment(padded_wave)

                    # 模型推理
                    with torch.no_grad():
                        global_logits, _, projected_vector = model(wave_tensor, spec_tensor, freq_aug)

                    global_logits_avg = global_logits.squeeze(0).cpu().numpy()
                    projected_vector_avg = projected_vector.squeeze(0).cpu().numpy()

                    # 计算伪造概率
                    real_logit = global_logits_avg[1]
                    fake_prob = 100 / (1 + np.exp(1.5 * (real_logit - THRESHOLD_GLOBAL)))
                    fake_prob = np.clip(fake_prob, 0, 100)

                    # 使用50%阈值判断
                    fake_segments = 1 if fake_prob > 50.0 else 0

                    segment_probs = [fake_prob]
                    segment_timestamps = [0.0]
                    num_segments = 1

                # 计算整体伪造概率
                avg_fake_prob = np.mean(segment_probs)
                conclusion = "伪造" if avg_fake_prob > 50 else "真实"

                # 格式化时长
                hours = int(audio_duration // 3600)
                minutes = int((audio_duration % 3600) // 60)
                seconds = int(audio_duration % 60)
                duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                # 存储结果
                result = {
                    'name': file_info['name'],
                    'path': file_path,
                    'duration': duration_str,
                    'fake_prob': round(avg_fake_prob, 2),
                    'conclusion': conclusion,
                    'fake_segments': fake_segments,
                    'segment_probs': segment_probs,
                    'num_segments': num_segments,
                    'segment_timestamps': segment_timestamps,
                    'slide_step': SLIDE_STEP / SR,
                    'audio_duration': audio_duration,
                    'frame_logits': None,
                    'vector': projected_vector_avg,
                    'waveform': wave,
                    'status': "成功"
                }
                self.batch_results[file_path] = result

                # 更新结果列表和进度
                self.update_batch_result_row(current_idx, result)
                self.update_batch_progress(current_idx, total_files, "完成")
                self.update_batch_stats()

                db_result_data = {
                    "file_name": result['name'],
                    "file_path": result['path'],
                    "audio_duration": result['duration'],
                    "fake_prob": result['fake_prob'],
                    "conclusion": result['conclusion'],
                    "fake_segments": result['fake_segments'],
                    "global_threshold": THRESHOLD_GLOBAL,
                    "status": result['status']
                }
                insert_detection_result_to_db(db_result_data)

            except Exception as e:
                # 记录失败结果
                result = {
                    'name': file_info['name'],
                    'path': file_path,
                    'duration': file_info['duration'],
                    'fake_prob': 0,
                    'conclusion': "未知",
                    'fake_segments': 0,
                    'segment_probs': [],
                    'num_segments': 0,
                    'segment_timestamps': [],
                    'audio_duration': 0,
                    'frame_logits': None,
                    'vector': None,
                    'waveform': None,
                    'status': f"失败：{str(e)}"
                }
                self.batch_results[file_path] = result

                db_result_data = {
                    "file_name": result['name'],
                    "file_path": result['path'],
                    "audio_duration": result['duration'],
                    "fake_prob": result['fake_prob'],
                    "conclusion": result['conclusion'],
                    "fake_segments": result['fake_segments'],
                    "global_threshold": THRESHOLD_GLOBAL,
                    "status": result['status']
                }

                insert_detection_result_to_db(db_result_data)

                # 更新结果列表和进度
                self.update_batch_result_row(current_idx, result)
                self.update_batch_progress(current_idx, total_files, "失败")
                self.update_batch_stats()

            # 处理UI事件，避免卡死
            QApplication.processEvents()
            # 进入下一个文件
            self.batch_current_index += 1

        # 检测完成/取消后更新UI
        if self.batch_is_cancelled:
            self.update_batch_ui_after_cancel()
        else:
            self.update_batch_ui_after_complete()

    def update_batch_progress(self, current_idx, total_files, status):
        """更新批量检测进度"""
        # 计算整体进度百分比
        progress_percent = int(((current_idx + 1) / total_files) * 100)

        # 计算用时和预估剩余时间
        elapsed_time = time.time() - self.batch_start_time
        elapsed_str = self.format_seconds(elapsed_time)

        if current_idx + 1 < total_files:
            avg_time_per_file = elapsed_time / (current_idx + 1)
            remaining_time = avg_time_per_file * (total_files - current_idx - 1)
            remaining_str = self.format_seconds(remaining_time)
        else:
            remaining_str = "00:00:00"

        # 更新进度条和信息
        QMetaObject.invokeMethod(self.global_progress_bar, "setValue", Qt.QueuedConnection,
                                 Q_ARG(int, progress_percent))
        QMetaObject.invokeMethod(self.progress_info_label, "setText", Qt.QueuedConnection,
                                 Q_ARG(str,
                                       f"已用时：{elapsed_str} | 预估剩余：{remaining_str} | 当前进度：{current_idx + 1}/{total_files}"))

        # 更新统计信息
        self.update_batch_stats()

    def update_batch_result_row(self, row_idx, result):
        # 确保表格有足够的行数
        if self.result_list_table.rowCount() <= row_idx:
            self.result_list_table.setRowCount(row_idx + 1)

        # 设置各列数据
        self.set_table_item(row_idx, 0, result['name'])
        self.set_table_item(row_idx, 1, result['path'])
        self.set_table_item(row_idx, 2, result['duration'])
        self.set_table_item(row_idx, 3, f"{result['fake_prob']}%")
        self.set_table_item(row_idx, 4, result['conclusion'])
        self.set_table_item(row_idx, 5, f"{result['fake_segments']}")
        self.set_table_item(row_idx, 6, result['status'])

    def set_table_item(self, row, col, text):
        if row >= self.result_list_table.rowCount():
            self.result_list_table.insertRow(row)
        item = QTableWidgetItem(text)
        # 设置文本左对齐
        item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.result_list_table.setItem(row, col, item)
        self.result_list_table.viewport().update()

    def update_batch_stats(self):
        """更新统计面板"""
        total = len(self.batch_file_list)
        success = len([r for r in self.batch_results.values() if r['status'] == "成功"])
        failed = total - success

        # 计算伪造/真实占比
        if success > 0:
            fake_count = len(
                [r for r in self.batch_results.values() if r['status'] == "成功" and r['conclusion'] == "伪造"])
            real_count = success - fake_count
            fake_ratio = (fake_count / success) * 100 if success > 0 else 0
            real_ratio = (real_count / success) * 100 if success > 0 else 0

            # 计算伪造概率均值
            prob_list = [r['fake_prob'] for r in self.batch_results.values() if r['status'] == "成功"]
            prob_mean = np.mean(prob_list) if prob_list else 0
        else:
            fake_ratio = 0
            real_ratio = 0
            prob_mean = 0

        # 更新UI
        QMetaObject.invokeMethod(self.count_value, "setText", Qt.QueuedConnection,
                                 Q_ARG(str, f"{total}/{success}/{failed}"))
        QMetaObject.invokeMethod(self.fake_ratio_value, "setText", Qt.QueuedConnection,
                                 Q_ARG(str, f"{round(fake_ratio, 2)}%"))
        QMetaObject.invokeMethod(self.real_ratio_value, "setText", Qt.QueuedConnection,
                                 Q_ARG(str, f"{round(real_ratio, 2)}%"))
        QMetaObject.invokeMethod(self.prob_mean_value, "setText", Qt.QueuedConnection,
                                 Q_ARG(str, f"{round(prob_mean, 2)}%"))

    def pause_resume_batch(self):
        """暂停/继续批量检测"""
        self.batch_is_paused = not self.batch_is_paused
        if self.batch_is_paused:
            self.pause_resume_button.setText("继续检测")
        else:
            self.pause_resume_button.setText("暂停检测")
            # 继续检测时触发UI事件处理
            QApplication.processEvents()

    def cancel_batch_detection(self):
        """取消批量检测"""
        self.batch_is_cancelled = True
        self.batch_is_paused = False  # 取消暂停，让循环退出
        # 立即更新UI状态
        self.update_batch_ui_after_cancel()

    def update_batch_ui_after_cancel(self):
        """取消后更新UI"""
        QMetaObject.invokeMethod(self.start_batch_button, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, True))
        QMetaObject.invokeMethod(self.pause_resume_button, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.cancel_batch_button, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, False))
        QMetaObject.invokeMethod(self.select_batch_button, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, True))
        QMetaObject.invokeMethod(self.progress_info_label, "setText", Qt.QueuedConnection,
                                 Q_ARG(str,
                                       f"已取消 | 当前进度：{self.batch_current_index}/{len(self.batch_file_list)}"))

    def update_batch_ui_after_complete(self):
        """完成后更新UI"""
        # 直接更新按钮状态
        self.start_batch_button.setEnabled(True)
        self.pause_resume_button.setEnabled(False)
        self.cancel_batch_button.setEnabled(False)
        self.select_batch_button.setEnabled(True)
        self.export_batch_button.setEnabled(True)

        # 计算总用时
        total_time = time.time() - self.batch_start_time
        total_time_str = self.format_seconds(total_time)
        self.progress_info_label.setText(
            f"检测完成 | 总用时：{total_time_str} | 进度：{len(self.batch_file_list)}/{len(self.batch_file_list)}"
        )

        self.show_batch_complete_message()

    def show_batch_complete_message(self):
        """显示完成提示"""
        QMessageBox.information(None, "完成", "批量检测已完成！")

    def preview_result_file(self, index):
        row = index.row()
        if row < 0:
            self.preview_waveform.set_waveform_data(None)
            self.preview_heatmap.set_segment_data(None, None, 0, 1.0, True)
            self.preview_vector.set_upload_vector(None)
            return

        # 从结果列表中获取文件路径
        path_item = self.result_list_table.item(row, 1)
        if not path_item:
            self.preview_waveform.set_waveform_data(None)
            self.preview_heatmap.set_segment_data(None, None, 0, 1.0, True)
            self.preview_vector.set_upload_vector(None)
            return

        file_path = path_item.text()
        if file_path not in self.batch_results:
            self.preview_waveform.set_waveform_data(None)
            self.preview_heatmap.set_segment_data(None, None, 0, 1.0, True)
            self.preview_vector.set_upload_vector(None)
            return

        result = self.batch_results[file_path]

        # 主线程更新预览
        def update_preview():
            if result['waveform'] is not None:
                self.preview_waveform.set_waveform_data(result['waveform'])
            else:
                self.preview_waveform.set_waveform_data(None)

            # 使用段级概率显示
            segment_probs = result.get('segment_probs', [])
            segment_timestamps = result.get('segment_timestamps', [])
            audio_duration = result.get('audio_duration', 0)
            slide_step = result.get('slide_step', 1.0)
            use_sliding_window = len(segment_probs) > 1  # 多段使用滑动窗口，单段使用圆环图

            if segment_probs:
                self.preview_heatmap.set_segment_data(
                    segment_probs,
                    segment_timestamps,
                    audio_duration,
                    slide_step,
                    use_sliding_window
                )
            else:
                self.preview_heatmap.set_segment_data(None, None, 0, 1.0, True)

            if result['vector'] is not None:
                self.preview_vector.set_upload_vector(result['vector'])
            else:
                self.preview_vector.set_upload_vector(None)

        QTimer.singleShot(0, update_preview)

    def export_batch_results(self):
        """导出批量检测结果"""
        if not self.batch_results:
            QMessageBox.warning(None, "警告", "暂无检测结果可导出！")
            return

        #格式选择对话框
        format_options = ["Excel (.xlsx)", "CSV (.csv)", "TXT (.txt)", "JSON (.json)"]
        choice, ok = QInputDialog.getItem(
            None,
            "选择导出格式",
            "请选择要导出的文件格式：",
            format_options,
            0,  # 默认选中第一个
            False
        )

        if not ok:
            return

        #根据选择设置文件后缀和过滤器
        format_map = {
            "Excel (.xlsx)": (".xlsx", "Excel Files (*.xlsx)"),
            "CSV (.csv)": (".csv", "CSV Files (*.csv)"),
            "TXT (.txt)": (".txt", "Text Files (*.txt)"),
            "JSON (.json)": (".json", "JSON Files (*.json)")
        }
        suffix, filter_text = format_map[choice]

        #打开保存对话框
        save_path, _ = QFileDialog.getSaveFileName(
            None,
            "导出检测结果",
            f"批量检测结果{suffix}",
            f"{filter_text};All Files (*.*)"
        )

        if not save_path:
            return

        # 确保文件后缀正确
        if not save_path.endswith(suffix):
            save_path += suffix

        #执行导出
        try:
            if choice == "Excel (.xlsx)":
                self.export_to_excel(save_path)
            elif choice == "CSV (.csv)":
                self.export_to_csv(save_path)
            elif choice == "TXT (.txt)":
                self.export_to_txt(save_path)
            elif choice == "JSON (.json)":
                self.export_to_json(save_path)

            QMessageBox.information(None, "成功", f"结果已成功导出为{choice}格式！")
        except Exception as e:
            QMessageBox.critical(None, "错误", f"导出失败：{str(e)}")

    def export_to_excel(self, save_path):
        """导出到Excel"""
        import pandas as pd
        df = pd.DataFrame(list(self.batch_results.values()))
        df = df[['name', 'path', 'duration', 'fake_prob', 'conclusion', 'fake_segments', 'status']]
        df.to_excel(save_path, index=False)

    def export_to_csv(self, save_path):
        """导出到CSV"""
        import pandas as pd
        df = pd.DataFrame(list(self.batch_results.values()))
        df = df[['name', 'path', 'duration', 'fake_prob', 'conclusion', 'fake_segments', 'status']]
        df.to_csv(save_path, index=False, encoding='utf-8-sig')

    def export_to_txt(self, save_path):
        """导出到TXT"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("音频伪造检测批量结果\n")
            f.write("=" * 80 + "\n")
            f.write(f"检测时间：{QDateTime.currentDateTime().toString('yyyy-MM-dd HH:mm:ss')}\n")
            f.write(f"总文件数：{len(self.batch_file_list)}\n")
            f.write(f"成功数：{len([r for r in self.batch_results.values() if r['status'] == '成功'])}\n")
            f.write(f"失败数：{len([r for r in self.batch_results.values() if r['status'] != '成功'])}\n")
            f.write("=" * 80 + "\n\n")

            for result in self.batch_results.values():
                f.write(f"文件名：{result['name']}\n")
                f.write(f"文件路径：{result['path']}\n")
                f.write(f"音频时长：{result['duration']}\n")
                f.write(f"伪造概率：{result['fake_prob']}%\n")
                f.write(f"判断结论：{result['conclusion']}\n")
                f.write(f"伪造段数：{result['fake_segments']}\n")
                f.write(f"检测状态：{result['status']}\n")
                f.write("-" * 50 + "\n")

    def export_to_json(self, save_path):
        """导出到JSON"""
        import json
        export_data = []
        for result in self.batch_results.values():
            export_item = {
                '文件名': result['name'],
                '文件路径': result['path'],
                '音频时长': result['duration'],
                '伪造概率(%)': float(result['fake_prob']),
                '判断结论': result['conclusion'],
                '伪造段数': int(result['fake_segments']),
                '检测状态': result['status']
            }
            export_data.append(export_item)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=4)

    def format_seconds(self, seconds):
        """将秒数格式化为HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def load_db_records(self):
        """加载OpenGauss数据库所有记录并更新表格"""

        try:
            self.db_table.setRowCount(0)
            conn = psycopg2.connect(
                host="127.0.0.1",
                port="7654",
                database="postgres",
                user="opengauss",
                password="123@Abc12"
            )
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audio_q ORDER BY id DESC")
            records = cursor.fetchall()
            # 填充表格
            for row_idx, record in enumerate(records):
                self.db_table.insertRow(row_idx)
                # 逐个字段填充
                self.db_table.setItem(row_idx, 0, QTableWidgetItem(str(record[0])))  # ID
                self.db_table.setItem(row_idx, 1, QTableWidgetItem(str(record[1])))  # 文件名
                self.db_table.setItem(row_idx, 2, QTableWidgetItem(str(record[2])))  # 文件路径
                self.db_table.setItem(row_idx, 3, QTableWidgetItem(str(record[3])))  # 音频时长
                self.db_table.setItem(row_idx, 4, QTableWidgetItem(f"{record[4]:.2f}"))  # 伪造概率
                self.db_table.setItem(row_idx, 5, QTableWidgetItem(str(record[5])))  # 判断结论
                self.db_table.setItem(row_idx, 6, QTableWidgetItem(str(record[6])))  # 伪造段数
                self.db_table.setItem(row_idx, 7, QTableWidgetItem(f"{record[7]:.4f}"))  # 全局阈值
                self.db_table.setItem(row_idx, 8, QTableWidgetItem(str(record[8])))  # 状态

            # 更新统计信息
            self.update_db_stats(records)

            cursor.close()
            conn.close()

        except Error as e:
            QMessageBox.critical(None, "数据库错误", f"查询记录失败：{str(e)}")

    def update_db_stats(self, records):
        """更新数据库统计信息"""
        if not records:
            self.db_total_value.setText("0")
            self.db_fake_value.setText("0")
            self.db_real_value.setText("0")
            self.db_avg_value.setText("0.00%")
            return

        # 总记录数
        total = len(records)
        # 伪造/真实记录数
        fake_count = len([r for r in records if r[5] == "伪造"])
        real_count = total - fake_count
        # 平均伪造概率
        prob_list = [r[4] for r in records if r[4] is not None]
        avg_prob = np.mean(prob_list) if prob_list else 0

        # 更新UI
        self.db_total_value.setText(str(total))
        self.db_fake_value.setText(str(fake_count))
        self.db_real_value.setText(str(real_count))
        self.db_avg_value.setText(f"{avg_prob:.2f}%")

    def delete_selected_db_record(self):
        """删除选中的数据库记录"""
        selected_rows = self.db_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(None, "警告", "请先选中要删除的记录！")
            return

        # 确认删除
        reply = QMessageBox.question(None, "确认删除", "确定要删除选中的记录吗？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        try:
            conn = psycopg2.connect(
                host="127.0.0.1",
                port="7654",
                database="postgres",
                user="opengauss",
                password="123@Abc12"
            )
            cursor = conn.cursor()

            for row in selected_rows:
                record_id = self.db_table.item(row.row(), 0).text()
                cursor.execute("DELETE FROM audio_q WHERE id = %s", (record_id,))

            conn.commit()
            cursor.close()
            conn.close()

            # 刷新列表
            self.load_db_records()
            QMessageBox.information(None, "成功", "选中记录已删除！")

        except Error as e:
            QMessageBox.critical(None, "数据库错误", f"删除记录失败：{str(e)}")

    def export_db_records(self):
        """导出数据库所有记录"""
        if self.db_table.rowCount() == 0:
            QMessageBox.warning(None, "警告", "暂无记录可导出！")
            return

        # 选择导出格式
        format_options = ["Excel (.xlsx)", "CSV (.csv)", "TXT (.txt)", "JSON (.json)"]
        choice, ok = QInputDialog.getItem(None, "选择导出格式", "请选择导出格式：", format_options, 0, False)
        if not ok:
            return

        # 选择保存路径
        format_map = {
            "Excel (.xlsx)": (".xlsx", "Excel Files (*.xlsx)"),
            "CSV (.csv)": (".csv", "CSV Files (*.csv)"),
            "TXT (.txt)": (".txt", "Text Files (*.txt)"),
            "JSON (.json)": (".json", "JSON Files (*.json)")
        }
        suffix, filter_text = format_map[choice]
        save_path, _ = QFileDialog.getSaveFileName(None, "导出数据库记录", f"数据库记录{suffix}", filter_text)
        if not save_path:
            return

        # 确保后缀正确
        if not save_path.endswith(suffix):
            save_path += suffix

        # 读取表格数据
        data = []
        headers = [self.db_table.horizontalHeaderItem(i).text() for i in range(self.db_table.columnCount())]

        for row in range(self.db_table.rowCount()):
            row_data = {}
            for col in range(self.db_table.columnCount()):
                item = self.db_table.item(row, col)
                row_data[headers[col]] = item.text() if item else ""
            data.append(row_data)

        # 导出数据
        try:
            if choice == "Excel (.xlsx)":
                import pandas as pd
                df = pd.DataFrame(data)
                df.to_excel(save_path, index=False)
            elif choice == "CSV (.csv)":
                import pandas as pd
                df = pd.DataFrame(data)
                df.to_csv(save_path, index=False, encoding='utf-8-sig')
            elif choice == "TXT (.txt)":
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write("数据库记录导出\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"导出时间：{QDateTime.currentDateTime().toString('yyyy-MM-dd HH:mm:ss')}\n")
                    f.write(f"总记录数：{len(data)}\n")
                    f.write("=" * 80 + "\n\n")
                    for row in data:
                        for key, value in row.items():
                            f.write(f"{key}：{value}\n")
                        f.write("-" * 50 + "\n")
            elif choice == "JSON (.json)":
                import json
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

            QMessageBox.information(None, "成功", f"数据库记录已导出为{choice}格式！")

        except Exception as e:
            QMessageBox.critical(None, "错误", f"导出失败：{str(e)}")

    def update_sidebar_triangle(self, active):
        """更新侧边栏三角形指示器"""
        transparent_style = "border: none; color: transparent; font-size: 12px;"
        white_style = "border: none; color: white; font-size: 12px;"

        self.single_triangle.setStyleSheet(transparent_style)
        self.batch_triangle.setStyleSheet(transparent_style)
        self.db_triangle.setStyleSheet(transparent_style)
        self.sys_triangle.setStyleSheet(transparent_style)

        if active == "single":
            self.single_triangle.setStyleSheet(white_style)
        elif active == "batch":
            self.batch_triangle.setStyleSheet(white_style)
        elif active == "db":
            self.db_triangle.setStyleSheet(white_style)
        elif active == "sys":
            self.sys_triangle.setStyleSheet(white_style)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QCoreApplication.translate("AudioForgeryDetection", u"Audio_Q音频伪造检测系统", None))
        self.VersionLabel.setText(QCoreApplication.translate("AudioForgeryDetection", u"v1.0", None))
        self.explain_title.setText(
            QCoreApplication.translate("AudioForgeryDetection", u"Audio_Q音频伪造检测系统", None))


# 主程序入口
if __name__ == "__main__":
    import sys
    from home_page import HomePage

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    home_page = HomePage()
    home_page.show()

    sys.exit(app.exec())