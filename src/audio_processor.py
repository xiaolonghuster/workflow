"""
音频处理模块 - 处理 opus 编码的音频文件
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    音频处理器
    - 解码 opus 音频
    - 获取音频时长
    - 音频格式转换
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        初始化音频处理器
        
        Args:
            sample_rate: 目标采样率
        """
        self.sample_rate = sample_rate
        self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> None:
        """检查 ffmpeg 是否可用"""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ffmpeg 未安装或不可用，请先安装 ffmpeg")
    
    def decode_opus(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        解码 opus 音频文件为 wav 格式
        
        Args:
            input_path: 输入 opus 文件路径
            output_path: 输出 wav 文件路径，如果为 None 则自动生成
            
        Returns:
            输出文件路径
        """
        input_path = Path(input_path)
        
        if output_path is None:
            output_path = input_path.with_suffix(".wav")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg",
            "-y",  # 覆盖输出文件
            "-i", str(input_path),
            "-ar", str(self.sample_rate),  # 采样率
            "-ac", "1",  # 单声道
            "-f", "wav",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                text=True
            )
            logger.debug(f"成功解码: {input_path} -> {output_path}")
            return str(output_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"解码失败: {input_path}, 错误: {e.stderr}")
            raise RuntimeError(f"opus 解码失败: {e.stderr}")
    
    def get_duration(self, audio_path: str) -> float:
        """
        获取音频文件时长（秒）
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频时长（秒）
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                text=True
            )
            duration = float(result.stdout.strip())
            return duration
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.error(f"获取时长失败: {audio_path}, 错误: {e}")
            raise RuntimeError(f"获取音频时长失败: {e}")
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        加载音频文件为 numpy 数组
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            (音频数据, 采样率)
        """
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path, dtype='float32')
            
            # 如果是多声道，转换为单声道
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # 如果采样率不匹配，进行重采样
            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate
            
            return audio, sr
        except ImportError:
            # 使用 ffmpeg 作为后备方案
            return self._load_with_ffmpeg(audio_path)
    
    def _load_with_ffmpeg(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """使用 ffmpeg 加载音频"""
        cmd = [
            "ffmpeg",
            "-i", str(audio_path),
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-"
        ]
        
        result = subprocess.run(cmd, capture_output=True, check=True)
        audio = np.frombuffer(result.stdout, dtype=np.float32)
        return audio, self.sample_rate
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sample_rate: Optional[int] = None
    ) -> str:
        """
        保存音频数据到文件
        
        Args:
            audio: 音频数据
            output_path: 输出路径
            sample_rate: 采样率
            
        Returns:
            输出文件路径
        """
        import soundfile as sf
        
        sr = sample_rate or self.sample_rate
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sf.write(str(output_path), audio, sr)
        return str(output_path)
    
    def split_audio(
        self,
        audio: np.ndarray,
        timestamps: list,
        sample_rate: Optional[int] = None
    ) -> list:
        """
        根据时间戳分割音频
        
        Args:
            audio: 音频数据
            timestamps: 时间戳列表 [(start_ms, end_ms), ...]
            sample_rate: 采样率
            
        Returns:
            分割后的音频片段列表
        """
        sr = sample_rate or self.sample_rate
        segments = []
        
        for start_ms, end_ms in timestamps:
            start_sample = int(start_ms * sr / 1000)
            end_sample = int(end_ms * sr / 1000)
            segment = audio[start_sample:end_sample]
            segments.append(segment)
        
        return segments
