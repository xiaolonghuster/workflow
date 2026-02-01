"""
VAD (Voice Activity Detection) 处理模块
使用 Silero VAD 进行语音活动检测和音频切分
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VADProcessor:
    """
    VAD 处理器
    - 检测语音活动区间
    - 将长音频切分成不超过指定时长的片段
    """
    
    def __init__(
        self,
        max_duration_seconds: int = 30,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        sample_rate: int = 16000
    ):
        """
        初始化 VAD 处理器
        
        Args:
            max_duration_seconds: 最大音频时长（秒）
            threshold: VAD 阈值
            min_speech_duration_ms: 最小语音段时长（毫秒）
            min_silence_duration_ms: 最小静音段时长（毫秒）
            speech_pad_ms: 语音填充时长（毫秒）
            sample_rate: 采样率
        """
        self.max_duration_seconds = max_duration_seconds
        self.max_duration_ms = max_duration_seconds * 1000
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate
        
        self.model = None
        self.utils = None
    
    def _load_model(self):
        """延迟加载 Silero VAD 模型"""
        if self.model is None:
            logger.info("加载 Silero VAD 模型...")
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            self.model = model
            self.utils = utils
            logger.info("Silero VAD 模型加载完成")
    
    def detect_speech(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        检测音频中的语音活动区间
        
        Args:
            audio: 音频数据 (numpy array)
            sample_rate: 采样率
            
        Returns:
            语音区间列表 [(start_ms, end_ms), ...]
        """
        self._load_model()
        
        sr = sample_rate or self.sample_rate
        
        # 转换为 torch tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio
        
        # 确保是一维
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.mean(dim=0)
        
        # 获取语音时间戳
        get_speech_timestamps = self.utils[0]
        
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=sr,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False
        )
        
        # 转换为毫秒时间戳
        timestamps = []
        for ts in speech_timestamps:
            start_ms = int(ts['start'] * 1000 / sr)
            end_ms = int(ts['end'] * 1000 / sr)
            timestamps.append((start_ms, end_ms))
        
        return timestamps
    
    def split_by_duration(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        将音频按最大时长切分，优先在静音处切分
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            
        Returns:
            切分后的时间戳列表 [(start_ms, end_ms), ...]
        """
        sr = sample_rate or self.sample_rate
        total_duration_ms = int(len(audio) * 1000 / sr)
        
        # 如果音频时长不超过最大时长，无需切分
        if total_duration_ms <= self.max_duration_ms:
            return [(0, total_duration_ms)]
        
        # 检测语音区间
        speech_timestamps = self.detect_speech(audio, sr)
        
        if not speech_timestamps:
            # 没有检测到语音，按固定时长切分
            return self._split_fixed_duration(total_duration_ms)
        
        # 合并相邻的语音区间，确保每个片段不超过最大时长
        segments = self._merge_segments(speech_timestamps, total_duration_ms)
        
        return segments
    
    def _split_fixed_duration(self, total_duration_ms: int) -> List[Tuple[int, int]]:
        """按固定时长切分"""
        segments = []
        start = 0
        
        while start < total_duration_ms:
            end = min(start + self.max_duration_ms, total_duration_ms)
            segments.append((start, end))
            start = end
        
        return segments
    
    def _merge_segments(
        self,
        speech_timestamps: List[Tuple[int, int]],
        total_duration_ms: int
    ) -> List[Tuple[int, int]]:
        """
        合并语音区间，确保每个片段不超过最大时长
        """
        if not speech_timestamps:
            return [(0, total_duration_ms)]
        
        segments = []
        current_start = 0
        current_end = 0
        
        for start_ms, end_ms in speech_timestamps:
            # 计算如果加入这个语音段后的总时长
            potential_end = end_ms
            potential_duration = potential_end - current_start
            
            if potential_duration <= self.max_duration_ms:
                # 可以合并到当前片段
                current_end = end_ms
            else:
                # 超过最大时长，需要保存当前片段并开始新片段
                if current_end > current_start:
                    segments.append((current_start, current_end))
                
                # 处理当前语音段
                segment_duration = end_ms - start_ms
                if segment_duration <= self.max_duration_ms:
                    current_start = start_ms
                    current_end = end_ms
                else:
                    # 单个语音段超过最大时长，需要强制切分
                    force_segments = self._force_split_segment(start_ms, end_ms)
                    segments.extend(force_segments[:-1])
                    current_start, current_end = force_segments[-1]
        
        # 添加最后一个片段
        if current_end > current_start:
            segments.append((current_start, current_end))
        
        # 确保覆盖整个音频（包括首尾的静音部分）
        if segments:
            # 扩展第一个片段到音频开始
            first_start, first_end = segments[0]
            if first_start > 0:
                new_duration = first_end - 0
                if new_duration <= self.max_duration_ms:
                    segments[0] = (0, first_end)
            
            # 扩展最后一个片段到音频结束
            last_start, last_end = segments[-1]
            if last_end < total_duration_ms:
                new_duration = total_duration_ms - last_start
                if new_duration <= self.max_duration_ms:
                    segments[-1] = (last_start, total_duration_ms)
                else:
                    # 需要添加新片段
                    segments.append((last_end, total_duration_ms))
        
        return segments
    
    def _force_split_segment(
        self,
        start_ms: int,
        end_ms: int
    ) -> List[Tuple[int, int]]:
        """强制切分超长片段"""
        segments = []
        current = start_ms
        
        while current < end_ms:
            segment_end = min(current + self.max_duration_ms, end_ms)
            segments.append((current, segment_end))
            current = segment_end
        
        return segments
    
    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        处理音频：检测语音并切分
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            
        Returns:
            (切分后的音频片段列表, 时间戳列表)
        """
        sr = sample_rate or self.sample_rate
        
        # 获取切分时间戳
        timestamps = self.split_by_duration(audio, sr)
        
        # 切分音频
        segments = []
        for start_ms, end_ms in timestamps:
            start_sample = int(start_ms * sr / 1000)
            end_sample = int(end_ms * sr / 1000)
            segment = audio[start_sample:end_sample]
            segments.append(segment)
        
        return segments, timestamps
