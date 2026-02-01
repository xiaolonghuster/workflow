"""
SenseVoice 语音识别模块
使用阿里 FunASR 的 SenseVoice 模型进行语音识别
"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SenseVoiceProcessor:
    """
    SenseVoice 语音识别处理器
    - 加载 SenseVoice 模型
    - 批量语音识别
    """
    
    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: str = "cuda",
        batch_size: int = 16,
        language: str = "auto"
    ):
        """
        初始化 SenseVoice 处理器
        
        Args:
            model_name: 模型名称或路径
            device: 设备类型 (cuda/cpu)
            batch_size: 批处理大小
            language: 语言设置 (auto/zh/en/ja/ko/yue)
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.language = language
        self.model = None
    
    def _load_model(self):
        """延迟加载 SenseVoice 模型"""
        if self.model is None:
            logger.info(f"加载 SenseVoice 模型: {self.model_name}")
            
            try:
                from funasr import AutoModel
                
                self.model = AutoModel(
                    model=self.model_name,
                    trust_remote_code=True,
                    device=self.device
                )
                logger.info("SenseVoice 模型加载完成")
                
            except ImportError:
                raise ImportError(
                    "请安装 funasr: pip install funasr"
                )
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                raise
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray, List],
        language: Optional[str] = None
    ) -> List[Dict]:
        """
        对音频进行语音识别
        
        Args:
            audio: 音频文件路径、numpy数组或音频列表
            language: 语言设置
            
        Returns:
            识别结果列表，每个元素包含 text, language 等字段
        """
        self._load_model()
        
        lang = language or self.language
        
        # 统一转换为列表
        if isinstance(audio, (str, np.ndarray)):
            audio_list = [audio]
        else:
            audio_list = audio
        
        results = []
        
        # 分批处理
        for i in range(0, len(audio_list), self.batch_size):
            batch = audio_list[i:i + self.batch_size]
            
            try:
                batch_results = self.model.generate(
                    input=batch,
                    cache={},
                    language=lang,
                    use_itn=True,
                    batch_size_s=300
                )
                
                for res in batch_results:
                    result = {
                        "text": res.get("text", ""),
                        "language": res.get("language", lang),
                    }
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"识别失败: {e}")
                # 对于失败的批次，添加空结果
                for _ in batch:
                    results.append({
                        "text": "",
                        "language": lang,
                        "error": str(e)
                    })
        
        return results
    
    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Dict:
        """
        对单个音频文件进行识别
        
        Args:
            audio_path: 音频文件路径
            language: 语言设置
            
        Returns:
            识别结果字典
        """
        results = self.transcribe(audio_path, language)
        return results[0] if results else {"text": "", "error": "No result"}
    
    def transcribe_batch(
        self,
        audio_paths: List[str],
        language: Optional[str] = None
    ) -> List[Dict]:
        """
        批量识别音频文件
        
        Args:
            audio_paths: 音频文件路径列表
            language: 语言设置
            
        Returns:
            识别结果列表
        """
        return self.transcribe(audio_paths, language)
    
    def transcribe_arrays(
        self,
        audio_arrays: List[np.ndarray],
        sample_rate: int = 16000,
        language: Optional[str] = None
    ) -> List[Dict]:
        """
        批量识别音频数组
        
        Args:
            audio_arrays: 音频数组列表
            sample_rate: 采样率
            language: 语言设置
            
        Returns:
            识别结果列表
        """
        self._load_model()
        
        lang = language or self.language
        results = []
        
        # 分批处理
        for i in range(0, len(audio_arrays), self.batch_size):
            batch = audio_arrays[i:i + self.batch_size]
            
            try:
                batch_results = self.model.generate(
                    input=batch,
                    cache={},
                    language=lang,
                    use_itn=True,
                    batch_size_s=300,
                    fs=sample_rate
                )
                
                for res in batch_results:
                    result = {
                        "text": res.get("text", ""),
                        "language": res.get("language", lang),
                    }
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"识别失败: {e}")
                for _ in batch:
                    results.append({
                        "text": "",
                        "language": lang,
                        "error": str(e)
                    })
        
        return results


class SenseVoiceActor:
    """
    SenseVoice Ray Actor
    用于在 Ray 集群中运行 SenseVoice 模型
    """
    
    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: str = "cuda",
        batch_size: int = 16,
        language: str = "auto"
    ):
        """初始化 Actor"""
        self.processor = SenseVoiceProcessor(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            language=language
        )
        # 预加载模型
        self.processor._load_model()
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray, List],
        language: Optional[str] = None
    ) -> List[Dict]:
        """执行语音识别"""
        return self.processor.transcribe(audio, language)
    
    def transcribe_batch(
        self,
        audio_paths: List[str],
        language: Optional[str] = None
    ) -> List[Dict]:
        """批量识别"""
        return self.processor.transcribe_batch(audio_paths, language)
