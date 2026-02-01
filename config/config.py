"""
语音数据处理工作流配置
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    """音频处理配置"""
    # 最大音频时长（秒）
    max_duration_seconds: int = 30
    # 采样率
    sample_rate: int = 16000
    # 音频格式
    output_format: str = "wav"


@dataclass
class VADConfig:
    """VAD 配置"""
    # VAD 模型类型: silero, webrtc
    vad_type: str = "silero"
    # 静音阈值
    threshold: float = 0.5
    # 最小语音段时长（毫秒）
    min_speech_duration_ms: int = 250
    # 最小静音段时长（毫秒）
    min_silence_duration_ms: int = 100
    # 语音填充时长（毫秒）
    speech_pad_ms: int = 30


@dataclass
class SenseVoiceConfig:
    """SenseVoice 模型配置"""
    # 模型名称/路径
    model_name: str = "iic/SenseVoiceSmall"
    # 设备类型
    device: str = "cuda"
    # 批处理大小
    batch_size: int = 16
    # 语言
    language: str = "auto"


@dataclass
class RayConfig:
    """Ray 集群配置"""
    # Ray 集群地址
    ray_address: str = "auto"
    # GPU 数量
    num_gpus: float = 1.0
    # CPU 数量
    num_cpus: int = 4
    # 每个 worker 处理的文件数
    batch_size: int = 50


@dataclass
class WorkflowConfig:
    """工作流总配置"""
    # 输入音频目录
    input_dir: str = "/data/audio/input"
    # 输出目录
    output_dir: str = "/data/audio/output"
    # 临时文件目录
    temp_dir: str = "/data/audio/temp"
    # 输出 JSONL 文件名
    output_filename: str = "transcriptions.jsonl"
    
    # 子配置
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    sensevoice: SenseVoiceConfig = field(default_factory=SenseVoiceConfig)
    ray: RayConfig = field(default_factory=RayConfig)


# 默认配置实例
default_config = WorkflowConfig()
