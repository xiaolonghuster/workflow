"""
Ray 分布式任务模块
实现基于 Ray 的分布式音频处理流水线
"""
import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

import ray
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioTask:
    """音频处理任务"""
    original_path: str           # 原始文件路径
    processed_path: str          # 处理后的文件路径
    segment_index: int           # 片段索引
    start_ms: int               # 开始时间（毫秒）
    end_ms: int                 # 结束时间（毫秒）


@dataclass
class TranscriptionResult:
    """识别结果"""
    original_path: str          # 原始文件路径
    segment_path: str           # 片段文件路径
    segment_index: int          # 片段索引
    start_ms: int              # 开始时间
    end_ms: int                # 结束时间
    text: str                  # 识别文本
    language: str              # 语言
    error: Optional[str] = None


@ray.remote
def scan_audio_files(input_dir: str, extensions: List[str] = None) -> List[str]:
    """
    扫描输入目录中的音频文件
    
    Args:
        input_dir: 输入目录
        extensions: 文件扩展名列表
        
    Returns:
        音频文件路径列表
    """
    if extensions is None:
        extensions = ['.opus', '.ogg', '.mp3', '.wav', '.flac', '.m4a']
    
    input_path = Path(input_dir)
    audio_files = []
    
    for ext in extensions:
        audio_files.extend(input_path.glob(f"**/*{ext}"))
    
    return [str(f) for f in sorted(audio_files)]


@ray.remote(num_cpus=1)
def process_single_audio(
    audio_path: str,
    output_dir: str,
    max_duration_seconds: int = 30,
    sample_rate: int = 16000
) -> List[AudioTask]:
    """
    处理单个音频文件：解码 + VAD 切分
    
    Args:
        audio_path: 音频文件路径
        output_dir: 输出目录
        max_duration_seconds: 最大时长
        sample_rate: 采样率
        
    Returns:
        音频任务列表
    """
    from src.audio_processor import AudioProcessor
    from src.vad_processor import VADProcessor
    
    audio_processor = AudioProcessor(sample_rate=sample_rate)
    vad_processor = VADProcessor(
        max_duration_seconds=max_duration_seconds,
        sample_rate=sample_rate
    )
    
    try:
        # 创建输出子目录
        audio_name = Path(audio_path).stem
        audio_output_dir = Path(output_dir) / audio_name
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 解码 opus 到 wav
        wav_path = str(audio_output_dir / f"{audio_name}.wav")
        wav_path = audio_processor.decode_opus(audio_path, wav_path)
        
        # 获取音频时长
        duration = audio_processor.get_duration(wav_path)
        
        tasks = []
        
        if duration <= max_duration_seconds:
            # 无需切分
            tasks.append(AudioTask(
                original_path=audio_path,
                processed_path=wav_path,
                segment_index=0,
                start_ms=0,
                end_ms=int(duration * 1000)
            ))
        else:
            # 需要 VAD 切分
            audio, sr = audio_processor.load_audio(wav_path)
            segments, timestamps = vad_processor.process_audio(audio, sr)
            
            for idx, (segment, (start_ms, end_ms)) in enumerate(zip(segments, timestamps)):
                segment_path = str(audio_output_dir / f"{audio_name}_seg{idx:04d}.wav")
                audio_processor.save_audio(segment, segment_path, sr)
                
                tasks.append(AudioTask(
                    original_path=audio_path,
                    processed_path=segment_path,
                    segment_index=idx,
                    start_ms=start_ms,
                    end_ms=end_ms
                ))
        
        logger.info(f"处理完成: {audio_path}, 生成 {len(tasks)} 个片段")
        return tasks
        
    except Exception as e:
        logger.error(f"处理失败: {audio_path}, 错误: {e}")
        return []


@ray.remote(num_gpus=1)
class SenseVoiceActor:
    """
    SenseVoice Ray Actor
    在 GPU 上运行语音识别模型
    """
    
    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        batch_size: int = 16,
        language: str = "auto"
    ):
        """初始化 Actor，加载模型到 GPU"""
        import torch
        from funasr import AutoModel
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.language = language
        
        # 确定设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"SenseVoice Actor 初始化，设备: {self.device}")
        
        # 加载模型
        self.model = AutoModel(
            model=model_name,
            trust_remote_code=True,
            device=self.device
        )
        logger.info("SenseVoice 模型加载完成")
    
    def transcribe_batch(
        self,
        tasks: List[AudioTask]
    ) -> List[TranscriptionResult]:
        """
        批量识别音频
        
        Args:
            tasks: 音频任务列表
            
        Returns:
            识别结果列表
        """
        results = []
        audio_paths = [task.processed_path for task in tasks]
        
        # 分批处理
        for i in range(0, len(audio_paths), self.batch_size):
            batch_paths = audio_paths[i:i + self.batch_size]
            batch_tasks = tasks[i:i + self.batch_size]
            
            try:
                batch_results = self.model.generate(
                    input=batch_paths,
                    cache={},
                    language=self.language,
                    use_itn=True,
                    batch_size_s=300
                )
                
                for task, res in zip(batch_tasks, batch_results):
                    results.append(TranscriptionResult(
                        original_path=task.original_path,
                        segment_path=task.processed_path,
                        segment_index=task.segment_index,
                        start_ms=task.start_ms,
                        end_ms=task.end_ms,
                        text=res.get("text", ""),
                        language=res.get("language", self.language)
                    ))
                    
            except Exception as e:
                logger.error(f"批量识别失败: {e}")
                for task in batch_tasks:
                    results.append(TranscriptionResult(
                        original_path=task.original_path,
                        segment_path=task.processed_path,
                        segment_index=task.segment_index,
                        start_ms=task.start_ms,
                        end_ms=task.end_ms,
                        text="",
                        language=self.language,
                        error=str(e)
                    ))
        
        return results


class AudioProcessingPipeline:
    """
    音频处理流水线
    协调 Ray 集群上的分布式处理
    """
    
    def __init__(
        self,
        ray_address: str = "auto",
        num_gpus: float = 1.0,
        num_cpus: int = 4,
        model_name: str = "iic/SenseVoiceSmall",
        batch_size: int = 16,
        max_duration_seconds: int = 30,
        sample_rate: int = 16000,
        language: str = "auto"
    ):
        """
        初始化处理流水线
        
        Args:
            ray_address: Ray 集群地址
            num_gpus: GPU 数量
            num_cpus: CPU 数量
            model_name: SenseVoice 模型名称
            batch_size: 批处理大小
            max_duration_seconds: 最大音频时长
            sample_rate: 采样率
            language: 语言设置
        """
        self.ray_address = ray_address
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_duration_seconds = max_duration_seconds
        self.sample_rate = sample_rate
        self.language = language
        
        self._initialized = False
        self._actor = None
    
    def init_ray(self):
        """初始化 Ray 连接"""
        if not self._initialized:
            if not ray.is_initialized():
                ray.init(address=self.ray_address)
            self._initialized = True
            logger.info(f"Ray 已连接: {ray.cluster_resources()}")
    
    def shutdown(self):
        """关闭 Ray 连接"""
        if self._actor is not None:
            ray.kill(self._actor)
            self._actor = None
        self._initialized = False
    
    def scan_files(self, input_dir: str) -> List[str]:
        """扫描音频文件"""
        self.init_ray()
        return ray.get(scan_audio_files.remote(input_dir))
    
    def process_audio_files(
        self,
        audio_files: List[str],
        output_dir: str
    ) -> List[AudioTask]:
        """
        处理音频文件（解码 + VAD 切分）
        
        Args:
            audio_files: 音频文件列表
            output_dir: 输出目录
            
        Returns:
            音频任务列表
        """
        self.init_ray()
        
        # 并行处理所有文件
        futures = [
            process_single_audio.remote(
                audio_path,
                output_dir,
                self.max_duration_seconds,
                self.sample_rate
            )
            for audio_path in audio_files
        ]
        
        # 收集结果
        all_tasks = []
        results = ray.get(futures)
        for tasks in results:
            all_tasks.extend(tasks)
        
        logger.info(f"音频处理完成，共 {len(all_tasks)} 个任务")
        return all_tasks
    
    def transcribe_tasks(
        self,
        tasks: List[AudioTask]
    ) -> List[TranscriptionResult]:
        """
        对处理后的音频进行语音识别
        
        Args:
            tasks: 音频任务列表
            
        Returns:
            识别结果列表
        """
        self.init_ray()
        
        # 创建 SenseVoice Actor
        if self._actor is None:
            self._actor = SenseVoiceActor.options(
                num_gpus=self.num_gpus
            ).remote(
                model_name=self.model_name,
                batch_size=self.batch_size,
                language=self.language
            )
        
        # 执行识别
        results = ray.get(self._actor.transcribe_batch.remote(tasks))
        
        logger.info(f"语音识别完成，共 {len(results)} 条结果")
        return results
    
    def run(
        self,
        input_dir: str,
        output_dir: str,
        output_jsonl: str
    ) -> str:
        """
        运行完整的处理流水线
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            output_jsonl: 输出 JSONL 文件路径
            
        Returns:
            输出 JSONL 文件路径
        """
        try:
            self.init_ray()
            
            # 1. 扫描文件
            logger.info(f"扫描音频文件: {input_dir}")
            audio_files = self.scan_files(input_dir)
            logger.info(f"发现 {len(audio_files)} 个音频文件")
            
            if not audio_files:
                logger.warning("未找到音频文件")
                return output_jsonl
            
            # 2. 处理音频（解码 + VAD）
            logger.info("开始音频处理...")
            tasks = self.process_audio_files(audio_files, output_dir)
            
            if not tasks:
                logger.warning("没有生成任何处理任务")
                return output_jsonl
            
            # 3. 语音识别
            logger.info("开始语音识别...")
            results = self.transcribe_tasks(tasks)
            
            # 4. 写入 JSONL
            logger.info(f"写入结果到: {output_jsonl}")
            self._write_jsonl(results, output_jsonl)
            
            return output_jsonl
            
        finally:
            self.shutdown()
    
    def _write_jsonl(
        self,
        results: List[TranscriptionResult],
        output_path: str
    ):
        """将结果写入 JSONL 文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                data = {
                    "original_path": result.original_path,
                    "segment_path": result.segment_path,
                    "segment_index": result.segment_index,
                    "start_ms": result.start_ms,
                    "end_ms": result.end_ms,
                    "text": result.text,
                    "language": result.language
                }
                if result.error:
                    data["error"] = result.error
                
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        logger.info(f"已写入 {len(results)} 条记录到 {output_path}")
