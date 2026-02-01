"""
语音数据处理 Airflow DAG

基于 Airflow 3.0.1 和 Ray 实现的语音数据处理工作流：
1. 扫描输入目录中的 opus 音频文件
2. 使用 VAD 进行切分（确保每个片段不超过 30s）
3. 使用 SenseVoice 模型进行语音识别
4. 将结果写入 JSONL 文件
"""
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import json
import logging

from airflow.sdk import DAG, task
from airflow.models import Variable

logger = logging.getLogger(__name__)

# DAG 默认参数
default_args = {
    "owner": "audio_processing",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# 从 Airflow Variables 获取配置，提供默认值
def get_config() -> Dict[str, Any]:
    """获取工作流配置"""
    return {
        # 路径配置
        "input_dir": Variable.get("audio_input_dir", default_var="/data/audio/input"),
        "output_dir": Variable.get("audio_output_dir", default_var="/data/audio/output"),
        "temp_dir": Variable.get("audio_temp_dir", default_var="/data/audio/temp"),
        "output_filename": Variable.get("audio_output_filename", default_var="transcriptions.jsonl"),
        
        # 音频配置
        "max_duration_seconds": int(Variable.get("audio_max_duration", default_var="30")),
        "sample_rate": int(Variable.get("audio_sample_rate", default_var="16000")),
        
        # Ray 配置
        "ray_address": Variable.get("ray_address", default_var="auto"),
        "ray_num_gpus": float(Variable.get("ray_num_gpus", default_var="1.0")),
        "ray_num_cpus": int(Variable.get("ray_num_cpus", default_var="4")),
        
        # SenseVoice 配置
        "model_name": Variable.get("sensevoice_model", default_var="iic/SenseVoiceSmall"),
        "batch_size": int(Variable.get("sensevoice_batch_size", default_var="16")),
        "language": Variable.get("sensevoice_language", default_var="auto"),
    }


with DAG(
    dag_id="audio_processing_pipeline",
    default_args=default_args,
    description="语音数据处理流水线：opus解码 -> VAD切分 -> SenseVoice识别 -> JSONL输出",
    schedule=None,  # 手动触发
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["audio", "speech", "sensevoice", "ray"],
    params={
        "input_dir": "/data/audio/input",
        "output_dir": "/data/audio/output",
    },
) as dag:
    
    @task
    def validate_inputs(**context) -> Dict[str, Any]:
        """
        验证输入参数和环境
        """
        config = get_config()
        
        # 从 DAG 运行参数中获取覆盖值
        params = context.get("params", {})
        if "input_dir" in params:
            config["input_dir"] = params["input_dir"]
        if "output_dir" in params:
            config["output_dir"] = params["output_dir"]
        
        input_dir = Path(config["input_dir"])
        output_dir = Path(config["output_dir"])
        temp_dir = Path(config["temp_dir"])
        
        # 验证输入目录存在
        if not input_dir.exists():
            raise ValueError(f"输入目录不存在: {input_dir}")
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"输入目录: {input_dir}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"Ray 地址: {config['ray_address']}")
        
        return config
    
    @task
    def scan_audio_files(config: Dict[str, Any]) -> List[str]:
        """
        扫描输入目录中的音频文件
        """
        import ray
        
        try:
            # 连接 Ray 集群
            if not ray.is_initialized():
                ray.init(address=config["ray_address"])
            
            input_dir = Path(config["input_dir"])
            extensions = ['.opus', '.ogg', '.mp3', '.wav', '.flac', '.m4a']
            
            audio_files = []
            for ext in extensions:
                audio_files.extend(input_dir.glob(f"**/*{ext}"))
            
            audio_files = [str(f) for f in sorted(audio_files)]
            
            logger.info(f"发现 {len(audio_files)} 个音频文件")
            
            return audio_files
            
        finally:
            if ray.is_initialized():
                ray.shutdown()
    
    @task
    def process_audio_with_vad(
        audio_files: List[str],
        config: Dict[str, Any]
    ) -> List[Dict]:
        """
        处理音频文件：解码 opus + VAD 切分
        使用 Ray 进行分布式处理
        """
        import ray
        
        if not audio_files:
            logger.warning("没有音频文件需要处理")
            return []
        
        try:
            # 连接 Ray 集群
            if not ray.is_initialized():
                ray.init(address=config["ray_address"])
            
            # 导入处理模块
            from src.audio_processor import AudioProcessor
            from src.vad_processor import VADProcessor
            
            @ray.remote(num_cpus=1)
            def process_single_audio_remote(
                audio_path: str,
                output_dir: str,
                max_duration_seconds: int,
                sample_rate: int
            ) -> List[Dict]:
                """Ray remote 函数：处理单个音频"""
                audio_processor = AudioProcessor(sample_rate=sample_rate)
                vad_processor = VADProcessor(
                    max_duration_seconds=max_duration_seconds,
                    sample_rate=sample_rate
                )
                
                try:
                    audio_name = Path(audio_path).stem
                    audio_output_dir = Path(output_dir) / audio_name
                    audio_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 解码
                    wav_path = str(audio_output_dir / f"{audio_name}.wav")
                    wav_path = audio_processor.decode_opus(audio_path, wav_path)
                    
                    # 获取时长
                    duration = audio_processor.get_duration(wav_path)
                    
                    tasks = []
                    
                    if duration <= max_duration_seconds:
                        tasks.append({
                            "original_path": audio_path,
                            "processed_path": wav_path,
                            "segment_index": 0,
                            "start_ms": 0,
                            "end_ms": int(duration * 1000)
                        })
                    else:
                        # VAD 切分
                        audio, sr = audio_processor.load_audio(wav_path)
                        segments, timestamps = vad_processor.process_audio(audio, sr)
                        
                        for idx, (segment, (start_ms, end_ms)) in enumerate(zip(segments, timestamps)):
                            segment_path = str(audio_output_dir / f"{audio_name}_seg{idx:04d}.wav")
                            audio_processor.save_audio(segment, segment_path, sr)
                            
                            tasks.append({
                                "original_path": audio_path,
                                "processed_path": segment_path,
                                "segment_index": idx,
                                "start_ms": start_ms,
                                "end_ms": end_ms
                            })
                    
                    return tasks
                    
                except Exception as e:
                    logger.error(f"处理失败: {audio_path}, 错误: {e}")
                    return []
            
            # 并行处理
            futures = [
                process_single_audio_remote.remote(
                    audio_path,
                    config["temp_dir"],
                    config["max_duration_seconds"],
                    config["sample_rate"]
                )
                for audio_path in audio_files
            ]
            
            # 收集结果
            all_tasks = []
            results = ray.get(futures)
            for tasks in results:
                all_tasks.extend(tasks)
            
            logger.info(f"音频处理完成，共 {len(all_tasks)} 个片段")
            return all_tasks
            
        finally:
            if ray.is_initialized():
                ray.shutdown()
    
    @task
    def run_speech_recognition(
        audio_tasks: List[Dict],
        config: Dict[str, Any]
    ) -> List[Dict]:
        """
        使用 SenseVoice 进行语音识别
        在 Ray 集群的 GPU 上运行
        """
        import ray
        
        if not audio_tasks:
            logger.warning("没有音频任务需要识别")
            return []
        
        try:
            # 连接 Ray 集群
            if not ray.is_initialized():
                ray.init(address=config["ray_address"])
            
            # 创建 GPU Actor
            @ray.remote(num_gpus=config["ray_num_gpus"])
            class SenseVoiceActorRemote:
                def __init__(self, model_name: str, batch_size: int, language: str):
                    import torch
                    from funasr import AutoModel
                    
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.batch_size = batch_size
                    self.language = language
                    
                    self.model = AutoModel(
                        model=model_name,
                        trust_remote_code=True,
                        device=self.device
                    )
                    logger.info(f"SenseVoice 模型已加载到 {self.device}")
                
                def transcribe_batch(self, tasks: List[Dict]) -> List[Dict]:
                    results = []
                    audio_paths = [t["processed_path"] for t in tasks]
                    
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
                                results.append({
                                    "original_path": task["original_path"],
                                    "segment_path": task["processed_path"],
                                    "segment_index": task["segment_index"],
                                    "start_ms": task["start_ms"],
                                    "end_ms": task["end_ms"],
                                    "text": res.get("text", ""),
                                    "language": res.get("language", self.language)
                                })
                                
                        except Exception as e:
                            logger.error(f"识别失败: {e}")
                            for task in batch_tasks:
                                results.append({
                                    "original_path": task["original_path"],
                                    "segment_path": task["processed_path"],
                                    "segment_index": task["segment_index"],
                                    "start_ms": task["start_ms"],
                                    "end_ms": task["end_ms"],
                                    "text": "",
                                    "language": self.language,
                                    "error": str(e)
                                })
                    
                    return results
            
            # 创建 Actor 并执行识别
            actor = SenseVoiceActorRemote.remote(
                model_name=config["model_name"],
                batch_size=config["batch_size"],
                language=config["language"]
            )
            
            results = ray.get(actor.transcribe_batch.remote(audio_tasks))
            
            logger.info(f"语音识别完成，共 {len(results)} 条结果")
            return results
            
        finally:
            if ray.is_initialized():
                ray.shutdown()
    
    @task
    def write_results_to_jsonl(
        results: List[Dict],
        config: Dict[str, Any]
    ) -> str:
        """
        将识别结果写入 JSONL 文件
        """
        output_dir = Path(config["output_dir"])
        output_file = output_dir / config["output_filename"]
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"结果已写入: {output_file}, 共 {len(results)} 条记录")
        
        return str(output_file)
    
    @task
    def generate_summary(
        output_file: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成处理摘要
        """
        output_path = Path(output_file)
        
        if not output_path.exists():
            return {
                "status": "no_output",
                "total_records": 0
            }
        
        # 统计结果
        total_records = 0
        total_errors = 0
        languages = {}
        
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                total_records += 1
                
                if record.get("error"):
                    total_errors += 1
                
                lang = record.get("language", "unknown")
                languages[lang] = languages.get(lang, 0) + 1
        
        summary = {
            "status": "completed",
            "output_file": str(output_file),
            "total_records": total_records,
            "total_errors": total_errors,
            "success_rate": (total_records - total_errors) / total_records if total_records > 0 else 0,
            "language_distribution": languages
        }
        
        logger.info(f"处理摘要: {summary}")
        
        return summary
    
    # 定义 DAG 任务依赖关系
    config = validate_inputs()
    audio_files = scan_audio_files(config)
    audio_tasks = process_audio_with_vad(audio_files, config)
    recognition_results = run_speech_recognition(audio_tasks, config)
    output_file = write_results_to_jsonl(recognition_results, config)
    summary = generate_summary(output_file, config)
