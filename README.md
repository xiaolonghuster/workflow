# 语音数据处理工作流

基于 **Airflow 3.0.1** 和 **Ray** 实现的分布式语音数据处理工作流。

## 功能特性

- **Opus 音频解码**: 支持 opus 压缩格式的音频文件
- **VAD 智能切分**: 使用 Silero VAD 进行语音活动检测，将长音频切分为不超过 30 秒的片段
- **SenseVoice 语音识别**: 集成阿里 FunASR 的 SenseVoice 模型进行高精度语音识别
- **Ray 分布式处理**: 利用 Ray 集群进行大规模并行处理，支持 GPU 加速
- **Airflow 工作流编排**: 使用 Airflow 3.0.1 进行任务调度和监控

## 项目结构

```
.
├── dags/
│   ├── __init__.py
│   └── audio_processing_dag.py    # Airflow DAG 定义
├── src/
│   ├── __init__.py
│   ├── audio_processor.py         # 音频处理（解码、格式转换）
│   ├── vad_processor.py           # VAD 语音切分
│   ├── sensevoice_processor.py    # SenseVoice 语音识别
│   └── ray_tasks.py               # Ray 分布式任务
├── config/
│   ├── __init__.py
│   └── config.py                  # 配置文件
├── requirements.txt
└── README.md
```

## 工作流程

```
┌─────────────────┐
│  1. 扫描音频文件  │
│  (输入目录)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. 音频解码     │
│  (Opus → WAV)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. VAD 切分    │
│  (≤30s 片段)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. 语音识别    │
│  (SenseVoice)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. 写入 JSONL  │
│  (输出结果)      │
└─────────────────┘
```

## 环境要求

- Python >= 3.9
- Airflow >= 3.0.1
- Ray >= 2.9.0
- CUDA >= 11.8 (GPU 加速)
- FFmpeg (音频解码)

## 安装

### 1. 安装系统依赖

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 Airflow

```bash
# 初始化 Airflow 数据库
airflow db migrate

# 创建管理员用户
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# 将 DAG 目录链接到 Airflow
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
```

### 4. 配置 Airflow Variables

在 Airflow UI 中设置以下 Variables，或使用命令行：

```bash
# 路径配置
airflow variables set audio_input_dir "/data/audio/input"
airflow variables set audio_output_dir "/data/audio/output"
airflow variables set audio_temp_dir "/data/audio/temp"
airflow variables set audio_output_filename "transcriptions.jsonl"

# Ray 配置
airflow variables set ray_address "ray://your-ray-cluster:10001"
airflow variables set ray_num_gpus "1.0"
airflow variables set ray_num_cpus "4"

# SenseVoice 配置
airflow variables set sensevoice_model "iic/SenseVoiceSmall"
airflow variables set sensevoice_batch_size "16"
airflow variables set sensevoice_language "auto"
```

## 使用方法

### 启动 Airflow

```bash
# 启动 Web 服务器
airflow webserver --port 8080

# 在另一个终端启动调度器
airflow scheduler
```

### 触发 DAG

**方式 1: 通过 Web UI**

1. 访问 http://localhost:8080
2. 找到 `audio_processing_pipeline` DAG
3. 点击触发按钮，可选择性地传入参数：
   - `input_dir`: 输入音频目录
   - `output_dir`: 输出目录

**方式 2: 通过命令行**

```bash
# 使用默认配置触发
airflow dags trigger audio_processing_pipeline

# 传入自定义参数
airflow dags trigger audio_processing_pipeline \
    --conf '{"input_dir": "/path/to/audio", "output_dir": "/path/to/output"}'
```

**方式 3: 通过 Python API**

```python
from src.ray_tasks import AudioProcessingPipeline

pipeline = AudioProcessingPipeline(
    ray_address="ray://your-ray-cluster:10001",
    num_gpus=1.0,
    model_name="iic/SenseVoiceSmall",
    max_duration_seconds=30
)

output_file = pipeline.run(
    input_dir="/data/audio/input",
    output_dir="/data/audio/output",
    output_jsonl="/data/audio/output/transcriptions.jsonl"
)
```

## 输出格式

输出为 JSONL 格式，每行一个 JSON 对象：

```json
{"original_path": "/data/audio/input/sample.opus", "segment_path": "/data/audio/temp/sample/sample_seg0000.wav", "segment_index": 0, "start_ms": 0, "end_ms": 15000, "text": "识别出的文本内容", "language": "zh"}
{"original_path": "/data/audio/input/sample.opus", "segment_path": "/data/audio/temp/sample/sample_seg0001.wav", "segment_index": 1, "start_ms": 15000, "end_ms": 28500, "text": "第二段识别文本", "language": "zh"}
```

字段说明：
- `original_path`: 原始音频文件路径
- `segment_path`: 切分后的音频片段路径
- `segment_index`: 片段索引（从 0 开始）
- `start_ms`: 片段在原始音频中的开始时间（毫秒）
- `end_ms`: 片段在原始音频中的结束时间（毫秒）
- `text`: 语音识别结果
- `language`: 检测到的语言

## 配置说明

### 音频处理配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_duration_seconds` | 30 | 最大音频片段时长（秒） |
| `sample_rate` | 16000 | 音频采样率 |

### VAD 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `threshold` | 0.5 | VAD 阈值 |
| `min_speech_duration_ms` | 250 | 最小语音段时长（毫秒） |
| `min_silence_duration_ms` | 100 | 最小静音段时长（毫秒） |

### Ray 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ray_address` | auto | Ray 集群地址 |
| `num_gpus` | 1.0 | GPU 资源数量 |
| `num_cpus` | 4 | CPU 资源数量 |

### SenseVoice 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | iic/SenseVoiceSmall | 模型名称 |
| `batch_size` | 16 | 批处理大小 |
| `language` | auto | 语言设置 (auto/zh/en/ja/ko/yue) |

## 性能优化建议

1. **GPU 利用**: 确保 Ray 集群有足够的 GPU 资源，SenseVoice 在 GPU 上运行效率更高
2. **批处理大小**: 根据 GPU 显存调整 `batch_size`，L40S (48GB) 建议使用 16-32
3. **并行度**: 音频预处理任务会自动在多个 CPU 核心上并行执行
4. **存储**: 建议使用 SSD 存储临时文件，提升 I/O 性能

## 故障排除

### 常见问题

1. **FFmpeg 未找到**
   ```
   RuntimeError: ffmpeg 未安装或不可用
   ```
   解决：安装 ffmpeg 并确保在 PATH 中

2. **Ray 连接失败**
   ```
   ConnectionError: Failed to connect to Ray cluster
   ```
   解决：检查 Ray 集群地址和网络连接

3. **GPU 内存不足**
   ```
   CUDA out of memory
   ```
   解决：减小 `batch_size` 或使用更大显存的 GPU

## License

MIT License
