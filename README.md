# Robola - MuJoCo MJCF Editor

本地 Python 库，用于与 Robola Web Editor 配合编辑 MuJoCo MJCF 模型文件。

## 安装

```bash
pip install robola
```

或从源码安装：

```bash
cd RobolaPyLib
pip install -e .
```

## 使用方法

### 命令行

```bash
# 启动编辑服务，指向你的 MJCF 文件
robola serve /path/to/your/model.xml

# 指定端口
robola serve /path/to/your/model.xml --port 9527

# 指定允许的来源（用于 CORS）
robola serve /path/to/your/model.xml --origin https://robola.com
```

### Python API

```python
from robola import serve

# 基本用法
serve("/path/to/your/model.xml")

# 自定义配置
serve(
    mjcf_path="/path/to/your/model.xml",
    port=9527,
    allowed_origin="https://robola.com"
)
```

### 工作流程

1. 在本地运行 `robola serve your_model.xml`
2. 打开浏览器访问 Robola Web Editor (https://robola.com/editor)
3. 登录后，编辑器会自动连接到本地的 WebSocket 服务
4. 开始编辑你的 MuJoCo 模型！

## 系统要求

- Python 3.10+
- MuJoCo 3.0+

## 许可证

MIT License
