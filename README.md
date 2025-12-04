# Robola – Local MuJoCo Runtime Companion

RobolaPyLib 为 [Robola Web Editor](https://robolaweb.com/) 提供本地运行时支持：

- 解析与加载 MuJoCo MJCF 模型
- 将几何、材质、关节等完整模型数据推送到浏览器
- 通过 WebSocket 在本地实现实时仿真、控制与文件保存

> ✅ 在线编辑器位于 **https://robolaweb.com/editor**。该网站不会读取你的文件系统；所有模型都由本库在本地加载和托管。

---

## 安装方式

### 使用 PyPI（推荐）

```bash
pip install --upgrade robola
```

### 从源码开发

```bash
git clone https://github.com/robola-dev/RobolaPyLib.git
cd RobolaPyLib
pip install -e .
```

依赖要求：

- Python 3.10 或更高版本
- 已安装 MuJoCo 3.0+（或在 `MUJOCO_PY_MJKEY_PATH` 等变量中正确配置）

---

## 5 分钟快速上手

1. **准备模型**  
    确保本地存在可用的 `*.xml` MJCF 文件（如 `~/robots/spot.xml`）。

2. **启动本地服务**
    ```bash
    robola serve ~/robots/spot.xml --port 9527
    ```
    - 默认监听 `ws://localhost:9527`
    - 终端会显示“Serving MJCF …”即表示启动成功

3. **打开在线编辑器**  
    访问 [https://robolaweb.com/editor](https://robolaweb.com/editor) 并登录。

4. **连接到本地运行时**  
    在编辑器右上角的 “WebSocket 端口” 输入框中填入 `9527`，点击 **连接**。

5. **开始建模与仿真**  
    - 编辑器会读取本地模型并显示场景
    - 你可以实时修改模型结构、材质、关节参数
    - 运行/暂停/停止仿真、保存模型，全部通过 Web UI 完成

> 📌 只要浏览器保持打开，本地 CLI 进程就需要保持运行，以便随时响应指令。

---

## 命令行用法

```bash
# 基本：加载模型并使用默认端口 9527
robola serve /path/to/model.xml

# 指定端口
robola serve /path/to/model.xml --port 9000

# 限制可连接的网页来源（CORS）
robola serve /path/to/model.xml --origin https://robolaweb.com

# 调整帧率、日志等级等
robola serve /path/to/model.xml --fps 35
```

常用选项：

| 选项 | 说明 | 默认 |
| --- | --- | --- |
| `--port` | WebSocket 监听端口 | `9527` |
| `--origin` | 允许连接的浏览器来源 | `*`（仅本地开发建议设置为 `https://robolaweb.com`） |
| `--fps` | 仿真帧率 (1-60 Hz) | `30` |


---

## Python API 嵌入

```python
from robola import serve

serve(
     mjcf_path="/abs/path/to/model.xml",
     port=9527,
     allowed_origin="https://robolaweb.com",
     fps=30,
)
```

在 Notebook、脚本或自定义应用中直接调用 `serve`，即可复用同一套本地 WebSocket 协议。

---

## 故障排查

| 问题 | 可能原因 | 解决方案 |
| --- | --- | --- |
| 编辑器显示“无法连接” | 端口或 CORS 设置不匹配 | 确认 CLI 使用的 `--port`、`--origin` 与网页一致 |
| 仿真启动即崩溃 | MJCF 引用丢失或 mesh 未找到 | 查看终端日志，确保所有资源路径相对/绝对正确 |
| 保存失败 | 浏览器无写入权限 | 终端会显示具体错误，确认文件可写且 CLI 仍在运行 |

---

## 许可协议

MIT License. 欢迎提交 Issue 与 PR，让更多研究者和机器人开发者使用 Robola。🤖🚀
