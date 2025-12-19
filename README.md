# Robola – Local MuJoCo Runtime Companion

RobolaPyLib powers the local runtime that feeds the [Robola Web Editor](https://www.robolaweb.com/):

- Load and validate MuJoCo MJCF scenes on your own machine
- Stream geometry/material/joint data to the browser in real time
- Expose a WebSocket API for simulation control, saving, and asset transfer

> ✅ Use the online editor at **https://www.robolaweb.com/editor**. The website never touches your disk—the CLI in this repo loads every file locally and only streams the data you approve.

---

## Installation

### From PyPI (recommended)

```bash
pip install robola
```

### From source (development)

```bash
git clone https://github.com/robola-dev/robola.git
cd robola
pip install -e .
```

Prerequisites:

- Python 3.10 or later
- MuJoCo 3.3+ (and its license/key environment variables configured if required)

---

## 5-Minute Quickstart

1. **Prepare a model** – keep an `*.xml` MJCF file handy (for example `./robots/spot.xml`).
2. **Start the local server**
   ```bash
   robola serve ./robots/spot.xml 
   ```
   or create mjcf from --new:
   ```
   robola serve ./robot.xml --new
   ```
   - Serves WebSocket traffic on `ws://localhost:9527`
   - The terminal banner confirms that the runtime is up
3. **Open the online editor** – visit [https://www.robolaweb.com/editor](https://www.robolaweb.com/editor) and sign in.
4. **Connect to the runtime** – enter `9527` in the “WebSocket Port” field (top-right) and click **Connect**.
5. **Model, simulate, iterate**
   - The editor fetches the local model and renders the scene
   - You can inspect/edit bodies, joints, materials, etc.
   - Start/pause/stop simulations or save back to disk from the browser

> 📌 Keep the CLI process running while the browser tab is open so the editor can stream frames and send commands.

---

## Example Models

You can use MJCF models from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) as examples. Clone the repository and serve any `.xml` file:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie
robola serve ./google_robot/scene.xml 
```

Then connect to the Robola Web Editor at [https://www.robolaweb.com/editor](https://www.robolaweb.com/editor) using port 9527.

---

## CLI Usage

```bash
# Load a model with the default port (9527)
robola serve /path/to/model.xml

# Bootstrap a blank scene and launch the server (path is created from the built-in template before serving)
robola serve /path/to/new_scene.xml --new

# Use a custom port
robola serve /path/to/model.xml --port 9000

# Restrict CORS origins (production)
robola serve /path/to/model.xml --origin https://robolaweb.com

# Tune streaming rate and logging
robola serve /path/to/model.xml --fps 30
```

### Common Options

| Option | Description | Default |
| --- | --- | --- |
| `--port` | WebSocket listening port | `9527` |
| `--fps` | Simulation streaming rate (1–60 Hz) | `30` |
| `--new` | Create the specified MJCF path with the built-in template before serving (fails if the file already exists) | `false` |

---

## Embedding via Python

```python
from robola import serve

serve(
    mjcf_path="path/to/model.xml",
    port=9527,
    fps=30,
)
```

Call `serve()` inside a notebook, a script, or your own application to reuse the same local WebSocket protocol that the CLI exposes.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| “Unable to connect” in the editor | Port/origin mismatch | Ensure the CLI `--port` and `--origin` match the values entered in the editor |
| Simulation crashes on start | Missing meshes/textures referenced by MJCF | Check the terminal logs and verify every asset path is valid relative to the MJCF file |
| Saving fails | File not writable | Confirm the XML can be written and that the CLI is still running |

> ⚠️ Current limitation: Robola does **not** support MuJoCo Flex (soft bodies) or height-field (`hfield`) based terrain rendering yet. These assets will be ignored/omitted when streaming to the editor.

---

## License

MIT License. PRs and issues are welcome—help more roboticists build faster with Robola. 🤖🚀
