from modelscope import snapshot_download
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(current_dir, "../model"))
model = snapshot_download(model_id="LLM-Research/Meta-Llama-3.1-8B-Instruct", cache_dir=model_path, revision="master")