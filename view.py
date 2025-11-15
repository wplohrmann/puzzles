import os
from utils import show_task, base_path




tasks = os.listdir(base_path)
for task_name in sorted(tasks):
    show_task(task_name)
