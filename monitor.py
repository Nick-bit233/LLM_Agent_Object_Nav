import time
import subprocess

while True:
    print("Starting main_run_script...")
    process = subprocess.Popen(["python", "main_run_script.py"])
    process.wait()

    if process.returncode == 0:  # 如果主要脚本正常退出
        break

    print("main_script.py exited unexpectedly, restarting in 5 seconds...")
    time.sleep(5)
