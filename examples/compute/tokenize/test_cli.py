import json
import subprocess
import sys
import os


def log_error(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)


def send_command(proc, func_name, args):
    cmd = {"func": func_name, "args": args}
    line = json.dumps(cmd, ensure_ascii=False)
    print(f"[TEST] stdin -> {line}")
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


def read_response(proc, timeout=30):
    delimiter = "==e6b7e03aa02b4ffe=="
    outputs = []
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.rstrip("\n")
        if delimiter in line:
            line = line.replace(delimiter, "").strip()
            if line:
                outputs.append(line)
            break
        line = line.strip()
        if line:
            outputs.append(line)
    return "\n".join(outputs)


def main():
    proc = subprocess.Popen(
        ["xmake", "run", "tokenizer"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=os.getcwd(),
    )

    try:
        # Read initial prompt
        init_line = proc.stdout.readline()
        if init_line:
            print(f"[TEST] stdout <- {init_line.rstrip()}")

        # Step 1: add_builder with .agents/ dir
        send_command(proc, "add_builder", ["../../.agents/"])
        handle_str = read_response(proc)
        if not handle_str or handle_str.startswith("error"):
            log_error(f"add_builder failed: {handle_str}")
            proc.stdin.write("exit\n")
            proc.stdin.flush()
            proc.wait()
            sys.exit(1)

        print(f"[TEST] Got builder handle: {handle_str}")

        # Step 2: search with random words
        keywords = "xmake"
        send_command(proc, "search", [handle_str, keywords])
        search_result = read_response(proc)
        if search_result.startswith("error"):
            log_error(f"search failed: {search_result}")
            proc.stdin.write("exit\n")
            proc.stdin.flush()
            proc.wait()
            sys.exit(1)

        print(f"[TEST] Search result:\n{search_result}")

        # Exit CLI
        proc.stdin.write("exit\n")
        proc.stdin.flush()

        # Collect remaining stderr
        try:
            _, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate()

        if stderr:
            print(f"[TEST] stderr:\n{stderr}")

        rc = proc.returncode
        if rc != 0:
            log_error(f"tokenizer exited with code {rc}")
            sys.exit(1)

        print("[TEST] SUCCESS: tokenizer CLI works correctly.")
        sys.exit(0)

    except Exception as e:
        log_error(f"Exception: {e}")
        proc.kill()
        proc.wait()
        sys.exit(1)


if __name__ == "__main__":
    main()
