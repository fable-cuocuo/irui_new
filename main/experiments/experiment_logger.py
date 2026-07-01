import argparse
import datetime as dt
import os
import re
import subprocess
import sys


def _sanitize_tag(text: str) -> str:
    text = text.strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "na"


def build_log_path(model: str, params: str, log_dir: str) -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = _sanitize_tag(model)
    params_tag = _sanitize_tag(params)
    filename = f"{ts}_{model_tag}_{params_tag}.log"
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, filename)


def main():
    parser = argparse.ArgumentParser(
        description="Run an experiment command and save full logs."
    )
    parser.add_argument("--model", required=True, help="Model tag in log filename.")
    parser.add_argument(
        "--params",
        required=True,
        help="2-3 main parameter tags, e.g. orth0.01_rec0.5_conf0.3",
    )
    parser.add_argument(
        "--log-dir",
        default="./log_runs",
        help="Directory to save experiment logs.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after '--', e.g. -- py -3.12 script.py",
    )
    args = parser.parse_args()

    if not args.command:
        raise ValueError("No command provided. Use: ... -- <command>")

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise ValueError("No command provided after '--'.")

    log_path = build_log_path(args.model, args.params, args.log_dir)
    print(f"[experiment_logger] log_path={log_path}")
    print(f"[experiment_logger] command={' '.join(cmd)}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"start_time={dt.datetime.now().isoformat()}\n")
        f.write(f"command={' '.join(cmd)}\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
        ret = process.wait()
        f.write(f"\nexit_code={ret}\n")

    print(f"[experiment_logger] done, exit_code={ret}")
    if ret != 0:
        raise SystemExit(ret)


if __name__ == "__main__":
    main()
