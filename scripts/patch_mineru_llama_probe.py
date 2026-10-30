"""修复 MinerU 在 Windows 上找不到 llama.cpp Vulkan 设备的问题。

现象：
    llama.cpp device discovery unavailable
    AttributeError: function 'ggml_backend_dev_type' not found

原因：
    mineru-llama-cpp 的 ggml.dll 只导出 load/count/get，
    dev_type/name/description/backend_reg 等函数在 ggml-base.dll 里。
    MinerU 4.1.0 的 llama_probe.py 只加载了 ggml.dll。

这个脚本会直接修改 .venv-mineru 里的 llama_probe.py。
以后重装 MinerU 后，重新运行一次即可：

    PYTHONPATH=src .venv/Scripts/python.exe scripts/patch_mineru_llama_probe.py

也可以用 --probe 指定其他路径。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE = (
    ROOT
    / ".venv-mineru"
    / "Lib"
    / "site-packages"
    / "mineru"
    / "model"
    / "runtime"
    / "llama_probe.py"
)
DEFAULT_PYTHON = ROOT / ".venv-mineru" / "Scripts" / "python.exe"

BASE_LOAD = """    # Windows 的 mineru-llama-cpp wheel 把设备查询函数拆到了
    # ggml-base.dll：ggml.dll 只有 load/count/get，类型/名称/后端函数在 base 里。
    base_names = (
        ("ggml-base.dll", "libggml-base.so")
        if sys.platform == "win32"
        else ("libggml-base.so", "libggml-base.so.0")
    )
    base_path = next(
        (
            root / folder / name
            for folder in ("lib", "bin")
            for name in base_names
            if (root / folder / name).is_file()
        ),
        None,
    )
    base = ct.CDLL(str(base_path)) if base_path is not None else library
"""


def patch(probe_path: Path) -> bool:
    source = probe_path.read_text(encoding="utf-8")
    if "ggml-base.dll" in source and "base.ggml_backend_dev_type" in source:
        print(f"已经修过，不用重复修改：{probe_path}")
        return False

    needle = "    library = ct.CDLL(str(path))\n"
    if needle not in source:
        raise SystemExit(f"找不到插入位置，文件版本可能变了：{probe_path}")
    source = source.replace(needle, needle + BASE_LOAD, 1)

    functions = [
        ("kind", "ggml_backend_dev_type"),
        ("name", "ggml_backend_dev_name"),
        ("description", "ggml_backend_dev_description"),
        ("registry", "ggml_backend_dev_backend_reg"),
        ("backend", "ggml_backend_reg_name"),
    ]
    for variable, function in functions:
        old = f"    {variable} = library.{function}\n"
        new = f"    {variable} = base.{function}\n"
        if old not in source:
            raise SystemExit(f"找不到要替换的代码：{old.strip()}")
        source = source.replace(old, new, 1)

    backup = probe_path.with_suffix(".py.bak")
    backup.write_text(
        probe_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    probe_path.write_text(source, encoding="utf-8")
    print(f"已修改：{probe_path}")
    print(f"原文件备份：{backup}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe",
        default=str(DEFAULT_PROBE),
        help="llama_probe.py 路径",
    )
    parser.add_argument(
        "--python",
        default=str(DEFAULT_PYTHON),
        help=".venv-mineru 里的 python.exe",
    )
    args = parser.parse_args()

    probe_path = Path(args.probe)
    if not probe_path.exists():
        print(f"找不到文件：{probe_path}")
        return 1

    patch(probe_path)

    python = Path(args.python)
    if python.exists():
        print("\n检查设备：")
        result = subprocess.run(
            [str(python), "-m", "mineru.model.runtime.llama_probe"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = (result.stdout or "") + (result.stderr or "")
        for line in output.splitlines():
            if "MINERU_LLAMA_DEVICES=" in line or "Vulkan" in line:
                print(line)
        if result.returncode != 0:
            print("\n探测仍然失败，完整输出：")
            print(output)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
