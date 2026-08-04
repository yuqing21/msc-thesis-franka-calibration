import ast
from pathlib import Path


def test_herbert_python_scripts_parse() -> None:
    scripts = Path(__file__).parents[1] / "herbert_ws" / "src" / "franka_control_bridge" / "scripts"
    for script in sorted(scripts.glob("*.py")):
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))

