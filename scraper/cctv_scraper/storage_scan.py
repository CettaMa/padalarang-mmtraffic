from datetime import date, datetime


def trim_stderr(stderr: str, max_lines: int = 12) -> str:
    lines = (stderr or "").splitlines()
    return " | ".join(lines[-max_lines:])


def parse_date_dir_name(name: str) -> date | None:
    try:
        return datetime.strptime(name, "%Y-%m-%d").date()
    except ValueError:
        return None
