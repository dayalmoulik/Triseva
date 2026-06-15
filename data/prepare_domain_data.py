import re
from pathlib import Path


RAW_ROOT = Path("data/raw")
DOMAINS = ("health", "legal", "agriculture")
MIN_LINE_LEN = 25
MAX_DUP_RATIO = 0.35


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def drop_noisy_lines(lines: list[str]) -> list[str]:
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if len(s) < MIN_LINE_LEN:
            continue
        lower = s.lower()
        # Common web scrape noise
        if any(
            token in lower
            for token in (
                "cookie",
                "privacy policy",
                "terms and conditions",
                "all rights reserved",
                "skip to main content",
                "javascript",
                "copyright",
            )
        ):
            continue
        cleaned.append(s)
    return cleaned


def dedupe_lines(lines: list[str]) -> list[str]:
    seen = set()
    unique = []
    for line in lines:
        key = re.sub(r"\W+", "", line.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(line)
    return unique


def quality_gate(lines_before: int, lines_after: int) -> bool:
    if lines_before == 0:
        return False
    dropped = lines_before - lines_after
    drop_ratio = dropped / lines_before
    return drop_ratio <= (1 - MAX_DUP_RATIO)


def clean_text_file(path: Path) -> tuple[bool, str]:
    original = path.read_text(encoding="utf-8", errors="ignore")
    normalized = normalize_whitespace(original)
    before_lines = [ln for ln in normalized.split("\n") if ln.strip()]

    lines = drop_noisy_lines(before_lines)
    lines = dedupe_lines(lines)

    if not quality_gate(len(before_lines), len(lines)):
        return False, "too much content filtered; check source manually"

    final_text = "\n".join(lines).strip()
    if not final_text:
        return False, "empty after cleaning"

    out_path = path.with_name(f"{path.stem}.clean.txt")
    out_path.write_text(final_text, encoding="utf-8")
    return True, f"wrote {out_path.name}"


def main() -> None:
    print("Preparing domain data (health/legal/agriculture)...")
    total = 0
    success = 0

    for domain in DOMAINS:
        domain_dir = RAW_ROOT / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        txt_files = sorted(
            p for p in domain_dir.glob("*.txt")
            if not p.name.endswith(".clean.txt")
        )
        if not txt_files:
            print(f"[{domain}] no .txt files found")
            continue

        print(f"\n[{domain}] processing {len(txt_files)} file(s)")
        for file_path in txt_files:
            total += 1
            ok, msg = clean_text_file(file_path)
            if ok:
                success += 1
                print(f"  OK   {file_path.name}: {msg}")
            else:
                print(f"  WARN {file_path.name}: {msg}")

    print("\nDone.")
    print(f"Cleaned successfully: {success}/{total}")
    print("Next: move vetted .clean.txt content into KB ingestion flow.")


if __name__ == "__main__":
    main()
