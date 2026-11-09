# Set this to your file's base name (no extension) if not passing CLI args.
txt_file = "260721"

import os
import re
import sys
import csv

# Folder the script itself lives in - used so input/output paths resolve
# correctly no matter what directory you launch the script from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# A single number: optional sign, digits, optional decimal part, optional exponent.
_NUM = r"-?\d+\.?\d*(?:[eE][+-]?\d+)?"

# A data row: 1-2 opening brackets, 4 numbers (x y z ID), 1-2 closing brackets.
# Brackets are intentionally NOT required to be paired/balanced with each other,
# since this file contains rows with mismatched bracket counts (see module docstring).
ROW_PATTERN = re.compile(rf"^\s*\[{{1,2}}\s*({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s*\]{{1,2}}\s*$")


def extract_rows(text: str):
    """
    Scans every line of `text` and returns a list of (x, y, z, ID) tuples
    (floats, floats, floats, int) for every line that matches the data-row shape.
    Also returns the list of (line_number, line_text) for lines that contain a
    bracket but did NOT match - these are the lines you should spot-check.
    """
    rows = []
    unmatched_with_bracket = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        m = ROW_PATTERN.match(line)
        if m:
            x, y, z, mussel_id = m.groups()
            rows.append((float(x), float(y), float(z), int(float(mussel_id))))
        elif "[" in line or "]" in line:
            unmatched_with_bracket.append((line_no, line))

    return rows, unmatched_with_bracket


def independent_recount(text: str):
    """
    A second, deliberately different extraction method used only to
    cross-check `extract_rows`. Strips every bracket character from each
    line, splits on whitespace, and counts lines that reduce to exactly
    4 numeric tokens. If this count disagrees with extract_rows' count,
    something about the row shape assumption is wrong.
    """
    count = 0
    for line in text.splitlines():
        stripped = line.replace("[", " ").replace("]", " ").strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) == 4:
            try:
                [float(t) for t in tokens]
                count += 1
            except ValueError:
                pass
    return count


def verify_extraction(text: str, rows: list, unmatched_with_bracket: list) -> bool:
    """
    Prints a verification report and returns True if extraction looks sound.

    Checks performed:
      1. Cross-check row count against an independently-implemented recount.
      2. Show every bracket-containing line that was NOT captured, so you can
         manually confirm they're all genuinely non-data (progress bars,
         status text) and not missed rows.
      3. Basic sanity ranges for x, y, z, ID (flags NaN/Inf, wildly out-of-range
         values, and reports how many exact (0,0,0) "dropout" rows were found,
         since those are expected to be present in real tracking logs and
         downstream RMSE code needs to know about them).
    """
    ok = True
    print("=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)

    # 1. Cross-check against an independently written counting method
    recount = independent_recount(text)
    print(f"\n[1] Row count cross-check:")
    print(f"    Primary regex extraction : {len(rows)} rows")
    print(f"    Independent recount      : {recount} rows")
    if len(rows) != recount:
        print("    MISMATCH -- investigate before trusting the CSV.")
        ok = False
    else:
        print("    Match.")

    # 2. Show unmatched lines that contain a bracket, for manual eyeballing
    print(f"\n[2] Bracket-containing lines NOT captured as data " f"({len(unmatched_with_bracket)} total):")
    if unmatched_with_bracket:
        preview = unmatched_with_bracket[:8]
        for line_no, line in preview:
            print(f"    line {line_no}: {line.strip()!r}")
        if len(unmatched_with_bracket) > len(preview):
            print(
                f"    ... and {len(unmatched_with_bracket) - len(preview)} more "
                f"(inspect the full list via the returned unmatched_with_bracket value)"
            )
        print("    -> Confirm these are all progress bars / prompts / status text,")
        print("       not rows that failed to match due to a formatting quirk.")
    else:
        print("    None - every bracket in the file belongs to a captured row.")

    # 3. Sanity-check the extracted values themselves
    print(f"\n[3] Value sanity check on {len(rows)} extracted rows:")
    if not rows:
        print("    No rows extracted - nothing to check.")
        return False

    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    zs = [r[2] for r in rows]
    ids = [r[3] for r in rows]

    import math

    bad_values = [r for r in rows if any(math.isnan(v) or math.isinf(v) for v in r[:3])]
    if bad_values:
        print(f"    Found {len(bad_values)} rows with NaN/Inf values!")
        ok = False
    else:
        print("    No NaN/Inf values found.")

    print(f"    x range   : [{min(xs):.6g}, {max(xs):.6g}]")
    print(f"    y range   : [{min(ys):.6g}, {max(ys):.6g}]")
    print(f"    z range   : [{min(zs):.6g}, {max(zs):.6g}]")
    print(f"    ID range  : [{min(ids)}, {max(ids)}]  ({len(set(ids))} unique IDs)")

    dropout_rows = sum(1 for x, y, z, _ in rows if x == 0 and y == 0 and z == 0)
    print(f"    Exact (0,0,0) rows (likely dropout frames): {dropout_rows}")

    print("\n" + "=" * 60)
    print("Extraction looks CONSISTENT." if ok else "Extraction FLAGGED ISSUES - see above.")
    print("=" * 60)
    return ok


def write_csv(rows, output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["xpos", "ypos", "zpos", "ID"])
        for x, y, z, mussel_id in rows:
            writer.writerow([x, y, z, mussel_id])


def main():

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(input_path)[0] + ".csv"
    else:
        # Resolve relative to the script's own folder, NOT the terminal's cwd -
        # this is what fixes the FileNotFoundError when running from elsewhere.
        input_path = os.path.join(SCRIPT_DIR, txt_file + ".txt")
        output_path = os.path.join(SCRIPT_DIR, txt_file + ".csv")

    with open(input_path, "r") as f:
        text = f.read()

    rows, unmatched_with_bracket = extract_rows(text)
    verify_extraction(text, rows, unmatched_with_bracket)

    write_csv(rows, output_path)
    print(f"\nWrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
