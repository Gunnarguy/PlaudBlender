#!/usr/bin/env python3
"""Find all @keyframes blocks with brace imbalances in style.css."""

CSS_PATH = "app_v2/assets/style.css"

with open(CSS_PATH) as f:
    lines = f.readlines()

# Find all @keyframes and print their blocks
in_keyframes = False
kf_name = ""
kf_start = 0
kf_depth = 0
kf_lines = []

for i, line in enumerate(lines, 1):
    if "@keyframes" in line and not in_keyframes:
        in_keyframes = True
        kf_name = line.strip()
        kf_start = i
        kf_depth = 0
        kf_lines = []

    if in_keyframes:
        kf_lines.append((i, line.rstrip()))
        kf_depth += line.count("{") - line.count("}")
        if kf_depth == 0 and len(kf_lines) > 1:
            # Check internal structure
            inner_depth = 0
            broken = False
            for ln, txt in kf_lines:
                inner_depth += txt.count("{") - txt.count("}")
                if inner_depth < 0:
                    broken = True
            # Print if broken or has unusual depth patterns
            has_from = any("from" in t for _, t in kf_lines)
            has_to = any("  to" in t or t.strip().startswith("to") for _, t in kf_lines)
            # Count inner blocks (from/to/0%/50%/100%)
            inner_blocks = (
                sum(1 for _, t in kf_lines if "{" in t) - 1
            )  # minus the @keyframes line
            inner_closes = (
                sum(t.count("}") for _, t in kf_lines) - 1
            )  # minus final close
            if inner_blocks != inner_closes:
                broken = True
            if broken:
                print(f"BROKEN: {kf_name} at line {kf_start}")
            else:
                print(f"OK: {kf_name} at line {kf_start}")
            for ln, txt in kf_lines:
                marker = (
                    " >>>"
                    if ("{" in txt and "}" not in txt and "@keyframes" not in txt)
                    else "    "
                )
                print(f"  {marker} {ln}: {txt}")
            print()
            in_keyframes = False

# Overall brace count
total_open = sum(l.count("{") for l in lines)
total_close = sum(l.count("}") for l in lines)
print(
    f"Total {{ = {total_open}, Total }} = {total_close}, Diff = {total_open - total_close}"
)
