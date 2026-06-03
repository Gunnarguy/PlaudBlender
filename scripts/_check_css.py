"""Quick CSS brace checker."""

css = open("app_v2/assets/style.css").read()
lines = css.split("\n")
depth = 0
rule_starts = []
for i, line in enumerate(lines, 1):
    for ch in line:
        if ch == "{":
            if depth == 0:
                rule_starts.append((i, line.strip()[:60]))
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and rule_starts:
                rule_starts.pop()
print(f"Final depth: {depth}")
print(f"Unclosed rules:")
for ln, txt in rule_starts:
    print(f"  L{ln}: {txt}")
