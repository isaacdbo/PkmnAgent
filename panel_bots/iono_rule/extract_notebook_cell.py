"""Extract the `%%writefile main.py` code cell from a Kaggle rendered notebook HTML."""
import html
import re
import sys

src = open(sys.argv[1], encoding="utf-8").read()

# Kaggle renders code cells inside <div class="highlight"><pre>...</pre></div>
blocks = re.findall(r"<pre[^>]*>(.*?)</pre>", src, flags=re.S)
print(f"pre blocks: {len(blocks)}", file=sys.stderr)

def detag(s):
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s)

hits = [detag(b) for b in blocks if "writefile" in b]
print(f"blocks containing writefile: {len(hits)}", file=sys.stderr)
if not hits:
    sys.exit(1)
text = max(hits, key=len)
# Drop the magic line itself; the file body is everything after it.
lines = text.split("\n")
for i, ln in enumerate(lines):
    if "%%writefile" in ln:
        lines = lines[i + 1 :]
        break
out = "\n".join(lines)
sys.stdout.write(out)
