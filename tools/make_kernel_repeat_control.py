"""Make an archived comment-only repeat for independent judge measurements.

The CLI caches identical source. This preserves the problem marker and proves
the entire Python AST, including embedded native source constants, is unchanged.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("destination", type=Path)
parser.add_argument("tag")
args = parser.parse_args()
assert not args.destination.exists(), "Do not overwrite archived controls"
assert "\n" not in args.tag and "\r" not in args.tag
source = args.source.read_text()
marker, rest = source.split("\n", 1)
assert marker.startswith("# okc: problem=")
repeat = marker + "\n# Independent measurement control: " + args.tag + "\n" + rest
assert ast.dump(ast.parse(source), include_attributes=False) == ast.dump(ast.parse(repeat), include_attributes=False)
patch = "*** Begin Patch\n*** Add File: " + str(args.destination) + "\n"
patch += "\n".join("+" + line for line in repeat.splitlines()) + "\n*** End Patch\n"
subprocess.run(["apply_patch", patch], check=True, capture_output=True)
actual = args.destination.read_text()
assert ast.dump(ast.parse(source), include_attributes=False) == ast.dump(ast.parse(actual), include_attributes=False)
print("AST/native constants identical:", args.destination)
print("source SHA256:", hashlib.sha256(args.source.read_bytes()).hexdigest())
print("repeat SHA256:", hashlib.sha256(args.destination.read_bytes()).hexdigest())
