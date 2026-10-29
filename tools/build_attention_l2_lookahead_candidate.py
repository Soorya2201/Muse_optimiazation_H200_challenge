"""Create an isolated attention port from the checked parent, using apply_patch."""
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parent = (ROOT / "muse_attention_decode/submission_v15_vjeux_f35a41e.py").read_text()
anchor = "          fetch_pages(x + PD);\n          load_pages_t(x, pg);\n"
assert parent.count(anchor) == 1
hook = r'''
          // One-tile L2 lookahead, adapted from Madhavan113/kerneloptimizer c1809247
          // OKC_PFL2. Apache-2.0; original vjeux license is retained alongside this file.
          // Runs only in the flat L=1 TMA page-index producer. Actual cp.async copies,
          // complete context coverage, activation precision and all barriers are unchanged.
          constexpr int LOOKAHEAD = 1;
          if (LOOKAHEAD < PD && pfast && ch == 0 && x + LOOKAHEAD < ntiles
              && start + (x + LOOKAHEAD) * TILE + TILE <= cache_end) {
            int pgf[PK];
            load_pages_t(x + LOOKAHEAD, pgf);
#pragma unroll
            for (int k = 0; k < PK; ++k) {
              const uint32_t off = ((uint32_t)pgf[k] << p.lrb) + pcol;
              asm volatile("prefetch.global.L2 [%0];" ::"l"(p.k_cache + off));
              asm volatile("prefetch.global.L2 [%0];" ::"l"(p.v_cache + off));
            }
          }
'''
target = "muse_attention_decode/submission_v16_l2_lookahead.py"
assert not (ROOT / target).exists(), "Do not overwrite an archived experiment"
candidate = parent.replace(anchor, anchor + hook).replace("okc_attn_v298{tag}_90a", "okc_attn_v316_l2{tag}_90a")
old_tree, new_tree = ast.parse(parent), ast.parse(candidate)
assert len(old_tree.body) == len(new_tree.body)
for old, new in zip(old_tree.body, new_tree.body):
    if isinstance(old, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in old.targets):
        assert ast.literal_eval(new.value).replace(hook, "") == ast.literal_eval(old.value)
    elif isinstance(old, ast.FunctionDef) and old.name == "_build":
        restored = ast.get_source_segment(candidate, new).replace("okc_attn_v316_l2{tag}_90a", "okc_attn_v298{tag}_90a")
        assert ast.dump(ast.parse(restored).body[0]) == ast.dump(old)
    else:
        assert ast.dump(old) == ast.dump(new)
compile(candidate, target, "exec")
patch = "*** Begin Patch\n*** Add File: " + target + "\n" + "\n".join("+" + s for s in candidate.splitlines()) + "\n*** End Patch\n"
subprocess.run(["apply_patch", patch], cwd=ROOT, check=True)
print("PASS: only the producer lookahead and extension namespace changed; syntax passes")
