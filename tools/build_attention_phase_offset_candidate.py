"""Isolate a one-time softmax warpgroup offset on the checked attention parent."""
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parent = (ROOT / "muse_attention_decode/submission_v15_vjeux_f35a41e.py").read_text()
before = r'''
    // One-time phase offset, adapted from c1809247 OKC_TURN=2 (Apache-2.0).
    // Keep the parent's per-head V-transpose proxy fence and warpgroup barrier.
    // Only flat L=8: 128 head-1 consumers wait for 128 head-0 arrivals, once.
    if constexpr (FLAT) {
      static_assert(NT == 256, "phase offset requires two 128-thread consumer groups");
      if (hk == 1 && jj == 0) bar_sync_n(6, 256);
    }
'''
after = r'''
    // Head 0 has issued the first PV before releasing head 1's exponentials.
    // All consumer threads still participate in the original stage-release protocol.
    if constexpr (FLAT) {
      if (hk == 0 && jj == 0) bar_arrive_n(6, 256);
    }
'''
anchor0 = "       // my turn for the exponentials\n"
anchor1 = "\n    PROF_TILE;\n    \n    \n      if (++st_c == STAGES)"
assert parent.count(anchor0) == parent.count(anchor1) == 1
candidate = (parent.replace(anchor0, anchor0+before).replace(anchor1, after+anchor1)
             .replace("okc_attn_v298{tag}_90a", "okc_attn_v318_offset{tag}_90a"))
target = "muse_attention_decode/submission_v18_softmax_phase_offset.py"
assert not (ROOT / target).exists(), "Do not overwrite an archived experiment"
old_tree, new_tree = ast.parse(parent), ast.parse(candidate)
for old, new in zip(old_tree.body, new_tree.body):
    if isinstance(old, ast.Assign) and any(isinstance(t,ast.Name) and t.id=="CUDA_SRC" for t in old.targets):
        assert ast.literal_eval(new.value).replace(before, "").replace(after, "") == ast.literal_eval(old.value)
    elif isinstance(old,ast.FunctionDef) and old.name=="_build":
        restored = ast.get_source_segment(candidate,new).replace("okc_attn_v318_offset{tag}_90a", "okc_attn_v298{tag}_90a")
        assert ast.dump(ast.parse(restored).body[0]) == ast.dump(old)
    else:
        assert ast.dump(old) == ast.dump(new)
compile(candidate,target,"exec")
patch = "*** Begin Patch\n*** Add File: "+target+"\n"+"\n".join("+"+s for s in candidate.splitlines())+"\n*** End Patch\n"
subprocess.run(["apply_patch",patch],cwd=ROOT,check=True)
print("PASS: only one-time named-barrier handoff and namespace differ; syntax passes")
