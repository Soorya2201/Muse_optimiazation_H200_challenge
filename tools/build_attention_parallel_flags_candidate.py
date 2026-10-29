"""Generate an isolated small-batch parallel completion-flag merge experiment."""
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parent = (ROOT / "muse_attention_decode/submission_v15_vjeux_f35a41e.py").read_text()
anchor = "  if ((FLAT && (L == 8)) && M_COOP) {\n"
assert parent.count(anchor) == 1
hook = r'''
  // Local phase-trace-driven experiment: replace the contended arrival RMW only
  // for a single sliding decode request with <=32 resident, non-cluster splits.
  // Workspace minimum is 32768 counter words; these slots are unused by parent.
  // All 32 flags are cleared before kernel completion, including inactive slots.
  // Serial CUDA stream/graph calls therefore start with zero flags. No epoch wraps.
  // No activation/output is cached.
  if constexpr (L == 1 && !FLAT) {
    if (p.B == 1 && p.Hkv == 2 && p.window > 0 && p.nc == 1
        && p.cs <= 1 && p.n_splits <= 32) {
      constexpr int FLAGS = 31104;
      if (tid == 0) {
        // The original CTA barrier above makes all threads' FP32 reds precede
        // this release publication, matching the parent's semaphore protocol.
        asm volatile("st.release.gpu.global.u32 [%0], %1;\n"
                     ::"l"(p.ws_cnt + FLAGS + blockIdx.x), "r"(1u) : "memory");
      }
      if (blockIdx.x == 0) {
        if (tid < 32) {
          const int active_splits = fdiv_i(n_keys + chunk - 1, chunk);
          SPIN_T0;
          for (;;) {
            bool ready = true;
            if (tid < active_splits) {
              uint32_t value;
              asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n"
                           : "=r"(value) : "l"(p.ws_cnt + FLAGS + tid) : "memory");
              ready = value == 1u;
            }
            if (__all_sync(0xffffffffu, ready)) break;
            SPIN_CHECK;
            __nanosleep(32);
          }
        }
        csync<WS, NT>();  // propagate all split acquires to the finalizer threads
        fin_nc1<L, NT>(p, b, pair, bp, vscale, 0);
        // One coalesced warp store resets every slot. The next graph node or
        // same-stream call starts after all this kernel's writes complete.
        if (tid < 32) p.ws_cnt[FLAGS + tid] = 0;
      }
      return true;
    }
  }
'''
candidate = parent.replace(anchor, hook+anchor).replace("okc_attn_v298{tag}_90a", "okc_attn_v319_flags{tag}_90a")
target = "muse_attention_decode/submission_v19_parallel_completion_flags.py"
assert not (ROOT / target).exists(), "Do not overwrite an archived experiment"
old_tree, new_tree = ast.parse(parent), ast.parse(candidate)
for old,new in zip(old_tree.body,new_tree.body):
    if isinstance(old,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="CUDA_SRC" for t in old.targets):
        assert ast.literal_eval(new.value).replace(hook,"") == ast.literal_eval(old.value)
    elif isinstance(old,ast.FunctionDef) and old.name=="_build":
        restored=ast.get_source_segment(candidate,new).replace("okc_attn_v319_flags{tag}_90a","okc_attn_v298{tag}_90a")
        assert ast.dump(ast.parse(restored).body[0])==ast.dump(old)
    else:
        assert ast.dump(old)==ast.dump(new)
assert 31104 % 32 == 0 and 31104+32 <= 1 << 15
compile(candidate,target,"exec")
patch="*** Begin Patch\n*** Add File: "+target+"\n"+"\n".join("+"+s for s in candidate.splitlines())+"\n*** End Patch\n"
subprocess.run(["apply_patch",patch],cwd=ROOT,check=True)
print("PASS: isolated completion protocol; aligned flags within workspace; syntax passes")
