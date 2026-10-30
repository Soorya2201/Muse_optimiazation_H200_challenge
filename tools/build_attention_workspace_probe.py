"""Build a bounded same-input workspace-shard/finalizer diagnostic on the parent."""
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parent = (ROOT / "muse_attention_decode/submission_v15_vjeux_f35a41e.py").read_text()
probe = (ROOT / "muse_attention_decode/submission_v17_phase_probe.py").read_text()
tree = ast.parse(probe)
helpers = "\n\n".join(ast.get_source_segment(probe,n) for n in tree.body
                       if isinstance(n,ast.FunctionDef) and n.name in ("_phase_fixture","_phase_pair"))
assert len(helpers)>1500
tail = r'''

# Development-only import-time probe. Timed entry above remains the checked parent.
def _workspace_shard_probe():
    import json
    import math
    import time
    started = time.perf_counter()
    mod = _module()
    dev = torch.device("cuda")
    flush = torch.zeros((64<<20,),dtype=torch.float32,device=dev)
    for B,L in ((1,1),(1,8),(16,1),(32,1),(16,8),(32,8)):
        if time.perf_counter()-started>240:
            print("SHARDS budget reached",flush=True)
            break
        base = _plan(B,1,L,2048,80000+L if B==1 else 86000+L,1,dev)
        base.update(_TUNED.get((B,L,True,1),{}))
        splits=base["n_splits"]
        # L1 defaults use last-arriver for these small split counts; L8 explicitly cooperates.
        default_coop=base.get("coop",0 if splits<64 else int(splits*B<=2*_num_sms(dev)))
        copies=(1,2,4,8,16,32) if B==1 else (1,2,4,8)
        plans=[dict(n_splits=splits,nc=nc,coop=coop) for nc in copies for coop in (0,1)
               if (nc,coop)!=(base["nc"],default_coop)]
        # Keep workspace storage stable before capturing either graph, even if expanded later.
        _workspace(B*max(copies)*L*16,dev)
        trials=[{"nc":p["nc"],"coop":p["coop"],"ratios":[],"delta_us":[],"error":0.0} for p in plans]
        for sample in range(3):
            data=_phase_fixture(B,L,2048, 57119+997*B+L+1543*sample,dev)
            fn0=lambda:_launch(data,_mod=mod)
            reference=fn0().float()
            for plan,record in zip(plans,trials):
                if time.perf_counter()-started>240:
                    break
                fn1=lambda:_launch(data,_mod=mod,**plan)
                got=fn1().float()
                error=float((got-reference).norm()/reference.norm().clamp_min(1e-12))
                record["error"]=max(record["error"],error if math.isfinite(error) else float("inf"))
                del got
                if record["error"]>0.002:
                    continue
                for _ in range(2):
                    old,new=_phase_pair(fn0,fn1,flush,reps=10)
                    record["ratios"].append(round(old/new,6))
                    record["delta_us"].append(round(old-new,3))
            del data,reference
        for record in trials:
            record["error"]=round(record["error"],9)
            record["qualifies"]=(len(record["ratios"])==6 and record["error"]<=0.002
                                  and all(v>1.02 for v in record["ratios"]))
        trials.sort(key=lambda r:min(r["ratios"],default=0.0),reverse=True)
        print("SHARDS "+json.dumps({"B":B,"L":L,"splits":splits,
              "parent_nc":base["nc"],"parent_coop":default_coop,"trials":trials}),flush=True)
    for buffers in _WS.values():
        for buffer in buffers:buffer.zero_()
    torch.cuda.synchronize()
    print(f"SHARDS done {time.perf_counter()-started:.1f}s; timed entry remains parent",flush=True)


if torch.cuda.is_available():
    try:
        _workspace_shard_probe()
    except Exception as _shard_error:
        print(f"SHARDS failed: {str(_shard_error)[-1500:]}",flush=True)
        for _shard_buffers in _WS.values():
            for _shard_buffer in _shard_buffers:_shard_buffer.zero_()
'''
target="muse_attention_decode/submission_v20_workspace_shard_probe.py"
assert not (ROOT/target).exists(),"Do not overwrite an archived experiment"
source=parent.replace("okc_attn_v298{tag}_90a","okc_attn_v320_shard_probe{tag}_90a")+"\n"+helpers+tail
assert source.startswith(parent.replace("okc_attn_v298{tag}_90a","okc_attn_v320_shard_probe{tag}_90a"))
compile(source,target,"exec")
patch="*** Begin Patch\n*** Add File: "+target+"\n"+"\n".join("+"+s for s in source.splitlines())+"\n*** End Patch\n"
subprocess.run(["apply_patch",patch],cwd=ROOT,check=True)
print("PASS: parent timed prefix unchanged; only independent import-time measurements appended")
