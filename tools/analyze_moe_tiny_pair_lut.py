"""Inspect the saved sm90a result and tighten finite-table bank bounds offline."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TRACK=ROOT/'muse_moe_w4a16'
archive=TRACK/'TINY_PAIR_LUT_CODEGEN_2026-10-31.json.gz'
data=json.load(gzip.open(archive,'rt'))
result=data['response']
assert result['code']==0 and not result['truncated']
source=(TRACK/'TINY_PAIR_LUT_FRAGMENT_2026-10-31.cu').read_bytes()
assert hashlib.sha256(source).hexdigest()==data['source_sha256']
assert source.decode()==data['request']['source']
devices=result['devices']
assert not any('truncat' in row['text'].lower() for device in devices.values() for row in device['asm'])
rows=devices['SASS']['asm']
cores={}
for name,label in [('parent','$parent_fragment$arithmetic_core:'),('table','$table_fragment$table_core:')]:
    start=next(i for i,row in enumerate(rows) if row['text'].strip()==label)
    instructions=[]
    for row in rows[start+1:]:
        line=row['text'].strip()
        assert not line.endswith(':')
        if line:instructions.append(line)
        if line.startswith('RET.'):break
    assert instructions[-1].startswith('RET.')
    histogram=collections.Counter(line.split()[0] for line in instructions)
    cores[name]=dict(instructions=instructions,static_sites=len(instructions),opcode_histogram=dict(histogram))
assert cores['parent']['static_sites']==16
assert cores['table']['static_sites']==26
assert cores['table']['opcode_histogram']['LDS']==4

# At most 32/D lanes use a bank in the interleaved layout. The finite table
# also has just 256*D/32 unique words per bank. Witnesses reach both limits.
bounds=[]
for copies in (1,8,16,32):
    indices=([32*(lane%8) for lane in range(32)] if copies==1 else
             [(32//copies)*(lane//copies) for lane in range(32)])
    banks=[set() for _ in range(32)]
    for lane,index in enumerate(indices):
        assert 0<=index<256
        word=index*copies+lane%copies
        banks[word%32].add(word)
    exact_max=min(32//copies,256*copies//32)
    assert max(map(len,banks))==exact_max
    bounds.append(dict(copies=copies,tight_max_unique_words_per_bank=exact_max,
                       attaining_indices=indices))

out=dict(source_sha256=data['source_sha256'],archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    compiler_exit_code=result['code'],untruncated_device_rows={k:len(v['asm']) for k,v in devices.items()},
    cores=cores,
    scoped_costs=dict(parent_decode_integer_sites=9,parent_bf16_scale_sites=4,
        table_address_integer_sites=15,table_shared_reads=4,table_bf16_scale_sites=4,
        table_origin_integer_sites_hoistable_in_full_kernel=3,
        parent_device_abi_and_return_sites=3,table_device_abi_and_return_sites=3),
    resources_stderr=[row['text'] for row in result['stderr']],
    finite_table_bank_bounds=bounds,
    bank_bound_correction='Original preflight maximum_word_waves=32/D is a valid but loose upper bound at D=1; finite 256-entry table tightens it to8. Original archive preserved.',
    tiny_case_opportunity=dict(scored_us=25.686852,other_loads_unchanged=5,
        target_board_speedup=1.03,required_case_speedup=1.03**6,
        required_case_time_cut_fraction=1-1/1.03**6,
        required_case_time_cut_us=25.686852*(1-1/1.03**6)),
    decision='Defer this exact standalone table form before GPU. Capacity is feasible only for the measured tiny build; no >=3% practical benefit case. Not integrated or GPU-tested.',
    limits='Static device-core sites exclude caller, table initialization and synchronization. Registers are standalone entry resources, not fused MoE resources. Hoisting is hypothetical. Counts and synthetic bank requests are not elapsed cycles or a proof of runtime regression.')
target=TRACK/'TINY_PAIR_LUT_ANALYSIS_2026-10-31.json'
target.write_text(json.dumps(out,indent=2)+'\n')
print('PASS', {name:row['static_sites'] for name,row in cores.items()},
      'tight bank maxima',[x['tight_max_unique_words_per_bank'] for x in bounds])
