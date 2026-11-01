"""Keep asynchronous wide RS operands live until the following wait completes.

Research variant of wide_batch_native. The post-wait compiler operand fence
reads the *previous* stage bank, which is now complete under wait_group<1>.
Both banks start initialized so the first dummy fence has defined inputs.
"""
from prefill_wide_batch_native import wide_batch_native


def wide_batch_fenced_native(source: str) -> str:
    source = wide_batch_native(source)
    before = '  int prev = -1;\n  const int woff ='
    assert source.count(before) == 1
    source = source.replace(before, '''  if constexpr (GROUP4 && KPS == 1 && N > 128) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int r = 0; r < 4; ++r) afrag[i][r] = 0;
  }
  int prev = -1;
  const int woff =''')
    before = '''          wg_wait<1>();
          // the previous stage's MMAs are complete after the first wait of this stage'''
    assert source.count(before) == 1
    source = source.replace(before, '''          wg_wait<1>();
          // Extend the previous stage's A lifetime past its completion wait.
          // This is a compiler operand fence, not another GPU instruction.
          if constexpr (KPS == 1 && N > 128) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
              for (int r = 0; r < 4; ++r)
                asm volatile("" : "+r"(afrag[((it ^ 1) & 1) * 4 + j][r]) :: "memory");
          }
          // the previous stage's MMAs are complete after the first wait of this stage''')
    before = '''  wg_wait<0>();
  rel_sync();
  if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev);
  }  // !TRI'''
    assert source.count(before) == 1
    source = source.replace(before, '''  wg_wait<0>();
  if constexpr (GROUP4 && KPS == 1 && N > 128) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        asm volatile("" : "+r"(afrag[i][r]) :: "memory");
  }
  rel_sync();
  if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev);
  }  // !TRI''')
    return source


def wide_batch_staged_native(source: str) -> str:
    """Compile-time A-bank stages avoid dynamic-index local arrays."""
    source = wide_batch_fenced_native(source)
    start = source.index('  for (int it = 0; it < KS / KPS; ++it) {', source.index('#define MOE_HOIST 1'))
    end = source.index('\n  wg_wait<0>();', start)
    original_loop = source[start:end]
    assert original_loop.endswith('\n  }')
    body = original_loop[original_loop.index('\n') + 1:-4]
    assert body.count('((KPS == 1 ? it : kk) & 1)') == 2
    body = body.replace('((KPS == 1 ? it : kk) & 1)', 'AP')
    assert body.count('((it ^ 1) & 1)') == 1
    body = body.replace('((it ^ 1) & 1)', '(AP ^ 1)')
    staged = '''  if constexpr (GROUP4 && KPS == 1 && N > 128) {
    auto wide_stage = [&]<int AP>(int it) {
'''+body+'''
    };
    static_assert((KS % 2) == 0);
    for (int it = 0; it < KS; it += 2) {
      wide_stage.template operator()<0>(it);
      wide_stage.template operator()<1>(it + 1);
    }
  } else {
'''+original_loop+'''
  }'''
    return source[:start]+staged+source[end:]


def wide_batch_drained_native(source: str) -> str:
    """Complete each wide group before A reuse; no overlapping wide groups."""
    source = wide_batch_native(source)
    before = '''          wg_wait<1>();
          // the previous stage's MMAs are complete after the first wait of this stage'''
    assert source.count(before) == 1
    return source.replace(before, '''          if constexpr (KPS == 1 && N > 128) wg_wait<0>();
          else wg_wait<1>();
          // Wide batches drain before A-register reuse; narrow overlap is unchanged.
          // the previous stage's MMAs are complete after the first wait of this stage''')
