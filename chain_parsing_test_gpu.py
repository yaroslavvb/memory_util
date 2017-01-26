import memory_util

log = open('test_data/tanh_chain_gpu.stderr').read()
#memory_util.print_memory_timeline(log, ignore_less_than_bytes=0)
assert memory_util.peak_memory(log) == 6001664
assert len(memory_util.memory_timeline(log)) == 30
