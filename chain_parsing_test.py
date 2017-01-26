import memory_util

log = open('test_data/tanh_chain.stderr').read()
assert memory_util.peak_memory(log) == 6000008
assert len(memory_util.memory_timeline(log)) == 28
