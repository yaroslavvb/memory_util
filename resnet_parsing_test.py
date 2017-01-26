import memory_util

log = open('test_data/cifar_resnet.stderr').read()
#memory_util.print_memory_timeline(log, ignore_less_than_bytes=0)
peak_memory = memory_util.peak_memory(log)
print("Peak memory: "+str(peak_memory))
assert peak_memory>500000000 and peak_memory < 600000000
