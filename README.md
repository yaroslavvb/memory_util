# memory_util

This utility parses LOG_MEMORY messages out of TensorFlow vlog output and builds timeline of memory allocations

## Example usage:

```
import memory_util
memory_util.vlog(1)
import tensorflow as tf

sess = tf.Session()
a = tf.random_uniform((1000))
b = tf.random_uniform((1000))
c = a + b
with memory_util.capture_stderr() as stderr:
    sess.run(c.op)
memory_util.print_memory_timeline(stderr)
```
