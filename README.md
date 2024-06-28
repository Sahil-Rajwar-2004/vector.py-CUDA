<h1 align="center" style="font-size: 3em">vector.py</h1>

# Usage:

```python
import vector_cuda as vc
x = vc.vector([1,2,3,4],device = "cuda")
y = vc.vector([1,2,3,4])                    # by default the device will be set as CPU

print(x)                                    # output: <vector object at 0x7f10f82f7fd0 size=4 device=cuda>
print(x.array())                            # output: [1 2 3 4]

print(y)                                    # output: <vector object at 0x7f10d4fdc310 size=4 device=cpu>
print(y.array())                            # output: [1 2 3 4]
```


