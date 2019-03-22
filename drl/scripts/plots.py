import json
import os
import sys

from matplotlib import pyplot as plt


def moving_average(values, ratio=0.99):
    result = list()
    val = values[0]
    for v in values:
        val = ratio * val + (1 - ratio) * v
        result.append(val)
    return result


if len(sys.argv) > 1:
    files = [sys.argv[-1]]
else:
    files = filter(lambda x: x.endswith('.json'), sorted(os.listdir()))
for name in files:
    print(name)
    try:
        data = json.load(open(name))
    except Exception as e:
        print(e)
    name = name.rstrip('.json')
    plt.plot(moving_average(data['stats']['r']['mean']), label=name, linewidth=1,)
    plt.legend()

plt.show()
