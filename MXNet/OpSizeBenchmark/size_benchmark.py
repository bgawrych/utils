import argparse
import csv
import inspect
import time
from pathlib import Path

import mxnet as mx
from mxnet import profiler


parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='./benchmarks/', help='path to directory where to save results')
parser.add_argument('--profile', action='store_true', help='whether to run MXNet profiler')
args = parser.parse_args()

save_path = args.dir
try:
    Path(save_path).mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"Failed to create/open directory {save_path}")

########################################################
################### HELPER FUNCTIONS ###################
########################################################
class named_lambda:
    def __init__(self, name, lambda_expr):
        self.name = name
        self.lambda_expr = lambda_expr
    
    @property
    def __name__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        return self.lambda_expr(*args, **kwargs)

    @property
    def __code__(self):
        return self.lambda_expr.__code__


def benchmark_it(func, data_generator):
    def data_header():
        header = inspect.getargs(func.__code__).args
        header = header[:len(next(data_generator()))] + ['time']
        return header

    def data_desc(*data):
        row = []
        for d in data:
            if isinstance(d, mx.np.ndarray):
                row += [d.shape]
            elif isinstance(d, list):
                row += [str(data_desc(*d))]
            else:
                row += [d]
        return row

    func_name = func.__name__
    print(f'{func_name}... ', end='', flush=True)

    with open(f'{save_path}/{func_name}.csv', 'w') as file:
        writer = csv.writer(file, delimiter=";", quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(data_header())
        for data in data_generator():
            tic = time.time()
            out = func(*data)
            if isinstance(out, list):
                [o.wait_to_read() for o in out]
            else:
                out.wait_to_read()
            elapsed_time = time.time() - tic
            row = data_desc(*data) + [elapsed_time]
            writer.writerow(row)
    print("done", flush=True)






#########################################################
#################### DATA GENERATORS ####################
#########################################################
def reshape_data():
    for i in range(1, 50):
        x = mx.np.random.uniform(-1, 1, (1,12, i, 64))
        for dst_shape in [(1, 12, -1), (1, -1, 12)]:
            yield (x, dst_shape)

def split_data():
    for i in range(1, 20):
        for j in range(1, 100):
            data = mx.np.random.uniform(-1, 1, (1, j*3, i*36))
            num_outputs = 3
            for ax in [-1, -2]:
                yield (data, num_outputs, ax)

def stack_data():
    for i in range(1, 50):
        data1 = mx.np.random.uniform(-1, 1, (1, 12, i, 64))
        data2 = mx.np.random.uniform(-1, 1, (1, 12, i, 64))
        for ax in range(0, 3):
            yield ([data1, data2], ax)

def slice_data():
    data = mx.np.random.uniform(-1, 1, (300, 300, 300))
    for a0 in range(1, 199, 10):
        for a1 in range(1, 199, 10):
            for a2 in range(1, 199, 10):
                for step in range(1, 100, 10):
                    r0 = [a0, a0 + step]
                    r1 = [a1, a1 + step]
                    r2 = [a2, a2 + step]
                    yield (data, r0, r1, r2)
            yield (data, r0, r1, [None, None])
        yield (data, r0, [None, None], r2)
    yield (data, [None, None], r1, r2)


def softmax_data():
    for i in range(1, 32):
        for j in range(1, 10):
            data = mx.np.random.uniform(-1, 1, (12, i, 64*j))
            for ax in range(3):
                yield (data, None, ax, 0.7)

def masked_softmax_data():
    for i in range(1, 32):
        for j in range(1, 10):
            data = mx.np.random.uniform(-1, 1, (12, i, 64*j))
            mask = mx.np.tril(mx.np.ones((12, i, 64*j))).astype(bool)
            for ax in range(3):
                yield (data, mask, ax, 0.7)

def fc_data():
    for i in range(1, 128, 3):
        for j in range(1, 10):
            data1 = mx.np.random.uniform(-1, 1, (i, j*64))
            data2 = mx.np.random.uniform(-1, 1, (i, j*64))
            data3 = mx.np.random.uniform(-1, 1, (j*64, j*64))
            yield (data1, data2, None, i)
            yield (data1, data3, None, j*64)

def batch_dot_data():
    for i in range(1, 128, 3):
        for j in range(1, 10):
            data1 = mx.np.random.uniform(-1, 1, (1, i, j*64))
            data2 = mx.np.random.uniform(-1, 1, (1, i, j*64))
            yield (data1, data2, None, i)

def swapaxes_data():
    for i in range(1, 32):
        for j in range(1, 12):
            data = mx.np.random.uniform(-1, 1, (1, j, i, 64))
            for a in range(4):
                for b in range(a + 1, 4):
                    yield (data, a, b)

def where_data():
    for i in range(1, 32):
        for j in range(1, 12):
            condition = mx.np.tril(mx.np.ones((j, i, 64))).astype(bool)
            iftrue = mx.np.random.uniform(-1, 1, (1, j, i, 64))
            iffalse = mx.np.random.uniform(-1, 1, (1, j, i, 64))
            yield (condition, iftrue, iffalse)

def layernorm_data():
    for i in range(1, 32):
        for j in range(1, 12):
            shape = [j, i, 64]
            data = mx.np.ones(shape)
            for ax in range(3):
                gamma = mx.np.ones((shape[ax],))
                beta = mx.np.ones((shape[ax],))
                yield (data, gamma, beta, ax)


def binary_data():
    for i in range(2, 512, 8):
        for j in range(2, 12):
            for bcast in [None, 0, 1, 2]:
                s0 = [j, i, 64]
                s1 = s0[:]
                if bcast is not None:
                    s1[bcast] = 1
                data0 = mx.np.ones(tuple(s0))
                data1 = mx.np.ones(tuple(s1))
                yield (data0, data1)





########################################################
#################### RUN BENCHMARKS ####################
########################################################
if args.profile:
    profiler.set_config(profile_all=True, aggregate_stats=True)
    profiler.set_state('run')

benchmark_it(mx.np.reshape, reshape_data)
benchmark_it(mx.np.split, split_data)
benchmark_it(mx.np.stack, stack_data)
benchmark_it(named_lambda('slice', lambda x, ax0, ax1, ax2: x[ax0[0]: ax0[1], ax1[0]: ax1[1], ax2[0]: ax2[1]]), slice_data)
benchmark_it(mx.np.concat, stack_data)
benchmark_it(mx.np.swapaxes, swapaxes_data)
benchmark_it(mx.np.where, where_data)
benchmark_it(mx.npx.layer_norm, layernorm_data)
benchmark_it(mx.npx.fully_connected, fc_data)
benchmark_it(mx.npx.batch_dot, batch_dot_data)
benchmark_it(mx.npx.softmax, softmax_data)
benchmark_it(mx.npx.masked_softmax, masked_softmax_data)
benchmark_it(named_lambda('add', lambda a,b : a+b), binary_data)
benchmark_it(named_lambda('substract', lambda a,b : a-b), binary_data)
benchmark_it(named_lambda('multiply', lambda a,b : a*b), binary_data)
benchmark_it(named_lambda('divide', lambda a,b : a/b), binary_data)

if args.profile:
    profiler.set_state('stop')
    print(profiler.dumps(reset=True))

