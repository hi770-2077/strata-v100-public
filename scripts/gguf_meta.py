import struct, sys

def read_string(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', errors='replace')

def read_value(f, vtype):
    if vtype == 8:
        return read_string(f)
    if vtype == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, et) for _ in range(n)]
    fmt = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
    return struct.unpack(fmt[vtype], f.read(struct.calcsize(fmt[vtype])))[0]

def read_gguf_meta(path, max_keys=400):
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', magic
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        print(f"version={version} tensors={n_tensors} kv={n_kv}")
        for _ in range(min(n_kv, max_keys)):
            key = read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            val = read_value(f, vtype)
            if isinstance(val, str) and len(val) > 90:
                val = val[:90] + '...'
            if isinstance(val, list) and len(val) > 20:
                val = val[:20] + [f'...({len(val)} total)']
            if key.startswith('qwen4exp') or key == 'general.architecture':
                print(f"{key} = {val}")

read_gguf_meta(sys.argv[1])
