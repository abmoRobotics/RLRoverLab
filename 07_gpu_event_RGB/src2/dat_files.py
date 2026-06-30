import numpy as np
import struct
from datetime import datetime

def load_dat_event(filename, start=0, stop=-1, display=False):
    """ Load .dat events from file.
        Args:
            filename: Path of the .dat file
            start: starting timestamp (us)
            stop: if different than -1, last timestamp
            display: display file info
        Returns:
             ts, x, y, pol numpy arrays of timestamps, positions, and polarities
     """
    f = open(filename, 'rb')
    if f == -1:
        print("The file does not exist")
        return
    else:
        if display: print("Load DAT Events: " + filename)
    l = f.readline()
    all_lines = l
    while l[0] == 37:
        p = f.tell()
        if display: print(l)
        l = f.readline()
        all_lines = all_lines + l
    # f.close()
    all_lines = str(all_lines)
    # f = open(filename, 'rb')
    f.seek(p, 0)
    evType = np.uint8(f.read(1)[0])
    evSize = np.uint8(f.read(1)[0])
    p = f.tell()
    l_last = f.tell()
    if start > 0:
        t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        while t < start:
            p = f.tell()
            t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
            dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])

    if stop > 0:
        t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        while t < stop:
            l_last = f.tell()
            t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
            dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
    else:
        l_last = f.seek(0, 2)

    num_b = ((l_last - p) // int(evSize)) * 2
    f.close()
    data = np.fromfile(filename, dtype=np.uint32, count=num_b, offset=p)
    ts = data[::2]
    v = 0
    ind = all_lines.find("Version")
    if ind > 0:
        v = int(all_lines[ind+8])
    if v >= 2:
        x_mask = np.uint32(0x00007FF)
        y_mask = np.uint32(0x0FFFC000)
        pol_mask = np.uint32(0x10000000)
        x_shift = 0
        y_shift = 14
        pol_shift = 28
    else:
        x_mask = np.uint32(0x00001FF)
        y_mask = np.uint32(0x0001FE00)
        pol_mask = np.uint32(0x00020000)
        x_shift = 0
        y_shift = 9
        pol_shift = 17
    x = data[1::2] & x_mask
    x = x >> x_shift
    y = data[1::2] & y_mask
    y = y >> y_shift
    pol = data[1::2] & pol_mask
    pol = pol >> pol_shift
    if len(ts) > 0:
        if display:
            print("First Event: ", ts[0], " us")
            print("Last Event: ", ts[-1], " us")
            print("Number of Events: ", ts.shape[0])
    return ts, x, y, pol


def write_event_dat(filename, ts, x, y, pol,
                    event_type='dvs', width=None, height=None):
    """ Write the events in a .DAT file
        The file header begins with %, then event type (one byte uint8) and event lenght
        (one byte uint8), then the data are stores ts (4 bytes uint32) and x-y-pol (4 bytes uint32)
        Args:
            filename: path of the file to create
            ts: stimestamp
            x, y: positions of the pixels
            p: polarities (0 or 1)
    """
    f = open(filename, 'wb')
    if f == -1:
        print("Cannot open the file")
        return
    if event_type in ['dvs', 'cd', 'td']:
        f.write(bytes("% Data file containing DVS events.\n", encoding='utf8'))
    elif event_type in ['aps', 'em']:
        f.write(bytes("% Data file containing EM events.\n", encoding='utf8'))
    else:
        raise Exception("Specify a valid event type: 'dvs', 'cd', 'td', 'aps' or 'em'")

    f.write(bytes("% Version 2\n", encoding='utf8'))
    f.write(bytes("% Date " + str(datetime.now().replace(microsecond=0)) + '\n', encoding='utf8'))

    if width is None:
        width = x.max() + 1
    if height is None:
        height = y.max() + 1
    f.write(bytes("% Height " + str(height) + '\n', encoding='utf8'))
    f.write(bytes("% Width " + str(width) + '\n', encoding='utf8'))

    f.write(bytes(np.uint8([0])))  # Event Type
    f.write(bytes(np.uint8([8])))  # Event length
    arr = np.zeros(2 * ts.shape[0], dtype=np.uint32)
    arr[::2] = ts
    x_mask = np.uint32(0x00007FF)
    y_mask = np.uint32(0x0FFFC000)
    pol_mask = np.uint32(0x10000000)
    x_shift = 0
    y_shift = 14
    pol_shift = 28
    buf = np.array(x, dtype=np.uint32) << x_shift
    arr[1::2] += x_mask & buf
    buf = np.array(y, dtype=np.uint32) << y_shift
    arr[1::2] += y_mask & buf
    buf = np.array(pol, dtype=np.uint32) << pol_shift
    arr[1::2] += pol_mask & buf
    arr.tofile(f)
    f.close()

if __name__ == '__main__':
    ts, x, y, pol =  load_dat_event("ev_100_10_100_40_0.4_0.01.dat", start=0, stop=-1, display=True)
    print(ts.shape)
