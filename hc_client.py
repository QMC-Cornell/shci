import numpy as np
import socket
import subprocess
import time
import json
import sys


class HcClient(object):
    """RPC client for H*c with SHCI"""

    def __init__(self, nProcs=1, runtimePath='.', shciPath='./shci', port=2018, verbose=True):
        self.nProcs = nProcs
        self.shciPath = shciPath
        self.port = port
        self.verbose = verbose
        self.runtimePath = runtimePath

    def startServer(self):
        print('Preparing SHCI Hc server...')
        config = open('config.json').read()
        config = json.loads(config)
        config['hc_server_mode'] = True
        with open('config.json', 'w') as config_file:
            json.dump(config, config_file, indent=2)
        cmd = 'mpirun -n %d %s' % (self.nProcs, self.shciPath)
        serverProcess = subprocess.Popen(
            cmd, cwd=self.runtimePath, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
        ready = False
        for line in iter(serverProcess.stdout.readline, ''):
            line = line.strip()
            if self.verbose:
                print(line)
            if line == 'Hc server ready':
                ready = True
            elif ready is True:
                self._n = int(line)
                print('n:', self._n)
                self._server = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM)
                self._server.connect(('127.0.0.1', self.port))
                return

        raise RuntimeError('Server failed to start.')

    def getN(self):
        return self._n

    def getCoefs(self):
        self._server.send('getCoefs')
        coefs = self._recvDoubleArr()
        return coefs

    def Hc(self, arr):
        self._server.send('Hc')
        res = self._server.recv(32)
        if res != 'ACK':
            raise RuntimeError('Server does not ack.')
        if np.iscomplexobj(arr):
            resReal = self.Hc(arr.real)
            resImag = self.Hc(arr.imag)
            return resReal + resImag * 1j
        else:
            self._server.send(arr.tobytes())
            res = self._recvDoubleArr()
            return res

    def exit(self):
        self._server.send('exit')
        self._server.close()

    def _recvDoubleArr(self):
        res = self._server.recv(8 * self._n)
        while len(res) < 8 * self._n:
            res += self._server.recv(8 * self._n - len(res))
        res = np.frombuffer(res, dtype=np.float64)
        return res


if __name__ == '__main__':
    # Test Hc = lam * c
    client = HcClient(nProcs=1)
    client.startServer()

    coefs = client.getCoefs()
    print('Coefs:')
    print(coefs)

    Hc = client.Hc(coefs)
    print("Hc:")
    print(Hc)

    eigenValue = Hc[0] / coefs[0]
    HcExpected = coefs * eigenValue

    try:
        np.testing.assert_allclose(Hc, HcExpected, atol=1e-2)
    except:
        print("Hc beyond threshold")
        client.exit()
        exit(0)

    for i in range(100):
        Hc = client.Hc(Hc)
    print("H^100 * c:")
    print(Hc)

    client.exit()
