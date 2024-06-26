#Take a complex field and force the 
#symmetry on it so that its fourier transform is real.
#It's really pretty stupid, you can just use a real fft.
#But I did it, so here it is.
from dtools.starter1 import *

def cube(N,rank,rando=False):

    m = np.arange(N**rank)+1
    m.shape=tuple([N]*rank)
    p = np.random.random(N**rank)*2*np.pi-np.pi
    p.shape=tuple([N]*rank)
    b = m*np.exp(1j*p)
    rhat = np.fft.fftn(p)
    chat = np.fft.fftn(b)
    if rando:
        return b,chat
    else:
        return p,rhat
    

class make_kvec():
    def __init__(self,N,rank):
        k = [list((np.fft.fftfreq(N)*N).astype('int'))]*rank
        self.kvec  = nar(np.meshgrid(*k,indexing='ij'))
        self.kind = tuple(self.kvec)
        self.minusk = tuple(-self.kvec)
        return 

def sym3d(arr):

    N = arr.shape[0]
    rank = len(arr.shape)
    coord = make_kvec(N,rank)

    #The origin must be real
    ok = (coord.kvec[0]==0)*(coord.kvec[1]==0)*(coord.kvec[2]==0)
    arr[ok]=np.abs(arr[ok])

    #the bulk
    ok = (coord.kvec[2]<0)*(coord.kvec[2]>-N//2) 
    arr[ok]=np.conj(arr[coord.minusk][ok])

    #The Z face
    ok = (coord.kvec[2]==0)*(coord.kvec[1]<0)*(coord.kvec[1]>-N//2)
    arr[ok]=np.conj(arr[coord.minusk][ok])

    #the edge
    ok = (coord.kvec[2]==0)*(coord.kvec[1]==0)*(coord.kvec[0]<0)
    arr[ok]=np.conj(arr[coord.minusk][ok])

    #The N//2 stripe has no conjugate, must be real.
    if np.mod(N,2) == 0 and True:
        ok = (coord.kvec[0]==-N//2)+(coord.kvec[1]==-N//2)+(coord.kvec[2]==-N//2)
        arr[ok] = np.abs(arr[ok])

        #the plane
        ok = (coord.kvec[2]==-N//2)*(coord.kvec[1]<0)*(coord.kvec[1]>-N//2)
        arr[ok]=np.conj(arr[coord.minusk][ok])

        #a stripe
        ok = (coord.kvec[2]==-N//2)*(coord.kvec[1]==-N//2)*(coord.kvec[0]<0)
        arr[ok]=np.conj(arr[coord.minusk][ok])
        ok = (coord.kvec[2]==-N//2)*(coord.kvec[1]==0)*(coord.kvec[0]<0)
        arr[ok]=np.conj(arr[coord.minusk][ok])

        #the plane
        ok = (coord.kvec[2]==-N//2)*(coord.kvec[1]==0)*(coord.kvec[0]<0)
        arr[ok]=np.conj(arr[coord.minusk][ok])


        #The Z face
        ok = (coord.kvec[2]==0)*(coord.kvec[1]==-N//2)*(coord.kvec[0]<0)
        arr[ok]=np.conj(arr[coord.minusk][ok])



def sym2d(arr):

    N = arr.shape[0]
    rank = len(arr.shape)
    coord = make_kvec(N,rank)
    #the bulk.
    ok = (coord.kvec[1]<0)*(coord.kvec[1]>-N//2) + (coord.kvec[1]==-N//2)*(coord.kvec[0]<0)
    arr[ok]=np.conj(arr[coord.minusk][ok])

    #The Y face
    ok = (coord.kvec[1]==0)*(coord.kvec[0]<0)
    arr[ok]=np.conj(arr[coord.minusk][ok])

    #The N//2 stripe has no conjugate, must be real.
    if np.mod(N,2) == 0:
        ok = (coord.kvec[0]==-N//2)+(coord.kvec[1]==-N//2)#+(kvec[2]==N//2)
        arr[ok] = np.abs(arr[ok])

    #The origin must be real
    ok = (coord.kvec[0]==0)*(coord.kvec[1]==0)
    arr[ok]=np.abs(arr[ok])

def test(arr):
    N = arr.shape[0]
    rank = len(arr.shape)
    coord = make_kvec(N,rank)
    diff = np.abs(arr[coord.kind]-np.conj(arr[coord.minusk]))
    if rank == 2:
        return diff
    if rank == 3:
        return diff.sum(axis=2)


