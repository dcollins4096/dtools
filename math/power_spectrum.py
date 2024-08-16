#1
from dtools.starter1 import *
import scipy.stats

class powerspectrum():
    def __init__(self,arr):
        Nhat=np.fft.fftn(arr)
        Nhat /= arr.size #dx^n for the fft
        rhohat = np.abs(Nhat)**2
        nz = rhohat.shape[0]
        kx = np.fft.fftfreq(nz)*nz
        self.dk = kx[1]-kx[0]
        self.dx = 1/arr.shape[0]
        #self.dk = 1
        #self.dx = 1
        #print('1/dk',1/self.dk)
        #print('1/dx',1/self.dx)
        kabs = np.sort(np.unique(np.abs(kx)))
        rank = len(arr.shape)
        if rank == 2:
            kkx,kky=np.meshgrid(kx,kx)
            k = np.sqrt(kkx**2+kky**2)
        elif rank == 3:
            kkx,kky,kkz=np.meshgrid(kx,kx,kx)
            k = np.sqrt(kkx**2+kky**2+kkz**2)
        power, bins, counts =scipy.stats.binned_statistic(k.flatten(), rhohat.flatten(), bins=kabs,statistic='sum')
        bc = 0.5*(bins[1:]+bins[:-1])
        self.Nhat=Nhat  
        self.rho=arr
        self.rhohat=rhohat
        self.k = k
        self.da = self.dk**(rank)
        self.power=power.real*self.da
        self.kcen=bc
        if rank == 2:
            volume = 2*np.pi*self.kcen
        if rank == 3:
            volume = 4*np.pi*self.kcen**2
        self.avgpower = self.power/volume

#import fourier_tools_py3.fourier_filter as Filter
#class powerspectrum_old():
#    def __init__(self,array):
#        self.array=array
#        self.fft = np.fft.fftn( self.array )
#        self.power=self.fft*np.conjugate(self.fft)
#        self.power/=self.power.size
#        ff = Filter.FourierFilter(self.power)
#        self.power_1d = np.array([self.power[ff.get_shell(bin)].sum() for bin in range(ff.nx)])
#        self.Nzones = np.array([ff.get_shell(bin).sum() for bin in range(ff.nx)])
#        self.kcen=ff.get_shell_k()
