
from dtools.starter1 import *
import yt
from dtools.math import volavg
import dtools.davetools as dt
import dtools.math.power_spectrum as ps
reload(ps)

def apodize1(rho2,projax=0):
    shape=np.array(rho2.shape)
    baseshape=(1.*shape).astype('int')
    base = np.zeros(baseshape)
    start = baseshape//2-shape//2

    base[start[0]:(start[0]+shape[0]), start[1]:(start[1]+shape[1])] = rho2

    if 1:
        #works pretty well.
        x1 = np.arange(baseshape[0])
        x2 = np.arange(baseshape[1])
        sigma_conv=2
        g1 = np.exp(-x1**2/(2*sigma_conv**2))**6
        g2 = np.exp(-x2**2/(2*sigma_conv**2))**6
        window = np.outer(g1,g2)

    window/=window.sum()

    a = np.fft.fftn(base)
    b = np.fft.fftn(window)
    c = a*b
    rho2a = np.fft.ifftn(c)
    q=np.abs(rho2a.imag).sum()/rho2a.imag.size
    if q>1e-13:
        print("!!!!!!!!!!!!!!!!!Imaginary",q)
    return rho2a.real

    


def get_cubes(sim,frame,do_rho_4=False):
    ds = yt.load("/data/cb1/Projects/P49_EE_BB/%s/DD%04d/data%04d"%(sim, frame, frame))
    print('get cg')
    cg = ds.covering_grid(0, [0.0]*3, [512]*3)

    dds = cg.dds
    rho_full = cg["density"].v
    rho = volavg.volavg(rho_full,rank=3,refine_by=2)
    output = rho_full, rho
    if do_rho_4:
        rho_4 = volavg.volavg(rho,rank=3,refine_by=2)
        output = rho_full, rho, rho_4

    return output



def plot_set(ftool,outname,axis=0):
    fig,ax=plt.subplots(2,2)
    ax0=ax[0][0];ax1=ax[0][1];ax2=ax[1][0];ax3=ax[1][1]
    ax0.imshow(ftool.rho.sum(axis=axis),interpolation='nearest',origin='lower')
    ax1.plot(ftool.ps2.kcen,ftool.ps2.power,c='r',label='p2d')
    ax1.set(xscale='log',yscale='log')
    fig.savefig(outname)



def plot_power(arr,ax):
    a = np.roll(arr,arr.shape[0]//2,axis=0)
    a = np.roll(a  ,arr.shape[1]//2,axis=1)
    a = np.abs(a)
    norm = mpl.colors.LogNorm(vmin=a[a>0].min(),vmax=a.max())
    cmap = copy.copy(mpl.cm.get_cmap('viridis'))
    cmap.set_under('w')
    ax.imshow(a,norm=norm, cmap=cmap)
def plot_set(ftool,outname,other=None):
    fig,ax=plt.subplots(2,2,figsize=(10,10))
    ax0=ax[0][0];ax1=ax[0][1]
    ax2=ax[1][0];ax3=ax[1][1]
    kwargs={'interpolation':'nearest', 'origin':'lower'}
    ax0.imshow(ftool.rho2,**kwargs)
    nhat=ftool.ps2.Nhat
    plot_power(nhat,ax=ax2)
    another_axis = np.mod(ftool.projax+1,3)
    sl = [slice(None)]*3
    sl[another_axis]=0
    #ax3.imshow( ftool.rho.sum(0),**kwargs)
    nhat=ftool.ps3.Nhat[tuple(sl)]
    plot_power(nhat,ax=ax3)
    plot_brunt(ftool,ax=ax1)
    ax0.set(title='projected density')
    ax2.set(title='proj. FFT')
    ax3.set(title='FFT3d[%s=0]'%'xyz'[another_axis])
    if other is not None:
        y=other.ps2.avgpower
        ok = y>1e-16
        ax1.plot(other.ps2.kcen[ok], y[ok], c='k')
        y=other.ps3.avgpower
        ok = y>1e-16
        ax1.plot(other.ps3.kcen[ok], other.ps3.avgpower[ok], c='k', linestyle='--')
    fig.savefig(outname)
def plot_brunt(ftool,outname=None, fitrange=None, method=None, ax=None):

    sigmas_full(ftool)
    savefig=False
    if ax is None:
        savefig=True
        fig,ax=plt.subplots(1,1)

    if fitrange is None:
        mask = slice(None)
    else:
        mask = slice(fitrange[0],fitrange[1])

    
    M1 = ftool.ps2.power>1e-16
    M2 = M1
    M3 = ftool.ps3.power>1e-16

    ax.plot( ftool.ps2.kcen[M1], ftool.ps2.avgpower[M1],c='m',label='P2d/V2d')
    ax.plot( ftool.ps3.kcen[M3], ftool.ps3.avgpower[M3],c='r',label='P3d/V3d')
    ax.set(xscale='log',yscale='log')
    ax.legend(loc=1)
    error = 1-ftool.ratio_1
    text_x=0.05
    text_y=0.35
    dy=0.07
    dx = 0.3
    ax.text(text_x,    text_y-0*dy,r"$\sigma_{x3}\ %5.3f $ "%ftool.sigma_x3d, transform=ax.transAxes)
    ax.text(text_x+dx, text_y-0*dy,r"$\sigma_B   \ %5.3f $ "%ftool.sigma_Brunt,  transform=ax.transAxes)
    ax.text(text_x,    text_y-1*dy,r"$ratio      \ %5.3f $ "%(ftool.ratio_1),  transform=ax.transAxes)
    ax.text(text_x+dx, text_y-1*dy,r"$error      \ %5.3f $ "%error,  transform=ax.transAxes)
    ax.text(text_x,    text_y-2*dy,r"$\sigma_{k3}\ %5.3f $ "%ftool.sigma_k3d, transform=ax.transAxes)
    ax.text(text_x+dx, text_y-2*dy,r"$\sigma_{k2k}\ %5.3f $ "%ftool.sigma_k2dk, transform=ax.transAxes)
    ax.text(text_x,    text_y-3*dy,r"$\sigma_{x2}\ %5.3f $ "%ftool.sigma_x2d, transform=ax.transAxes)
    ax.text(text_x+dx, text_y-3*dy,r"$\sigma_{k2}\ %5.3f $ "%ftool.sigma_k2d, transform=ax.transAxes)
    ax.text(text_x,    text_y-4*dy,r"$R          \  %5.3f $ "%(1./ftool.Rinv), transform=ax.transAxes)
    print("Sx3 %0.3f  SB %0.3f"%(ftool.sigma_x3d,ftool.sigma_Brunt))
    print("rat %0.3f  er %0.3f"%(ftool.ratio_1,error))
    print("Sk3 %0.3f  Sk2k %0.3f"%(ftool.sigma_k3d,ftool.sigma_k2dk))
    print("Sx2 %0.3f  Sk2 %0.3f"%(ftool.sigma_x2d, ftool.sigma_k2d))



    if savefig:
        fig.savefig(outname)

def sigmas_2donly(self):
    dx = self.ps2.dx
    da=dx**2
    self.sigma_x2d = np.sqrt(((self.rho2)**2*da).sum().real)
    self.sigma_k2d = np.sqrt((self.ps2.power*self.ps2.dk).sum().real) #kludge
    self.sigma_k2dk= np.sqrt((2* self.ps2.kcen*self.ps2.power*self.ps2.dk**2).sum())
    self.Rinv = self.sigma_k2dk.real/self.sigma_k2d.real
    self.sigma_Brunt = self.sigma_x2d.real*self.Rinv

def sigmas_full(self):
    dx = self.ps2.dx
    dv=dx**3
    da=dx**2
    self.sigma_x2d = np.sqrt(((self.rho2)**2*da).sum().real)
    self.sigma_k2d = np.sqrt((self.ps2.power*self.ps2.dk).sum().real) #kludge
    self.sigma_k2dk= np.sqrt((2* self.ps2.kcen*self.ps2.power*self.ps2.dk**2).sum())
    self.Rinv = self.sigma_k2dk.real/self.sigma_k2d.real
    self.sigma_Brunt = self.sigma_x2d.real*self.Rinv

    self.sigma_x3d = np.sqrt((self.rho**2*dv).sum().real)
    self.sigma_k3d = np.sqrt((self.ps3.power*self.ps3.dk).sum().real)
    self.ratio_1 =self.sigma_Brunt/self.sigma_x3d
    #just to check that everything works right, do it with the actual 3d power spectrum.
    #R2 should be 1
    self.Rinv_actual = np.sqrt(self.ps3.power.sum()/self.ps2.power.sum())
    self.sigma_Brunt_actual = self.sigma_x2d*self.Rinv_actual
    self.ratio_2 = self.sigma_Brunt_actual/self.sigma_x3d



class fft_tool():
    def __init__(self,rho, rho2=None):
        self.rho=rho
        self.rho2=rho2
        self.rho2p=None
        self.done2=False
        self.done3=False
        self.projax=None

    def do3(self):
        self.ps3 = ps.powerspectrum(self.rho)

    def do2(self,projax=0, apodize=0):
        self.projax=projax
        if self.rho2 is None:
            self.rho2=self.rho.sum(axis=projax)
        if apodize == 1:
            self.rho2 = apodize1(self.rho2)

        self.ps2 = ps.powerspectrum(self.rho2)

