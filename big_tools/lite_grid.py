from dtools.starter1 import *

class lg():
    def __init__(self,grid_id):
        self.grid_id=grid_id

class fake_hierarchy():
    def __init__(self,base,frame):

        self.base = base
        self.frame=frame
        self.fname= '%s/DD%04d/data%04d.hierarchy'%(base,frame,frame)
        self.grids=[]
    def parse(self):
        fptr = open(self.fname,'r')
        lines = fptr.readlines()
        fptr.close()
        
        grid_id = None
        this_grid=None
        for line in lines:
            san = line.strip()
            if len(san) == 0:
                continue
            pair = line.split('=')
            key = pair[0].strip()
            val = pair[1].strip()
            if key=='Grid':
                grid_id = int(pair[1])
                this_grid = lg(grid_id)
            elif key in ['GridLeftEdge','GridRightEdge']:
                three = nar(val.split()).astype('float')
                this_grid.__dict__[key] =three
            elif key.startswith('BaryonFileName'):
                this_grid.fname = val
            elif key.startswith('Pointer') and this_grid is not None:
                #the last entry
                self.grids.append(this_grid)
                this_grid=None

def get_cube(this_h, TopGridDimensions, field):

    output = np.zeros(TopGridDimensions)
    for ng,ggg in enumerate(this_h.grids):
        h5ptr = h5py.File('%s/%s'%(this_h.base,ggg.fname),'r')
        group = h5ptr['Grid%08d'%ggg.grid_id]
        dset = group[field][()].swapaxes(0,2)
        h5ptr.close()
        left =  (ggg.GridLeftEdge*TopGridDimensions).astype('int')
        right = (ggg.GridRightEdge*TopGridDimensions).astype('int')
        output[left[0]:right[0],left[1]:right[1],left[2]:right[2]]=dset
    return output

def proj(this_h,top_grid_size,field,axis,fname):
    N = top_grid_size
    dim1 = [1,0,0][axis]
    dim2 = [2,2,1][axis]
    output = np.zeros([top_grid_size,top_grid_size])
    for ng,ggg in enumerate(this_h.grids):
        print('grid',ng)

        left =  (ggg.GridLeftEdge*N).astype('int')
        right = (ggg.GridRightEdge*N).astype('int')
        h5ptr = h5py.File('%s/%s'%(this_h.base,ggg.fname),'r')
        group = h5ptr['Grid%08d'%ggg.grid_id]
        dset = group[field][()].swapaxes(0,2)
        h5ptr.close()
        proj = dset.sum(axis=axis)
        output[left[dim1]:right[dim1],left[dim2]:right[dim2]] += proj
    do_log=True
    if do_log:
        output = np.log10(output)
    #mpl.image.imsave(fname,output)
    plt.clf()
    plt.imshow(output.T)
    plt.savefig(fname)






                




