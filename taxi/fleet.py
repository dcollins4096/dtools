#For analyzing several simulations.
from dtools.starter1 import *
import dtools.taxi.taxi as taxi

class fleet():
    def __init__(self,taxi_list=[]):
        self.taxi_list = []
        self.next_taxi_index=0
        for car in taxi_list:
            if type(car) is str:
                self.taxi_list.append(load(car))
            else:
                self.taxi_list.append(car)
        self.namelength = max([len(t.name) for t in self.taxi_list])
        self.nt = "%"+str(self.namelength+1)+"s "

    def next(self):
        this_taxi_index = self.next_taxi_index
        self.next_taxi_index += 1
        if self.next_taxi_index >= len(self.taxi_list) + 1:raise StopIteration
        return self.taxi_list[this_taxi_index]
    def __iter__(self):
        self.next_taxi_index=0
        return self
    def __getitem__(self,item):
        out = []
        if type(item) is int: #isinstance(item,types.IntType):
            out = self.taxi_list[item]
        else:
            for car in self.taxi_list:
                print(self.nt%car.name, car.__dict__[item])
                out.append(car.__dict__[item])
        return out
    def __setitem__(self,item,value ):
        for car in self.taxi_list:
            car.__dict__[item] = value
            
    def __call__(self,string, frames=None):
        """Execute arbitrary code on the cars in the taxi fleet.
        output can be used to return values."""
        output = []
        for car in self.taxi_list:
            if frames is None:
                exec(string)
            else:
                for frame in frames:
                    car.load(frame)
                    exec(string)
        return output
    def output(self,command, frames=None):
        """Call *command* on all cars, return the output as a list."""
        output = []
        for car in self.taxi_list:
            if frames is None:
                output.append(eval(command))
            else:
                for frame in frames:
                    car.load(frame)
                    output.append(eval(command))
        return output
    def outname(self,prefix):

        """sets car.outname = prefix+car.name"""
        for car in self.taxi_list:
            car.outname = prefix + car.name
    def plot(self,*args, **kwargs):
        #check for fixed colorbars.
        prior_zlim=None
        if 'prefix' in kwargs:
            self.outname(kwargs.pop('prefix'))

        for car in self.taxi_list:
            if car.Colorbar is 'fixed' and prior_zlim is not None and car.zlim is not None:
                car.zlim = prior_zlim
            car.plot(*args, **kwargs)
            prior_zlim = car.zlim
    def save(self,suffix=""):
        for car in self.taxi_list:
            thisname = car.name+suffix
            car.save(thisname)
    def allnames(self):
        ncar=len(self.taxi_list)
        return "%s_"*ncar%tuple([car.name for car in self.taxi_list])
    def dumb_profile(self,fields,**kwargs):
        for car in self.taxi_list:
            car.dumb_profile(fields,**kwargs)
    def phase(self,*args,**kwargs):
        for car in self.taxi_list:
            car.phase(*args,**kwargs)
        
    def profile(self,*args,**kwargs):
        """Runs profile on all using *args and **kwargs.
        Also plots the combined set"""
        for car in self.taxi_list:
            car.profile(*args,**kwargs)
        plt.clf()
        color_by = kwargs.pop('color_by','sim') #either color with sim or frame. The other gets line style
        ntotal = max([len(car.frames) for car in self.taxi_list])
        simtotal = len(self.taxi_list)
        if color_by=='frame':
            rm = davetools.rainbow_map(ntotal)
        elif color_by=='sim':
            rm = davetools.rainbow_map(simtotal)
        plt.clf()
        linelist = {0:'-',1:'--',2:':'}
        all_frames=[]
        plot_args={'linewidth':0.3}  #I can probably make this the same as kwargs
        prior_y = None
        for n,car in enumerate(self.taxi_list):
            all_xbins = car.profile_data['all_xbins']
            all_profiles = car.profile_data['all_profiles']
            for i,frame in enumerate(car.return_frames()):
                thex= all_xbins[i]
                they=all_profiles[i]
                if 'units' in kwargs:
                    units=kwargs['units']
                    fractional=kwargs.get('fractional',True)
                    if units[0] is not None:
                        thex = thex.in_units(units[0])
                    if units[1] is not None and fractional is not True:
                        they = they.in_units(units[1])
                plot_args['label']='%s %04d'%(car.name,frame)
                if prior_y is not None:
                    l1norm = np.mean(np.abs( they - prior_y ))/np.mean(they)
                    print(l1norm)
                    prior_y = they
                    if l1norm > 1e-8:
                        they *= 1.1
                        plot_args['label'] += "+off"
                else:
                    prior_y = they
    
                if color_by=='sim':
                    plot_args['c']=rm(n)
                    plot_args['linestyle']=linelist.get(i,'-')
                elif color_by=='frame':
                    plot_args['c']=rm(i)
                    plot_args['linestyle']=linelist.get(n,'-')
                plt.plot( thex, they,**plot_args)
                #plt.plot( all_xbins[i], all_profiles[i],c=rm(i),linestyle=linelist[n],label='%s %04d'%(car.name,frame))
                if frame not in all_frames:
                    all_frames.append(frame)
        plt.xlabel(car.profile_data['fields'][0])
        plt.ylabel(car.profile_data['fields'][1])
        plt.xscale(car.profile_data['scales'][0]); plt.yscale(car.profile_data['scales'][1])
        plt.legend(loc=0)
        frame_name = "_%04d"*len(all_frames)%tuple(all_frames)
        profname = '%s_prof_%s_%s_n%s.pdf'%(self.allnames(), car.profile_data['fields'][0], car.profile_data['fields'][1], frame_name)
        print(profname)
        plt.savefig(profname)

    def stat(self,field,frame=None):
        """print min and max of *field*"""
        ds = self.load(frame)
        minTuple = ds.find_min(field)
        maxTuple = ds.find_max(field)
        return {'min':minTuple,'max':maxTuple}


    def mstat(self,fields=None,frames=None, format = "%9.2e",Norm=False):
        """1 Min and max for all fields in self.fields.
        Field list overridden by *fields*.
        *Norm* subtracts off the volume-weighted mean."""
        print(fields)
        if frames is None:
            frames = self.return_frames()
        for frame in frames:
            if fields is None:
                fields = self.fields
            self.load(frame)
            out = self.region.quantities['Extrema'](fields)
            if len(fields) == 1:
                out = [out]
            if Norm is True:
                for n, field in enumerate(fields):
                    avg = self.region.quantities['WeightedAverageQuantity'](field,'CellVolume')
                    out[n] = out[n][0]-avg, out[n][1]-avg
            if hasattr(Norm,'has_key'):
                for n, field in enumerate(fields):
                    avg =  0
                    if Norm.has_key(field):
                        avg = Norm[field]
                        print(field,avg)
                    out[n] = out[n][0]-avg, out[n][1]-avg
            format_string = "%s %s %s"%(format,format,"%s")
            for n, field in enumerate(fields):
                print(format_string%(out[n][0], out[n][1], field))

    def find_extrema(self,fields=None,frames=None, manual_positive=False):
        all_fields = []
        for car in self: #I feel like there's an easier way to do this.
            for field in car.fields:
                if field not in all_fields:
                    all_fields.append(field)

        extrema_store={}
        for n,car in enumerate(self.taxi_list):
            car.find_extrema(fields,frames, manual_positive=manual_positive)
            for field in all_fields:
                if car.extrema.has_key(field):
                    if extrema_store.has_key(field):
                        extrema_store[field][0] = min([extrema_store[field][0],car.extrema[field][0]])
                        extrema_store[field][1] = max([extrema_store[field][1],car.extrema[field][1]])
                    else:
                        extrema_store[field]=np.zeros(2)
                        extrema_store[field][0] = car.extrema[field][0]
                        extrema_store[field][1] = car.extrema[field][1]
        for car in self.taxi_list:
            car.extrema = copy.copy(extrema_store)

