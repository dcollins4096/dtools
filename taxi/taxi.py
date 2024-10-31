"""The taxi object, for keeping track of yt things."""
""" used to be called uber, but that got taken. """
#checkrel
from dtools.starter1 import *
import warnings
not_ported = False
import yt
array=np.array
#import h5py
import types, time,weakref
#import dave_callbacks
from dtools.davetools import   ensure_list
import dtools.davetools as dt

def load(filename):
    try:
        filename.index('/')
        actual_filename = filename
    except:
        actual_filename = "taxi_stand/%s"%filename
    fptr = open(actual_filename,'r')
    taxi_type = "'yellow'"
    for line in fptr:
        a, b = line.split("=")
        if a.strip() == 'self.taxi_type':
            taxi_type = b.strip()
    if taxi_type == "'yellow'":
        this_taxi = taxi()
        this_taxi.read(actual_filename)
    elif taxi_type == "'bb'":
        this_taxi = bb_taxi()
        this_taxi.read(actual_filename)
    elif taxi_type == "'athena'":
        this_taxi = athena_taxi()
        this_taxi.read(actual_filename)
    else:
        print("UNKNOWN TAXI TYPE %s"%taxi_type)
        pdb.set_trace()
        this_taxi=None
    return this_taxi


class PortException(Exception):
    def __init__(self, value = ""):
        self.value = "Taxi function not yet ported" + value
    def __str__(self):
        return repr(self.value)
class OperationException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class dummy_YTArray():
    """YTarrays need to have datasets associated.  This is a container that assumes you know what you're doing.
    Has a *value* and *units*, the latter stored as a string.
    This is for easy caching of things to text.
    This might need to be cleaned up, there are probably better ways to handle this."""
    def __init__(self,value=0,units=None):
        self.units = units
        if hasattr(value,'units'):
            self.value=value.v
            self.units=value.units
        #elif isinstance(value,types.TupleType) or isinstance(value,types.ListType)
        elif type(value) is tuple or type(value) is list: #isinstance(value,types.TupleType) or isinstance(value,types.ListType):
            if type( value[-1]) is str: #isinstance(value[-1], types.StringType)
                self.value = value[:-1]
                if len(self.value) == 1:
                    self.value=self.value[0]
                    

                self.units= value[-1]
            else:
                self.value = value
                
        else:
            self.value=value
            self.units=units
        self.v = self.value
    def smarten(self,ds):
        verb = ds.quan
        try:
            if len(self.value) > 1:
                verb = ds.arr
        except:
            pass
        return verb(self.v,self.units)
    def __str__(self):
        return str(self.value) + " " + str(self.units)
    def __repr__(self):
        return repr(self.value) + " " + repr(self.units)


def writefunction(thing):
    """
    taxi.read(file) populates itself from file by executing each line in the file.
    writefunction(thing) returns a string that, when executed, repopulates uber.
    Assumes all numpy nd arrays are 1d.  This can be fixed with recursion, but I'm lazy."""

    output = ""
    if isinstance(thing, dummy_YTArray):
        output += "dummy_YTArray("
        output += writefunction(thing.v)
        output += ",'%s')"%thing.units
    elif isinstance(thing, yt.YTArray):
        output += "dummy_YTArray("
        if hasattr(thing,'size') and thing.size > 1:
            output+="["+str(thing[0].v)
            for L in thing[1:]:
                output += ","+str(L.v)
            output += "]"
        else:
            output += str(thing.v)

        output += ", '%s')"%str(thing.units)

    elif type(thing) is dict: #isinstance(thing, types.DictType)
        output += "{"
        keys = sorted(thing.keys()); 
        if len(keys):
            key=keys[0]
            output += '"%s":%s'%(key,writefunction(thing[key]))
            for key in keys[1:]:
                output += ',"%s":%s'%(key,writefunction(thing[key]))
        output += "}"

    elif type( thing) is list: #isinstance(thing, types.ListType)
        output += "["
        for L in thing:
            output += writefunction(L)
            output += ", "

        #There are probably too many commas.
        if len(thing) > 0:
            output = output[0:-2]
        output += "]"
    elif type(thing) is str: #isinstance(thing, types.StringType)
        output += "'"+thing+"'"
    elif isinstance(thing, np.ndarray):
        output += "np.array("
        if thing.size > 1:
            output += "[%s"%str(thing[0])
            for L in thing[1:]:
                output += ",%s"%str(L)
            output += "])"
        else:
            output += "%s"%str(thing)
    else:
        output += str(thing)
    return output

class taxi:
    """Taxi is a container for yt options, because the author is both forgetful and lazy.
    Primarily use for repeatability and reduced startup for plotting simple things that I do
    all the time, with default values suitably chosen.  See the source code for the options
    and defaults.  The won't be repeated here because there are several, and that's asking
    for a documentation inconsistency.
    """
    
    def smarten(self,value, units=None, frame=None):
        if self.ds is None or frame is not None:
            self.load(frame)
        return dummy_YTArray(value,units).smarten(self.ds)
    def __init__(self, filename=None,dir=None, name=None,**kwargs):
        """
        Either reads in taxifile *fileame* or just sets some defaults."""

        #List of members that will NOT be written.  

        self.ExcludeFromWrite = ['ExcludeFromWrite']
        self.ExcludeFromWrite.append('data')
        

        #To make the file easier to use, some of the uber members are written first
        self.taxi_type='yellow'
        self.WriteMeFirst = ['name','directory','outname','outdir','operation','frames','fields','axis',
                            'name_syntax', 'name_files','name_dir','GlobalParameters',
                             'ProfileDir','ProfileName', 'taxi_type']
        self.ExcludeFromWrite.append('WriteMeFirst')
        self.ExcludeFromWrite.append('profile_data')
        self.ExcludeFromWrite.append('last_prof')

        #Some members may need special treatment.
        self.WriteSpecial = {}
        self.ExcludeFromWrite.append('WriteSpecial')
        #externals
        self.OutputLog = "OutputLog"
        self.plot_origin = 'domain'  #see yt docs.
        self.axis = 0                #Axis for the plot
        self.frames = 'all'             #Dump numbers to be plotted.
        self.fields = 'Density'      #Field (hopefully to be list) for the plot
        self.weight_field = None           #Weight field.  Projections?    
        self.name   = 'sim'          #identifier for this taxi instance
        self.outname = 'Image'       #Prefix for output
        self.outdir = plot_dir
        self.directory = '.'         #Directory where the simulation was run. (Where the DD* dirs are)
        self.ProfileDir= "./ProfileFiles/"
        self.ProfileName = None
        self.set_log = {}
        self.clobber_plot = True     #check plots against plot_log_<outname>.txt
        self.subdir = True           #Are there DD0000 directories, or is it just data0000.grid*?
        self.operation = 'Full'      #The operation that will be done.  taxi.operations() for a list.
                                     #    or physically motivated ones.  Options are 'Code' or 'Physics'
        self.format = 'png'
        self.plot_args = {}
        #The actual yt objects.  Not saved on output.
        self.ds = None               #The lagos parameter file.  Gets re-made at each plot.
        self.ExcludeFromWrite.append('ds')

        self.pc = None               #The plot collection.
        self.ExcludeFromWrite.append('pc')
        self.index = None                #A weak proxy to ds.h
        self.ExcludeFromWrite.append('index')
        self.h = None                #A weak proxy to ds.h
        self.ExcludeFromWrite.append('h')
        self.reg = None              #Weak proxy to the most recent region
        self.ExcludeFromWrite.append('reg')
        self.proj = None             #Weak proxy to the most recent projection
        self.ExcludeFromWrite.append('proj')

        self.extra_save = False       #Forces an extra plot save to get colormaps correct.
        self.ExcludeFromWrite.append('extra_save')        

        self.ExcludeFromWrite.append('region')
        self.last_particle_index_step = None
        self.ExcludeFromWrite.append('last_particle_index_step')
        self.last_particle_indices = None
        self.ExcludeFromWrite.append('last_particle_indices')
        self.field_parameters = {}
        self.left=np.array([0.0,0.0,0.0])      #The left of the region extraction.
        self.right=np.array([1.0,1.0,1.0])     #The right of the extracted region.
        self.radius=None
        self.height=None
        self.restrict = False                   #Restric plot to img_left, img_right
        self.img_left = None                   #left edge of image
        self.img_right = None                  #right edge of image.
        self.img_width = None                  #width of image.
        self.width = None
        self.zoom_sequence=None
        self.region_type = 'all'
        self.center= 0.5*(self.left+self.right)#Center, used for all things that need a center
        self.periodic = False              #For projection shift
        self.cmap = {'default':None}       #color map, field driven. Everyone uses default.
        self.Colorbar = None               #Monotone: monotonically increaseing, forced to min,max
        self.use_colorbar = True           #use color bar for plots.  Bad naming.
        self.zlim = {}
        self.slice_zlim = {}
        self.proj_zlim = {}
        self.callbacks = []                #Which callbacks to use.  taxi.callbacks() for options
        self.callback_args={'particles':{'args':[1],'kwargs':{'col':'r'}},'select_particles':{'args':1,'kwargs':{'col':'y'}}}
        self.callback_args['star_particles']={'args':[1],'kwargs':{'col':'g'}}
        self.callback_args['nparticles']= {'args': [[0.03, 0.03]], 'kwargs': {'coord_system': 'axis', 'text_args': {'color': 'red'}}}
        #self.callback_args['magnetic_field']={'kwargs':{'use_streamlines':True,'plot_args':{'color':'b'}}}
        self.callback_args['magnetic_field']={'kwargs':{'plot_args':{'color':'b'}}}
        self.callback_args['velocity']={'kwargs':{'use_streamlines':False,'plot_args':{'color':'y'}}}
        #self.ExcludeFromWrite.append('callbacks')
        self.WriteSpecial['callbacks'] = self.write_callbacks

        self.dt = None
        self.MaxLevel=-1                   #max analysis level, currently does nothing
        
        #for strings on the image
        self.UnitMultiplier = 1.0          #Info for the PrintTime callback.  Kind of a hack.
        self.UnitName = 'FemtoParsec'      #Info for the PrintTime callback.  Kind of a hack.
        
        #for clump reading
        self.clump_prefix = None
        self.clumps = None
        self.ExcludeFromWrite.append('clumps')
        
        #periodic?
        self.Periodic = False              #For periodically shifting boxes when plotting.
        
        #internals
        self.old_outname = None
        self.frame_template = None
        self.basename_template = "data"
        self.name_syntax = 'outputlog'  #outputlog or preset
        self.name_files = "data"        #prefix on data file
        self.name_dir = "DD"  #Prefix on said directories
        self.extrema = {}
        self.frame_dict = None
        self.ExcludeFromWrite.append('frame_dict')
        self.basename = None
        self.verbose = True
        self.mark_time = True
        #for storing global parameters
        #Updates each parameter file on read.
        self.GlobalParameters = {}

        #quan box keeps track of a wide array of quantities.
        self.ExcludeFromWrite.append('qb')
        self.qb=None

        #Covering grid.
        self.cg = None
        self.ExcludeFromWrite.append('cg')
        self.ExcludeFromWrite.append('the_plot')
        self.hold_values={}

        self.particle_filter_names=[]  #Filters get added directly to the ds, e.g.
        #http://yt-project.org/doc/cookbook/calculating_information.html?highlight=formed_star

        self.ExcludeFromWrite.append('derived_fields')
        self.derived_fields={} #Derived fields can also be added to the ds directly.
        #This is not fully developed, syntax should be sth like
        #self.derived_fields['field_name'] = {'function':{dict of args}}
        #which then gets called on ds as its loaded.
 
        #dummy YTArrays are just for easy disk writing.  They get promoted to YTArrays 
        #once there's a ds, here we keep track of which ones need to be promoted.
        self.dummy_YTArray_list=[] 


        if filename != None:
            print("USE THE MAIN taxi.load FUNCION")
            return None
            #If no directories are mentioned, assume the actual file is ./taxi_stand/filename
            try:
                filename.index('/')
                actual_filename = filename
            except:
                actual_filename = "taxi_stand/%s"%filename
            self.read(actual_filename)
        if dir != None:
            self.directory=dir
        if name != None:
            self.name = name
            self.outname = name
        self.__dict__.update(**kwargs)

    def read(self,filename='DefaultFile'):
        """Simply executes each line in the file *filename* in the namespace of this uber."""

        file = open(filename,"r")
        for line in file:
            exec(line)
        for key in self.__dict__:
            if isinstance(self.__dict__[key],dummy_YTArray):
                self.dummy_YTArray_list.append(key)
        file.close()

    def save(self,filename='DefaultFile'):
        """Writes out the taxi file.  Writes the entire contents of taxi.__dict__,
        making a hopefully appropriate assumption about the type."""
        try:
            filename.index('/')
            actual_filename = filename
        except:
            actual_filename = "taxi_stand/%s"%filename
        file = open(actual_filename,"w")
        for i in self.WriteMeFirst:
            file.write("self."+i + " = " + writefunction(self.__dict__[i]) + "\n")            

        for i in self.__dict__.keys():
            if i in self.ExcludeFromWrite:
                continue
            if i in self.WriteMeFirst:
                continue
            if i in self.WriteSpecial.keys():
                continue
            file.write("self."+i + " = " + writefunction(self.__dict__[i]) + "\n")
        for i in self.WriteSpecial.keys():
            file.write( self.WriteSpecial[i]() )
        file.close()
        print("saved", actual_filename)

    def write_callbacks(self):
        """checkrel"""
        output = []
        for i in self.callbacks:
            if type(i) is str: #isinstance(i,types.StringType):
                output.append(i)
        return "self.callbacks = " + writefunction(output)


    def __str__(self):
        out = ""
        for i in self.WriteMeFirst:
            out += "self."+i + " = " + writefunction(self.__dict__[i]) + "\n"
        for i in self.__dict__.keys():
            if i not in self.ExcludeFromWrite and i not in self.WriteMeFirst:
                out += "self."+i + " = " + writefunction(self.__dict__[i]) + "\n"
        return out

    #
    #
    # filenames are a mild hassle.
    # Either they can be preset, e.g DD????/data???? where DD and data are variables.
    # OR can be taken sequentially from the OutputLog.
    # 

    def set_filename(self,frame=None):
        if self.name_syntax == 'preset':
            self.set_filename_preset_syntax(frame)
        elif self.name_syntax == 'outputlog':
            self.set_filename_from_outputlog(frame)
        elif self.name_syntax == 'athena':
            self.basename = "%s/parthenon.prim.%05d.phdf"%(self.directory,frame)
            
        else:
            print("Error: Poorly defined name syntax.  set oober.name_syntax = 'preset'||'outputlog'")
            raise


    def return_filename(self,frame=None):
        self.set_filename(frame)
        return self.basename

    def set_filename_preset_syntax(self,frame=None):
        if frame == None:
            frame = self.return_frames()[-1]
        #take 1: no subdirectory (very old style)
        if glob.glob(self.basename+".hierarchy") == []:
            self.basename_template = self.directory+'/'+self.name_files+'%04i'
            self.basename = self.basename_template %(frame)
            self.basename_template = self.directory + "/" + self.name_dir + '%04i/'\
                                     + self.name_files + "%04i"
            self.basename = self.basename_template %(frame,frame)
        if self.verbose == True:
            print("==== %s ===="%(self.basename))

    def print_frames(self):
        self.get_frames()
        if self.name_syntax == 'preset':
            print("oober.name_syntax == 'preset'.  Nothing to list")
        elif self.name_syntax == 'outputlog':
            self.get_frames()
            nframe = self.frame_dict.keys()
            nframe.sort()
            output_format = "%s%d%s"%("%(dsname)",self.frame_dict_string_length ,"s %(cycle)d %(time)0.6e")

            for n in nframe:
                print("%4d"%n, output_format%(self.frame_dict[n]))
        else:
            print("set oober.name_syntax = 'preset'||'outputlog'")

    def returnsetnumber(self,frame=None):
        return frame
        if self.name_syntax == 'preset':
            return frame
        else:
            self.get_frames()
            if frame==None:
                frame = self.return_frames()[0]
            return self.frame_dict[frame]['SetNumber']

    def get_frames(self):
        """parses OutputLog to populate self.frame_dict"""
        DirSetNumber = re.compile(r'([^\d]*)(\d\d\d\d)/([^\d]*)(\d\d\d\d)')
        SetNumber = re.compile(r'([^\d]*)(\d\d\d\d)')
        SetNumber = re.compile(r'(.*)(\d\d\d\d)')
        debug = 0
        if self.name_syntax == 'preset':
            self.frame_dict={}
            return
        if not hasattr(self,'frame_dict') or self.frame_dict is None:
            self.frame_dict = {}
            output_name = "%s/%s"%(self.directory , self.OutputLog)
            output_log = open(output_name, 'r')
            print("OutputName", self.name, output_name)

            if debug > 0:
                print(output_log)
            lines = output_log.readlines()
            self.frame_dict_string_length = 0
            for nframe,line in enumerate(lines):
                if debug>0:
                    print(nframe, line[:-1])
                linesplit = dt.no_whites(line.split(" "))
                self.frame_dict[ nframe ] = {}
                self.frame_dict[ nframe ]['dsname'] = linesplit[2]
                self.frame_dict_string_length = max( self.frame_dict_string_length , len(linesplit[2]) )

                try:
                    self.frame_dict[ nframe ]['cycle']  = int(linesplit[3])
                except:
                    self.frame_dict[ nframe ]['cycle']  = -1
                match = DirSetNumber.match(linesplit[2])
                match2 = SetNumber.match(linesplit[2])
                if match is not None:
                    self.frame_dict[nframe]['DirPrefix']=match.group(1)
                    self.frame_dict[nframe]['name_files'] = match.group(3)
                    self.frame_dict[nframe]['SetNumber'] = int(match.group(2))
                elif match2 is not None:
                    self.frame_dict[nframe]['DirPrefix']=None
                    self.frame_dict[nframe]['name_files'] = match2.group(1)
                    self.frame_dict[nframe]['SetNumber'] = int(match2.group(2))
                else:
                    print("error in set parsing")
                    print(linesplit[2])

                    self.frame_dict[nframe]['DirPrefix']=None
                    self.frame_dict[nframe]['name_files']  =None
                    self.frame_dict[nframe]['SetNumber']=-1

                try:
                    self.frame_dict[ nframe ]['time']   = float(linesplit[4])
                except:
                    self.frame_dict[ nframe ]['time']   = -1
            self.frame_list = nar(sorted(self.frame_dict.keys()))
            self.times = nar([self.frame_dict[nframe]['time'] for nframe in self.frame_list])
    def get_movie_frames(self,dt=None,thresh=1e-7):
        self.get_frames()
        if dt==None:
            first_frame = min(self.frame_list)
            self.load(first_frame)
            dt = self.ds['dtDataDump']
        mask = (self.times % dt)/dt < thresh
        return mask
    def get_tff(self):
        self.load()
        self.G = self.ds['GravitationalConstant']/(4*np.pi)
        self.tff = np.sqrt(3*np.pi/(32*G*1))
        return self.tff


    def set_filename_from_outputlog(self,frame=None):
        self.get_frames()
        if frame == None:
            frame = self.return_frames()[-1]
        self.basename = "%s/%s"%(self.directory,self.frame_dict[frame]['dsname'])
        if self.verbose == True:
            print("==== %s ===="%(self.basename))

    def load(self, frame = None):
        """populate parameter file, plot collection, hierarchy, region if desired.
        Possibly could be renamed 'load' """
        self.set_filename(frame)
        self.ds = yt.load(self.basename)

        for filter_name in self.particle_filter_names:
            self.ds.add_particle_filter(filter_name)

        for field in self.derived_fields:
            field_stuff = self.derived_fields[field]
            print("LOADING FIELD",field)
            if type(field_stuff) is dict:
                self.ds.add_field(field,**self.derived_fields[field])
            else:
                field_stuff(self.ds)
        for key in self.dummy_YTArray_list:
            dya = self.__dict__[key] 
            #if it's a dummy array, or even it it's suggest that it should be,
            #make it a smarter YTArray.
            self.__dict__[key] = dummy_YTArray(dya).smarten(self.ds)
        self.dummy_YTArray_list = [] #only need to do this once.
        return self.ds

    def get_region(self, frame=None):
        if frame is not None or self.ds is None:
            self.load(frame)
        if self.region_type.lower()=='sphere':
            reg = self.ds.sphere(self.center, self.radius)
        if self.region_type.lower() in ['rectangle','region']:
            self.center = 0.5*(self.left + self.right)
            reg = self.ds.region(self.center, self.left,self.right)
        if self.region_type.lower() in ['all','all_data']:
            reg = self.ds.all_data()
        if self.region_type.lower() in ['disk']:
            print("OK!")
            reg = self.ds.disk(self.center,self.normal,self.radius,self.height)
        if self.region_type.startswith('grid'):
            """gridN gets grid N."""
            N = int( self.region_type[4:])
            g = self.ds.index.grids[N]
            center = 0.5*(g.LeftEdge+g.RightEdge)
            reg = self.ds.region(center,g.LeftEdge,g.RightEdge)
        for parameter in self.field_parameters.keys():
            reg.set_field_parameter(parameter, self.field_parameters[parameter])
        #self.reg  = weakref.proxy(reg)
        return reg

    def arg_setter(self,kwargs):
        """For updating arguments by way of the function call."""
        for arg in kwargs.keys():
            if arg.startswith("_"):
                continue
            if self.__dict__.has_key(arg):
                self.__dict__[arg] = kwargs[arg]


    def get_all_frames(self):
        if not hasattr(self,'frame_dict') or self.frame_dict is None:
            self.get_frames()
        all_frames = sorted(self.frame_dict.keys())
        return all_frames
    def return_frames(self):
        if self.name_syntax == 'preset':
            if type( self.frames) is not list: #isinstance(self.frames, types.ListType)
                print("With preset name syntax, frames must be a list of integers.")
                raise
            return self.frames
        if self.name_syntax == 'athena':
            return self.frames
        all_frames=self.get_all_frames()
        #pdb.set_trace()
        return_frames = 'type error'
        if type(self.frames) is str: #isinstance(self.frames,types.StringType):
            if self.frames=='all':
                return_frames = all_frames
            elif self.frames.startswith('every'):
                interval = int(self.frames.split(" ")[-1])
                return_frames = all_frames[::interval]
                if return_frames[-1] != all_frames[-1]:
                    return_frames += all_frames[-1:]

            elif self.frames == 'last':
                return_frames = all_frames[-1:]
            elif self.frames == 'all_reverse':
                return_frames = all_frames[::-1]
            elif self.frames == 'movie_frames':
                mask = self.get_movie_frames(self.dt)
                #pdb.set_trace()
                return_frames = list(self.frame_list[mask])
        elif type(self.frames) is slice: # isinstance(self.frames, types.SliceType):
            return_frames = all_frames[self.frames]
        else:
            return_frames = self.frames

        return return_frames

    def plot(self,local_frames=None, **kwargs):
        self.arg_setter(kwargs)
        """Makes the uber plot.  Options for plot can be found in the uber description,
        plot operations can be found by typing uber.operations()"""
        start_time = time.time()    
        
        frame_template = "%s/%s_%s"%(self.outdir,self.outname, "_%04i")
        FirstOne = True #For monotonic plots.

        if local_frames==None:
            local_frames=self.return_frames()

        for frame in ensure_list(local_frames):

            self.load(frame=frame)

            print("==== %s ===="%(self.basename))

            #Find the density peak for all field,axis plots for this snapshot.
            if self.operation == 'DensityPeakSlice':
                print("getting peak")
                value,center = self.ds.find_max('Density')
                self.center = np.array(center)

            for field in ensure_list(self.fields):

                #Find the center, if the center is to come from the min/max.
                #If a multi plot, use the first plot given.  
                
                if self.operation == 'PeakSlice':
                    value,center = self.ds.find_max( ensure_list(field)[0] )
                    self.center = np.array(center)
                elif self.operation == 'MinSlice':
                    value,self.center = self.ds.find_min( ensure_list(field)[0] )
                    self.center = np.array(center)

                #start axing
                for axis in ensure_list(self.axis):
                    #if multiple plots are used, do_plot will return a figure object.
                    plot_or_fig = self.do_plots(field,axis,FirstOne)

                    if field in self.set_log: #.has_key(field):
                        plot_or_fig.set_log( field, self.set_log[field])

                    if hasattr(plot_or_fig,'savefig'):
                        
                        filename = frame_template%(self.returnsetnumber(frame))
                        filename += "_"+self.operation
                        filename += "_"+['x','y','z'][axis]

                        for i in field:
                            filename += "_"+i
                        if self.format:
                            filename += "."+self.format
                        else:
                            filename += ".png"
                        print(filename)
                        #fig.savefig(filename)
                        #dsave(fig,filename,field_name=field,ds_list=[self.basename],script_name='uber')
                        plot_or_fig.savefig(filename)
                        print(filename)
                    else:
                        plot_or_fig.set_origin('domain')
                        if self.zoom_sequence is not None:
                            for nz,width in enumerate(self.zoom_sequence):
                                plot_or_fig.set_width(width)
                                this_frame_template = frame_template + "_zoom%02d"%nz
                                print(plot_or_fig.save(this_frame_template%frame))
                        else:
                            print(plot_or_fig.save(frame_template%frame))
    def set_center_max(self,field='density', frame=None):
        if self.ds is None or frame is not None:
            self.load(frame)
        value,center = self.ds.find_max(field)
        self.center = np.array(center)
    def set_center_min(self,field='density', frame=None):
        if self.ds is None or frame is not None:
            self.load(frame)
        value,center = self.ds.find_min(field)
        self.center = np.array(center)
    def do_plots(self,sub_field,axis,FirstOne):

        """Adds the plot to the plot collection.  Loops over plots for multi-plots.
        FirstOne tracks the first plot, for colorbar purposes"""
        sub_field_list = ensure_list(sub_field)
        orient = 'horizontal' #hard coded for devel!  
        use_colorbar = self.use_colorbar #local copy made for multi plots
        extra_args = self.plot_args

        #set up the multi plot if this field is a list.
        if len(sub_field_list) > 1:
            config = guess_multi_configuration(len(sub_field_list))
            #fig, axes, colorbars = raven.get_multi_plot( config[0], config[1], colorbar=orient, bw = 4)
            fig, axes, colorbars = get_multi_plot( 2, 1, colorbar=orient, bw = 4)
            use_colorbar = False 
            extra_args = {'figure':fig,'axes':0}
        for num, field in enumerate(sub_field_list):
            if 'axes' in extra_args: #.has_key('axes'):
                #extra_args['axes'] = axes[wp(num)[0]][wp(num)[1]]
                extra_args['axes'] = wp(axes,num) #axes[wp(num)[0]][wp(num)[1]]

            if self.operation == 'multi_test':
                the_plot = yt.SlicePlot(self.ds,axis,field,center=self.center,origin=self.plot_origin,**extra_args)
                #not tested.
                #the_plot = self.pc.add_slice(field,axis, use_colorbar=use_colorbar,
                #                             **extra_args)
            elif self.operation == 'Full':
                the_plot = yt.ProjectionPlot(self.ds,axis,field, center=self.center, weight_field=self.weight_field, **extra_args)
                #self.pc.add_projection(field,axis, use_colorbar=use_colorbar,
                #                            **extra_args)
                self.zlim = self.proj_zlim

            elif self.operation == 'RegionProjection':
                #setup a projection
                reg = self.get_region()
                self.proj = self.ds.proj(field,axis,center=self.center,data_source=reg, weight_field=self.weight_field) #,periodic=self.periodic)
                the_plot = self.proj.to_pw()
                self.zlim = self.proj_zlim
            elif self.operation in ['MinSlice','PeakSlice','DensityPeakSlice','CenterSlice']:
                """peak taken outside the axis loop"""
                the_plot = yt.SlicePlot(self.ds,axis,field,center=self.center,origin=self.plot_origin,**extra_args)
                self.zlim = self.slice_zlim
            else:
                raise OperationException(self.operation)

            #this is to clean up old implementations
            if not self.cmap:
                self.cmap = {'default':None}
            if field not in self.cmap:#.has_key(field):
                self.cmap[field] = self.cmap['default']

            the_plot.set_cmap( field, self.cmap[field] )
            #the_plot.label_kws['size'] = 'x-large'

            if self.Colorbar:
                field_with_weight = field
                if self.weight_field is not None:
                    field_with_weight += "_%s"%self.weight_field
                do_log = True #please get this from the yt object.
                ok_zones = np.isnan(the_plot.data_source[field]) == False
                if do_log:
                    ok_zones = np.logical_and(ok_zones, the_plot.data_source[field] != 0)
                if ok_zones.sum() > 0:
                    this_min = the_plot.data_source[field][ok_zones].min()
                    this_max = the_plot.data_source[field][ok_zones].max()
                if self.Colorbar in ['Monotone', 'monotonic']:
                    if FirstOne:
                        self.zlim[field_with_weight] = [this_min,this_max]
                    else:
                        self.zlim[field_with_weight][0] = min([self.zlim[field_with_weight][0],this_min])
                        self.zlim[field_with_weight][1] = max([self.zlim[field_with_weight][1],this_max])
                    if self.verbose:
                        print("set lim", self.zlim[field_with_weight])
                elif self.Colorbar in  ['Fixed', 'fixed'] :
                    if field_with_weight not in  self.zlim:#.has_key(field_with_weight)
                        do_log = True #please get this from the yt object.
                        self.zlim[field_with_weight] = [this_min,this_max]
                    if self.verbose:
                        print("set lim", self.zlim[field_with_weight])

                if hasattr(self.zlim[field_with_weight][0],'v'):
                    the_plot.set_zlim(field, self.zlim[field_with_weight][0].v, self.zlim[field_with_weight][1].v)
                else:
                    the_plot.set_zlim(field, self.zlim[field_with_weight][0], self.zlim[field_with_weight][1])
                if self.operation in ['Full','RegionProjection']:
                    self.proj_zlim = self.zlim
                elif self.operation in ['MinSlice','PeakSlice','DensityPeakSlice','CenterSlice']:
                    self.slice_zlim = self.zlim

        self.the_plot = the_plot #this does actually need to be a list.
        FirstOne = False
        try:
            self.add_callbacks(the_plot, axis=axis)
        except:
            raise
        if self.restrict:
            if self.width:
                the_plot.set_width(self.width)                    

        if len(sub_field_list) > 1:
            raise PortException("multiplots")
            for p,cax in zip(self.pc.plots, colorbars):
                # Now we make a colorbar, using the 'image' attribute of the plot.
                # 'image' is usually not accessed; we're making a special exception here,
                # though.  'image' will tell the colorbar what the limits of the data are.
                cbar = cb.Colorbar(cax, p.image, orientation=orient)
                # Now, we have to do a tiny bit of magic -- we tell the plot what its
                # colorbar is, and then we tell the plot to set the label of that colorbar.
                p.colorbar = cbar
                p._autoset_label()                
            return fig
            
        else:
            return the_plot

    def product_dir(self,frame):
        """Destination of intermediate products.  Typically of the form 
        simdir/DD0001.products
        """
        if self.name_syntax == 'preset':
            return self.pdir_old(frame)
        elif self.name_syntax == 'outputlog':
            self.get_frames()
            return "%s/%s%04d.products"%(self.directory,
                    self.frame_dict[frame]['DirPrefix'],
                    self.frame_dict[frame]['SetNumber'])


    def make_cg(self,frame=None,field=None, num_ghost_zones=0):
        """Makes a covering grid for the root grid."""
        if field == None:
            field = self.fields[0]
        self.load(frame)
        left=[0.0]*3
        resolution = self.ds['TopGridDimensions'] + 2*num_ghost_zones
        self.cg=self.ds.covering_grid(0,left,resolution,num_ghost_zones=num_ghost_zones)#,fields=fields)

    def extract_root(self,frame=None,field=None,make_cg=True,num_ghost_zones=0,debug=0,dtype='float32'):
        """Extracts a grid the size of the root grid.  
        Cached in self.directory+".products"+"cube_<field>.single" exists, reads that.
        *num_ghost_zones* is for fields that require derivatives, etc.
        Currently defaults to first frame, all fields"""
        if frame == None:
            frame = self.return_frames()[0]
        if field == None:
            field = self.fields[0]
        directory = self.product_dir(frame)
        filename_no_ghost = "%s/cube_%s.%s"%(directory,field,dtype)
        if debug > 0:
            print(filename_no_ghost)
        remove_gz = False
        if num_ghost_zones >= 0:
            filename = "%s.%d"%(filename_no_ghost,num_ghost_zones)
        else:
            """If num_ghost_zones < 0, try to read with ngz=0.  If not there, try larger values.
            If extant, use that and remove ghost zones."""
            filename = "%s.%d"%(filename_no_ghost,0)
            if glob.glob(filename) == []:
                with_ghost_list = glob.glob(filename_no_ghost+"*")
                if with_ghost_list != []:
                    ghost_zones_available = [ int(a.split(".")[-1]) for a in with_ghost_list]
                    ngz = min(ghost_zones_available)
                    filename = "%s.%d"%(filename_no_ghost, ngz)
                    if debug>0:
                        print("Found",filename, "with ngz",ngz)
                    remove_gz = True

        if glob.glob(filename):
            if debug > 0:
                print("open set from disk")
            file = h5py.File(filename,'r')
            set = file[field][:]
            file.close()
            if remove_gz:
                if debug>0:
                    print("stripping")
                set = set[ngz:-ngz,ngz:-ngz,ngz:-ngz]
        else:
            if glob.glob(directory) == []:
                print("making directory",directory)
                os.mkdir(directory)
            if debug > 0:
                print("Create set")
            if make_cg or self.cg == None:
                if debug > 0:
                    print("make cg")
                self.make_cg(frame,field,num_ghost_zones)
            set = self.cg[field]
            file = h5py.File(filename,'w')
            file.create_dataset(field,set.shape, data=set, dtype=dtype)
            file.close()
        return set

    def fft(self,frame=None,field=None,data=None,make_cg=True,num_ghost_zones=0,dtype='float32',debug=0,fft_func=np.fft.fftn, project=False):
        """Make an fft of a field.
        As with extract_root, look for a cached version on disk first.
        Defaults to the ghost zone guess.
        if project= 'x' or 'y' or 'z', a projection is done."""
        if dtype == 'float32':
            fft_dtype = 'complex64'
        elif dtype == 'float64':
            fft_dtype = 'complex128'
        elif dtype not in ['complex64','complex128']:
            print("Can't cast type ",dtype, "to a complex type.")
            return None
        if frame == None:
            frame = self.return_frames()[0]
        if field == None:
            field = self.fields[0]
        field_name = field
        if project:
            field_name = field + "_%s"%project

        directory = self.product_dir(frame) #"%s/%s%04d.products/"%(self.directory,self.name_dir,frame)
        filename = "%s/fft_%s.%s"%(directory,field_name,dtype)
        if debug > 0:
            print(filename)
        if glob.glob(filename):
            if debug > 0:
                print("open FFT from disk")
            file = h5py.File(filename,'r')
            fft = file[field][:]
            file.close()
        else:
            if glob.glob(directory) == []:
                print("making directory",directory)
                os.mkdir(directory)
            if debug > 0:
                print("Create FFT")
            if data == None:
                this_set = self.extract_root(frame=frame,field=field,make_cg=make_cg,num_ghost_zones=num_ghost_zones,dtype=dtype)
            else:
                this_set = data
            if project:
                this_set_3d = this_set
                this_set = np.sum(this_set_3d,'xyz'.index(project)) #this is obtuse.

            if debug > 0:
                print("Do fft.")
            fft = fft_func(this_set)/this_set.size
            if debug > 0:
                print("save")
            fptr = h5py.File(filename,'w')

            fptr.create_dataset(field,fft.shape, data=fft, dtype=fft_dtype)
            fptr.close()
        return fft

    def find_extrema(self,fields=None,frames=None,manual_positive=False):
        if fields is not None:
            local_fields = fields
        else:
            local_fields = self.fields
        if frames is not None:
            local_frames = frames
        else:
            local_frames = self.return_frames()
        for frame in local_frames:
            self.load(frame)
            reg=self.get_region(frame)
            for field in local_fields:
                if manual_positive:
                    vals = reg[field]
                    positive = vals > 0
                    this_extrema=[0,0]
                    this_extrema[0] = vals[positive].min()
                    this_extrema[1] = vals[positive].max()
                else:
                    this_extrema = reg.quantities['Extrema'](field,non_zero=False)
                if field not in self.extrema:
                    self.extrema[field] = [this_extrema[0].v, this_extrema[1].v]
                else:
                    self.extrema[field][0] = min([this_extrema[0].v, self.extrema[field][0]])
                    self.extrema[field][1] = max([this_extrema[1].v, self.extrema[field][1]])


    def count_particles(self, frame=None, ptype='all'):
        """Turns out Metadata.NumberOfParticles isn't always updated close to the output in Enzo."""
        """Also, if ds is loaded already, just leave frame as None and it won't reload"""
        nparticles = 0
        reg = self.get_region(frame)
        try: 
            nparticles = reg[ptype,'particle_index'].size
        except:
            pass
        return nparticles

    def add_callbacks(self,the_plot,axis=None):
        for i,callback in enumerate(self.callbacks):
            args = self.callback_args.get(callback,{'args':[],'kwargs':{}})
            myargs=args.get('args',[])
            mykwargs=args.get('kwargs',{})
            print("my args",myargs)
            print("my kwargs", mykwargs)
            if type( callback) is str: #isinstance(callback,types.StringType):
                if callback == 'stokes_angles':
                    self.kludge_my_package={}
                    the_plot.annotate_stokes_angles(kludge_my_package=self.kludge_my_package)
                    #the_plot.annotate_stokes_angle(myargs, **mykwargs)
                elif callback == 'pol_frac':
                    the_plot.annotate_polarized_angles()
                elif callback == 'magnetic_angles':
                    the_plot.annotate_magnetic_angles()
                elif callback == 'velocity':
                    the_plot.annotate_velocity(*myargs,**mykwargs)
                elif callback == 'magnetic_field':
                    the_plot.annotate_magnetic_field(*myargs,**mykwargs)
                elif callback == 'grids':
                    the_plot.annotate_grids()
                elif callback == 'particles':
                    nparticles = self.count_particles()
                    if nparticles>0:
                        the_plot.annotate_particles(*myargs,**mykwargs)
                elif callback == 'text':
                        the_plot.annotate_text(myargs[0],myargs[1],**mykwargs)
                elif callback == 'nparticles':
                        the_plot.annotate_text(myargs[0],r'$n_p=%d$'%self.count_particles(),**mykwargs)
                elif callback == 'halos':
                    halos = self.callback_args['halos']['halos']
                    circle_args = self.callback_args['halos'].get('circle_args',{})
                    text_args = self.callback_args['halos'].get('text_args',{})
                    id_offset = self.callback_args['halos'].get('id_offset',0)
                    if 'color' not in text_args.has_key() and 'color' in  circle_args:
                        text_args['color']=circle_args['color']

                    for halo in halos:
                        halo_position = self.ds.arr([halo['position'][0].in_units('unitary').d,
                            halo['position'][1].in_units('unitary').d,
                            halo['position'][2].in_units('unitary').d], "unitary")
                        halo_radius = self.ds.quan(halo['rvir'].in_units('unitary').d, "unitary")
                        text = int(halo['tree_id'] - id_offset)
                        #halo_sphere = self.ds.sphere(halo_position,
                        #    halo_radius)
                        the_plot.annotate_sphere(halo_position,halo_radius,text=text,circle_args=circle_args,
                                                text_args=text_args)
                elif callback == 'star_particles':
                    nparticles = self.count_particles()
                    if nparticles>0:
                        reg = self.get_region()
                        these_indices = reg['formed_star','particle_index']
                        mykwargs['ptype']='formed_star'
                        the_plot.annotate_select_particles(myargs[0], **mykwargs)
                        pargs = self.callback_args['nparticles']['args']
                        pkwargs=self.callback_args['nparticles']['kwargs']
                        the_plot.annotate_text(pargs[0],r'$n_{\rm{new}}=%d$'%these_indices.shape,**pkwargs)
                elif callback == 'time_title':
                    units  = self.callback_args.get('time_title',{'units':'Myr'}).get('units','Myr')
                    format = self.callback_args.get('time_title',{'format':'%0.2f'}).get('format','%0.2f')
                    output = r"$"
                    time = self.ds.current_time.in_units(units)
                    output += format%time
                    output += "\ %s$"%time.units
                    the_plot.annotate_title(output)
                elif callback == 'z_title':
                    output = r"$z = "
                    z = self.ds['CosmologyCurrentRedshift']
                    output += "%0.2f$"%z
                    the_plot.annotate_title(output)
                elif callback == 'new_particles':
                    """This callback does not work with multiple plots for each frame.
                    This is due to the fact that the old particle list is updated each call
                    A more sophisiticated cache system, or perhaps updating the 'last particle list'
                    in the plot loop is necessary."""
                    nparticles = self.count_particles()
                    if nparticles>0:
                        these_indices = self.get_new_indices()
                        mykwargs['indices'] = these_indices
                        mykwargs['col'] = 'r'
                        the_plot.annotate_select_particles(myargs[0], **mykwargs)
                        pargs = self.callback_args['nparticles']['args']
                        pkwargs=self.callback_args['nparticles']['kwargs']
                        the_plot.annotate_text(pargs[0],r'$n_{\rm{new}}=%d$'%these_indices.shape,**pkwargs)

                elif callback == 'magnetic_field':
                    the_plot.annotate_magnetic_field()
                else:
                    raise PortError("Callback %s not supported"%callback)
            else:
                print("Where the heck did you get that callback at?")

    def stat(self,field,frame=None):
        """print min and max of *field*"""
        ds = self.load(frame)
        minTuple = ds.find_min(field)
        maxTuple = ds.find_max(field)
        return {'min':minTuple,'max':maxTuple}

    def mstat(self,fields=None,frames=None, format = "%9.2e",Norm=False):
        """2 Min and max for all fields in self.fields.
        Field list overridden by *fields*.
        *Norm* subtracts off the volume-weighted mean."""
        print(fields)
        if frames is None:
            frames = self.return_frames()
        for frame in frames:
            if fields is None:
                fields = self.fields
            self.load(frame)
            region=self.get_region(frame)
            out = region.quantities['Extrema'](fields)
            if Norm is True:
                for n, field in enumerate(fields):
                    avg = region.quantities['WeightedAverageQuantity'](field,'CellVolume')
                    out[n] = out[n][0]-avg, out[n][1]-avg

            if Norm is not False:
                for n, field in enumerate(fields):
                    avg =  0
                    if Norm.has_key(field):
                        avg = Norm[field]
                        print(field,avg)
                    out[n] = out[n][0]-avg, out[n][1]-avg
            format_string = "%s %s %s"%(format,format,"%s")
            for n, field in enumerate(fields):
                print(format_string%(out[n][0], out[n][1], field))

    def conservation(self, frames=None, field='density', units='code_mass'):
        """This should be done in a more general manner, later."""
        if frames is None:
            frames = self.return_frames()
        mass = []
        mass_delta = []
        time = []
        volume = []
        for nf, frame in enumerate(frames):
            ad = self.load(frame).all_data()
            total_volume = ad.quantities['TotalQuantity']('cell_volume')
            total_mass=(ad.quantities['WeightedAverageQuantity'](field, 'cell_volume')*total_volume).in_units(units)
            time.append(self.ds.current_time)
            volume.append(total_volume)
            mass.append(total_mass)
            mass_delta.append( (total_mass - mass[0])/mass[0] )

        mass=nar(mass); mass_delta=nar(mass_delta); time=nar(time); volume=nar(volume)
        print("total mass deviation = sum(|delta|)", np.abs(mass_delta).sum())
        return {'time':time,'mass':mass,'volume':volume, 'mass_delta':mass_delta}
