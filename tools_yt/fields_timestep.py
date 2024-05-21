import yt
import numpy as np
from dtools.tools_yt.yt_names import *
def dt_rk2(field,data):
    """Casting all components to code units eliminates some source of reduced precision that
    was being incurred during thing 1/dx step (I think?)"""
    aye=1.0
    if data.ds['ComovingCoordinates']:
        aye=(data.ds['CosmologyInitialRedshift']+1)/\
            ( data.ds['CosmologyCurrentRedshift']+1)

    if hasattr(data.dds,'in_units'):
        dx, dy, dz = data.dds.in_units('code_length').v
    else:
        dx, dy, dz = data.dds
    dxinv = 1./(aye*dx)
    dyinv = 1./(aye*dy)
    dzinv = 1./(aye*dz)
    
    if data.ds['EquationOfState'] == 1:
        cs2 = data.ds['IsothermalSoundSpeed']**2
    else:
        cs2 = (data['sound_speed'].in_units('code_velocity').v)**2
    Bx = data['Bx']
    By = data['By']
    Bz = data['Bz']
    if hasattr(Bx,'in_units'):
        Bx = Bx.in_units('code_magnetic').v
        By = By.in_units('code_magnetic').v
        Bz = Bz.in_units('code_magnetic').v
    vx = data['velocity_x'].in_units('code_velocity').v
    vy = data['velocity_y'].in_units('code_velocity').v
    vz = data['velocity_z'].in_units('code_velocity').v
    rho = data['density'].in_units('code_density').v
    B2 = Bx*Bx+By*By+Bz*Bz

    temp1 = cs2 + B2/rho
    ca2 = Bx*Bx/rho;
    cf2 = 0.5 * (temp1 + np.sqrt(temp1*temp1 - 4.0*cs2*ca2));
    cf = np.sqrt(cf2);
    v_signal_x = (cf + np.abs(vx));
    GridRank = 3 #fix later
    if GridRank > 1:
      ca2 = By*By/rho;
      cf2 = 0.5 * (temp1 + np.sqrt(temp1*temp1 - 4.0*cs2*ca2));
      cf = np.sqrt(cf2);
      v_signal_y = (cf + np.abs(vy));
    
    if GridRank > 2:
      ca2 = Bz*Bz/rho;
      cf2 = 0.5 * (temp1 + np.sqrt(temp1*temp1 - 4.0*cs2*ca2));
      cf = np.sqrt(cf2);
      v_signal_z = (cf + np.abs(vz));
    
    dt_x = v_signal_x * dxinv;
    dt_y = v_signal_y * dyinv
    dt_z = v_signal_z * dzinv
    
    dt = np.max([dt_x, dt_y, dt_z],axis=0);
    dt = data.ds.parameters['CourantSafetyNumber'] /dt
    dt = data.ds.arr(dt,'code_time')
    return dt

def dt_enzo(field,data):
    """Casting all components to code units eliminates some source of reduced precision that
    was being incurred during thing 1/dx step (I think?)"""
    if data.ds['EquationOfState'] == 1:
        Cs = data.ds['IsothermalSoundSpeed']
    else:
        Cs = (data['sound_speed'].in_units('code_velocity').v)
    aye=1.0
    if data.ds['ComovingCoordinates']:
        aye=(data.ds['CosmologyInitialRedshift']+1)/\
            ( data.ds['CosmologyCurrentRedshift']+1)

    if hasattr(data.dds,'in_units'):
        dx, dy, dz = data.dds.in_units('code_length').v
    else:
        dx, dy, dz = data.dds
    #using harmonic mean.
    dti  = (Cs + np.abs(  data['velocity_x'].in_units('code_velocity').v )) /dx
    dti  += (Cs + np.abs( data['velocity_y'].in_units('code_velocity').v ))/dy
    dti  += (Cs + np.abs( data['velocity_z'].in_units('code_velocity').v ))/dz
    dti /= data.ds.parameters['CourantSafetyNumber'] #yes divided: still in reciporical space
    dt = data.ds.arr(aye/dti, 'code_time')
    return dt
def add_dt(obj):
    obj.add_field(YT_dt_enzo,function=dt_enzo, validators=[yt.ValidateGridType()],sampling_type='cell', units='code_time') #, units='dimensionless')
    obj.add_field(YT_dt_rk2,function=dt_rk2, validators=[yt.ValidateGridType()],sampling_type='cell', units='code_time') #, units='dimensionless')
