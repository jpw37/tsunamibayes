  
 --------------------------------------------
 Physics Parameters:
 -------------------
    gravity:   9.8100000000000005     
    density water:   1025.0000000000000     
    density air:   1.1499999999999999     
    ambient pressure:   101300.00000000000     
    earth_radius:   6367500.0000000000     
    coordinate_system:           2
    sea_level:   0.0000000000000000     
  
    coriolis_forcing: F
    theta_0:   0.0000000000000000     
    friction_forcing: T
    manning_coefficient:   2.5000000000000001E-002
    friction_depth:   1000000.0000000000     
  
    dry_tolerance:   1.0000000000000000E-003
  
 --------------------------------------------
 Refinement Control Parameters:
 ------------------------------
    wave_tolerance:  0.10000000000000001     
    speed_tolerance:   1000000000000.0000        1000000000000.0000        1000000000000.0000        1000000000000.0000        1000000000000.0000        1000000000000.0000     
    Variable dt Refinement Ratios: T
 
  
 --------------------------------------------
 SETDTOPO:
 -------------
    num dtopo files =            0
  
 --------------------------------------------
 SETTOPO:
 ---------
    mtopofiles =            1
    
    /nobackup/archive/grp/fslg_tsunami/push_forward_adjoint_nephi/etopo.tt3                                                                               
   itopotype =            3
   mx =         3120   x = (   116.00208333333350      ,   128.99791666770651      )
   my =         2880   y = (  -1.9979166666665000      ,   9.9979166676264999      )
   dx, dy (meters/degrees) =    4.1666666669999998E-003   4.1666666669999998E-003
  
   Ranking of topography files  finest to coarsest:            1
  
  
 --------------------------------------------
 SETQINIT:
 -------------
 /nobackup/archive/grp/fslg_tsunami/push_forward_adjoint_nephi/adjoint/hump.xyz                                                                        
   
 Reading qinit data from
 /nobackup/archive/grp/fslg_tsunami/push_forward_adjoint_nephi/adjoint/hump.xyz                                                                        
   
  
 --------------------------------------------
 Multilayer Parameters:
 ----------------------
    check_richardson: T
    richardson_tolerance:  0.94999999999999996     
    eigen_method:           4
    inundation_method:           2
    dry_tolerance:   1.0000000000000000E-003
