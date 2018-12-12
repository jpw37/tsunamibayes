  
 --------------------------------------------
 Physics Parameters:
 -------------------
    gravity:   9.8100000000000005     
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
    maxleveldeep:           3
    depthdeep:   100.00000000000000     
    Variable dt Refinement Ratios: T
 
  
 --------------------------------------------
 SETDTOPO:
 -------------
    num dtopo files =            1
    fname:/tsunami/codykesler/tsunamibayes/Model_Scripts/Data/dtopo.tt3                                                                                         
    topo type:           3
    minlevel, maxlevel:
           3           3
  
 --------------------------------------------
 SETTOPO:
 ---------
    mtopofiles =            1
    
    /tsunami/codykesler/tsunamibayes/Model_Scripts/Data/etopo.tt3                                                                                         
   itopotype =            3
   minlevel, maxlevel =            1           3
   tlow, thi =    0.0000000000000000        10000000000.000000     
   mx =          421   x = (   127.49166666666700      ,   134.49166666680699      )
   my =          421   y = (  -9.5083333333330007      ,  -2.5083333331930007      )
   dx, dy (meters/degrees) =    1.6666666667000000E-002   1.6666666667000000E-002
  
   Ranking of topography files  finest to coarsest:            2           1
  
  
 --------------------------------------------
 SETQINIT:
 -------------
   qinit_type = 0, no perturbation
  
 --------------------------------------------
 Multilayer Parameters:
 ----------------------
    check_richardson: T
    richardson_tolerance:  0.94999999999999996     
    eigen_method:           4
    inundation_method:           2
    dry_tolerance:   1.0000000000000000E-003   1.0000000000000000E-003
