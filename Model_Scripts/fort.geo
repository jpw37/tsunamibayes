  
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
    fname:/tsunami/codykesler/tsunamibayes/Model_Scripts/InputData/dtopo.tt3                                                                                    
    topo type:           3
    minlevel, maxlevel:
           3           3
  
 --------------------------------------------
 SETTOPO:
 ---------
    mtopofiles =            6
    
    /tsunami/codykesler/tsunamibayes/Model_Scripts/InputData/etopo.tt3                                                                                    
   itopotype =            3
   minlevel, maxlevel =            1           3
   tlow, thi =    0.0000000000000000        10000000000.000000     
   mx =          571   x = (   124.99166666666700      ,   134.49166666685701      )
   my =          421   y = (  -9.5083333333330007      ,  -2.5083333331930007      )
   dx, dy (meters/degrees) =    1.6666666667000000E-002   1.6666666667000000E-002
    
    /tsunami/codykesler/tsunamibayes/Model_Scripts/InputData/banda_map_merged.tt3                                                                         
   itopotype =            3
   minlevel, maxlevel =            1           3
   tlow, thi =    0.0000000000000000        10000000000.000000     
   mx =          460   x = (   129.60000012106661      ,   129.98250012106647      )
   my =          160   y = (  -4.5999994188648348      ,  -4.4674994188648878      )
   dx, dy (meters/degrees) =    8.3333333333299999E-004   8.3333333333299999E-004
    
    /tsunami/codykesler/tsunamibayes/Model_Scripts/InputData/gauge10002_merged3.tt3                                                                       
   itopotype =            3
   minlevel, maxlevel =            1           3
   tlow, thi =    0.0000000000000000        10000000000.000000     
   mx =          460   x = (   128.00000012106659      ,   128.38250012106644      )
   my =          580   y = (  -3.9999994188648360      ,  -3.5174994188650288      )
   dx, dy (meters/degrees) =    8.3333333333299999E-004   8.3333333333299999E-004
    
    /tsunami/codykesler/tsunamibayes/Model_Scripts/InputData/gauge10004_merged3.tt3                                                                       
   itopotype =            3
   minlevel, maxlevel =            1           3
   tlow, thi =    0.0000000000000000        10000000000.000000     
   mx =          460   x = (   126.90000012106670      ,   127.28250012106655      )
   my =          460   y = (  -3.5999994188648392      ,  -3.2174994188649921      )
   dx, dy (meters/degrees) =    8.3333333333299999E-004   8.3333333333299999E-004
    
    /tsunami/codykesler/tsunamibayes/Model_Scripts/InputData/gauge10005_merged3.tt3                                                                       
   itopotype =            3
   minlevel, maxlevel =            1           3
   tlow, thi =    0.0000000000000000        10000000000.000000     
   mx =          460   x = (   128.50000012106659      ,   128.88250012106644      )
   my =          460   y = (  -3.7999994188648381      ,  -3.4174994188649910      )
   dx, dy (meters/degrees) =    8.3333333333299999E-004   8.3333333333299999E-004
    
    /tsunami/codykesler/tsunamibayes/Model_Scripts/InputData/gauge10006_merged3.tt3                                                                       
   itopotype =            3
   minlevel, maxlevel =            1           3
   tlow, thi =    0.0000000000000000        10000000000.000000     
   mx =          700   x = (   128.50000012106659      ,   129.08250012106635      )
   my =          700   y = (  -3.5999994188648392      ,  -3.0174994188650723      )
   dx, dy (meters/degrees) =    8.3333333333299999E-004   8.3333333333299999E-004
  
   Ranking of topography files  finest to coarsest:            6           5           4           3           2           7           1
  
  
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
