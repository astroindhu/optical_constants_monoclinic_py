This script derives optical constants for the a-c plane monoclinic crystals from oriened reflectance data. 

The functions called by the main script are:
disp_model_wrap_ac.m  :Wrapper function needed to ease use of
                            lsqcurvefit.
                            
dispersion_model_ac.m :Actual dispersion model called by the
                            wrapper function. Outputs optical constants 
                           (n,k). 
                           
fresnel_ac.m          :Fresnel equations for non-normal incidence. Calculates 
                            reflectance from optical constants. Incidence
                            angle set to 30 degrees. Edit this funcion to
                            change that.
                            
isint.m                    :Short boolean function that gives 'Yes' if
                            the input is an integer. 

inputs: param_values.txt - oscillator parameter estimates

 output: new_param_values.txt
        optical_constants_ac.txt