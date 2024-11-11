unzip xsm data -> put it in scripts folder, for eg something like SuryaDrishti/scripts/data/2022/08/04/calibrated/{.lc,...} run the scripts in order -> 

extract_pha
extract_lc
list_data
xsm_genspec_batch
list_data
xsm_gen_lc

now you will get a folder in scripts/data/XSM_Generated_LightCurve in this copy all the file to suryadrishti/data/lightcurves
now in log_main change the variable start_date = date(2024, 4, 3)
numdays = 7  
to define start dates and num days

then two run it in suryadrishti folder run python logmain.py <random name>
