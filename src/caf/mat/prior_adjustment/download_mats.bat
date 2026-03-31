@echo off
setlocal enabledelayedexpansion

set "mat_dir=I:\NorMITs Distribution\ntem_emp_rail"
set "out_dir=D:\NorMITs Demand\ntem_emp_test"
set "run_name=6_sec_adj"
set "mode=6"
set "tps=1 2 3"

for %%t in (%tps%) do (
    for %%p in (1 2 3 4 5 6 7 8) do (
        set /a nhb_p = %%p + 10
        
        rem Copy fr file
        robocopy "%mat_dir%\p%%p\m%mode%_p%%p_ts%%t_fr\%run_name%" "%out_dir%" "m%mode%_p%%p_ts%%t_fr_matrix.csv" /NJH /NJS /NP /R:1 /W:1
        if exist "%out_dir%\m%mode%_p%%p_ts%%t_fr_matrix.csv" (
            rename "%out_dir%\m%mode%_p%%p_ts%%t_fr_matrix.csv" "m%mode%_p%%p_ts%%t_fr.csv"
        )
        
        rem Copy to file
        robocopy "%mat_dir%\p%%p\m%mode%_p%%p_ts%%t_to\%run_name%" "%out_dir%" "m%mode%_p%%p_ts%%t_to_matrix.csv" /NJH /NJS /NP /R:1 /W:1
        if exist "%out_dir%\m%mode%_p%%p_ts%%t_to_matrix.csv" (
            rename "%out_dir%\m%mode%_p%%p_ts%%t_to_matrix.csv" "m%mode%_p%%p_ts%%t_to.csv"
        )
        
        rem Copy nhb file
        robocopy "%mat_dir%\p!nhb_p!\m%mode%_p!nhb_p!_ts%%t_nhb\%run_name%" "%out_dir%" "m%mode%_p!nhb_p!_ts%%t_nhb_matrix.csv" /NJH /NJS /NP /R:1 /W:1
        if exist "%out_dir%\m%mode%_p!nhb_p!_ts%%t_nhb_matrix.csv" (
            rename "%out_dir%\m%mode%_p!nhb_p!_ts%%t_nhb_matrix.csv" "m%mode%_p!nhb_p!_ts%%t_nhb.csv"
        )
    )
)

echo Done.
pause