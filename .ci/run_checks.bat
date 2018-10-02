set DEVICE=%1

set DIR=%cd%
set PYOPENCL_CTX=0

pylint ../nengo_dl --rcfile=../setup.cfg
pytest --pyargs nengo --device=%DEVICE% --dtype=float32 --unroll_simulation=1 || goto :exit
pytest ../nengo_dl --device=%DEVICE% --dtype=float32 --unroll_simulation=1 || goto :exit
python ../nengo_dl/benchmarks.py performance_samples --device %DEVICE% || goto :exit

cd %TEMP%
python %DIR%/../docs/whitepaper/whitepaper2018_plots.py --no-show --reps 1 test || goto :exit

:exit
  cd %DIR%
  exit /b %errorlevel%
