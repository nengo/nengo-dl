set DEVICE=%1
set DIR=%cd%
set PYOPENCL_CTX=0

pylint ../nengo_dl --rcfile=../setup.cfg > ../tmp/run_checks.txt 2>&1 || goto :exit
pytest --pyargs nengo --device=%DEVICE% --dtype=float32 --unroll-simulation=1 >> ../tmp/run_checks.txt 2>&1 || goto :exit
pytest ../nengo_dl --device=%DEVICE% --dtype=float32 --unroll-simulation=1 >> ../tmp/run_checks.txt 2>&1 || goto :exit

cd /d %TEMP%
python %DIR%/../docs/whitepaper/whitepaper2018_plots.py --no-show --reps 1 test >> %DIR%/../tmp/run_checks.txt 2>&1 || goto :exit

:exit
  cd /d %DIR%
  exit /b %errorlevel%
