for %%f in (examples/*.ipynb) do (
ren "%%~dpfexamples\%%~nxf" %%~nxf.saved && ^
jupyter nbconvert --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.iopub_timeout=30 --to notebook --execute "%%~dpfexamples\%%~nxf.saved" --output "%%~dpfexamples\%%~nxf"
)

sphinx-build -b html -D nbsphinx_execute=never . _build/

for %%f in (examples/*.saved) do (
del "%%~dpfexamples\%%~nf" && ^
ren "%%~dpfexamples\%%~nxf" %%~nf
)