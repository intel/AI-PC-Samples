@echo off
setlocal
pushd "%~dp0" || exit /b 1

REM Install the fully resolved, hash-pinned export toolchain.
pip install --require-hashes -r requirements-export.txt
if errorlevel 1 (
    popd
    exit /b 1
)

REM Export the pinned Hugging Face image-classification model to ONNX.
python export_model.py
set "exit_code=%errorlevel%"
popd
exit /b %exit_code%
