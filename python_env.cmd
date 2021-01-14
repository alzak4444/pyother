rem turn off execution alias in windows 10 from app & features settings
set PATH=%PATH%;%ProgramFiles(x86)%\Microsoft Visual Studio\Shared\Python37_64\
set PATH=%PATH%;%ProgramFiles(x86)%\Microsoft Visual Studio\Shared\Python37_64\Scripts
cd c:\pyother
rem call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
cmd /k
