$env:LUISA_PATH=Get-Location
$env:PATH=$env:PATH + ";" + $env:LUISA_PATH + "\build\bin"
$env:PYTHONPATH=$env:PYTHONPATH + ";" + $env:LUISA_PATH + "\build\bin"