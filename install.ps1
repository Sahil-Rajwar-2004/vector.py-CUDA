$pkg = "vector_cuda"

pip show $pkg > $null 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "$pkg lib is already installed"
    exit 0
} else {
    Write-Host "$pkg isn't installed yet"
}

Write-Host "Initiating the installation process..."

if (Test-Path "./setup.py") {
    Write-Host "setup.py has been found!"
} else {
    Write-Host "setup.py file not found!"
    Write-Host "it seems like you tempered with this dir (vector.py-CUDA) or"
    Write-Host "maybe the owner who own this repo has made a mistake!"
    Write-Host "clone this repo and try again"
    exit 1
}

python ./setup.py sdist bdist_wheel
$whlPath = Get-ChildItem -Recurse -Force ".\dist\*.whl"
pip install $whlPath

Remove-Item -Recurse -Force "./dist", "./build", "./vector_cuda.egg-info" -ErrorAction SilentlyContinue

pip show $pkg > null 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "$pkg lib has been installed successfully"
} else {
    Write-Host "something bad has happened! try to install manually use (python setup.py sdist bdist_wheel)"
    exit 1
}

Remove-Item -Force "./null"
