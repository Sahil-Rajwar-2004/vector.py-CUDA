
pkg="vector_cuda"

pip show "$pkg" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "$pkg lib is already installed"
    exit 0
else
    echo "$pkg lib isn't installed yet"
fi

echo "Initiating the installation process..."

if [ -e "./setup.py" ]; then
    echo "./setup.py has been found!"
else
    echo "setup.py file not found!"
    echo "it seems like you tempered with this dir (vector.py-CUDA) or"
    echo "maybe the owner who own this repo has made a mistake!"
    echo "clone this repo and try again"
    exit 1
fi

set -e

python3 ./setup.py sdist bdist_wheel
cd ./dist
pip install *.whl
cd ..
rm -rf dist build vector_cuda.egg-info

pip show "$pkg" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "$pkg lib has been installed successfully"
else
    echo "something bad has happened! try to install manually"
    exit 1
fi
