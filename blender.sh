echo "###################"
echo "# blender"

path_blender=/Applications/blender.app/Contents/MacOS/Blender

version_blender=$(${path_blender} --version)
version_blender=$(cut -d' ' -f 2 <<< ${version_blender})
version_blender=${version_blender%.*}
echo "blender version: ${version_blender}"

path_python=$(dirname ${path_blender})/../Resources/${version_blender}/python/bin
path_python=$(find ${path_python} -maxdepth 1 -type f -name 'python*')
echo "path python: ${path_python}"

version_python=$(${path_python} --version)
version_python=$(cut -d' ' -f 2 <<< ${version_python})
version_python=${version_python%.*}
echo "version python: ${version_python}"

<< COMMENTOUT
brew install "python@${version_python}"
path_include_from=/usr/local/opt/python@${version_python}/Frameworks/Python.framework/Headers
path_include_to=/Applications/Blender.app/Contents/Resources/${version_blender}/python/include/python${version_python}m

echo ${path_include_from}
echo ${path_include_to}
cp -r ${path_include_from}/*.h ${path_include_to}
cp -r ${path_include_from}/internal ${path_include_to}

${path_python} -m ensurepip     # make sure pip is installed
${path_python} -m pip install --upgrade pip # upgrade pip
${path_python} -m pip install --upgrade opencv-python  # upgrade pip
${path_python} -m pip install --upgrade numpy  # upgrade pip
COMMENTOUT

${path_blender} -P blender.py
#${path_blender} -b -P 02_envtex.py -- --render