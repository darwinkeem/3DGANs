echo "ModelNet40 Download..."
cd ..
wget http://modelnet.cs.princeton.edu/ModelNet40.zip
unzip ModelNet40.zip
echo "Binvox converter Download..."
cd utils
wget http://www.patrickmin.com/binvox/linux64/binvox?rnd=1520896952313989 -O binvox
chmod 755 binvox
