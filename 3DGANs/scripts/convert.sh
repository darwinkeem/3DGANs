echo "making env..."
cd ..
python3 -m venv .env
source .env/bin/activate
pip3 install -r requirements.txt
echo ".off file convert..."
python3 utils/off2vox.py ModelNet40/ --remove-all-dupes