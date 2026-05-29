#!/bin/bash

# ============================================================
# AGRISHIELD AI - LOCAL LAUNCHER SCRIPT
# ============================================================

# Color codes for premium terminal feedback
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

clear
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}               AGRISHIELD AI - CROP PROTECTION              ${NC}"
echo -e "${GREEN}             PHD SCHOLAR WEED DIAGNOSIS ENGINE              ${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

echo -e "${CYAN}[1/3] Verifying System Environments...${NC}"
python3 --version
pip3 --version

echo -e "${CYAN}[2/3] Checking dependencies...${NC}"
# Check if required libraries are installed
python3 -c "
libs = ['flask', 'tensorflow', 'numpy', 'PIL']
for lib in libs:
    __import__(lib)
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}WARNING: Standard libraries missing. Installing via requirements.txt...${NC}"
    pip3 install -r requirements.txt
else
    echo -e "${GREEN} -> All libraries verified! (Flask, TensorFlow, NumPy, Pillow)${NC}"
fi

echo ""
echo -e "${CYAN}[3/3] Initializing Deep Learning Server Core...${NC}"
echo -e "${GREEN} -> Serving locally at http://127.0.0.1:8000${NC}"
echo -e "${YELLOW} -> Automatically launching default web browser...${NC}"

# Launch browser silently in the background
sleep 1.5
xdg-open http://127.0.0.1:8000 >/dev/null 2>&1 &

# Execute Flask application
python3 app.py
