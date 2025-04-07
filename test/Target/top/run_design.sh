#!/bin/bash
if [ -z "$XILINX_PATH" ]; then
    echo "XILINX_PATH not set"
    exit 1
fi
if [ -z "$XILINX_VERSION" ]; then
    echo "XILINX_VERSION not set"
    exit 1
fi
VITIS_HLS="$XILINX_PATH/Vitis/$XILINX_VERSION/bin/vitis-run"
VIVADO="$XILINX_PATH/Vivado/$XILINX_VERSION/bin/vivado"
echo "Runing Vitis HLS"
"$VITIS_HLS" --mode hls --tcl run_hls.tcl
if [ $? -ne 0 ]; then
    echo "ERROR: Vitis HLS execution failed"
    exit 1
fi
echo "Runing Vivado"
"$VIVADO" -mode tcl -source run_vivado.tcl
if [ $? -ne 0 ]; then
    echo "ERROR: Vivado execution failed"
    exit 1
fi
echo "Successfully generate design"
DESIGN_DIR=./vivado_project
BASENAME="top_bd"
XSA_FILENAME="$BASENAME.xsa"
XSA_FILE="$DESIGN_DIR/$XSA_FILENAME"
if [ ! -d "$DESIGN_DIR" ]; then
    echo "Directory $DESIGN_DIR doesn't exist"
    exit 1
fi
if [ ! -f "$XSA_FILE" ]; then
    echo "XSA File $XSA_FILE doesn't exist"
    exit 1
fi
echo "Found XSA File: $XSA_FILE"
TEMP_DIR="./xsa_temp"
TARGET_DIR="./driver/bitfile"
echo "Extracting "$XSA_FILE""
unzip -q "$XSA_FILE" -d "$TEMP_DIR"
if [ $? -ne 0 ]; then
    echo "Failed to extract "$XSA_FILE""
    rm -rf "$TEMP_DIR"
    exit 1
fi
mkdir -p "$TARGET_DIR"
BIT_FILE=$(find "$TEMP_DIR" -name "$BASENAME.bit" | head -n 1)
if [ -z "$BIT_FILE" ]; then
    echo "Failed to find "$BASENAME.bit""
    rm -rf "$TEMP_DIR"
    exit 1
else
    cp "$BIT_FILE" "$TARGET_DIR"
    echo "Copied "$BASENAME.bit" into "$TARGET_DIR""
fi
HWH_FILE=$(find "$TEMP_DIR" -name "$BASENAME.hwh" | head -n 1)
if [ -z "$HWH_FILE" ]; then
    echo "Failed to find "$BASENAME.hwh""
    rm -rf "$TEMP_DIR"
    exit 1
else
    cp "$HWH_FILE" "$TARGET_DIR"
    echo "Copied "$BASENAME.hwh" into "$TARGET_DIR""
fi
rm -rf "$TEMP_DIR"
echo "Extract bitfile done!"
