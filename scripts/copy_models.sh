#!/bin/bash
# Helper script to copy HSEmotion ONNX models into /home/mikey/fuck/models/
set -e

DEST="/home/mikey/fuck/models"
mkdir -p "$DEST"

if [ -f "$HOME/.hsemotion/enet_b2_7.onnx" ]; then
    cp -v "$HOME/.hsemotion/enet_b2_7.onnx" "$DEST/"
fi

if [ -f "$HOME/.hsemotion/enet_b0_8_best_vgaf.onnx" ]; then
    cp -v "$HOME/.hsemotion/enet_b0_8_best_vgaf.onnx" "$DEST/"
fi

if [ -f "/home/mikey/fuck/face_landmarker.task" ] && [ ! -f "$DEST/face_landmarker.task" ]; then
    cp -v "/home/mikey/fuck/face_landmarker.task" "$DEST/"
fi

echo "Models available in $DEST:"
ls -la "$DEST"

