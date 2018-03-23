#!/usr/bin/env bash

echo "Loading PyTorch AMD patch differentials."

SETUP_PATCH_RESULT=$(git apply ../amd_patch_files/setup.patch 2>&1)

cd torch
TORCH_PATCH_RESULT=$(git apply ../amd_patch_files/torch.patch 2>&1)
cd ..

cd aten
ATEN_PATCH_RESULT=$((git apply ../amd_patch_files/aten.patch) 2>&1)
cd ..

if [ ! -z $SETUP_PATCH_RESULT]; then
  echo "Failed to patch Setup:" $SETUP_PATCH_RESULT
fi

if [ ! -z $TORCH_PATCH_RESULT]; then
  echo "Failed to patch Torch:" $TORCH_PATCH_RESULT
fi

if [ ! -z $ATEN_PATCH_RESULT]; then
  echo "Failed to patch ATen:" $ATEN_PATCH_RESULT
fi

if [[ $TORCH_PATCH_RESULT != ""  ||  $ATEN_PATCH_RESULT != ""  || $SETUP_PATCH_RESULT != "" ]]; then
  echo "Failed to load AMD patch differentials."
else
  echo "PyTorch AMD patch differentials successfully installed!"
fi

echo "Running the PyTorch hipify script."
echo .
. "do_hipify.sh"
