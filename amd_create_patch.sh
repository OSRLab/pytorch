# Automatically create the PATCH files.
mkdir -p amd_patch_files

# Swap out the HIP files before the patching process.
# Search through each (.hip) file and replace with the real one.
FILES=$(find . -name "*.hip")
for hip_file in $FILES
do
  new_file="${hip_file%%.hip}"
  echo "Swapping .hip file $hip_file with $new_file"
  # take action on each file. $f store current file name
  cp "$hip_file" "$new_file"
done

# Create Patch Files.
git diff upstream/master:torch origin/hc2_latest_rebased:torch > amd_patch_files/torch.patch
git diff upstream/master:aten origin/hc2_latest_rebased:aten > amd_patch_files/aten.patch
git diff upstream/master:setup.py origin/hc2_latest_rebased:setup.py > amd_patch_files/setup.patch

echo "Successfully created PATCH files."

#Process)
#1 - Starting from the hc2_latest_rebased, rebase to master.
#2 - Create the patch files.

#Usage
#1 - git checkout pytorch on a specific version.
#2 - Execute the patch files.
#3 - Run the Hipify script.
