import subprocess
import os

cwd = os.getcwd()
proj_dir = os.path.dirname(cwd)
out_dir = os.path.join(os.path.dirname(proj_dir), "pytorch_amd")
exclude_dirs = [
    "aten/src/TH",
    "aten/src/THNN",
    "aten/src/THS",
    "caffe2",
    "third_party"
]

yaml_file = os.path.join(cwd, "disabled.yaml")

# Apply the Patch File.
subprocess.Popen("git apply --stat", os.path.join(cwd, "patch084e3a7.patch"))

# Execute the Hipify Script.
subprocess.Popen(
    "/opt/rocm/bin/hipify-python.py",
    "--project-directory %s" % proj_dir,
    "--output-directory %s" % out_dir,
    "--exclude-dirs %s" % " ".join(exclude_dirs),
    "--yaml-settings %s" % yaml_file,
    "--add-static-casts %s" % "True"
)
