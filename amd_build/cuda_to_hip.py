#!/usr/bin/python
""" The Python Hipify script.
##
# Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""

import argparse
import constants
import clang.cindex
import re
import shutil
import sys
import os

from functools import reduce
from cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS


def update_progress_bar(total, progress):
    """
    Displays and updates a console progress bar.
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()


def walk_over_directory(path, extensions, show_detailed):
    """ Walks over the entire directory and applies the function (func) on each file encountered.

    func (path as string): void
    """
    cur = 0
    total = sum([sum([reduce(lambda result, ext: filename.endswith("." + ext) or result, extensions, False) for filename in files]) for r, d, files in os.walk(path)])
    stats = {"unsupported_calls": [], "kernel_launches": []}

    for (dirpath, _dirnames, filenames) in os.walk(path):
        for filename in filenames:
            # Extract the (.hip)
            if filename.endswith(".hip"):
                hip_source = os.sep.join([dirpath, filename])
                dest_file = hip_source[0:-4]

                # Execute the preprocessor on the specified file.
                shutil.copy(hip_source, dest_file)

                # Assume (.hip) files are already preprocessed. Continue.
                continue

            if reduce(
                lambda result, ext: filename.endswith("." + ext) or result,
                    extensions, False):
                filepath = os.sep.join([dirpath, filename])

                # Execute the preprocessor on the specified file.
                preprocessor(filepath, stats)

                # Update the progress
                print(os.path.join(dirpath, filename))
                update_progress_bar(total, cur)

                cur += 1

    print("Finished")
    compute_stats(stats, show_detailed)


def compute_stats(stats, show_detailed):
    unsupported_calls = set(cuda_call for (cuda_call, _filepath) in stats["unsupported_calls"])

    # Print the number of unsupported calls
    print("Total number of unsupported CUDA function calls: %d" % (len(unsupported_calls)))

    # Print the list of unsupported calls
    print(", ".join(unsupported_calls))

    # Print the number of kernel launches
    print("\nTotal number of replaced kernel launches: %d" % (len(stats["kernel_launches"])))

    if show_detailed:
        print("\n".join(stats["kernel_launches"]))

        for unsupported in stats["unsupported_calls"]:
            print("Detected an unsupported function %s in file %s" % unsupported)


def processKernelLaunches(string, stats):
    """ Replace the CUDA style Kernel launches with the HIP style kernel launches."""
    def grab_method_and_template(in_kernel):
        # The positions for relevant kernel components.
        pos = {
            "kernel_launch": {"start": in_kernel.start(), "end": in_kernel.end()},
            "kernel_name": {"start": -1, "end": -1},
            "template": {"start": -1, "end": -1}
        }

        # Count for balancing template
        count = {"<>": 0}

        # Status for whether we are parsing a certain item.
        START = 0
        AT_TEMPLATE = 1
        AFTER_TEMPLATE = 2
        AT_KERNEL_NAME = 3

        status = START

        # Parse the string character by character
        for i in range(pos["kernel_launch"]["start"]-1, -1, -1):
            char = string[i]

            # Handle Templating Arguments
            if status == START or status == AT_TEMPLATE:
                if char == ">":
                    if status == START:
                        status = AT_TEMPLATE
                        pos["template"]["end"] = i
                    count["<>"] += 1

                if char == "<":
                    count["<>"] -= 1
                    if count["<>"] == 0 and (status == AT_TEMPLATE):
                        pos["template"]["start"] = i
                        status = AFTER_TEMPLATE

            # Handle Kernel Name
            if status != AT_TEMPLATE:
                if string[i] == "(" or string[i] == ")" or string[i] == "_" or string[i].isalnum():
                    if status != AT_KERNEL_NAME:
                        status = AT_KERNEL_NAME
                        pos["kernel_name"]["end"] = i

                    # Case: Kernel name starts the string.
                    if i == 0:
                        pos["kernel_name"]["start"] = 0

                        # Finished
                        return [(pos["kernel_name"]), (pos["template"]), (pos["kernel_launch"])]
                else:
                    # Potential ending point if we're already traversing a kernel's name.
                    if status == AT_KERNEL_NAME:
                        pos["kernel_name"]["start"] = i

                        # Finished
                        return [(pos["kernel_name"]), (pos["template"]), (pos["kernel_launch"])]

    # Grab positional ranges of all kernel launchces
    get_kernel_positions = [k for k in re.finditer("<<<\s*(.+)\s*,\s*(.+)\s*(,\s*(.+)\s*)?(,\s*(.+)\s*)?>>>", string)]
    output_string = string

    # Replace each CUDA kernel with a HIP kernel.
    for kernel in get_kernel_positions:
        # Get kernel components
        params = grab_method_and_template(kernel)

        # Find paranthesis after kernel launch
        paranthesis = string.find("(", kernel.end())

        # Extract cuda kernel
        cuda_kernel = string[params[0]["start"]:paranthesis+1]

        # Transform cuda kernel to hip kernel
        hip_kernel = "hipLaunchKernelGGL(" + cuda_kernel[0:-1].replace("<<<", ", ").replace(">>>", ", ")

        # Replace cuda kernel with hip kernel
        output_string = output_string.replace(cuda_kernel, hip_kernel, 1)

        # Update the statistics
        stats["kernel_launches"].append(hip_kernel)

    return output_string


def disable_asserts(input_string):
    """ Disables regular assert statements
    e.g. "assert(....)" -> "/*assert(....)*/"
    """
    def whitelist(input):
        return input.group(1) + "/*" + input.group(2) + "*/" + input.group(3)

    # Calling asserts from device code results in errors.
    result = re.sub(r'(^|[^a-zA-Z0-9_.\n]+)(assert\(.*\))([^a-zA-Z0-9_.\n]+)', whitelist, input_string)

    # PyTorch Specific - Calling THAssert when compiling for device causes errors.
    result = re.sub(r'(^|[^a-zA-Z0-9_.\n]+)(THError\(.*\))([^a-zA-Z0-9_.\n]+)', whitelist, result)
    # Solution around this is to stub <assert.h> and then make sure __CUDA_ARCH__ is defined.
    # Potential issue is that setting __CUDA_ARCH__ may lead to issues in the code base / understandability.
    return result


def preprocessor(filepath, stats):
    """ Executes the CUDA -> HIP conversion on the specified file. """
    with open(filepath, "r+") as fileobj:
        output_source = fileobj.read()

        # Perform type, method, constant replacements
        for mapping in CUDA_TO_HIP_MAPPINGS:
            for key, value in mapping.iteritems():
                # Extract relevanft info
                cuda_type = key
                hip_type = value[0]
                meta_data = value[1:]

                if output_source.find(cuda_type) > -1:
                    # Check if supported
                    if constants.HIP_UNSUPPORTED in meta_data:
                        stats["unsupported_calls"].append((cuda_type, filepath))

                if cuda_type in output_source:
                    output_source = re.sub(r'\b(%s)\b' % cuda_type, lambda x: hip_type, output_source)
                    #output_source = output_source.replace(cuda_type, hip_type)

        # Perform Kernel Launch Replacements
        output_source = processKernelLaunches(output_source, stats)

        # Disable asserts
        output_source = disable_asserts(output_source)

        # Overwrite file contents
        fileobj.seek(0)
        fileobj.write(output_source)
        fileobj.truncate()
        fileobj.flush()

        # Flush to disk
        os.fsync(fileobj)


def file_specific_replacement(filepath, search_string, replace_string):
    with open(filepath, "r+") as f:
        contents = f.read()
        contents = contents.replace(search_string, replace_string)
        f.seek(0)
        f.write(contents)
        f.truncate()
        f.flush()
        os.fsync(f)


def file_add_header(filepath, header):
    with open(filepath, "r+") as f:
        contents = f.read()
        if header[0] != "<" and header[1] != ">":
            header = '"%s"' % header
        contents = ('#include "%s" \n' % header) + contents
        f.seek(0)
        f.write(contents)
        f.truncate()
        f.flush()
        os.fsync(f)


def pytorch_specific_fixes(amd_pytorch_directory):
    """Load the PyTorch specific patches"""
    aten_src_directory = os.path.join(amd_pytorch_directory, "aten/src/")

    # Due to an issue in HCC, change filename of CuDNN batch norm
    shutil.move(os.path.join(aten_src_directory, "ATen/native/cudnn/BatchNorm.cpp"), os.path.join(aten_src_directory, "ATen/native/cudnn/BatchNormCuDNN.cpp"))

    # Disable OpenMP in aten/src/TH/generic/THTensorMath.c
    file_specific_replacement(os.path.join(aten_src_directory, "TH/generic/THTensorMath.c"), "_OPENMP", "_OPENMP_STUBBED")

    # Swap the math functions from std:: to hcc device functions.
    file_specific_replacement(os.path.join(aten_src_directory, "ATen/native/cuda/Embedding.cu"), "std::pow", "powf")
    file_specific_replacement(os.path.join(aten_src_directory, "ATen/native/cuda/Embedding.cu"), "std::abs", "fabs")

    # Swap abs w/ fabsf for device code.
    file_specific_replacement(os.path.join(aten_src_directory, "THCUNN/Abs.cu"), "abs(", "fabs(")

    # Disable the loading of the CUDA runtime in torch/cuda/__init__.py
    torch_cuda_init = os.path.join(amd_pytorch_directory, "torch/cuda/__init__.py")
    file_specific_replacement(torch_cuda_init, "_cudart = _load_cudart()", "# _cudart = _load_cudart()")
    file_specific_replacement(torch_cuda_init, "_cudart.cudaGetErrorName.restype = ctypes.c_char_p", "#_cudart.cudaGetErrorName.restype = ctypes.c_char_p")
    file_specific_replacement(torch_cuda_init, "_cudart.cudaGetErrorString.restype = ctypes.c_char_p", "#_cudart.cudaGetErrorString.restype = ctypes.c_char_p")
    file_specific_replacement(torch_cuda_init, "_lazy_call(_check_capability)", "#_lazy_call(_check_capability)")

    # Add include to ATen.h
    file_add_header(
        os.path.join(aten_src_directory, "ATen/ATen.h"),
        "hip/hip_runtime.h"
    )

    # Add include to THCStream.h
    file_add_header(
        os.path.join(aten_src_directory, "THC/THCStream.h"),
        "<thrust/execution_policy.h>"
    )

    # Add include to THCTensorIndex.cu
    file_add_header(
        os.path.join(aten_src_directory, "THCTensorIndex.cu"),
        "<thrust/execution_policy.h>"
    )

    # Wrap all calls to THError(...) with #if defined(__HIP_DEVICE_COMPILE__) \n THError(...) \n #endif

    # Replace "cudaStreamCreateWithPriority(&self->stream, flags, priority)" with cudaStreamCreateWithFlags(&self->stream, flags)
    # Inside of aten/src/THC/THCStream.cpp
    file_specific_replacement(
        os.path.join(aten_src_directory, "THC/THCStream.cpp"),
        "cudaStreamCreateWithPriority(&self->stream, flags, priority)",
        "cudaStreamCreateWithFlags(&self->stream, flags)")

    print("Successfully loaded PyTorch specific modifications.")


def main():
    """Example invocation

    python hipify.py --project-directory /home/myproject/ --extensions cu cuh h cpp --output-directory /home/gains/
    """

    parser = argparse.ArgumentParser(
        description="The Python Hipify Script.")

    parser.add_argument(
        '--project-directory',
        type=str,
        default=os.getcwd(),
        help="The root of the project.",
        required=True)

    parser.add_argument(
        '--show-detailed',
        type=bool,
        default=False,
        help="Show detailed summary of the hipification process.",
        required=False)

    parser.add_argument(
        '--extensions',
        nargs='+',
        default=["cu", "cuh", "c", "cpp", "h", "in"],
        help="The extensions for files to run the Hipify script over.",
        required=False)

    parser.add_argument(
        '--output-directory',
        type=str,
        default="",
        help="The directory to store the hipified project.",
        required=False)

    args = parser.parse_args()

    # Sanity check arguments
    if not os.path.exists(args.project_directory):
        print("The project folder specified does not exist.")
        return

    # If output directory not set, provide a default output directory.
    if args.output_directory is "":
        args.project_directory = args.project_directory[0:-1] if args.project_directory.endswith("/") else args.project_directory
        args.output_directory = args.project_directory + "_amd"

    # Make sure output directory doesn't already exist.
    if os.path.exists(args.output_directory):
        print("The provided output directory already exists. Please move or delete it to prevent overwriting of content.")
        return

    # Remove periods from extensions
    args.extensions = map(lambda ext: ext[1:] if ext[0] is "." else ext, args.extensions)

    # Copy the directory
    shutil.copytree(args.project_directory, args.output_directory)

    # PyTorch Specific Modifications
    pytorch_specific_fixes(args.output_directory)

    # Start Preprocessor
    walk_over_directory(
        args.output_directory,
        extensions=args.extensions,
        show_detailed=args.show_detailed)


if __name__ == '__main__':
    main()
