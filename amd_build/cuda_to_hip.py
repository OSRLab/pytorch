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
import re
import shutil
import sys
import os
import yaml

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

    # Number of blocks to display. Used to visualize progress.
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)

    # Send the progress to stdout.
    sys.stdout.write(text)

    # Send the buffered text to stdout!
    sys.stdout.flush()


def walk_over_directory(path, extensions, show_detailed=False, exclude_dirs=None):
    """ Recursively walk over directory and preprocesses files
    with specified extensions.

    extensions - A list of file extensions excluding the leading "."
                (e.g. ['cuh'] not ['.cuh'])

    show_detailed - If set to true, then show all replaced kernels.
    """

    # Default argument for excluded directories.
    if exclude_dirs is None:
        exclude_dirs = []

    # Dissalow the 'hip' as an extension.
    # .hip has a special meaning
    #    => i.e. "This file has already been processed and simply needs to be moved"
    #    => e.g. moving "kernel_file.cu.hip" to "kernel_file.cu"
    if "hip" in extensions:
        print(".hip files are assumed to already be preprocessed. \
        Please remove 'hip' from the extensions list.")

    # Compute the total number of files to be traversed.
    total_files = 0
    for (dirpath, _dirnames, filenames) in os.walk(path):
        if dirpath not in exclude_dirs:
            for filename in filenames:
                total_files += reduce(
                    lambda result, ext: filename.endswith("." + ext) or result,
                    extensions, False)

    current_file = 0

    # Preprocessing statistics.
    stats = {"unsupported_calls": [], "kernel_launches": []}

    # Begin traversing the files.
    for (dirpath, _dirnames, filenames) in os.walk(path):
        # Check if file ends with a valid extensions
        if sum([dirpath.endswith(ext) for ext in extensions]) == 0:
            continue

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
                update_progress_bar(total_files, current_file)

                current_file += 1

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

    # Show a detailed summary
    if show_detailed:
        print("\n".join(stats["kernel_launches"]))


def processKernelLaunches(string, stats):
    """ Replace the CUDA style Kernel launches with the HIP style kernel launches."""
    # Concat the namespace with the kernel names. (Find cleaner way of doing this later).
    string = re.sub(r'([ ]+)(detail+)::[ ]+\\\n[ ]+', lambda inp: "%s%s::" % (inp.group(1), inp.group(2)), string)

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
                if string[i] == "(" or string[i] == ")" or string[i] == "_" or string[i].isalnum() or string[i] == ":":
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
        output_string = output_string.replace(cuda_kernel, hip_kernel)

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
    return result


def disable_function(input_string, function, replace_style):
    """ Finds and disables a function in a particular file.

    If type(function) == List
        function - The signature of the function to disable.
            e.g. ["bool", "overlappingIndices", "(const Tensor& t)"]
            disables function -> "bool overlappingIndices(const Tensor& t)"

    If type(function) == String
        function - Disables the function by name only.
            e.g. "overlappingIndices"

    replace_style - The style to use when stubbing functions.
        0 - Remove the function entirely (includes the signature).
        1 - Stub the function and return an empty object based off the type.
        2 - Add !defined(__HIP_PLATFORM_HCC__) preprocessors around the function.
        3 - Add !defined(__HIP_DEVICE_COMPILE__) preprocessors around the function.
    """

    info = {
        "function_start": -1,
        "function_end": -1,
        "bracket_count": 0
    }

    STARTED = 0
    INSIDE_FUNCTION = 1
    BRACKET_COMPLETE = 2

    STATE = STARTED

    if type(function) == list:
        # Extract components from function signature.
        func_info = {
            "return_type": function[0].strip(),
            "function_name": function[1].strip(),
            "function_args": function[2].strip()
        }

        # Create function string to search for
        function_string = "%s%s%s" % (
            func_info["return_type"],
            func_info["function_name"],
            func_info["function_args"]
        )

        # Find the starting position for the function
        info["function_start"] = input_string.find(function_string)
    else:
        # Automatically detect signature.
        the_match = re.search("((.*) (\*)?)(%s)(\(.*\))" % (function.replace("(", "\(").replace(")", "\)")), input_string)

        func_info = {
            "return_type": the_match.group(1).strip(),
            "function_name": the_match.group(4).strip(),
            "function_args": the_match.group(5).strip(),
        }

        # Find the starting position for the function
        info["function_start"] = the_match.start()
        function_string = input_string[the_match.span()[0]:the_match.span()[1]]

    # The function can't be found anymore.
    if info["function_start"] == -1:
        return input_string

    # Find function block start.
    pos = info["function_start"] + len(function_string)
    while pos < len(input_string) - 1 and STATE != BRACKET_COMPLETE:
        pos += 1
        if input_string[pos] == "{":
            if STATE != INSIDE_FUNCTION:
                STATE = INSIDE_FUNCTION
                bracket_count = 1
            else:
                bracket_count += 1
        elif input_string[pos] == "}":
            bracket_count -= 1

            if bracket_count == 0 and STATE == INSIDE_FUNCTION:
                STATE = BRACKET_COMPLETE
                info["function_end"] = pos

    # Never found the function end. Corrupted file!
    if STATE != BRACKET_COMPLETE:
        return input_string

    # Preprocess the source by removing the function.
    function_body = input_string[info["function_start"]:info["function_end"]+1]

    # Remove the entire function body
    if replace_style == 0:
        output_string = input_string.replace(function_body, "")

    # Stub the function based off its return type.
    elif replace_style == 1:
        # void return type
        if func_info["return_type"] == "void" or func_info["return_type"] == "static void":
            stub = "%s{\n}" % (function_string)
        # pointer return type
        elif "*" in func_info["return_type"]:
            stub = "%s{\nreturn %s;\n}" % (function_string, "NULL")
        else:
            stub = "%s{\n%s stub_var;\nreturn stub_var;\n}" % (function_string, func_info["return_type"])

        output_string = input_string.replace(function_body, stub)

    # Add HIP Preprocessors.
    elif replace_style == 2:
        output_string = input_string.replace(
            function_body,
            "#if !defined(__HIP_PLATFORM_HCC__)\n%s\n#endif" % function_body)

    # Add HIP Preprocessors.
    elif replace_style == 3:
        output_string = input_string.replace(
            function_body,
            "#if !defined(__HIP_DEVICE_COMPILE__)\n%s\n#endif" % function_body)

    return output_string


def preprocessor(filepath, stats):
    """ Executes the CUDA -> HIP conversion on the specified file. """
    with open(filepath, "r+") as fileobj:
        output_source = fileobj.read()

        # Perform type, method, constant replacements
        for mapping in CUDA_TO_HIP_MAPPINGS:
            for cuda_type, value in mapping.iteritems():
                # Extract relevant information
                hip_type = value[0]
                meta_data = value[1:]

                if output_source.find(cuda_type) > -1:
                    # Check if supported
                    if constants.HIP_UNSUPPORTED in meta_data:
                        stats["unsupported_calls"].append((cuda_type, filepath))

                if cuda_type in output_source:
                    output_source = re.sub(r'\b(%s)\b' % cuda_type, lambda x: hip_type, output_source)

        # Perform Kernel Launch Replacements
        output_source = processKernelLaunches(output_source, stats)

        # Disable asserts
        if not filepath.endswith("THCGeneral.h.in"):
            output_source = disable_asserts(output_source)

        # Overwrite file contents
        fileobj.seek(0)
        fileobj.write(output_source)
        fileobj.truncate()
        fileobj.flush()

        # Flush to disk
        os.fsync(fileobj)


def file_specific_replacement(filepath, search_string, replace_string, strict = False):
    with open(filepath, "r+") as f:
        contents = f.read()
        if strict:
            contents = re.sub(r'\b(%s)\b' % search_string, lambda x: replace_string, contents)
        else:
            contents = contents.replace(search_string, replace_string)
        f.seek(0)
        f.write(contents)
        f.truncate()
        f.flush()
        os.fsync(f)


def file_add_header(filepath, header):
    with open(filepath, "r+") as f:
        contents = f.read()
        if header[0] != "<" and header[-1] != ">":
            header = '"%s"' % header
        contents = ('#include %s \n' % header) + contents
        f.seek(0)
        f.write(contents)
        f.truncate()
        f.flush()
        os.fsync(f)


def get_kernel_template_params(the_file, KernelDictionary):
    """Scan for __global__ kernel definitions then extract its argument types, and static cast as necessary"""
    # Read the kernel file.
    with open(the_file, "r") as f:
        # Extract all kernels with their templates inside of the file
        string = f.read()

        get_kernel_definitions = [k for k in re.finditer(r"(template[ ]*<(.*)>\n.*\n?)?__global__ void (\w+(\(.*\))?)\(", string)]

        # Create new launch syntax
        for kernel in get_kernel_definitions:
            template_arguments = kernel.group(2).split(",") if kernel.group(2) else ""
            template_arguments = [x.replace("template", "").replace("typename", "").strip() for x in template_arguments]
            kernel_name = kernel.group(3)

            # Kernel starting / ending positions
            arguments_start = kernel.end()
            argument_start_pos = arguments_start
            current_position = arguments_start + 1

            # Search for final parenthesis
            arguments = []
            closures = {"(": 1, "<": 0}
            while current_position < len(string):
                if string[current_position] == "(":
                    closures["("] += 1
                elif string[current_position] == ")":
                    closures["("] -= 1
                elif string[current_position] == "<":
                    closures["<"] += 1
                elif string[current_position] == ">":
                    closures["<"] -= 1

                # Finished all arguments
                if closures["("] == 0 and closures["<"] == 0:
                    # Add final argument
                    arguments.append({"start": argument_start_pos, "end": current_position})
                    break

                # Finished current argument
                if closures["("] == 1 and closures["<"] == 0 and string[current_position] == ",":
                    arguments.append({"start": argument_start_pos, "end": current_position})
                    argument_start_pos = current_position + 1

                current_position += 1

            # Grab range of arguments
            arguments_string = [string[arg["start"]: arg["end"]] for arg in arguments]

            argument_types = [None] * len(arguments_string)
            for arg_idx, arg in enumerate(arguments_string):
                for i in range(len(arg)-1, -1, -1):
                    if arg[i] == "*" or arg[i] == " ":
                        argument_types[arg_idx] = re.sub(' +',' ', arg[0:i+1].replace("\n", "").strip())
                        break
            if len(template_arguments) == 1 and template_arguments[0].strip() in ["Dtype", "T"]:
                # Updates kernel
                kernel_with_template = "%s<real>" % (kernel_name)
            else:
                kernel_with_template = kernel_name
            formatted_args = {}
            for idx, arg_type in enumerate(argument_types):
                formatted_args[idx] = arg_type

            KernelDictionary[kernel_name] = {"kernel_with_template": kernel_with_template, "arg_types": formatted_args}

        # Extract generated kernels
        get_generated_kernels = [k for k in re.finditer(r"GENERATE_KERNEL([1-9])\((.*)\)", string)]
        # curandStateMtgp32 *state, int size, T *result, ARG1
        for kernel in get_generated_kernels:
            kernel_gen_type = int(kernel.group(1))
            kernel_name = kernel.group(2).split(",")[0]
            kernel_params = kernel.group(2).split(",")[1:]

            if kernel_gen_type == 1:
                kernel_args = {1: "int", 2: "%s *" % kernel_params[0], 3: kernel_params[1]}

            if kernel_gen_type == 2:
                kernel_args = {1: "int", 2: "%s *" % kernel_params[0], 3: kernel_params[1], 4: kernel_params[2]}

            # Argument at position 1 should be int
            KernelDictionary[kernel_name] = {"kernel_with_template": kernel_name, "arg_types": kernel_args}


def pytorch_specific_fixes(amd_pytorch_directory):
    """Load the PyTorch specific patches"""
    aten_src_directory = os.path.join(amd_pytorch_directory, "aten/src/")

    # The strings in certain files to replace.
    REPLACEMENTS = {
        # ATen & TH* files.
        os.path.join(aten_src_directory, "TH/THAtomic.c"): {"assert(sizeof(int) == sizeof(int32_t));": ""}, # Remove the assert from THAtomic.c
        os.path.join(aten_src_directory, "TH/generic/THTensorMath.c"): {"_OPENMP": "_OPENMP_STUBBED"}, # Disable OpenMP in aten/src/TH/generic/THTensorMath.c
        os.path.join(aten_src_directory, "ATen/native/cuda/Embedding.cu"): {"std::pow": "powf"}, # Swap the math functions from std:: to hcc device functions.
        os.path.join(aten_src_directory, "ATen/native/cuda/Embedding.cu"): {"std::abs": "fabs"}, # Swap the math functions from std:: to hcc device functions.
        os.path.join(aten_src_directory, "THCUNN/Abs.cu"): {"abs(": "fabs("}, # Swap abs w/ fabsf for device code.
        os.path.join(aten_src_directory, "THC/THCTensorRandom.cpp"): {"struct curandStateMtgp32*": "curandStateMtgp32*"}, # Remove the "struct" from the return type since hipStageMtgp32* implies pointer to struct.

        # Torch files
        os.path.join(amd_pytorch_directory, "torch/csrc/cuda/Module.cpp"): {"CUDA_VERSION": "0"}, # Update CUDA_VERSION to 0.

        # Disable the loading of the CUDA runtime in torch/cuda/__init__.py
        os.path.join(amd_pytorch_directory, "torch/cuda/__init__.py"): {
            "_cudart = _load_cudart()": "# _cudart = _load_cudart()",
            "_cudart.cudaGetErrorName.restype = ctypes.c_char_p": "#_cudart.cudaGetErrorName.restype = ctypes.c_char_p",
            "_cudart.cudaGetErrorString.restype = ctypes.c_char_p": "#_cudart.cudaGetErrorString.restype = ctypes.c_char_p",
            "_lazy_call(_check_capability)": "#_lazy_call(_check_capability)"
        },

        os.path.join(aten_src_directory, "THC/THCAllocator.c"): {"cudaMallocManaged(&ptr, size, cudaMemAttachGlobal)": "cudaSuccess"}, # Replace with THCudaCheck(hipSuccess)

        # Replace "cudaStreamCreateWithPriority(&self->stream, flags, priority)" with cudaStreamCreateWithFlags(&self->stream, flags)
        os.path.join(aten_src_directory, "THC/THCStream.cpp"): {"cudaStreamCreateWithPriority(&self->stream, flags, priority)": "cudaStreamCreateWithFlags(&self->stream, flags)"}
    }

    # Handle the necessary replacements
    for filepath in REPLACEMENTS:
        for key in REPLACEMENTS[filepath]:
            file_specific_replacement(filepath, key, REPLACEMENTS[filepath][key])

    # Due to an issue in HCC, change filename of CuDNN batch norm
    shutil.move(os.path.join(aten_src_directory, "ATen/native/cudnn/BatchNorm.cpp"), os.path.join(aten_src_directory, "ATen/native/cudnn/BatchNormCuDNN.cpp"))

    # Replace WITH_CUDA with WITH_ROCM for all locations inside for /torch/
    torch_directory = os.path.join(amd_pytorch_directory, 'torch/')
    extensions = ["h", "cpp", "hpp"]
    exclude_files = ["torch/csrc/autograd/profiler.h", "torch/csrc/autograd/profiler.cpp", "torch/csrc/cuda/cuda_check.h", "torch/csrc/jit/fusion_compiler.h", "torch/csrc/jit/fusion_compiler.cpp"]
    for (dirpath, _dirnames, filenames) in os.walk(torch_directory):
        for filename in filenames:
            if reduce(lambda result, ext: filename.endswith("." + ext) or result, extensions, False):
                the_file = os.sep.join([dirpath, filename])
                if not reduce(lambda result, exclude: the_file.endswith(exclude) or result, exclude_files, False):
                    file_specific_replacement(the_file, "WITH_CUDA", "WITH_ROCM")

    # Add include to ATen.h
    file_add_header(os.path.join(aten_src_directory, "ATen/ATen.h"), "hip/hip_runtime.h")

    # Add include to THCTensorIndex.cu
    file_add_header(os.path.join(aten_src_directory, "THC/THCTensorIndex.cu"), "<thrust/execution_policy.h>")

    # Add templating to all of the kernel calls inside THCUNN.
    extensions = ["cu", "cuh", "h"]
    KernelTemplateParams = {}
    for (dirpath, _dirnames, filenames) in os.walk(aten_src_directory):
        for filename in filenames:
            if reduce(lambda result, ext: filename.endswith("." + ext) or result, extensions, False):
                the_file = os.sep.join([dirpath, filename])

                # Store param information inside KernelTemplateParams
                get_kernel_template_params(the_file, KernelTemplateParams)

    # Walk over entire source tree and replace kernel launches with templated kernels.
    return KernelTemplateParams
    print("Successfully loaded PyTorch specific modifications.")


def add_static_casts(args, KernelTemplateParams):
    """Added necessary static casts to kernel launches due to issue in HIP"""

    # Add static_casts<> to all kernel launches.
    for (dirpath, _dirnames, filenames) in os.walk(args.output_directory):
        for filename in filenames:
            if reduce(
                lambda result, ext: filename.endswith("." + ext) or result,
                    args.extensions, False):
                filepath = os.sep.join([dirpath, filename])
                with open(filepath, "r+") as fileobj:
                    output_source = fileobj.read()
                    new_output_source = output_source
                    get_kernel_definitions = [k for k in re.finditer("hipLaunchKernelGGL\(", output_source)]
                    for kernel in get_kernel_definitions:
                        arguments = []
                        closures = {
                            "<": 0,
                            "(": 1
                        }
                        current_position = kernel.end()
                        argument_start_pos = current_position

                        # Search for final parenthesis
                        while current_position < len(output_source):
                            if output_source[current_position] == "(":
                                closures["("] += 1
                            elif output_source[current_position] == ")":
                                closures["("] -= 1
                            elif output_source[current_position] == "<":
                                closures["<"] += 1
                            elif output_source[current_position] == ">" and output_source[current_position-1] != "-":
                                closures["<"] -= 1

                            # Finished all arguments
                            if closures["("] == 0 and closures["<"] == 0:
                                # Add final argument
                                arguments.append({"start": argument_start_pos, "end": current_position})
                                break

                            # Finished current argument
                            if closures["("] == 1 and closures["<"] == 0 and output_source[current_position] == ",":
                                arguments.append({"start": argument_start_pos, "end": current_position})
                                argument_start_pos = current_position + 1

                            current_position += 1

                        # Check if we have templating + static_cast information
                        argument_strings = [output_source[arg["start"]:arg["end"]] for arg in arguments]
                        kernel_name = argument_strings[0].strip()
                        ignore = ["upscale"]
                        if kernel_name in KernelTemplateParams and kernel_name not in ignore:
                            # Add template to the kernel
                            # Add static_casts to relevant arguments
                            kernel_name_with_template = KernelTemplateParams[kernel_name]["kernel_with_template"]
                            argument_types = KernelTemplateParams[kernel_name]["arg_types"]

                            old_kernel_launch = output_source[arguments[0]["start"]:arguments[-1]["end"]]
                            new_kernel_launch = old_kernel_launch

                            kernel_params = argument_strings[5:]
                            for arg_idx, arg in enumerate(kernel_params):
                                if arg_idx in argument_types:
                                    arg = kernel_params[arg_idx].strip()
                                    the_type = argument_types[arg_idx]
                                    the_arg = arg.replace("\n", "").strip()
                                    if the_type in ["int", "const int", "int64_t", "THCIndex_t *", "const int *", "ptrdiff_t", "long", "const int64_t*", "int64_t *", "double"]:
                                        static_argument = "static_cast<%s>(%s)" % (the_type, the_arg)
                                        static_argument = arg.replace(the_arg, static_argument)

                                        # Update to static_cast
                                        new_kernel_launch = re.sub(r'\b(%s)\b' % arg, lambda x: static_argument, new_kernel_launch)

                            # Add template type
                            if "THCUNN" in filepath.split("/") and "generic" not in filepath.split("/"):
                                kernel_name_with_template = kernel_name_with_template.replace("<real>", "<Dtype>")
                            new_kernel_launch = re.sub(r'\b(%s)\b' % kernel_name, lambda x: kernel_name_with_template, new_kernel_launch)


                            # Replace Launch
                            new_output_source = new_output_source.replace(old_kernel_launch, new_kernel_launch)

                    # Overwrite file contents
                    fileobj.seek(0)
                    fileobj.write(new_output_source)
                    fileobj.truncate()
                    fileobj.flush()

                    # Flush to disk
                    os.fsync(fileobj)


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
    KernelTemplateParams = pytorch_specific_fixes(args.output_directory)

    # Start Preprocessor
    walk_over_directory(
        args.output_directory,
        extensions=args.extensions,
        show_detailed=args.show_detailed,
        exclude_dirs=["aten/src/TH", "aten/src/THNN", "aten/src/THS"])

    # Add static_casts
    add_static_casts(args, KernelTemplateParams)

    # Disable functions in certain files according to YAML description
    with open("/Users/gains/amd_build/disabled_funcs.yaml", "r") as lines:
        yaml_data = yaml.load(lines)

        for disable_info in yaml_data["disabled"]:
            filepath = os.path.join(args.output_directory, disable_info["path"])
            functions = disable_info["functions"]

            with open(filepath, "r+") as f:
                txt = f.read()
                for func in functions:
                    txt = disable_function(txt, func, 1)
                f.seek(0)
                f.write(txt)
                f.truncate()
                f.close()


if __name__ == '__main__':
    main()
