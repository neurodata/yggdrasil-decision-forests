load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
"tool_path", "feature", "flag_group", "flag_set")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

all_link_actions = [
ACTION_NAMES.cpp_link_executable,
ACTION_NAMES.cpp_link_dynamic_library,
ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

all_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
]

def _impl(ctx):
    tool_paths = [
        tool_path(name = "gcc", path = "/opt/intel/oneapi/compiler/2025.2/bin/icx"),
        tool_path(name = "ld", path = "/opt/intel/oneapi/compiler/2025.2/bin/icx"),
        tool_path(name = "ar", path = "/usr/bin/ar"),
        tool_path(name = "cpp", path = "/opt/intel/oneapi/compiler/2025.2/bin/icx"),
        tool_path(name = "gcov", path = "/bin/false"),
        tool_path(name = "nm", path = "/usr/bin/nm"),
        tool_path(name = "objdump", path = "/usr/bin/objdump"),
        tool_path(name = "strip", path = "/usr/bin/strip"),
    ]
    
    features = [    # This will always apply (you marked this one enabled=True).
        feature(
            name = "default_linker_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = ["-lstdc++"])],
                ),
            ],
        ),
        # Enabled automatically when you pass -c opt
        feature(
            name = "opt",
            flag_sets = [
                flag_set(
                    actions = all_compile_actions,
                    flag_groups = [flag_group(flags = ["-O2", "-DNDEBUG"])],
                ),
            ],
        ),
        # Enabled automatically when you pass -c dbg
        feature(
            name = "dbg",
            flag_sets = [
                flag_set(
                    actions = all_compile_actions,
                    flag_groups = [flag_group(flags = ["-g", "-O0"])],
                ),
            ],
        ),
        # Enabled automatically when you pass -c fastbuild
        feature(
            name = "fastbuild",
            flag_sets = [
                flag_set(
                    actions = all_compile_actions,
                    flag_groups = [flag_group(flags = ["-O2"])],
                ),
            ],
        ),
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        cxx_builtin_include_directories = [
            "/opt/intel/oneapi/compiler/2025.2/lib/clang/21/include",
            "/opt/intel/oneapi/compiler/2025.2/opt/compiler/include",
            "/opt/intel/oneapi/compiler/2025.2/linux/include",
            "/opt/intel/oneapi/compiler/2025.2/linux/include/sycl",
            "/usr/include",
        ],
        toolchain_identifier = "intel-linux-toolchain",
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "x86_64",
        target_libc = "glibc",
        compiler = "icx",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        tool_paths = tool_paths,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)