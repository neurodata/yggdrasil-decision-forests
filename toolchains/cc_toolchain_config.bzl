load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
     "tool_path", "feature", "flag_group", "flag_set")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

all_link_actions = [
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

def _impl(ctx):
    tool_paths = [
        tool_path(name = "gcc", path = "/opt/intel/oneapi/compiler/2025.1/bin/icx"),
        tool_path(name = "ld", path = "/opt/intel/oneapi/compiler/2025.1/bin/icx"),
        tool_path(name = "ar", path = "/usr/bin/ar"),
        tool_path(name = "cpp", path = "/opt/intel/oneapi/compiler/2025.1/bin/icx"),
        tool_path(name = "gcov", path = "/bin/false"),
        tool_path(name = "nm", path = "/usr/bin/nm"),
        tool_path(name = "objdump", path = "/usr/bin/objdump"),
        tool_path(name = "strip", path = "/usr/bin/strip"),
    ]

    features = [
        feature(
            name = "default_linker_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [
                        flag_group(flags = ["-lstdc++"])
                    ]
                ),
            ],
        ),
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        cxx_builtin_include_directories = [
            "/opt/intel/oneapi/compiler/2025.1/linux/include",
            "/opt/intel/oneapi/compiler/2025.1/linux/include/sycl",
            "/usr/include",
        ],
        toolchain_identifier = "intel-linux-toolchain",
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "local",
        target_libc = "glibc",
        compiler = "intel",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        tool_paths = tool_paths,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)