package(default_visibility = ["//visibility:public"])

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

cc_toolchain_config(
    name = "intel_toolchain_config",
)

cc_toolchain(
    name = "intel_cc_toolchain",
    toolchain_identifier = "intel-linux-toolchain",
    all_files = ":intel_toolchain_config",
    compiler_files = ":intel_toolchain_config",
    dwp_files = ":intel_toolchain_config",
    linker_files = ":intel_toolchain_config",
    strip_files = ":intel_toolchain_config",
    objcopy_files = ":intel_toolchain_config",
    toolchain_config = ":intel_toolchain_config",
)

cc_toolchain_suite(
    name = "intel_cc_toolchain_suite",
toolchains = {
    "local|local": ":intel_cc_toolchain",
}
)