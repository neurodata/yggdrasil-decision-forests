"""Eigen project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    http_archive(
        name = "eigen_archive",
        urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"],
        strip_prefix = "eigen-3.4.0",
        sha256 = "eba3f3d414d2f8cba2919c78ec6daab08fc71ba2ba4ae502b7e5d4d99fc02cda",
        build_file_content =
            """
cc_library(
    name = 'eigen3_internal',
    srcs = [],
    includes = ['.'],
    hdrs = glob(['Eigen/**']),
    visibility = ['//visibility:public'],
)
alias(
    name = "eigen3",
    actual = "@eigen_archive//:eigen3_internal",
    visibility = ["//visibility:public"],
)
""",
    )
