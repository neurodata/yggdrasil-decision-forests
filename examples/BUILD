package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Usage example of Yggdrasil Decision Forests as a library.
#
# Compile and run the example with:
#   bazel build //examples:beginner_cc --config=linux_cpp17 --config=linux_avx2
#   bazel-bin/examples/beginner_cc --alsologtostderr
#
# See the "Using the C++ library" section in the user manual for more details about the API.
# See the "Compile command-line-interface from source" section in the user manual for more details about the compilation flags.
#
cc_binary(
    name = "beginner_cc",
    srcs = ["beginner.cc"],
    # Because this binary is defined in the Yggdrasil project directly, the
    # following dependencies are local. In your project you will probably use
    # a "http_archive" dependency:
    # https://docs.bazel.build/versions/master/repo/http.html#http_archive*
    #
    # See TensorFlow Decision Forests for an example (TF-DF imports YDF):
    #   https://github.com/tensorflow/decision-forests
    deps = [
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/learner:all_learners",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/metric:report",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
"//yggdrasil_decision_forests/dataset:csv_example_reader",
    ],
)

cc_binary(
    name = "fast_8bits_numerical",
    srcs = ["fast_8bits_numerical.cc"],
    deps = [
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/metric:report",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/serving/decision_forest:8bits_numerical_features",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)


cc_binary(
    name = "train_oblique_forest",
    srcs = ["train_oblique_forest.cc"],
    # copts = ["-pg"], # Ariel - for profiling
    # linkopts = ["-pg"],
    # features = ["c++17"],
    deps = [
"//yggdrasil_decision_forests/learner/random_forest:random_forest",
"//yggdrasil_decision_forests/learner:learner_library",

"//yggdrasil_decision_forests/model/random_forest:random_forest",
"//yggdrasil_decision_forests/model:model_library",

        "//yggdrasil_decision_forests/dataset:data_spec",
"//yggdrasil_decision_forests/dataset:csv_example_reader",
    "//yggdrasil_decision_forests/utils:logging",
    "//yggdrasil_decision_forests/utils:distribution",
    "@com_google_absl//absl/log",

        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)


cc_binary(
    name = "convert_csv_to_tfrecord",
    srcs = ["convert_csv_to_tfrecord.cc"],
    deps = [
        # ── YDF helpers
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:csv_example_reader",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/dataset/tensorflow_no_dep:tf_record_tf_example",
        # ── Abseil
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
    ],
    features = ["c++17"],
)