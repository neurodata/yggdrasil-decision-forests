cc_binary(
    name = "fisher-yates-vs-binomial-floyds",
    srcs = ["fisher-yates-vs-binomial-floyds.cc"],
    deps = [
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto", 
        # "//yggdrasil_decision_forests/model:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/utils:random",
        "@com_google_absl//absl/types:span",
    ],
)