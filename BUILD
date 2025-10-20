# Tell Bazel where to find the rule definition.
load("@hedron_compile_commands//:refresh_compile_commands.bzl",
     "refresh_compile_commands")

# Generate compile_commands.json only for the target(s) we care about.
refresh_compile_commands(
    name = "refresh_compile_commands",

    # If no extra flags are needed:
    targets = ["//examples:train_oblique_forest",
            "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_learner",
],

    # ‑---- OR, if you normally build that binary with extra --config flags ‑----
    # targets = {
    #     "//examples:train_oblique_forest":
    #         "--config=linux_cpp17 --config=linux_avx2",
    # },
)