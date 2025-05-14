// csv_to_ydf.cc  ─────────────────────────────────────────────────────────
// Usage:
//   bazel run :csv_to_ydf -- \
//        --csv_path=path/to/file.csv \
//        --label_col=Target \
//        --out_path=<non-existing-folder-path> // MAKE SURE FOLDER DOESN'T EXIST!
// Generates  /tmp/synth_ds/{data_spec.pb, shard-00000-of-00001}

#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"

using namespace yggdrasil_decision_forests;

ABSL_FLAG(std::string, csv_path,  "", "Path to input CSV (required)");
ABSL_FLAG(std::string, label_col, "", "Name of label column (required)");
ABSL_FLAG(std::string, out_path,  "", "Output directory, e.g. /tmp/ds (required)");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string csv  = absl::GetFlag(FLAGS_csv_path);
  const std::string lbl  = absl::GetFlag(FLAGS_label_col);
  const std::string out  = absl::GetFlag(FLAGS_out_path);

  if (csv.empty() || lbl.empty() || out.empty()) {
    std::cerr << "csv_to_ydf --csv_path=... --label_col=... --out_path=...\n";
    return 1;
  }

  // ── 1. infer DataSpec from CSV (same code you already trust) ──────────
  dataset::proto::DataSpecification spec;
  dataset::proto::DataSpecificationGuide guide;
  auto* g = guide.add_column_guides();
  g->set_column_name_pattern(lbl);
  g->set_type(dataset::proto::CATEGORICAL);

  dataset::CreateDataSpec("csv:" + csv,
                                   /*require_same_dataset_fields=*/false,
                                   guide, &spec);

  // ── 2. read the CSV into a VerticalDataset ────────────────────────────
  dataset::VerticalDataset ds;
  CHECK_OK(dataset::LoadVerticalDataset("csv:" + csv, spec, &ds));

  // ── 3. save in YDF native format (“tfrecord:” prefix) ─────────────────
  const std::string out_dir = absl::GetFlag(FLAGS_out_path);   // must not exist
  // 3. save the TF-Record (creates one file at FLAGS_out_path)
CHECK_OK(dataset::SaveVerticalDataset(ds, "tfrecord:" + out));

// 4. save the spec *next to* it
CHECK_OK(file::SetBinaryProto(out + ".data_spec.pb", spec, file::Defaults()));



  std::cout << "✓ wrote dataset to  tfrecord:" << out << '\n';
  return 0;
}
