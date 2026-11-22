// End-to-end check against the tree printed in the benepar README.
// Needs the model: BENEPAR_MODEL=/path/to/benepar_en3 cargo test -- --ignored

use std::process::Command;

#[test]
#[ignore]
fn readme_sentence() {
    let model = std::env::var("BENEPAR_MODEL").expect("set BENEPAR_MODEL");
    let out = Command::new(env!("CARGO_BIN_EXE_benepar-rs"))
        .args(["--model", &model, "The time for action is now."])
        .output()
        .expect("run benepar-rs");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "(S (NP (NP (DT The) (NN time)) (PP (IN for) (NP (NN action)))) (VP (VBZ is) (ADVP (RB now))) (. .))"
    );
}
