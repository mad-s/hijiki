use std::error::Error;

extern crate spirv_builder;
use spirv_builder::{SpirvBuilder, MetadataPrintout};

fn main() -> Result<(), Box<dyn Error>> {
    let result = SpirvBuilder::new("crates/hijiki-kernel", "spirv-unknown-spv1.5")
        .print_metadata(MetadataPrintout::Full)
        .build()?;
    std::fs::copy(result.module.unwrap_single(), "./shader.spv")?;
    Ok(())
}
