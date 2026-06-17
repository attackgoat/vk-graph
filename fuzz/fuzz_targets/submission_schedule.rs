#![no_main]

use {
    arbitrary::{Arbitrary, Unstructured},
    libfuzzer_sys::fuzz_target,
    vk_graph::submission::fuzz::{ResourceAccess, check_schedule_reordering},
};

#[derive(Debug)]
struct FuzzCase {
    pass_count: usize,
    resource_accesses: Vec<Vec<ResourceAccess>>,
}

impl<'a> Arbitrary<'a> for FuzzCase {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let pass_count = usize::from(u.int_in_range::<u8>(0..=64)?);
        let resource_count = usize::from(u.int_in_range::<u8>(0..=48)?);

        let mut resource_accesses = Vec::with_capacity(resource_count);
        for _ in 0..resource_count {
            let use_count = usize::from(u.int_in_range::<u8>(0..=96)?);
            let mut accesses = Vec::with_capacity(use_count);
            for _ in 0..use_count {
                accesses.push(ResourceAccess {
                    cmd_idx: usize::from(u.arbitrary::<u16>()?),
                    write: u.arbitrary()?,
                });
            }

            resource_accesses.push(accesses);
        }

        Ok(Self {
            pass_count,
            resource_accesses,
        })
    }
}

fuzz_target!(|case: FuzzCase| {
    check_schedule_reordering(case.pass_count, &case.resource_accesses);
});
