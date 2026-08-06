#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() -> anyhow::Result<()> {
    tofy::alloc::init();
    let _perf = tofy::perf::install()?;
    tofy::experiment::run_cli()
}
