# IPI Arena OS upstream data

The behavior JSON files in `data/` come from
[`GraySwanAI/ipi_arena_os`](https://github.com/GraySwanAI/ipi_arena_os) at commit
`5e2e285910d581c97767195f1d113ef323f9ab07`.

They are vendored because version 0.1.0 of the upstream wheel does not include
its repository-level `data/` directory. Runtime logic remains in the upstream
`ipi-arena-bench` package. See `UPSTREAM_LICENSE` for its MIT license.
