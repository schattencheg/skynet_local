"""Application entrypoint for launching Skynet Local."""

from skynet_local.bootstrap import build_runtime


def main() -> None:
    """Build the runtime container and start the main processing loop."""
    runtime = build_runtime()
    runtime.run()


if __name__ == "__main__":
    main()
