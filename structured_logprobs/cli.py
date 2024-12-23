"""Console script for structured_logprobs."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("structured-logprobs")
    click.echo("=" * len("structured-logprobs"))
    click.echo("OpenAI's Structured Outputs with Logprobs")


if __name__ == "__main__":
    main()  # pragma: no cover
