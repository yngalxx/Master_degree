import click

from constants import General


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--main_dir",
    default=General.MAIN_DIR,
    type=str,
    help="Working directory path.",
    show_default=True,
)
def gui_runner(main_dir):
    pass 


if __name__ == "__main__":
    gui_runner()
