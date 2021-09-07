import click

@click.command()
@click.option('--count', default=1)
@click.option('--name', prompt='Your name')
def hello(count, name):
    for _ in range(count):
        click.echo("Hi {}".format(name))


if __name__ == '__main__':
    hello()