import typer


# def main(name: str):
#     print(f" hello {name}")
    

app = typer.Typer()


@app.command()
def hello(name: str):
    print(f'hello {name}')
    
    
@app.command()
def greet(name: str, norm: bool=False):
    if norm:
        print(f"good: {name}")
    else:
        print(f"name")
        
        
    
if __name__ == '__main__':
    app()