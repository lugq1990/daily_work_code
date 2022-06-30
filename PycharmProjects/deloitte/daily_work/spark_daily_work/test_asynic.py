import asyncio
import time

async def main():
    print('hi')
    await asyncio.sleep(1)
    print('world')
    
async def say_hi(delay, word):
    await asyncio.sleep(delay)
    print(word)
    

async def main():
    print(f"start at {time.strftime('%X')} ")
    
    await say_hi(1, 'hi')
    await say_hi(2, 'word')
    
    print(f"end at {time.strftime('%X')}")
    

async def main():
    task1 = asyncio.create_task(say_hi(1, 'hi'))
    task2 = asyncio.create_task(say_hi(2, 'world'))
    
    print(f"start at {time.strftime('%X')} ")
    
    await task1
    await task2

    print(f"end at {time.strftime('%X')}")
    
    
async def nested():
    print(10)

async def main():
    task = asyncio.create_task(nested())
    await task
    
asyncio.run(main())