import asyncio 
import time

async def f1():
    print('f1 start')
    await asyncio.sleep(1)
    print('f1 finish')
    
async def f2():
    print("f2 start")
    await asyncio.sleep(2)
    print("f2 end")
    
    
async def main():
    loop = asyncio.get_event_loop()
    
    
    t1 = asyncio.create_task(f1())
    t2 = asyncio.create_task(f2())
    
    await asyncio.gather(t1, t2)
    

asyncio.run(main())