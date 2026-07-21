import asyncio
import edge_tts

async def main():
    communicate = edge_tts.Communicate("Hello world, this is a test.", "en-US-AndrewNeural")
    await communicate.save("test.mp3")
    print("Done - saved test.mp3")

asyncio.run(main())