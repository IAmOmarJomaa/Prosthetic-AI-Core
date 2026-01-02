# backend/server.py
import asyncio
import websockets
import json
import random  # We'll use fake data first

async def send_hand_data(websocket):
    print("Client connected")
    try:
        while True:
            # Generate fake hand data for testing
            ground_truth = [random.random() for _ in range(22)]  # 22 random values 0-1
            prediction = [random.random() for _ in range(22)]    # 22 random values 0-1
            
            data = {
                "ground_truth": ground_truth,
                "prediction": prediction,
                "metrics": {"mse": random.random() * 0.1}
            }
            
            await websocket.send(json.dumps(data))
            await asyncio.sleep(0.1)  # Send 10 times per second
            
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(send_hand_data, "localhost", 8765):
        print("WebSocket server running on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())