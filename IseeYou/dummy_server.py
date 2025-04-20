import asyncio
import websockets
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - Server - %(message)s')

async def handler(websocket, path):
    client_id = websocket.remote_address
    logging.info(f"Client connected: {client_id}")
    message_count = 0
    try:
        async for message in websocket:
            message_count += 1
            received_time = time.monotonic()
            # Simulate *very minimal* processing - just get length
            msg_len = len(message)
            logging.info(f"Received message #{message_count} ({msg_len} bytes) from {client_id}")

            # Create a simple JSON response immediately
            response = json.dumps({
                "received_bytes": msg_len,
                "server_time": time.time(),
                "message_num": message_count
            })

            # Send response
            await websocket.send(response)
            send_time = time.monotonic()
            logging.info(f"Sent response for message #{message_count} to {client_id} (latency: {(send_time - received_time)*1000:.2f} ms)")

    except websockets.exceptions.ConnectionClosedOK:
        logging.info(f"Client {client_id} disconnected normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        logging.warning(f"Client {client_id} connection closed with error: {e}")
    except Exception as e:
        logging.error(f"Error handling client {client_id}: {e}", exc_info=True)
    finally:
        logging.info(f"--- Client {client_id} handler finished ---")

async def main():
    # Use same port as your real server for easy client testing
    port = 8080
    # Use slightly relaxed ping settings just for this test
    async with websockets.serve(handler, "0.0.0.0", port, ping_interval=20, ping_timeout=60):
        logging.info(f"Dummy Echo Server started on ws://0.0.0.0:{port}")
        await asyncio.Future() # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server stopped.")