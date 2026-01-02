// websocket-client.js
export class WebSocketClient {
    constructor(url) {
        this.url = url;
        this.socket = null;
        this.onMessageCallback = null;
    }

    connect() {
        this.socket = new WebSocket(this.url);

        this.socket.onopen = (event) => {
            console.log("WebSocket connection established");
            // Request the first frame
            this.socket.send("next_frame");
        };

        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Pass the data to your visualization update function
            if (this.onMessageCallback) {
                this.onMessageCallback(data);
            }
            
            // Optionally, automatically request the next frame after a delay
            setTimeout(() => {
                if (this.socket.readyState === WebSocket.OPEN) {
                    this.socket.send("next_frame");
                }
            }, 100); // 10 FPS for stability, adjust as needed
        };

        this.socket.onclose = (event) => {
            console.log("WebSocket connection closed");
        };

        this.socket.onerror = (error) => {
            console.error("WebSocket error:", error);
        };
    }

    setOnMessageCallback(callback) {
        this.onMessageCallback = callback;
    }

    disconnect() {
        if (this.socket) {
            this.socket.close();
        }
    }
}