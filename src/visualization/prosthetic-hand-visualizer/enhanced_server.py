# enhanced_server.py
import numpy as np

class BiomechanicalConstraints:
    """Apply biomechanical constraints to joint angles"""
    
    @staticmethod
    def apply_finger_constraints(glove_values):
        """Ensure finger movements are biomechanically realistic"""
        constrained_values = glove_values.copy()
        
        # Thumb constraints
        constrained_values[0] = np.clip(glove_values[0], 0, 0.8)  # CMC flexion
        constrained_values[1] = np.clip(glove_values[1], 0, 0.6)  # CMC abduction
        
        # Finger coupling constraints (PIP/DIP relationship)
        for finger_start in [4, 8, 12, 16]:  # MCP indices for each finger
            pip_index = finger_start + 2
            dip_index = finger_start + 3
            
            # Ensure DIP flexion doesn't exceed PIP flexion
            if constrained_values[dip_index] > constrained_values[pip_index]:
                constrained_values[dip_index] = constrained_values[pip_index] * 0.8
        
        return constrained_values

# Enhanced WebSocket endpoint with constraints
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    biomechanical = BiomechanicalConstraints()
    
    try:
        while True:
            data = await websocket.receive_text()
            if data == "next_frame":
                # ... existing data loading code ...
                
                # Apply biomechanical constraints
                constrained_truth = biomechanical.apply_finger_constraints(ground_truth_glove_values)
                constrained_pred = biomechanical.apply_finger_constraints(predicted_glove_values)
                
                payload = {
                    "ground_truth": constrained_truth.tolist(),
                    "prediction": constrained_pred.tolist(),
                    "metrics": {
                        "mse": mse,
                        "constrained_mse": calculate_mse(constrained_truth, constrained_pred)
                    }
                }
                
                await websocket.send_text(json.dumps(payload))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)