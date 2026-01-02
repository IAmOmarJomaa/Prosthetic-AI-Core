// frontend/js/main.js
import * as THREE from 'three';

class SimpleHandVisualizer {
    constructor() {
        // Use the improved setupScene method
        this.setupScene();
        
        // Create simple geometric hands
        this.createHands();
        this.setupWebSocket();
        this.animate();
    }
    
    setupScene() {
        // Create scene, camera, renderer
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer();
        
        // Renderer setup
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x222222);
        document.body.appendChild(this.renderer.domElement);
        
        // Better camera position
        this.camera.position.set(0, 0.5, 4);
        this.camera.lookAt(0, 0, 0);
        
        // Improved lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 7);
        this.scene.add(directionalLight);
        
        // Add a floor for reference
        const floor = new THREE.Mesh(
            new THREE.PlaneGeometry(10, 10),
            new THREE.MeshPhongMaterial({ color: 0x444444 })
        );
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -1;
        this.scene.add(floor);
        
        // Add coordinate axes helper
        const axesHelper = new THREE.AxesHelper(1);
        this.scene.add(axesHelper);
    }
    
    createHands() {
        // Ground truth hand (left - GREEN)
        this.groundTruthHand = this.createSimpleHand(0x00ff00); // Green
        this.groundTruthHand.position.x = -1.5;
        
        // Prediction hand (right - BLUE)  
        this.predictionHand = this.createSimpleHand(0x0088ff); // Blue
        this.predictionHand.position.x = 1.5;
        
        this.scene.add(this.groundTruthHand);
        this.scene.add(this.predictionHand);
    }
    
    createSimpleHand(color) {
        const hand = new THREE.Group();
        
        // Improved Palm (more hand-like)
        const palm = new THREE.Mesh(
            new THREE.BoxGeometry(0.4, 0.15, 0.25),
            new THREE.MeshPhongMaterial({ color: color })
        );
        hand.add(palm);

        // Better finger creation with individual joints
        const fingerConfigs = [
            { name: 'thumb', pos: [0.12, 0, 0.08], lengths: [0.12, 0.1, 0.08] },
            { name: 'index', pos: [0.18, 0, 0.04], lengths: [0.15, 0.12, 0.09] },
            { name: 'middle', pos: [0.18, 0, 0], lengths: [0.16, 0.13, 0.1] },
            { name: 'ring', pos: [0.18, 0, -0.04], lengths: [0.14, 0.11, 0.09] },
            { name: 'pinky', pos: [0.15, 0, -0.08], lengths: [0.12, 0.09, 0.07] }
        ];

        // Initialize fingers array on this hand
        hand.fingers = [];

        fingerConfigs.forEach((config, fingerIndex) => {
            const finger = this.createRealisticFinger(config, color);
            finger.position.set(config.pos[0], config.pos[1], config.pos[2]);
            hand.add(finger);
            
            // Store finger reference for animation
            hand.fingers[fingerIndex] = finger;
        });

        return hand;
    }

    createRealisticFinger(config, color) {
        const finger = new THREE.Group();
        finger.name = config.name;

        let currentY = 0;
        
        // Create multiple segments for each finger joint
        config.lengths.forEach((length, segmentIndex) => {
            const segment = new THREE.Mesh(
                new THREE.CylinderGeometry(0.03, 0.03, length, 8),
                new THREE.MeshPhongMaterial({ color: color })
            );
            
            // Position each segment end-to-end
            segment.position.y = currentY + (length / 2);
            currentY += length;
            
            // Rotate to horizontal position
            segment.rotation.z = Math.PI / 2;
            
            finger.add(segment);
        });

        return finger;
    }
    
    setupWebSocket() {
        this.ws = new WebSocket('ws://localhost:8765');
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Update hand positions based on received data
            this.updateHandPose(this.groundTruthHand, data.ground_truth);
            this.updateHandPose(this.predictionHand, data.prediction);
            
            // Update metrics display
            document.getElementById('metrics').innerHTML = 
                `MSE: ${data.metrics.mse.toFixed(4)}`;
        };
        
        this.ws.onopen = () => console.log("Connected to server");
        this.ws.onerror = (error) => console.error("WebSocket error:", error);
    }
    
    updateHandPose(hand, gloveValues) {
        if (!hand.fingers) return;
        
        // Animate each finger based on glove values
        hand.fingers.forEach((finger, fingerIndex) => {
            const baseValueIndex = fingerIndex * 4; // 4 values per finger
            
            // Animate each segment in the finger
            finger.children.forEach((segment, segmentIndex) => {
                if (baseValueIndex + segmentIndex < gloveValues.length) {
                    // Make fingers curl based on glove values
                    const curlAmount = gloveValues[baseValueIndex + segmentIndex] * Math.PI / 3;
                    segment.rotation.x = curlAmount;
                }
            });
        });
        
        // Simple wrist movement (using last two glove values)
        if (gloveValues.length >= 22) {
            hand.rotation.x = (gloveValues[20] - 0.5) * Math.PI / 6; // Wrist flexion
            hand.rotation.z = (gloveValues[21] - 0.5) * Math.PI / 6; // Wrist deviation
        }
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }
}

// Start the application
new SimpleHandVisualizer();