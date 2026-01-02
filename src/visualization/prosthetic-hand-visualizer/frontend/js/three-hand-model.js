// three-hand-model.js
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { HandJointMapper } from './joint-mapping.js';

export class ProstheticHandVisualizer {
    constructor(scene) {
        this.scene = scene;
        this.handModel = null;
        this.bones = {};
        this.jointMapper = new HandJointMapper();
        this.loader = new GLTFLoader();
    }
    
    async loadHandModel(modelPath) {
        return new Promise((resolve, reject) => {
            this.loader.load(
                modelPath,
                (gltf) => {
                    this.handModel = gltf.scene;
                    this.scene.add(this.handModel);
                    
                    // Extract and store bones by name
                    this.handModel.traverse((child) => {
                        if (child.isBone) {
                            this.bones[child.name] = child;
                        }
                    });
                    
                    console.log('Hand model loaded with bones:', Object.keys(this.bones));
                    resolve(this.handModel);
                },
                undefined,
                (error) => {
                    console.error('Error loading hand model:', error);
                    reject(error);
                }
            );
        });
    }
    
    updatePose(gloveValues) {
        if (!this.handModel) return;
        
        // Update bone rotations based on glove values
        this.jointMapper.updateHandPose(this, gloveValues);
        
        // Update matrix world for proper rendering
        this.handModel.traverse((child) => {
            if (child.isBone) {
                child.updateMatrixWorld();
            }
        });
    }
    
    createVisualHand() {
        // Fallback: Create a simple procedural hand if no model is available
        const handGroup = new THREE.Group();
        
        // Create palm
        const palmGeometry = new THREE.BoxGeometry(0.3, 0.1, 0.15);
        const palmMaterial = new THREE.MeshPhongMaterial({ color: 0xffaa00 });
        const palm = new THREE.Mesh(palmGeometry, palmMaterial);
        handGroup.add(palm);
        
        // Create simple finger bones
        this.createFinger(handGroup, 'thumb', new THREE.Vector3(0.1, 0, 0.1));
        this.createFinger(handGroup, 'index', new THREE.Vector3(0.15, 0, 0.05));
        // Add other fingers...
        
        this.handModel = handGroup;
        this.scene.add(this.handModel);
        
        return handGroup;
    }
    
    createFinger(parent, name, position) {
        const fingerGroup = new THREE.Group();
        fingerGroup.position.copy(position);
        
        // Create bone segments
        const segmentGeometry = new THREE.CylinderGeometry(0.02, 0.02, 0.1, 8);
        const material = new THREE.MeshPhongMaterial({ color: 0xffaa00 });
        
        for (let i = 0; i < 3; i++) {
            const segment = new THREE.Mesh(segmentGeometry, material);
            segment.position.y = i * 0.1;
            fingerGroup.add(segment);
        }
        
        parent.add(fingerGroup);
        return fingerGroup;
    }
}