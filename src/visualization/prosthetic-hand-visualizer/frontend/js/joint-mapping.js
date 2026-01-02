// joint-mapping.js
export class HandJointMapper {
    constructor() {
        this.jointLimits = {
            // THUMB JOINTS
            thumb_0: { // CMC Flexion/Extension
                axis: 'x',
                min: -0.2,  // -15 degrees in radians
                max: 0.7,   // 40 degrees in radians
                mapping: (gloveValue) => this.mapLinear(gloveValue, -0.2, 0.7)
            },
            thumb_1: { // CMC Abduction/Adduction
                axis: 'z', 
                min: -0.3,  // -20 degrees
                max: 0.5,   // 30 degrees
                mapping: (gloveValue) => this.mapLinear(gloveValue, -0.3, 0.5)
            },
            thumb_2: { // MCP Flexion
                axis: 'x',
                min: 0,     // 0 degrees
                max: 1.2,   // 70 degrees
                mapping: (gloveValue) => this.mapLinear(gloveValue, 0, 1.2)
            },
            thumb_3: { // IP Flexion
                axis: 'x',
                min: 0,     // 0 degrees  
                max: 1.4,   // 80 degrees
                mapping: (gloveValue) => this.mapLinear(gloveValue, 0, 1.4)
            },
            
            // FINGER JOINTS (Index, Middle, Ring, Pinky follow similar patterns)
            index_0: { // MCP Flexion
                axis: 'x',
                min: 0,
                max: 1.4,   // 80 degrees
                mapping: (gloveValue) => this.mapLinear(gloveValue, 0, 1.4)
            },
            index_1: { // MCP Abduction
                axis: 'z',
                min: -0.35, // -20 degrees
                max: 0.35,  // 20 degrees
                mapping: (gloveValue) => this.mapLinear(gloveValue, -0.35, 0.35)
            },
            index_2: { // PIP Flexion
                axis: 'x',
                min: 0,
                max: 1.7,   // 100 degrees
                mapping: (gloveValue) => this.mapLinear(gloveValue, 0, 1.7)
            },
            index_3: { // DIP Flexion  
                axis: 'x',
                min: 0,
                max: 1.2,   // 70 degrees
                mapping: (gloveValue) => this.mapLinear(gloveValue, 0, 1.2)
            },
            
            // WRIST JOINTS
            wrist_0: { // Flexion/Extension
                axis: 'x',
                min: -0.7,  // -40 degrees
                max: 0.7,   // 40 degrees
                mapping: (gloveValue) => this.mapLinear(gloveValue, -0.7, 0.7)
            },
            wrist_1: { // Radial/Ulnar Deviation
                axis: 'z',
                min: -0.5,  // -30 degrees
                max: 0.3,   // 20 degrees
                mapping: (gloveValue) => this.mapLinear(gloveValue, -0.5, 0.3)
            }
        };
        
        // Bone naming convention that matches Blender export
        this.boneMapping = {
            // Thumb bones
            0: "thumb_0",  // CMC
            1: "thumb_0",  // CMC (second axis)
            2: "thumb_1",  // MCP
            3: "thumb_2",  // IP
            
            // Index finger
            4: "index_0",  // MCP flexion
            5: "index_0",  // MCP abduction  
            6: "index_1",  // PIP
            7: "index_2",  // DIP
            
            // Add middle, ring, pinky mappings...
            
            // Wrist
            20: "wrist_0", // Flexion/Extension
            21: "wrist_1"  // Radial/Ulnar
        };
    }
    
    mapLinear(gloveValue, minAngle, maxAngle) {
        // Convert 0-1 glove value to joint angle range
        return minAngle + (gloveValue * (maxAngle - minAngle));
    }
    
    applyRotationToBone(bone, axis, angle) {
        switch(axis) {
            case 'x':
                bone.rotation.x = angle;
                break;
            case 'y':
                bone.rotation.y = angle;
                break;
            case 'z':
                bone.rotation.z = angle;
                break;
        }
    }
    
    updateHandPose(handModel, gloveValues) {
        for (let i = 0; i < gloveValues.length; i++) {
            const boneName = this.boneMapping[i];
            const jointConfig = this.jointLimits[boneName];
            
            if (jointConfig && handModel.bones[boneName]) {
                const angle = jointConfig.mapping(gloveValues[i]);
                this.applyRotationToBone(handModel.bones[boneName], jointConfig.axis, angle);
            }
        }
    }
}