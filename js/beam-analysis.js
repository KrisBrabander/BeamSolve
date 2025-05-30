/**
 * Beam Analysis Module
 * Interactive beam analysis with support for various profiles, loads, and supports
 */

// Main Beam Analysis Class
class BeamAnalysis {
    constructor() {
        // Beam properties
        this.length = 6; // meters
        this.eModulus = 210000; // MPa (N/mm²)
        this.momentOfInertia = 8360000; // mm⁴
        
        // Canvas and interaction properties
        this.canvas = null;
        this.ctx = null;
        this.scale = 1;
        this.offsetX = 50;
        this.offsetY = 100;
        this.beamHeight = 20;
        
        // Elements on the beam
        this.supports = [];
        this.loads = [];
        
        // Currently selected element
        this.selectedElement = null;
        this.dragStartX = 0;
        this.dragStartY = 0;
        this.isDragging = false;
        
        // Analysis results
        this.results = {
            reactions: [],
            shear: [],
            moment: [],
            deflection: [],
            rotation: []
        };
        
        // Calculation resolution
        this.numPoints = 100;
    }
    
    // Initialize the beam analysis module
    init() {
        console.log('Initializing beam analysis module');
        this.initCanvas();
        this.initEventListeners();
        this.initProfileSelector();
        this.drawBeam();
    }
    
    // Initialize the canvas
    initCanvas() {
        this.canvas = document.getElementById('beam-canvas');
        if (!this.canvas) {
            console.error('Beam canvas not found');
            return;
        }
        
        this.ctx = this.canvas.getContext('2d');
        
        // Set canvas dimensions
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }
    
    // Resize canvas to fit container
    resizeCanvas() {
        if (!this.canvas) return;
        
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = 300;
        
        // Calculate scale factor based on beam length and canvas width
        this.scale = (this.canvas.width - this.offsetX * 2) / this.length;
        
        // Redraw beam after resize
        this.drawBeam();
    }
    
    // Initialize event listeners for user interaction
    initEventListeners() {
        if (!this.canvas) return;
        
        // Canvas event listeners for drag and drop functionality
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', () => this.handleMouseUp());
        this.canvas.addEventListener('mouseleave', () => this.handleMouseUp());
        
        // Button event listeners
        document.getElementById('add-support')?.addEventListener('click', () => this.addSupport());
        document.getElementById('add-load')?.addEventListener('click', () => this.addLoad());
        document.getElementById('clear-beam')?.addEventListener('click', () => this.clearBeam());
        document.getElementById('analyze-beam')?.addEventListener('click', () => this.analyze());
        document.getElementById('export-analysis')?.addEventListener('click', () => this.exportToPDF());
        
        // Property panel event listeners
        document.getElementById('close-properties')?.addEventListener('click', () => this.closePropertiesPanel());
        document.getElementById('apply-support')?.addEventListener('click', () => this.applySupport());
        document.getElementById('apply-load')?.addEventListener('click', () => this.applyLoad());
        document.getElementById('delete-support')?.addEventListener('click', () => this.deleteSupport());
        document.getElementById('delete-load')?.addEventListener('click', () => this.deleteLoad());
        
        // Load type change event
        document.getElementById('load-type')?.addEventListener('change', (e) => this.handleLoadTypeChange(e));
        
        // Beam length change event
        document.getElementById('beam-length')?.addEventListener('input', (e) => {
            this.length = parseFloat(e.target.value) || 6;
            this.resizeCanvas();
            this.analyze();
        });
        
        // Custom properties change events
        document.getElementById('e-modulus')?.addEventListener('input', (e) => {
            this.eModulus = parseFloat(e.target.value) || 210000;
            this.analyze();
        });
        
        document.getElementById('moment-of-inertia')?.addEventListener('input', (e) => {
            this.momentOfInertia = parseFloat(e.target.value) || 8360000;
            this.analyze();
        });
    }
    
    // Initialize profile selector
    initProfileSelector() {
        const profileSelector = document.getElementById('profile-selector');
        const customPropertiesDiv = document.getElementById('custom-properties');
        
        if (profileSelector && customPropertiesDiv) {
            profileSelector.addEventListener('change', (e) => {
                const selectedValue = e.target.value;
                
                if (selectedValue === 'custom') {
                    customPropertiesDiv.style.display = 'block';
                } else {
                    customPropertiesDiv.style.display = 'none';
                    
                    // Set properties based on selected profile
                    const selectedProfile = this.getProfileProperties(selectedValue);
                    if (selectedProfile) {
                        this.eModulus = selectedProfile.eModulus;
                        this.momentOfInertia = selectedProfile.momentOfInertia;
                        this.analyze();
                    }
                }
            });
        }
    }
    
    // Get properties for a specific profile
    getProfileProperties(profileId) {
        // Common steel profiles with their properties
        const profiles = {
            'ipn100': { eModulus: 210000, momentOfInertia: 171000 },
            'ipn200': { eModulus: 210000, momentOfInertia: 2140000 },
            'ipn300': { eModulus: 210000, momentOfInertia: 9800000 },
            'hea100': { eModulus: 210000, momentOfInertia: 349000 },
            'hea200': { eModulus: 210000, momentOfInertia: 3690000 },
            'hea300': { eModulus: 210000, momentOfInertia: 18300000 },
            'heb100': { eModulus: 210000, momentOfInertia: 450000 },
            'heb200': { eModulus: 210000, momentOfInertia: 5700000 },
            'heb300': { eModulus: 210000, momentOfInertia: 25200000 }
        };
        
        return profiles[profileId] || null;
    }
    
    // Draw the beam on the canvas
    drawBeam() {
        if (!this.ctx || !this.canvas) return;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw beam
        this.ctx.beginPath();
        this.ctx.rect(
            this.offsetX, 
            this.offsetY - this.beamHeight / 2, 
            this.length * this.scale, 
            this.beamHeight
        );
        this.ctx.fillStyle = '#e0e0e0';
        this.ctx.fill();
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        // Draw ruler
        this.drawRuler();
        
        // Draw supports
        this.supports.forEach(support => this.drawSupport(support));
        
        // Draw loads
        this.loads.forEach(load => this.drawLoad(load));
    }
    
    // Draw ruler below the beam
    drawRuler() {
        if (!this.ctx) return;
        
        const rulerY = this.offsetY + 30;
        const tickHeight = 5;
        const meterWidth = this.scale; // Width of 1 meter in pixels
        
        this.ctx.beginPath();
        this.ctx.moveTo(this.offsetX, rulerY);
        this.ctx.lineTo(this.offsetX + this.length * this.scale, rulerY);
        this.ctx.strokeStyle = '#999';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
        
        // Draw ticks and labels
        this.ctx.textAlign = 'center';
        this.ctx.font = '10px Arial';
        this.ctx.fillStyle = '#666';
        
        for (let i = 0; i <= this.length; i++) {
            const x = this.offsetX + i * meterWidth;
            
            // Draw tick
            this.ctx.beginPath();
            this.ctx.moveTo(x, rulerY - tickHeight);
            this.ctx.lineTo(x, rulerY + tickHeight);
            this.ctx.stroke();
            
            // Draw label
            this.ctx.fillText(i + 'm', x, rulerY + 20);
        }
    }
    
    // Draw a support on the beam
    drawSupport(support) {
        if (!this.ctx) return;
        
        const x = this.offsetX + support.position * this.scale;
        const y = this.offsetY + this.beamHeight / 2;
        
        this.ctx.save();
        
        // Highlight if selected
        if (this.selectedElement && this.selectedElement.type === 'support' && 
            this.selectedElement.id === support.id) {
            this.ctx.shadowColor = '#4285f4';
            this.ctx.shadowBlur = 10;
        }
        
        // Draw based on support type
        switch (support.type) {
            case 'fixed': // Fixed support (encastrement)
                this.ctx.beginPath();
                this.ctx.rect(x - 5, y, 10, 30);
                this.ctx.fillStyle = '#555';
                this.ctx.fill();
                
                // Draw hatching
                this.ctx.beginPath();
                for (let i = 0; i < 6; i++) {
                    this.ctx.moveTo(x - 15, y + 5 + i * 5);
                    this.ctx.lineTo(x + 15, y + 5 + i * 5);
                }
                this.ctx.strokeStyle = '#333';
                this.ctx.lineWidth = 1;
                this.ctx.stroke();
                break;
                
            case 'pinned': // Pinned support (articulation)
                // Draw triangle
                this.ctx.beginPath();
                this.ctx.moveTo(x, y);
                this.ctx.lineTo(x - 15, y + 25);
                this.ctx.lineTo(x + 15, y + 25);
                this.ctx.closePath();
                this.ctx.fillStyle = '#555';
                this.ctx.fill();
                this.ctx.strokeStyle = '#333';
                this.ctx.lineWidth = 1;
                this.ctx.stroke();
                
                // Draw base
                this.ctx.beginPath();
                this.ctx.rect(x - 20, y + 25, 40, 5);
                this.ctx.fillStyle = '#555';
                this.ctx.fill();
                this.ctx.stroke();
                break;
                
            case 'roller': // Roller support
                // Draw triangle
                this.ctx.beginPath();
                this.ctx.moveTo(x, y);
                this.ctx.lineTo(x - 15, y + 20);
                this.ctx.lineTo(x + 15, y + 20);
                this.ctx.closePath();
                this.ctx.fillStyle = '#555';
                this.ctx.fill();
                this.ctx.strokeStyle = '#333';
                this.ctx.lineWidth = 1;
                this.ctx.stroke();
                
                // Draw circles (rollers)
                this.ctx.beginPath();
                this.ctx.arc(x - 10, y + 25, 5, 0, Math.PI * 2);
                this.ctx.arc(x + 10, y + 25, 5, 0, Math.PI * 2);
                this.ctx.fillStyle = '#777';
                this.ctx.fill();
                this.ctx.stroke();
                break;
        }
        
        // Draw position label
        this.ctx.fillStyle = '#333';
        this.ctx.font = '10px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`${support.position.toFixed(1)}m`, x, y + 45);
        
        this.ctx.restore();
    }
    
    // Draw a load on the beam
    drawLoad(load) {
        if (!this.ctx) return;
        
        // Highlight if selected
        if (this.selectedElement && this.selectedElement.type === 'load' && 
            this.selectedElement.id === load.id) {
            this.ctx.shadowColor = '#4285f4';
            this.ctx.shadowBlur = 10;
        }
        
        switch (load.type) {
            case 'point':
                this.drawPointLoad(load);
                break;
            case 'distributed':
                this.drawDistributedLoad(load);
                break;
            case 'moment':
                this.drawMomentLoad(load);
                break;
        }
        
        this.ctx.shadowColor = 'transparent';
        this.ctx.shadowBlur = 0;
    }
    
    // Draw a point load
    drawPointLoad(load) {
        if (!this.ctx) return;
        
        const x = this.offsetX + load.position * this.scale;
        const y = this.offsetY - this.beamHeight / 2;
        const arrowLength = 40;
        const arrowHeadSize = 8;
        
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
        this.ctx.lineTo(x, y - arrowLength);
        
        // Arrow head
        this.ctx.lineTo(x - arrowHeadSize, y - arrowLength + arrowHeadSize);
        this.ctx.moveTo(x, y - arrowLength);
        this.ctx.lineTo(x + arrowHeadSize, y - arrowLength + arrowHeadSize);
        
        this.ctx.strokeStyle = '#e53935';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        // Draw value label
        this.ctx.fillStyle = '#e53935';
        this.ctx.font = 'bold 12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`${Math.abs(load.value)}kN`, x, y - arrowLength - 10);
        
        // Draw position label
        this.ctx.fillStyle = '#666';
        this.ctx.font = '10px Arial';
        this.ctx.fillText(`${load.position.toFixed(1)}m`, x, y - 15);
    }
    
    // Draw a distributed load
    drawDistributedLoad(load) {
        if (!this.ctx) return;
        
        const startX = this.offsetX + load.position * this.scale;
        const endX = this.offsetX + load.endPosition * this.scale;
        const y = this.offsetY - this.beamHeight / 2;
        const arrowLength = 30;
        const arrowHeadSize = 6;
        const arrowSpacing = 30; // Pixels between arrows
        
        // Draw distributed load arrows
        this.ctx.strokeStyle = '#e53935';
        this.ctx.lineWidth = 2;
        
        // Draw top line
        this.ctx.beginPath();
        this.ctx.moveTo(startX, y - arrowLength);
        this.ctx.lineTo(endX, y - arrowLength);
        this.ctx.stroke();
        
        // Calculate number of arrows to draw
        const loadWidth = endX - startX;
        const numArrows = Math.max(2, Math.floor(loadWidth / arrowSpacing));
        const actualSpacing = loadWidth / (numArrows - 1);
        
        // Draw arrows
        for (let i = 0; i < numArrows; i++) {
            const arrowX = startX + i * actualSpacing;
            
            this.ctx.beginPath();
            this.ctx.moveTo(arrowX, y - arrowLength);
            this.ctx.lineTo(arrowX, y);
            
            // Arrow head
            this.ctx.lineTo(arrowX - arrowHeadSize, y - arrowHeadSize);
            this.ctx.moveTo(arrowX, y);
            this.ctx.lineTo(arrowX + arrowHeadSize, y - arrowHeadSize);
            
            this.ctx.stroke();
        }
        
        // Draw value label
        this.ctx.fillStyle = '#e53935';
        this.ctx.font = 'bold 12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`${Math.abs(load.value)}kN/m`, (startX + endX) / 2, y - arrowLength - 10);
        
        // Draw position labels
        this.ctx.fillStyle = '#666';
        this.ctx.font = '10px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`${load.position.toFixed(1)}m`, startX, y - 15);
        this.ctx.fillText(`${load.endPosition.toFixed(1)}m`, endX, y - 15);
    }
    
    // Draw a moment load
    drawMomentLoad(load) {
        if (!this.ctx) return;
        
        const x = this.offsetX + load.position * this.scale;
        const y = this.offsetY;
        const radius = 25;
        
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 1.5, false);
        this.ctx.strokeStyle = '#9c27b0';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        // Draw arrow head
        const arrowHeadSize = 8;
        const arrowX = x;
        const arrowY = y - radius;
        
        this.ctx.beginPath();
        this.ctx.moveTo(arrowX, arrowY);
        this.ctx.lineTo(arrowX - arrowHeadSize, arrowY - arrowHeadSize);
        this.ctx.moveTo(arrowX, arrowY);
        this.ctx.lineTo(arrowX + arrowHeadSize, arrowY - arrowHeadSize);
        this.ctx.stroke();
        
        // Draw value label
        this.ctx.fillStyle = '#9c27b0';
        this.ctx.font = 'bold 12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`${Math.abs(load.value)}kNm`, x, y - radius - 10);
        
        // Draw position label
        this.ctx.fillStyle = '#666';
        this.ctx.font = '10px Arial';
        this.ctx.fillText(`${load.position.toFixed(1)}m`, x, y + radius + 15);
    }
    
    // Handle mouse down event
    handleMouseDown(e) {
        if (!this.canvas) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        // Check if clicking on an existing element
        const element = this.findElementAtPosition(mouseX, mouseY);
        
        if (element) {
            // Select the element
            this.selectedElement = element;
            this.dragStartX = mouseX;
            this.dragStartY = mouseY;
            this.isDragging = true;
            
            // Show properties panel
            this.showPropertiesPanel(element);
        } else {
            // Deselect if clicking on empty space
            this.selectedElement = null;
            this.closePropertiesPanel();
        }
        
        this.drawBeam();
    }
    
    // Find element (support or load) at a specific position
    findElementAtPosition(x, y) {
        // Check supports
        for (const support of this.supports) {
            const supportX = this.offsetX + support.position * this.scale;
            const supportY = this.offsetY + this.beamHeight / 2;
            
            // Check if within support hitbox
            if (Math.abs(x - supportX) < 20 && y > supportY && y < supportY + 45) {
                return { type: 'support', id: support.id, data: support };
            }
        }
        
        // Check loads
        for (const load of this.loads) {
            if (load.type === 'point') {
                const loadX = this.offsetX + load.position * this.scale;
                const loadY = this.offsetY - this.beamHeight / 2;
                
                // Check if within load hitbox
                if (Math.abs(x - loadX) < 15 && y < loadY && y > loadY - 45) {
                    return { type: 'load', id: load.id, data: load };
                }
            } else if (load.type === 'distributed') {
                const startX = this.offsetX + load.position * this.scale;
                const endX = this.offsetX + load.endPosition * this.scale;
                const loadY = this.offsetY - this.beamHeight / 2;
                
                // Check if within distributed load hitbox
                if (x >= startX && x <= endX && y < loadY && y > loadY - 35) {
                    return { type: 'load', id: load.id, data: load };
                }
            } else if (load.type === 'moment') {
                const loadX = this.offsetX + load.position * this.scale;
                const loadY = this.offsetY;
                const radius = 25;
                
                // Check if within moment load hitbox
                const distance = Math.sqrt((x - loadX) ** 2 + (y - loadY) ** 2);
                if (distance < radius + 10) {
                    return { type: 'load', id: load.id, data: load };
                }
            }
        }
        
        return null;
    }
    
    // Handle mouse move event
    handleMouseMove(e) {
        if (!this.canvas || !this.isDragging || !this.selectedElement) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        
        // Calculate new position based on mouse movement
        const deltaX = mouseX - this.dragStartX;
        const newPositionInPixels = this.offsetX + this.selectedElement.data.position * this.scale + deltaX;
        const newPosition = (newPositionInPixels - this.offsetX) / this.scale;
        
        // Ensure position is within beam bounds
        const clampedPosition = Math.max(0, Math.min(this.length, newPosition));
        
        // Update element position
        if (this.selectedElement.type === 'support') {
            this.selectedElement.data.position = clampedPosition;
            
            // Update position field in properties panel
            const supportPositionInput = document.getElementById('support-position');
            if (supportPositionInput) {
                supportPositionInput.value = clampedPosition.toFixed(1);
            }
        } else if (this.selectedElement.type === 'load') {
            if (this.selectedElement.data.type === 'distributed') {
                // Calculate width of distributed load
                const width = this.selectedElement.data.endPosition - this.selectedElement.data.position;
                
                // Update both positions maintaining the width
                this.selectedElement.data.position = clampedPosition;
                this.selectedElement.data.endPosition = Math.min(this.length, clampedPosition + width);
                
                // Update position fields in properties panel
                const loadPositionInput = document.getElementById('load-position');
                const loadEndPositionInput = document.getElementById('load-end-position');
                
                if (loadPositionInput) {
                    loadPositionInput.value = clampedPosition.toFixed(1);
                }
                
                if (loadEndPositionInput) {
                    loadEndPositionInput.value = this.selectedElement.data.endPosition.toFixed(1);
                }
            } else {
                this.selectedElement.data.position = clampedPosition;
                
                // Update position field in properties panel
                const loadPositionInput = document.getElementById('load-position');
                if (loadPositionInput) {
                    loadPositionInput.value = clampedPosition.toFixed(1);
                }
            }
        }
        
        // Update drag start position
        this.dragStartX = mouseX;
        
        // Redraw beam
        this.drawBeam();
        
        // Run analysis if we have enough supports
        if (this.supports.length >= 2) {
            this.analyze();
        }
    }
    
    // Handle mouse up event
    handleMouseUp() {
        this.isDragging = false;
    }
    
    // Show properties panel for the selected element
    showPropertiesPanel(element) {
        const propertiesPanel = document.getElementById('element-properties');
        const supportProperties = document.getElementById('support-properties');
        const loadProperties = document.getElementById('load-properties');
        const elementTitle = document.getElementById('element-title');
        
        if (!propertiesPanel || !supportProperties || !loadProperties || !elementTitle) return;
        
        // Show the properties panel
        propertiesPanel.style.display = 'block';
        
        if (element.type === 'support') {
            // Show support properties
            elementTitle.textContent = 'Support Properties';
            supportProperties.style.display = 'block';
            loadProperties.style.display = 'none';
            
            // Set values
            const supportType = document.getElementById('support-type');
            const supportPosition = document.getElementById('support-position');
            
            if (supportType) supportType.value = element.data.type;
            if (supportPosition) supportPosition.value = element.data.position.toFixed(1);
        } else if (element.type === 'load') {
            // Show load properties
            elementTitle.textContent = 'Load Properties';
            supportProperties.style.display = 'none';
            loadProperties.style.display = 'block';
            
            // Set values
            const loadType = document.getElementById('load-type');
            const loadValue = document.getElementById('load-value');
            const loadPosition = document.getElementById('load-position');
            const loadEndPosition = document.getElementById('load-end-position');
            const loadUnit = document.getElementById('load-unit');
            const distributedLoadOptions = document.querySelectorAll('.distributed-load-option');
            
            if (loadType) loadType.value = element.data.type;
            if (loadValue) loadValue.value = Math.abs(element.data.value);
            if (loadPosition) loadPosition.value = element.data.position.toFixed(1);
            
            // Update unit based on load type
            if (loadUnit) {
                if (element.data.type === 'point') loadUnit.textContent = 'kN';
                else if (element.data.type === 'distributed') loadUnit.textContent = 'kN/m';
                else if (element.data.type === 'moment') loadUnit.textContent = 'kNm';
            }
            
            // Show/hide distributed load options
            distributedLoadOptions.forEach(option => {
                option.style.display = element.data.type === 'distributed' ? 'block' : 'none';
            });
            
            // Set end position for distributed loads
            if (element.data.type === 'distributed' && loadEndPosition) {
                loadEndPosition.value = element.data.endPosition.toFixed(1);
            }
        }
    }
    
    // Close the properties panel
    closePropertiesPanel() {
        const propertiesPanel = document.getElementById('element-properties');
        if (propertiesPanel) propertiesPanel.style.display = 'none';
    }
    
    // Handle load type change in properties panel
    handleLoadTypeChange(e) {
        const loadType = e.target.value;
        const loadUnit = document.getElementById('load-unit');
        const distributedLoadOptions = document.querySelectorAll('.distributed-load-option');
        
        // Update unit based on load type
        if (loadUnit) {
            if (loadType === 'point') loadUnit.textContent = 'kN';
            else if (loadType === 'distributed') loadUnit.textContent = 'kN/m';
            else if (loadType === 'moment') loadUnit.textContent = 'kNm';
        }
        
        // Show/hide distributed load options
        distributedLoadOptions.forEach(option => {
            option.style.display = loadType === 'distributed' ? 'block' : 'none';
        });
    }
    
    // Apply support changes from properties panel
    applySupport() {
        if (!this.selectedElement || this.selectedElement.type !== 'support') return;
        
        const supportType = document.getElementById('support-type');
        const supportPosition = document.getElementById('support-position');
        
        if (supportType) this.selectedElement.data.type = supportType.value;
        if (supportPosition) {
            const position = parseFloat(supportPosition.value);
            if (!isNaN(position)) {
                this.selectedElement.data.position = Math.max(0, Math.min(this.length, position));
            }
        }
        
        this.drawBeam();
        this.analyze();
    }
    
    // Apply load changes from properties panel
    applyLoad() {
        if (!this.selectedElement || this.selectedElement.type !== 'load') return;
        
        const loadType = document.getElementById('load-type');
        const loadValue = document.getElementById('load-value');
        const loadPosition = document.getElementById('load-position');
        const loadEndPosition = document.getElementById('load-end-position');
        
        // Update load type if changed
        if (loadType && loadType.value !== this.selectedElement.data.type) {
            this.selectedElement.data.type = loadType.value;
            
            // Initialize new properties for different load types
            if (loadType.value === 'distributed' && !this.selectedElement.data.endPosition) {
                this.selectedElement.data.endPosition = Math.min(this.length, this.selectedElement.data.position + 2);
            }
        }
        
        // Update load value
        if (loadValue) {
            const value = parseFloat(loadValue.value);
            if (!isNaN(value)) this.selectedElement.data.value = value;
        }
        
        // Update load position
        if (loadPosition) {
            const position = parseFloat(loadPosition.value);
            if (!isNaN(position)) {
                this.selectedElement.data.position = Math.max(0, Math.min(this.length, position));
            }
        }
        
        // Update end position for distributed loads
        if (this.selectedElement.data.type === 'distributed' && loadEndPosition) {
            const endPosition = parseFloat(loadEndPosition.value);
            if (!isNaN(endPosition)) {
                this.selectedElement.data.endPosition = Math.max(
                    this.selectedElement.data.position + 0.1, 
                    Math.min(this.length, endPosition)
                );
            }
        }
        
        this.drawBeam();
        this.analyze();
    }
    
    // Delete selected support
    deleteSupport() {
        if (!this.selectedElement || this.selectedElement.type !== 'support') return;
        
        // Remove support from array
        this.supports = this.supports.filter(support => support.id !== this.selectedElement.data.id);
        
        // Close properties panel
        this.closePropertiesPanel();
        
        // Clear selection
        this.selectedElement = null;
        
        // Redraw beam
        this.drawBeam();
        this.analyze();
    }
    
    // Delete selected load
    deleteLoad() {
        if (!this.selectedElement || this.selectedElement.type !== 'load') return;
        
        // Remove load from array
        this.loads = this.loads.filter(load => load.id !== this.selectedElement.data.id);
        
        // Close properties panel
        this.closePropertiesPanel();
        
        // Clear selection
        this.selectedElement = null;
        
        // Redraw beam
        this.drawBeam();
        this.analyze();
    }
    
    // Add a new support to the beam
    addSupport() {
        // Create a new support
        const support = {
            id: Date.now(),
            type: 'pinned',
            position: this.supports.length === 0 ? 0 : Math.min(this.length, this.supports[0].position + 2)
        };
        
        // Add support to array
        this.supports.push(support);
        
        // Select the new support
        this.selectedElement = { type: 'support', id: support.id, data: support };
        
        // Show properties panel
        this.showPropertiesPanel(this.selectedElement);
        
        // Redraw beam
        this.drawBeam();
        
        // Run analysis if we have enough supports
        if (this.supports.length >= 2) {
            this.analyze();
        }
    }
    
    // Add a new load to the beam
    addLoad() {
        // Create a new load
        const load = {
            id: Date.now(),
            type: 'point',
            value: 10,
            position: this.length / 2
        };
        
        // Add load to array
        this.loads.push(load);
        
        // Select the new load
        this.selectedElement = { type: 'load', id: load.id, data: load };
        
        // Show properties panel
        this.showPropertiesPanel(this.selectedElement);
        
        // Redraw beam
        this.drawBeam();
        this.analyze();
    }
    
    // Clear all elements from the beam
    clearBeam() {
        this.supports = [];
        this.loads = [];
        this.selectedElement = null;
        this.closePropertiesPanel();
        this.drawBeam();
        
        // Clear results
        this.clearResults();
    }
    
    // Clear analysis results
    clearResults() {
        // Clear result arrays
        this.results = {
            reactions: [],
            shear: [],
            moment: [],
            deflection: [],
            rotation: []
        };
        
        // Clear result displays
        document.getElementById('max-moment')?.textContent = '-';
        document.getElementById('max-shear')?.textContent = '-';
        document.getElementById('max-deflection')?.textContent = '-';
        document.getElementById('max-stress')?.textContent = '-';
        
        // Clear support reactions
        const supportReactions = document.getElementById('support-reactions');
        if (supportReactions) supportReactions.innerHTML = '';
        
        // Clear diagrams
        this.clearDiagrams();
    }
    
    // Clear diagram canvases
    clearDiagrams() {
        // Diagrammen zijn verwijderd uit de HTML, dus deze functie hoeft niets meer te doen
        // ['moment-diagram', 'shear-diagram', 'deflection-diagram', 'rotation-diagram'].forEach(id => {
        //     const canvas = document.getElementById(id);
        //     if (canvas) {
        //         const ctx = canvas.getContext('2d');
        //         ctx.clearRect(0, 0, canvas.width, canvas.height);
        //     }
        // });
    }
    
    // Perform beam analysis
    analyze() {
        // Check if we have enough supports
        if (this.supports.length < 2) {
            this.clearResults();
            return;
        }
        
        // TODO: Implement full beam analysis
        // For now, we'll just generate some sample data for visualization
        this.generateSampleResults();
        
        // Update result displays
        this.updateResultDisplays();
        
        // Diagrammen zijn verwijderd uit de HTML, dus we hoeven ze niet meer te tekenen
        // this.drawDiagrams();
    }
    
    // Generate sample results for visualization
    generateSampleResults() {
        // Clear previous results
        this.results = {
            reactions: [],
            shear: [],
            moment: [],
            deflection: [],
            rotation: []
        };
        
        // Generate sample data points
        const numPoints = this.numPoints;
        const dx = this.length / (numPoints - 1);
        
        // Generate reactions
        this.supports.forEach(support => {
            this.results.reactions.push({
                position: support.position,
                value: Math.random() * 20 - 10 // Random value between -10 and 10
            });
        });
        
        // Generate shear diagram
        let prevShear = 0;
        for (let i = 0; i < numPoints; i++) {
            const x = i * dx;
            
            // Check for supports
            const support = this.supports.find(s => Math.abs(s.position - x) < 0.1);
            if (support) {
                // Add reaction at support
                const reaction = this.results.reactions.find(r => r.position === support.position);
                prevShear = reaction ? reaction.value : prevShear;
            }
            
            // Check for point loads
            const pointLoad = this.loads.find(l => l.type === 'point' && Math.abs(l.position - x) < 0.1);
            if (pointLoad) {
                prevShear -= pointLoad.value;
            }
            
            // Add distributed loads effect
            const distributedLoads = this.loads.filter(l => 
                l.type === 'distributed' && 
                x >= l.position && 
                x <= l.endPosition
            );
            
            distributedLoads.forEach(load => {
                prevShear -= load.value * dx;
            });
            
            this.results.shear.push({ x, value: prevShear });
        }
        
        // Generate moment diagram
        let prevMoment = 0;
        for (let i = 0; i < numPoints; i++) {
            const x = i * dx;
            const shear = this.results.shear[i].value;
            
            // Moment changes based on shear
            prevMoment += shear * dx;
            
            // Check for moment loads
            const momentLoad = this.loads.find(l => l.type === 'moment' && Math.abs(l.position - x) < 0.1);
            if (momentLoad) {
                prevMoment += momentLoad.value;
            }
            
            this.results.moment.push({ x, value: prevMoment });
        }
        
        // Generate rotation diagram (first derivative of deflection)
        let prevRotation = 0;
        for (let i = 0; i < numPoints; i++) {
            const x = i * dx;
            const moment = this.results.moment[i].value;
            
            // Rotation changes based on moment
            prevRotation += moment * dx / (this.eModulus * this.momentOfInertia / 1e6);
            
            this.results.rotation.push({ x, value: prevRotation });
        }
        
        // Generate deflection diagram (integral of rotation)
        let prevDeflection = 0;
        for (let i = 0; i < numPoints; i++) {
            const x = i * dx;
            const rotation = this.results.rotation[i].value;
            
            // Deflection changes based on rotation
            prevDeflection += rotation * dx;
            
            this.results.deflection.push({ x, value: prevDeflection * 1000 }); // Convert to mm
        }
        
        // Apply boundary conditions (zero deflection at supports)
        this.applyBoundaryConditions();
    }
    
    // Apply boundary conditions to results
    applyBoundaryConditions() {
        // Find deflection at supports
        const supportDeflections = [];
        
        this.supports.forEach(support => {
            // Find closest point in deflection array
            const index = Math.round(support.position / this.length * (this.numPoints - 1));
            if (index >= 0 && index < this.results.deflection.length) {
                supportDeflections.push({
                    position: support.position,
                    index,
                    value: this.results.deflection[index].value
                });
            }
        });
        
        // Simple correction: shift all deflections to make supports have zero deflection
        if (supportDeflections.length > 0) {
            const avgDeflection = supportDeflections.reduce((sum, d) => sum + d.value, 0) / supportDeflections.length;
            
            // Shift all deflection values
            this.results.deflection.forEach((point, i) => {
                point.value -= avgDeflection;
            });
        }
    }
    
    // Update result displays with calculated values
    updateResultDisplays() {
        // Find maximum values
        const maxMoment = Math.max(...this.results.moment.map(p => Math.abs(p.value)));
        const maxShear = Math.max(...this.results.shear.map(p => Math.abs(p.value)));
        const maxDeflection = Math.max(...this.results.deflection.map(p => Math.abs(p.value)));
        
        // Calculate maximum stress (M*y/I)
        const profileHeight = 0.2; // meters (assumed)
        const maxStress = maxMoment * (profileHeight / 2) / (this.momentOfInertia / 1e12) * 1000; // MPa
        
        // Update display elements
        document.getElementById('max-moment')?.textContent = maxMoment.toFixed(2);
        document.getElementById('max-shear')?.textContent = maxShear.toFixed(2);
        document.getElementById('max-deflection')?.textContent = maxDeflection.toFixed(2);
        document.getElementById('max-stress')?.textContent = maxStress.toFixed(2);
        
        // Update support reactions
        const supportReactions = document.getElementById('support-reactions');
        if (supportReactions) {
            let html = '<table class="table table-sm table-bordered"><tbody>';
            
            this.results.reactions.forEach((reaction, index) => {
                const support = this.supports.find(s => s.position === reaction.position);
                if (support) {
                    html += `
                        <tr>
                            <td class="small fw-bold">Support ${index + 1} (${support.type})</td>
                            <td class="text-end">${reaction.value.toFixed(2)} <span class="small text-muted">kN</span></td>
                        </tr>
                    `;
                }
            });
            
            html += '</tbody></table>';
            supportReactions.innerHTML = html;
        }
    }
    
    // Draw analysis diagrams
    drawDiagrams() {
        // Diagrammen zijn verwijderd uit de HTML, dus deze functie hoeft niets meer te doen
        return;
    }
    
    // Draw deflection diagram
    drawDeflectionDiagram() {
        // Diagrammen zijn verwijderd uit de HTML, dus deze functie hoeft niets meer te doen
        return;
    }
    
    // Draw rotation diagram
    drawRotationDiagram() {
        // Diagrammen zijn verwijderd uit de HTML, dus deze functie hoeft niets meer te doen
        return;
    }
    
    // Export analysis results to PDF
    exportToPDF() {
        console.log('Exporting beam analysis to PDF');
        
        // Check if we have analysis results
        if (this.results.shear.length === 0) {
            alert('Please analyze the beam first before exporting to PDF.');
            return;
        }
        
        // Create a new window for PDF content
        const printWindow = window.open('', '_blank');
        
        // Generate PDF content
        let content = `
            <!DOCTYPE html>
            <html>
            <head>
                <title>Beam Analysis Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2962ff; }
                    .section { margin-bottom: 20px; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 15px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .beam-canvas { border: 1px solid #ddd; margin-bottom: 15px; }
                    .footer { margin-top: 30px; font-size: 12px; color: #666; text-align: center; }
                </style>
            </head>
            <body>
                <h1>Beam Analysis Report</h1>
                
                <div class="section">
                    <h2>Beam Properties</h2>
                    <table>
                        <tr>
                            <th>Property</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Length</td>
                            <td>${this.length} m</td>
                        </tr>
                        <tr>
                            <td>Elastic Modulus</td>
                            <td>${this.eModulus} MPa</td>
                        </tr>
                        <tr>
                            <td>Moment of Inertia</td>
                            <td>${this.momentOfInertia} mm⁴</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Supports</h2>
                    <table>
                        <tr>
                            <th>Type</th>
                            <th>Position</th>
                        </tr>
                        ${this.supports.map(support => `
                            <tr>
                                <td>${support.type}</td>
                                <td>${support.position} m</td>
                            </tr>
                        `).join('')}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Loads</h2>
                    <table>
                        <tr>
                            <th>Type</th>
                            <th>Value</th>
                            <th>Position</th>
                            ${this.loads.some(load => load.type === 'distributed') ? '<th>End Position</th>' : ''}
                        </tr>
                        ${this.loads.map(load => `
                            <tr>
                                <td>${load.type}</td>
                                <td>${load.value} ${load.type === 'point' ? 'kN' : load.type === 'distributed' ? 'kN/m' : 'kNm'}</td>
                                <td>${load.position} m</td>
                                ${load.type === 'distributed' ? `<td>${load.endPosition} m</td>` : ''}
                            </tr>
                        `).join('')}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Analysis Results</h2>
                    <table>
                        <tr>
                            <th>Result</th>
                            <th>Maximum Value</th>
                        </tr>
                        <tr>
                            <td>Maximum Moment</td>
                            <td>${Math.max(...this.results.moment.map(p => Math.abs(p.value))).toFixed(2)} kNm</td>
                        </tr>
                        <tr>
                            <td>Maximum Shear</td>
                            <td>${Math.max(...this.results.shear.map(p => Math.abs(p.value))).toFixed(2)} kN</td>
                        </tr>
                        <tr>
                            <td>Maximum Deflection</td>
                            <td>${Math.max(...this.results.deflection.map(p => Math.abs(p.value))).toFixed(2)} mm</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Support Reactions</h2>
                    <table>
                        <tr>
                            <th>Position</th>
                            <th>Reaction</th>
                        </tr>
                        ${this.results.reactions.map(reaction => `
                            <tr>
                                <td>${reaction.position} m</td>
                                <td>${reaction.value.toFixed(2)} kN</td>
                            </tr>
                        `).join('')}
                    </table>
                </div>
                
                <div class="footer">
                    <p>Generated by Steel Profile Calculator Elite on ${new Date().toLocaleDateString()}</p>
                </div>
            </body>
            </html>
        `;
        
        // Write content to the new window
        printWindow.document.open();
        printWindow.document.write(content);
        printWindow.document.close();
        
        // Print the window after it's loaded
        printWindow.onload = function() {
            printWindow.print();
        };
    }
}

// Initialize the beam analysis module when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing beam analysis');
    
    // Check if any beam analysis elements exist on the page
    const beamContent = document.getElementById('beam-content');
    const analyzeBeamButton = document.getElementById('analyze-beam');
    
    // Initialize if beam analysis elements are found
    if (beamContent) {
        console.log('Beam analysis elements found, initializing');
        
        // Create and initialize beam analysis
        const beamAnalysis = new BeamAnalysis();
        beamAnalysis.init();
        
        // Add beam analysis to window for debugging
        window.beamAnalysis = beamAnalysis;
    } else {
        console.log('Beam analysis elements not found, skipping initialization');
    }
    
    // Initialize other parts of the application
    initializeProfilesTab();
});

// Initialize the profiles tab functionality
function initializeProfilesTab() {
    console.log('Initializing profiles tab');
    
    // Add row button
    const addRowButton = document.getElementById('add-row');
    if (addRowButton) {
        addRowButton.addEventListener('click', function() {
            addNewRow();
        });
    }
    
    // Initialize existing rows
    initializeExistingRows();
}

// Add a new row to the materials table
function addNewRow() {
    console.log('Adding new row');
    
    const materialsTable = document.querySelector('.materials-table tbody');
    if (!materialsTable) {
        console.error('Materials table not found');
        return;
    }
    
    // Clone the row template
    const rowTemplate = document.getElementById('row-template');
    if (!rowTemplate) {
        console.error('Row template not found');
        return;
    }
    
    const newRow = document.importNode(rowTemplate.content, true).querySelector('tr');
    materialsTable.appendChild(newRow);
    
    // Initialize the new row
    initializeRow(newRow);
}

// Initialize existing rows in the table
function initializeExistingRows() {
    const rows = document.querySelectorAll('.materials-table tbody tr');
    rows.forEach(row => initializeRow(row));
}

// Initialize a single row with event listeners
function initializeRow(row) {
    // Profile type change event
    const profileType = row.querySelector('.profile-type');
    const profileSize = row.querySelector('.profile-size');
    
    if (profileType && profileSize) {
        profileType.addEventListener('change', function() {
            // Enable size dropdown when type is selected
            profileSize.disabled = !this.value;
            
            // Clear and populate size options based on selected type
            profileSize.innerHTML = '<option value="">Select size</option>';
            
            if (this.value) {
                // Add sizes based on profile type
                const sizes = getProfileSizes(this.value);
                sizes.forEach(size => {
                    const option = document.createElement('option');
                    option.value = size;
                    option.textContent = size;
                    profileSize.appendChild(option);
                });
            }
        });
    }
}

// Get profile sizes based on profile type
function getProfileSizes(profileType) {
    const sizeMap = {
        'IPE': ['80', '100', '120', '140', '160', '180', '200', '220', '240', '270', '300', '330', '360', '400', '450', '500', '550', '600'],
        'HEA': ['100', '120', '140', '160', '180', '200', '220', '240', '260', '280', '300', '320', '340', '360', '400', '450', '500', '550', '600', '650', '700', '800', '900', '1000'],
        'HEB': ['100', '120', '140', '160', '180', '200', '220', '240', '260', '280', '300', '320', '340', '360', '400', '450', '500', '550', '600', '650', '700', '800', '900', '1000'],
        'UNP': ['80', '100', '120', '140', '160', '180', '200', '220', '240', '260', '280', '300', '320', '350', '380', '400'],
        'RHS': ['50x30x3', '60x40x3', '80x40x3', '100x50x3', '120x60x4', '140x80x4', '160x80x5', '180x100x5', '200x100x6', '250x150x6', '300x200x8'],
        'SHS': ['20x20x2', '25x25x2', '30x30x2', '40x40x2', '50x50x3', '60x60x3', '80x80x4', '100x100x4', '120x120x5', '140x140x5', '150x150x6', '160x160x6', '180x180x6', '200x200x6'],
        'ROUND_PIPE': ['21.3x2.3', '26.9x2.3', '33.7x2.6', '42.4x2.6', '48.3x2.6', '60.3x2.9', '76.1x2.9', '88.9x3.2', '114.3x3.6', '139.7x4', '168.3x4.5', '219.1x6.3'],
        'FLAT_BAR': ['20x3', '20x5', '25x4', '25x6', '30x5', '30x8', '40x5', '40x8', '50x5', '50x8', '50x10', '60x8', '60x10', '80x8', '80x10', '100x8', '100x10', '100x12']
    };
    
    return sizeMap[profileType] || [];
}
