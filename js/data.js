/**
 * Steel Profile Calculator - Data Module
 * Handles loading and processing of steel profile data
 */

// Global object to store profile data
const profileData = {
    // Will hold the data loaded from profiles.js
    profiles: null,
    
    // Load profile data from PROFILE_DATA global object
    async loadProfiles() {
        try {
            // Use the PROFILE_DATA object defined in profiles.js
            if (typeof PROFILE_DATA !== 'undefined') {
                this.profiles = PROFILE_DATA;
                console.log('Profile data loaded successfully');
                return this.profiles;
            } else {
                throw new Error('PROFILE_DATA not found');
            }
        } catch (error) {
            console.error('Error loading profile data:', error);
            alert('Failed to load profile data. Please refresh the page and try again.');
            return null;
        }
    },
    
    // Get profile types as array of strings
    getProfileTypes() {
        if (!this.profiles) return [];
        return Object.keys(this.profiles);
    },
    
    // Get profiles for a specific type
    getProfilesByType(type) {
        if (!this.profiles || !this.profiles[type]) return [];
        return this.profiles[type];
    },
    
    // Get a specific profile by type and name
    getProfile(type, name) {
        if (!this.profiles || !this.profiles[type]) return null;
        return this.profiles[type].find(profile => profile.name === name) || null;
    },
    
    // Calculate weight for a specific profile
    calculateWeight(type, name, length, quantity = 1, unit = 'mm') {
        const profile = this.getProfile(type, name);
        if (!profile) return 0;
        
        // Convert length to meters based on unit
        let lengthInMeters;
        switch(unit) {
            case 'mm':
                lengthInMeters = length / 1000;
                break;
            case 'inch':
                lengthInMeters = length * 0.0254; // 1 inch = 0.0254 meters
                break;
            case 'ft':
                lengthInMeters = length * 0.3048; // 1 foot = 0.3048 meters
                break;
            default: // 'm' or any other unit
                lengthInMeters = length;
        }
        
        // Calculate total weight (weight per meter * length in meters * quantity)
        return profile.weightPerMeter * lengthInMeters * quantity;
    },
    
    // Calculate volume for a specific profile
    calculateVolume(type, name, length, quantity = 1, unit = 'mm', density = 7850) {
        const totalWeight = this.calculateWeight(type, name, length, quantity, unit);
        // Volume in cubic meters = weight in kg / density in kg/m³
        return totalWeight / density;
    },
    
    // Calculate surface area for a specific profile
    calculateSurfaceArea(type, name, length, quantity = 1, unit = 'mm') {
        const profile = this.getProfile(type, name);
        if (!profile) return 0;
        
        // Convert length to meters based on unit
        let lengthInMeters;
        switch(unit) {
            case 'mm':
                lengthInMeters = length / 1000;
                break;
            case 'inch':
                lengthInMeters = length * 0.0254; // 1 inch = 0.0254 meters
                break;
            case 'ft':
                lengthInMeters = length * 0.3048; // 1 foot = 0.3048 meters
                break;
            default: // 'm' or any other unit
                lengthInMeters = length;
        }
        
        // Calculate total surface area (surface area per meter * length in meters * quantity)
        return profile.surfaceAreaPerMeter * lengthInMeters * quantity;
    }
};

if (typeof module !== 'undefined') {
  module.exports = profileData;
}
