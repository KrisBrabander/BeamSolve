/**
 * Steel Profile Calculator - Main Application
 * Handles user interactions and calculations
 */

// DOM Elements
const materialTableBody = document.getElementById('materials-table-body');
const addRowButton = document.getElementById('add-row');
const exportPdfButton = document.getElementById('export-pdf');
const exportCsvButton = document.getElementById('export-csv');
const projectNameInput = document.getElementById('project-name');
const companyNameInput = document.getElementById('company-name');
const materialDensityInput = document.getElementById('material-density');
const totalWeightElement = document.getElementById('total-weight');
const totalVolumeElement = document.getElementById('total-volume');
const totalSurfaceAreaElement = document.getElementById('total-surface-area');
const rowTemplate = document.getElementById('row-template');
const currentYearElement = document.getElementById('current-year');
const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');

// Set current year in footer
currentYearElement.textContent = new Date().getFullYear();

// Dark mode functionality
function initTheme() {
    // Check for saved theme preference or use system preference
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);
}

function setTheme(theme) {
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        // Direct styling voor de dark mode
        document.body.style.backgroundColor = '#121212';
        document.querySelector('.container-fluid').style.backgroundColor = '#121212';
        
        // Change icon to sun when in dark mode
        themeIcon.innerHTML = `
            <path d="M8 11a3 3 0 1 1 0-6 3 3 0 0 1 0 6zm0 1a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM8 0a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 0zm0 13a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 13zm8-5a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2a.5.5 0 0 1 .5.5zM3 8a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2A.5.5 0 0 1 3 8zm10.657-5.657a.5.5 0 0 1 0 .707l-1.414 1.415a.5.5 0 1 1-.707-.708l1.414-1.414a.5.5 0 0 1 .707 0zm-9.193 9.193a.5.5 0 0 1 0 .707L3.05 13.657a.5.5 0 0 1-.707-.707l1.414-1.414a.5.5 0 0 1 .707 0zm9.193 2.121a.5.5 0 0 1-.707 0l-1.414-1.414a.5.5 0 0 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .707zM4.464 4.465a.5.5 0 0 1-.707 0L2.343 3.05a.5.5 0 1 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .708z"/>
        `;
        themeIcon.classList.remove('bi-moon-stars');
        themeIcon.classList.add('bi-sun');
    } else {
        document.documentElement.removeAttribute('data-theme');
        // Verwijder directe styling voor light mode
        document.body.style.backgroundColor = '';
        document.querySelector('.container-fluid').style.backgroundColor = '';
        
        // Change icon to moon when in light mode
        themeIcon.innerHTML = `
            <path d="M6 .278a.768.768 0 0 1 .08.858 7.208 7.208 0 0 0-.878 3.46c0 4.021 3.278 7.277 7.318 7.277.527 0 1.04-.055 1.533-.16a.787.787 0 0 1 .81.316.733.733 0 0 1-.031.893A8.349 8.349 0 0 1 8.344 16C3.734 16 0 12.286 0 7.71 0 4.266 2.114 1.312 5.124.06A.752.752 0 0 1 6 .278zM4.858 1.311A7.269 7.269 0 0 0 1.025 7.71c0 4.02 3.279 7.276 7.319 7.276a7.316 7.316 0 0 0 5.205-2.162c-.337.042-.68.063-1.029.063-4.61 0-8.343-3.714-8.343-8.29 0-1.167.242-2.278.681-3.286z"/>
            <path d="M10.794 3.148a.217.217 0 0 1 .412 0l.387 1.162c.173.518.579.924 1.097 1.097l1.162.387a.217.217 0 0 1 0 .412l-1.162.387a1.734 1.734 0 0 0-1.097 1.097l-.387 1.162a.217.217 0 0 1-.412 0l-.387-1.162A1.734 1.734 0 0 0 9.31 6.593l-1.162-.387a.217.217 0 0 1 0-.412l1.162-.387a1.734 1.734 0 0 0 1.097-1.097l.387-1.162zM13.863.099a.145.145 0 0 1 .274 0l.258.774c.115.346.386.617.732.732l.774.258a.145.145 0 0 1 0 .274l-.774.258a1.156 1.156 0 0 0-.732.732l-.258.774a.145.145 0 0 1-.274 0l-.258-.774a1.156 1.156 0 0 0-.732-.732l-.774-.258a.145.145 0 0 1 0-.274l.774-.258c.346-.115.617-.386.732-.732L13.863.1z"/>
        `;
        themeIcon.classList.remove('bi-sun');
        themeIcon.classList.add('bi-moon-stars');
    }
    localStorage.setItem('theme', theme);
}

// Toggle theme when the button is clicked
themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    setTheme(currentTheme);
});

// Application state
let rows = [];
let isDataLoaded = false;

// Initialize the application
async function init() {
    try {
        // Initialize theme
        initTheme();
        console.log('Initializing application...');
        
        // Load profile data
        const profiles = await profileData.loadProfiles();
        if (!profiles) {
            console.error('Failed to load profile data');
            return;
        }
        
        console.log('Profile data loaded successfully', profiles);
        isDataLoaded = true;
        
        // Load saved data from localStorage if exists
        loadSavedData();
        
        // If no rows exist yet (no saved data), add initial row
        if (materialTableBody.children.length === 0) {
            addRow();
        }
        
        // Attach event listeners
        addRowButton.addEventListener('click', addRow);
        exportPdfButton.addEventListener('click', exportToPdf);
        exportCsvButton.addEventListener('click', exportToCsv);
        materialDensityInput.addEventListener('input', recalculateAll);
        
        // Set up autosave
        setupAutosave();
        
        console.log('Initialization complete!');
    } catch (error) {
        console.error('Error during initialization:', error);
    }
}

// Save data to localStorage
function saveData() {
    const rowsData = [];
    
    document.querySelectorAll('.material-row').forEach(row => {
        const profileType = row.querySelector('.profile-type').value;
        const profileSize = row.querySelector('.profile-size').value;
        const length = row.querySelector('.length').value;
        const unit = row.querySelector('.unit').value;
        const quantity = row.querySelector('.quantity').value;
        
        rowsData.push({
            profileType,
            profileSize,
            length,
            unit,
            quantity
        });
    });
    
    const data = {
        projectName: projectNameInput.value,
        companyName: companyNameInput.value,
        materialDensity: materialDensityInput.value,
        rows: rowsData
    };
    
    localStorage.setItem('steelProfileCalculator', JSON.stringify(data));
}

// Load saved data from localStorage
function loadSavedData() {
    const savedData = localStorage.getItem('steelProfileCalculator');
    if (!savedData) return;
    
    try {
        const data = JSON.parse(savedData);
        
        // Set project info
        projectNameInput.value = data.projectName || '';
        companyNameInput.value = data.companyName || '';
        materialDensityInput.value = data.materialDensity || 7850;
        
        // Add rows from saved data
        if (data.rows && data.rows.length > 0) {
            data.rows.forEach(rowData => {
                const newRow = addRow(false); // Don't recalculate yet
                
                // Set values for the new row
                const profileTypeSelect = newRow.querySelector('.profile-type');
                profileTypeSelect.value = rowData.profileType;
                
                // Populate profile sizes dropdown
                if (rowData.profileType) {
                    populateProfileSizes(profileTypeSelect, newRow.querySelector('.profile-size'));
                    // Set the size value after populating
                    newRow.querySelector('.profile-size').value = rowData.profileSize;
                }
                
                newRow.querySelector('.length').value = rowData.length;
                newRow.querySelector('.unit').value = rowData.unit;
                newRow.querySelector('.quantity').value = rowData.quantity;
            });
            
            // Now recalculate all rows
            recalculateAll();
        }
    } catch (error) {
        console.error('Error loading saved data:', error);
    }
}

// Set up autosave functionality
function setupAutosave() {
    const elements = [
        projectNameInput, 
        companyNameInput,
        materialDensityInput
    ];
    
    elements.forEach(element => {
        element.addEventListener('change', saveData);
        element.addEventListener('blur', saveData);
    });
    
    // Save when the page is about to unload
    window.addEventListener('beforeunload', saveData);
}

// Add a new row to the materials table
function addRow(shouldRecalculate = true) {
    // Clone the template
    const newRow = document.importNode(rowTemplate.content, true).querySelector('tr');
    newRow.classList.add('fade-in');
    materialTableBody.appendChild(newRow);
    
    // Set up event listeners for the new row
    setupRowEventListeners(newRow);
    
    // Recalculate if needed
    if (shouldRecalculate) {
        recalculateAll();
        saveData();
    }
    
    // Focus on the profile type dropdown to make it easier to start using
    setTimeout(() => {
        const profileTypeSelect = newRow.querySelector('.profile-type');
        if (profileTypeSelect) profileTypeSelect.focus();
    }, 100);
    
    return newRow;
}

// Set up event listeners for a row
function setupRowEventListeners(row) {
    const profileTypeSelect = row.querySelector('.profile-type');
    const profileSizeSelect = row.querySelector('.profile-size');
    const lengthInput = row.querySelector('.length');
    const unitSelect = row.querySelector('.unit');
    const quantityInput = row.querySelector('.quantity');
    const deleteButton = row.querySelector('.delete-row');
    
    // Profile type change
    profileTypeSelect.addEventListener('change', () => {
        console.log('Profile type changed to:', profileTypeSelect.value);
        // Make sure we have profile data loaded
        if (!profileData.profiles) {
            console.error('Profile data not loaded yet');
            // Try to load it again if needed
            profileData.loadProfiles().then(() => {
                populateProfileSizes(profileTypeSelect, profileSizeSelect);
                recalculateRow(row);
                saveData();
            });
        } else {
            populateProfileSizes(profileTypeSelect, profileSizeSelect);
            recalculateRow(row);
            saveData();
        }
    });
    
    // Profile size change
    profileSizeSelect.addEventListener('change', () => {
        recalculateRow(row);
        saveData();
    });
    
    // Other input changes
    [lengthInput, unitSelect, quantityInput].forEach(input => {
        input.addEventListener('input', () => {
            recalculateRow(row);
            saveData();
        });
        
        input.addEventListener('change', () => {
            recalculateRow(row);
            saveData();
        });
    });
    
    // Delete row
    deleteButton.addEventListener('click', () => {
        row.remove();
        recalculateAll();
        saveData();
    });
}

// Populate profile sizes dropdown based on selected profile type
function populateProfileSizes(profileTypeSelect, profileSizeSelect) {
    const selectedType = profileTypeSelect.value;
    
    // Clear existing options
    profileSizeSelect.innerHTML = '<option value="">Select size</option>';
    
    // If no type selected, disable the size dropdown
    if (!selectedType) {
        profileSizeSelect.disabled = true;
        return;
    }
    
    // Get profiles for the selected type
    const profiles = profileData.getProfilesByType(selectedType);
    
    // Add options for each profile
    profiles.forEach(profile => {
        const option = document.createElement('option');
        option.value = profile.name;
        option.textContent = profile.name;
        profileSizeSelect.appendChild(option);
    });
    
    // Enable the size dropdown
    profileSizeSelect.disabled = false;
}

// Recalculate values for a specific row
function recalculateRow(row) {
    if (!isDataLoaded) return;
    
    const profileType = row.querySelector('.profile-type').value;
    const profileSize = row.querySelector('.profile-size').value;
    const length = parseFloat(row.querySelector('.length').value) || 0;
    const unit = row.querySelector('.unit').value;
    const quantity = parseInt(row.querySelector('.quantity').value) || 0;
    const density = parseFloat(materialDensityInput.value) || 7850;
    
    const weightPerMeterElement = row.querySelector('.weight-per-meter');
    const totalWeightElement = row.querySelector('.total-weight');
    const volumeElement = row.querySelector('.volume');
    const surfaceAreaElement = row.querySelector('.surface-area');
    
    // If type or size not selected, clear values
    if (!profileType || !profileSize) {
        weightPerMeterElement.value = '';
        totalWeightElement.value = '';
        volumeElement.value = '';
        surfaceAreaElement.value = '';
        return;
    }
    
    // Get profile data
    const profile = profileData.getProfile(profileType, profileSize);
    if (!profile) return;
    
    // Set weight per meter
    weightPerMeterElement.value = profile.weightPerMeter.toFixed(2);
    
    // Calculate values
    const totalWeight = profileData.calculateWeight(profileType, profileSize, length, quantity, unit);
    const volume = profileData.calculateVolume(profileType, profileSize, length, quantity, unit, density);
    const surfaceArea = profileData.calculateSurfaceArea(profileType, profileSize, length, quantity, unit);
    
    // Update display
    totalWeightElement.value = totalWeight.toFixed(2);
    volumeElement.value = volume.toFixed(4);
    surfaceAreaElement.value = surfaceArea.toFixed(2);
    
    // Update project totals
    recalculateProjectTotals();
}

// Recalculate all rows
function recalculateAll() {
    document.querySelectorAll('.material-row').forEach(row => {
        recalculateRow(row);
    });
}

// Recalculate project totals
function recalculateProjectTotals() {
    let totalWeight = 0;
    let totalVolume = 0;
    let totalSurfaceArea = 0;
    
    document.querySelectorAll('.material-row').forEach(row => {
        const weightElement = row.querySelector('.total-weight');
        const volumeElement = row.querySelector('.volume');
        const surfaceAreaElement = row.querySelector('.surface-area');
        
        if (weightElement.value) totalWeight += parseFloat(weightElement.value);
        if (volumeElement.value) totalVolume += parseFloat(volumeElement.value);
        if (surfaceAreaElement.value) totalSurfaceArea += parseFloat(surfaceAreaElement.value);
    });
    
    totalWeightElement.value = totalWeight.toFixed(2);
    totalVolumeElement.value = totalVolume.toFixed(4);
    totalSurfaceAreaElement.value = totalSurfaceArea.toFixed(2);
}

// Export data to PDF
function exportToPdf() {
    // Import the jsPDF library (loaded in the HTML)
    const { jsPDF } = window.jspdf;
    
    // Create a new PDF document
    const doc = new jsPDF();
    
    // Get project info
    const projectName = projectNameInput.value || 'Steel Profiles Calculation';
    const companyName = companyNameInput.value || '';
    const date = new Date().toLocaleDateString();
    
    // Define colors to match our app style
    const primaryColor = [93, 123, 153]; // #5d7b99
    const secondaryColor = [44, 62, 80]; // #2c3e50
    const accentColor = [192, 208, 224]; // #c0d0e0
    const lightGray = [245, 247, 250]; // #f5f7fa
    
    // Add logo (simplified version for PDF)
    function drawLogo(x, y, size) {
        // Background rectangle
        doc.setFillColor(...lightGray);
        doc.setDrawColor(...primaryColor);
        doc.roundedRect(x, y, size, size, 2, 2, 'FD');
        
        // I-beam
        doc.setFillColor(...secondaryColor);
        const beamWidth = size * 0.4;
        const beamX = x + (size - beamWidth) / 2;
        const beamY = y + size * 0.2;
        const beamHeight = size * 0.6;
        doc.rect(beamX, beamY, beamWidth, beamHeight, 'F');
        
        // Top and bottom flanges
        const flangeHeight = size * 0.1;
        const flangeWidth = size * 0.6;
        const flangeX = x + (size - flangeWidth) / 2;
        doc.rect(flangeX, beamY, flangeWidth, flangeHeight, 'F');
        doc.rect(flangeX, beamY + beamHeight - flangeHeight, flangeWidth, flangeHeight, 'F');
    }
    
    // Draw header background
    doc.setFillColor(...lightGray);
    doc.rect(0, 0, 210, 40, 'F');
    
    // Draw logo
    drawLogo(14, 12, 16);
    
    // Add header
    doc.setFontSize(18);
    doc.setTextColor(...secondaryColor);
    doc.text(projectName, 35, 20);
    
    doc.setFontSize(10);
    doc.setTextColor(80, 80, 80);
    if (companyName) {
        doc.text(`Company: ${companyName}`, 35, 28);
    }
    doc.text(`Date: ${date}`, companyName ? 120 : 35, companyName ? 28 : 28);
    
    // Add title bar
    doc.setFillColor(...primaryColor);
    doc.roundedRect(10, 45, 190, 8, 1, 1, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(11);
    doc.text('Steel Profile Calculator Pro - Material List', 105, 50, { align: 'center' });
    
    // Add material table
    const tableData = [];
    const tableColumns = [
        { header: 'Profile', dataKey: 'profile' },
        { header: 'Length', dataKey: 'length' },
        { header: 'Qty', dataKey: 'quantity' },
        { header: 'Weight/m', dataKey: 'weightPerMeter' },
        { header: 'Total Weight', dataKey: 'totalWeight' },
        { header: 'Volume', dataKey: 'volume' },
        { header: 'Surface Area', dataKey: 'surfaceArea' }
    ];
    
    document.querySelectorAll('.material-row').forEach(row => {
        const profileType = row.querySelector('.profile-type').value;
        const profileSize = row.querySelector('.profile-size').value;
        const length = row.querySelector('.length').value;
        const unit = row.querySelector('.unit').value;
        const quantity = row.querySelector('.quantity').value;
        const weightPerMeter = row.querySelector('.weight-per-meter').value;
        const totalWeight = row.querySelector('.total-weight').value;
        const volume = row.querySelector('.volume').value;
        const surfaceArea = row.querySelector('.surface-area').value;
        
        if (profileType && profileSize) {
            // Format the unit display for the PDF
            let unitDisplay = unit;
            if (unit === 'inch') {
                unitDisplay = 'inches';
            } else if (unit === 'ft') {
                unitDisplay = 'feet';
            }
            
            tableData.push({
                profile: `${profileType} ${profileSize}`,
                length: `${length} ${unitDisplay}`,
                quantity: quantity,
                weightPerMeter: `${weightPerMeter} kg/m`,
                totalWeight: `${totalWeight} kg`,
                volume: `${volume} m³`,
                surfaceArea: `${surfaceArea} m²`
            });
        }
    });
    
    // Add table to PDF with modern styling
    doc.autoTable({
        startY: 58,
        head: [tableColumns.map(col => col.header)],
        body: tableData.map(item => tableColumns.map(col => item[col.dataKey])),
        styles: { 
            fontSize: 9,
            cellPadding: 3,
            lineColor: [220, 220, 220]
        },
        headStyles: { 
            fillColor: primaryColor,
            textColor: [255, 255, 255],
            fontStyle: 'bold',
            lineWidth: 0.1,
            halign: 'center'
        },
        alternateRowStyles: { 
            fillColor: [248, 250, 252]
        },
        columnStyles: {
            0: { fontStyle: 'bold' },
            2: { halign: 'center' }
        },
        margin: { top: 60, left: 10, right: 10 },
        tableLineColor: [200, 200, 200],
        tableLineWidth: 0.1
    });
    
    // Add totals with rounded corners and modern styling
    const finalY = doc.lastAutoTable.finalY + 10;
    
    // Draw totals box with rounded corners
    doc.setFillColor(...lightGray);
    doc.roundedRect(10, finalY - 5, 190, 32, 3, 3, 'F');
    
    // Add title bar for totals
    doc.setFillColor(...primaryColor);
    doc.roundedRect(10, finalY - 5, 190, 8, 3, 3, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(11);
    doc.text('Project Totals', 105, finalY, { align: 'center' });
    
    // Add totals content
    doc.setFontSize(10);
    doc.setTextColor(...secondaryColor);
    
    // Create two columns for totals
    doc.text(`Total Weight:`, 20, finalY + 10);
    doc.text(`${totalWeightElement.value} kg`, 70, finalY + 10);
    
    doc.text(`Total Volume:`, 20, finalY + 18);
    doc.text(`${totalVolumeElement.value} m³`, 70, finalY + 18);
    
    doc.text(`Total Surface Area:`, 110, finalY + 10);
    doc.text(`${totalSurfaceAreaElement.value} m²`, 170, finalY + 10);
    
    doc.text(`Material Density:`, 110, finalY + 18);
    doc.text(`${materialDensityInput.value} kg/m³`, 170, finalY + 18);
    
    // Add footer with line
    const pageCount = doc.internal.getNumberOfPages();
    doc.setDrawColor(200, 200, 200);
    
    for (let i = 1; i <= pageCount; i++) {
        doc.setPage(i);
        
        // Footer line
        doc.line(10, 280, 200, 280);
        
        // Footer text
        doc.setFontSize(8);
        doc.setTextColor(120, 120, 120);
        doc.text('Steel Profile Calculator Pro © ' + new Date().getFullYear(), 105, 287, { align: 'center' });
        doc.text('Page ' + i + ' of ' + pageCount, 195, 287, { align: 'right' });
    }
    
    // Save PDF
    doc.save(`${projectName.replace(/\s+/g, '_')}_${date.replace(/\//g, '-')}.pdf`);
}

// Export data to CSV
function exportToCsv() {
    // Headers for CSV
    let csvContent = 'Profile Type,Profile Size,Length,Unit,Quantity,Weight per meter (kg/m),Total Weight (kg),Volume (m³),Surface Area (m²)\n';
    
    // Add rows
    document.querySelectorAll('.material-row').forEach(row => {
        const profileType = row.querySelector('.profile-type').value;
        const profileSize = row.querySelector('.profile-size').value;
        const length = row.querySelector('.length').value;
        const unit = row.querySelector('.unit').value;
        const quantity = row.querySelector('.quantity').value;
        const weightPerMeter = row.querySelector('.weight-per-meter').value;
        const totalWeight = row.querySelector('.total-weight').value;
        const volume = row.querySelector('.volume').value;
        const surfaceArea = row.querySelector('.surface-area').value;
        
        if (profileType && profileSize) {
            csvContent += `${profileType},${profileSize},${length},${unit},${quantity},${weightPerMeter},${totalWeight},${volume},${surfaceArea}\n`;
        }
    });
    
    // Add totals row
    csvContent += `\nTOTALS,,,,,,"${totalWeightElement.value}","${totalVolumeElement.value}","${totalSurfaceAreaElement.value}"\n`;
    csvContent += `Material Density,${materialDensityInput.value} kg/m³\n`;
    
    // Download CSV
    const projectName = projectNameInput.value || 'Steel_Profiles_Calculation';
    const date = new Date().toLocaleDateString().replace(/\//g, '-');
    const fileName = `${projectName.replace(/\s+/g, '_')}_${date}.csv`;
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', fileName);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', init);
