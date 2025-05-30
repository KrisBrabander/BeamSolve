# Steel Profile Calculator

A comprehensive web application for calculating the weight, volume, and surface area of various steel profiles.

## Features

- Calculate weight, volume, and surface area for multiple steel profile types
- Support for IPE, HEA, HEB, UNP, rectangular/square hollow sections, round pipes, and flat bars
- Add multiple materials and quantities
- Automatic calculations based on profile database
- Unit conversion between mm and meters
- Export results to PDF or CSV
- Auto-save data to localStorage

## Profile Types Supported

- IPE beams
- HEA beams
- HEB beams
- UNP channels
- Rectangular hollow sections (RHS)
- Square hollow sections (SHS)
- Round pipes
- Flat bars

## Technical Details

- Built with HTML, CSS, and JavaScript
- Mobile-responsive design
- No external dependencies required for core functionality
- Uses jsPDF for PDF export functionality

## Usage

1. Select a profile type from the dropdown
2. Choose a specific profile size
3. Enter the length (in mm or m)
4. Specify quantity
5. Results are calculated automatically
6. Add more rows as needed
7. Export to PDF or CSV when finished

## Data Storage

The application stores your last session data in your browser's localStorage. This allows you to return to your calculations later without data loss.

## Material Database

The application includes a comprehensive database of steel profiles with accurate dimensions and properties. All weight calculations are based on standard steel density (7850 kg/m³ by default), which can be customized.
