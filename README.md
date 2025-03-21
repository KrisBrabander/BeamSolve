# Buigingsberekeningen

Een interactieve web-applicatie voor het berekenen en visualiseren van balkdoorbuigingen.

## Features

- **Profielkeuze**: Koker, I-profiel en U-profiel
- **Steunpunten**: Keuze uit 1 (inklemming), 2 of 3 steunpunten
- **Belastingen**: 
  - Puntlasten
  - Gelijkmatig verdeelde lasten
- **Real-time visualisatie**:
  - Doorbuigingslijn
  - Krachten en steunpunten
  - Maximale doorbuiging

## Installatie

1. Clone de repository:
```bash
git clone [repository-url]
cd BuigingsCalculator
```

2. Installeer de vereiste packages:
```bash
pip install -r requirements.txt
```

## Gebruik

Start de applicatie:
```bash
streamlit run streamlit_app.py
```

De applicatie opent automatisch in je standaard webbrowser.

## Invoer

1. **Profielgegevens**:
   - Kies profieltype
   - Voer afmetingen in (hoogte, breedte, wanddikte)
   - Voor I- en U-profielen: flensdikte
   - E-modulus (standaard 210000 N/mm² voor staal)

2. **Overspanning**:
   - Lengte van de balk
   - Aantal en type steunpunten
   - Posities van de steunpunten

3. **Belastingen**:
   - Type belasting
   - Grootte
   - Positie
   - Voor verdeelde last: lengte

## Output

- Grafische weergave van de balk
- Vervormde vorm (doorbuigingslijn)
- Maximale doorbuiging in mm
