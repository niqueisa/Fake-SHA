# FAKE-SHA UI Style Guide

This document defines the visual design standards for the FAKE-SHA browser extension. All contributors must follow this guide to ensure consistency across the popup, analysis view, and history page.

---

## 1. Brand Identity

### Primary Header Background

* **Color:** `#1e2c3e`
* Usage:

  * Top navigation/header bar
  * Primary action areas
  * Key emphasis elements

### Primary Font

* **Font Family:** Canva Sans
* Fallback: sans-serif

All headings, labels, and body text should use Canva Sans where available.

---

## 2. Layout System

### Popup Width

* Fixed width: **280px**

### Page Background

* Light neutral background (subtle gray tone)

### Card Design

* Background: White
* Rounded corners
* Soft shadow
* Consistent internal padding

Cards are used for:

* Analysis results
* Summary sections
* History records

---

## 3. Typography Hierarchy

### App Title (FAKE-SHA)

* Bold
* Clear and readable
* White text when placed on dark header

### Article Title

* Slightly larger than body text
* Semi-bold

### Section Titles (e.g., Key Indicators, Summary)

* Medium emphasis
* Clear visual separation from content

### Body Text

* Clean and readable
* Secondary gray tone

### Confidence Text

* Prominent
* Colored based on classification (Real/Fake)

---

## 4. Classification Color System

### Real News (Positive State)

#### Result Banner

* Text Color: `#035323`
* Indicator Bar Background: `#dfffe9`
* Indicator Bar Progress: `#16a34a`

#### Top Contributing to Authenticity (3 Circles)

* Low Impact: `#d0e6de`
* Medium Impact: `#a5dfbe`
* High Impact: `#83cfa0`

Usage:

* Real news detection banner
* Positive key indicators
* Authenticity token highlights

---

### Fake News (Negative State)

#### Result Banner

* Text Color: `#ad0516`
* Indicator Bar Background: `#f6c6c8`
* Indicator Bar Progress: `#f56f70`

#### Top Contributing to Fake News (3 Circles)

* Low Impact: `#f9cbc7`
* Medium Impact: `#f8a19e`
* High Impact: `#f25e5d`

Usage:

* Fake news detection banner
* Negative key indicators
* Sensational or misinformation token highlights

---

## 5. Key Indicators Section

Each indicator consists of:

* Label (left-aligned)
* SHAP Value (right-aligned)
* Horizontal bar visualization

### SHAP Value Number

* Background Color: `#e7e7e7`
* Rounded container
* Small but readable font

### Bar Rules

* Real-positive contributions use green palette.
* Fake-negative contributions use red palette.
* Bar length represents magnitude of contribution.

---

## 6. Summary Section

### Design

* Background: `#edf6ff`
* Border (outline): `#afcef0`
* Header/Text Accent: `#2958cb`
* Rounded corners
* Clear visual separation from indicators

Used to display:

* Primary reasoning
* Model explanation summary
* Key decision justification

---

## 7. Buttons

### Primary Button (e.g., Report Issue)

* Dark background (brand navy `#1e2c3e`)
* White text
* Rounded corners
* Clear hover feedback

### Secondary Buttons (Search / Actions)

* Clean, modern appearance
* Consistent spacing
* Rounded edges

---

## 8. History Page Design

### Search Field

* Rounded input box
* Clean border
* Prominent search button

### Record Cards

Each record includes:

* Status icon (green check or red warning)
* Article title
* Classification result
* Confidence percentage
* Date

Card appearance:

* White background
* Rounded corners
* Subtle border or shadow
* Consistent vertical spacing between cards

---

## 9. Visual Consistency Rules

All contributors must:

* Use consistent spacing between sections.
* Maintain consistent rounded corners across components.
* Use color strictly according to classification state.
* Avoid mixing green and red styles within the same classification context.
* Ensure sufficient contrast between text and background.

---

## 10. Accessibility Considerations

* Do not rely on color alone; include icons for Real and Fake states.
* Maintain strong contrast for red and green text.
* Keep all text readable within a 280px width constraint.

---

This style guide ensures consistency across:

* Article analysis popup
* Real/Fake classification states
* Indicator visualizations
* Token contribution sections
* History page
* Summary explanations

All UI changes must follow this document.
