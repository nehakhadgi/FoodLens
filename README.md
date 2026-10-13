# FoodLens
FoodLens is a multimodal meal coach that analyzes a meal photo with Qwen3-VL inside Google Colab.

## What it does
- Describes the likely dish and visible ingredients.
- Uses dietary preferences and portion context to shape its guidance.
- Separates visible evidence, estimates, and unknowns.

## Requirements
- A Google account with access to Google Colab.
- A meal photo that contains no private information.
- An optional Colab GPU for faster inference.

## Dependencies
`transformers==5.19.0`, `qwen-vl-utils==0.0.14`, `accelerate==1.15.0`, and `gradio==6.30.0`.

## Run in Colab
1. Open `FoodLens.ipynb` in Google Colab.
2. Select Colab runtime version `2026.07`.
3. Run the notebook cells in order.
4. Upload a privacy-safe meal photo.
5. Add dietary preferences and portion context.
6. Click **Analyze meal** to generate the report.

## Important limitations
FoodLens produces model-generated estimates from incomplete visual evidence.
Do not use its output as medical, allergy, or food-safety advice.
## Project file
`FoodLens.ipynb` contains the model setup, inference pipeline, interface, tests, and cautious meal-coach prompt.
