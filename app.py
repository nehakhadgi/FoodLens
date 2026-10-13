import torch

# Report whether this runtime can use CUDA acceleration.
print("CUDA available:", torch.cuda.is_available())

# Name the GPU when Colab assigned one to this session.
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Device: CPU")

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch

# Keep every model call pointed at the same repository.
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
)
print("Model device:", model.device)

def prepare_inputs(image, prompt):
    # Pair the meal image with the visitor's request.
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    # Format the conversation and extract visual inputs.
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # Convert both modalities into tensors on the model device.
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(model.device)

print("Input preparation ready.")

def run_model(image, prompt):
    # Return a useful message before attempting inference.
    if image is None:
        return "Please upload a meal photo."

    inputs = prepare_inputs(image, prompt)

    # Generate only the response tokens, then decode them as text.
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    trimmed_ids = [
        output[len(source):]
        for source, output in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
    )[0]

print("FoodLens inference ready.")

import gradio as gr

# Build a notebook interface around the inference function.
with gr.Blocks(title="FoodLens") as demo:
    gr.Markdown("# FoodLens")
    image_input = gr.Image(type="pil", label="Meal photo")
    prompt_input = gr.Textbox(label="Prompt")
    describe_button = gr.Button("Describe food")
    output = gr.Markdown()

    # Send both visitor inputs through the model when clicked.
    describe_button.click(
        fn=run_model,
        inputs=[image_input, prompt_input],
        outputs=output,
    )

demo.launch(debug=True)

def build_meal_prompt(preferences, portion):
    # Replace empty fields with explicit context limits.
    preferences = preferences.strip() or "No preferences supplied."
    portion = portion.strip() or "No portion context supplied."

    # Require a predictable report with visible safety boundaries.
    return f"""
Act as a cautious meal-photo assistant.
Dietary preferences: {preferences}
Portion context: {portion}

Return Markdown with exactly these headings:
## What I can see
List only visible foods and evidence.
## Likely meal
Label uncertain ingredients as likely or possible.
## Preference fit
Never guarantee that an allergen is absent from an image.
## Nutrition estimate
Use only rough ranges and state every portion assumption.
## Recipe adaptation
Suggest one practical adaptation for the preferences.
## Uncertainty and safety
List hidden ingredients, preparation, portion mass, cross-contact,
and medical suitability as limits. State that this is not medical advice.
""".strip()

print(build_meal_prompt("vegetarian", "one bowl").splitlines()[0])

def analyze_meal(image, preferences, portion):
    # Turn visitor context into the grounded report request.
    prompt = build_meal_prompt(preferences, portion)
    return run_model(image, prompt)

print("Cautious analysis ready.")

# Build the full meal-coach interface inside the notebook.
with gr.Blocks(title="FoodLens") as demo:
    gr.Markdown(
        "# FoodLens\nResults are estimates, not medical advice."
    )
    image_input = gr.Image(type="pil", label="Meal photo")
    preferences_input = gr.Textbox(
        label="Dietary preferences or restrictions"
    )
    portion_input = gr.Textbox(label="Portion context")
    analyze_button = gr.Button("Analyze meal")
    output = gr.Markdown()

    # Send the image and both context fields to one callback.
    analyze_button.click(
        fn=analyze_meal,
        inputs=[image_input, preferences_input, portion_input],
        outputs=output,
    )

demo.launch(debug=True)

def audit_response(image, draft):
    # Challenge the first report against the original photo.
    audit_prompt = f"""
Audit the meal analysis below against the image.

Classify important claims under exactly these headings:
### Directly visible
Claims supported by visible evidence.
### Inferred
Plausible claims that are not directly visible.
### Unknown or unsupported
Claims the image cannot establish, especially quantities,
hidden ingredients, allergens, preparation, and medical suitability.

Do not add new nutrition numbers.

Draft to audit:
{draft}
""".strip()
    return run_model(image, audit_prompt)

print("Evidence audit ready.")

def analyze_meal(image, preferences, portion):
    # Create the cautious first-pass meal report.
    prompt = build_meal_prompt(preferences, portion)
    draft = run_model(image, prompt)
    if image is None:
        return draft

    # Audit the draft against the same image before returning it.
    audit = audit_response(image, draft)
    return f"{draft}\n\n---\n\n## Evidence audit\n\n{audit}"

print("Two-pass analysis ready.")

