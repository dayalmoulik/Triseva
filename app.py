import os
import sys
import tempfile
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Import core LangGraph engine
from main import ask

# ZeroGPU Compatibility: Hugging Face ZeroGPU requires at least one @spaces.GPU decorated function
try:
    import spaces
    def gpu_decorator(func):
        return spaces.GPU(func)
    print("  [ZeroGPU] spaces module detected and @spaces.GPU decorator initialized.")
except ImportError:
    def gpu_decorator(func):
        return func
    print("  [CPU Mode] spaces module not present, running standard CPU mode.")

@gpu_decorator
def process_query(user_message, image, domain_choice):
    if not user_message and image is None:
        return "Please enter a question or upload an image to analyze."
    
    image_path = None
    if image is not None:
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "input_image.png")
        image.save(image_path)
    
    override = None if domain_choice == "Auto-Detect" else domain_choice.lower()
    if override == "health":
        override = "health"
    elif override == "legal":
        override = "legal"
    elif override == "agriculture":
        override = "agriculture"

    query_str = user_message.strip() if user_message and user_message.strip() else "Analyze the text and information in the uploaded image."
    
    try:
        res = ask(
            query=query_str,
            session_id="hf_space_session",
            image_path=image_path,
            domain_override=override
        )
        
        answer = res.get("answer", "No answer generated.")
        domain = str(res.get("domain", "unknown")).upper()
        score = res.get("score")
        score_str = f"{score:.2%}" if isinstance(score, (int, float)) else "N/A"
        
        sources_list = res.get("sources", [])
        sources_md = ""
        if sources_list:
            sources_md = "\n\n### 📚 Sources & Attribution:\n" + "\n".join([f"- {s}" for s in sources_list])
            
        telemetry = res.get("telemetry", {})
        latency = telemetry.get("latency", 0.0)
        
        footer = f"\n\n---\n**Domain Routed**: `{domain}` | **Faithfulness Score**: `{score_str}` | **Latency**: `{latency}s`"
        
        return answer + sources_md + footer
    except Exception as e:
        return f"❌ An error occurred during execution: {str(e)}"

# Define custom Gradio interface
demo = gr.Interface(
    fn=process_query,
    inputs=[
        gr.Textbox(
            lines=3,
            placeholder="Type your healthcare, legal/government scheme, or agriculture question in English, Hindi, or Hinglish...",
            label="User Question"
        ),
        gr.Image(type="pil", label="📷 Upload Document / Prescription / Leaf Image (Optional)"),
        gr.Dropdown(
            choices=["Auto-Detect", "Health", "Legal", "Agriculture"],
            value="Auto-Detect",
            label="🎯 Domain Scoping / Intent Override"
        )
    ],
    outputs=gr.Markdown(label="🤖 TriSeva Response & Sources"),
    title="🏥 ⚖️ 🌾 TriSeva: Multi-Agent AI Assistant",
    description="**TriSeva** is a multi-agent retrieval-augmented generation assistant for Healthcare, Legal/Government Schemes, and Agriculture domains with multimodal OCR and factuality verification.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
